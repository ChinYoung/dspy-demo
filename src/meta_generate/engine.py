# engine.py
import uuid
import random
import string
from datetime import datetime, timedelta
from typing import Dict, List, Any, Set
from faker import Faker
from sqlalchemy import create_engine, text
from sqlalchemy.exc import IntegrityError
from .schema import MockDataSpec, TableRule, FieldRule


class MockDataEngine:
    def __init__(self, db_url: str, locale: str = "en_US"):
        self.engine = create_engine(db_url)
        self.faker = Faker(locale)
        self.generated_ids: Dict[str, List[Any]] = {}  # table_name -> [ids]
        self.unique_values: Dict[str, Set[str]] = {}  # "table.field" -> {values}

    def generate(self, spec: MockDataSpec):
        # 1. 拓扑排序（先父表后子表）
        tables = self._topological_sort(spec.tables)

        # 2. 逐表生成并插入
        for table_rule in tables:
            records = self._generate_table(table_rule, spec.tables)
            self._insert_batch(table_rule.name, records)
            self.generated_ids[table_rule.name] = [
                r[table_rule.primary_key] for r in records
            ]
            print(f"✓ Generated {len(records)} records for {table_rule.name}")

    def _topological_sort(self, tables: List[TableRule]) -> List[TableRule]:
        """简单拓扑排序：无依赖表优先"""
        table_map = {t.name: t for t in tables}
        sorted_tables = []
        remaining = set(table_map.keys())

        while remaining:
            progress = False
            for name in list(remaining):
                table = table_map[name]
                # 检查是否有未生成的依赖表
                has_unresolved_ref = any(
                    field.ref_table in remaining
                    for field in table.fields.values()
                    if field.type == "reference" and field.ref_table
                )
                if not has_unresolved_ref:
                    sorted_tables.append(table)
                    remaining.remove(name)
                    progress = True
            if not progress:
                raise ValueError(f"Circular dependency detected in tables: {remaining}")
        return sorted_tables

    def _generate_table(
        self, table_rule: TableRule, all_tables: List[TableRule]
    ) -> List[Dict[str, Any]]:
        records = []
        table_name = table_rule.name

        for idx in range(table_rule.count):
            record = {}
            for field_name, field_rule in table_rule.fields.items():
                key = f"{table_name}.{field_name}"
                value = self._generate_field(field_rule, idx, table_rule, all_tables)

                # 唯一性校验（本地维护集合）
                if field_rule.unique:
                    attempt = 0
                    while value in self.unique_values.get(key, set()) and attempt < 10:
                        value = self._generate_field(
                            field_rule, idx + attempt + 1, table_rule, all_tables
                        )
                        attempt += 1
                    if attempt >= 10:
                        raise ValueError(
                            f"Failed to generate unique value for {key} after 10 attempts"
                        )
                    self.unique_values.setdefault(key, set()).add(value)

                record[field_name] = value
            records.append(record)
        return records

    def _generate_field(
        self,
        rule: FieldRule,
        idx: int,
        table_rule: TableRule,
        all_tables: List[TableRule],
    ) -> Any:
        if rule.type == "auto_increment":
            # 从数据库获取当前最大ID（避免冲突）
            with self.engine.connect() as conn:
                result = conn.execute(
                    text(f"SELECT MAX({table_rule.primary_key}) FROM {table_rule.name}")
                ).scalar()
                base = (result or 0) + 1
            return base + idx

        elif rule.type == "uuid":
            return str(uuid.uuid4())

        elif rule.type == "string":
            if rule.pattern:
                return rule.pattern.format(
                    idx=idx, random="".join(random.choices(string.ascii_lowercase, k=5))
                )
            return self.faker.word()

        elif rule.type == "email":
            base = self.faker.user_name()
            domain = self.faker.free_email_domain()
            return f"{base}.{idx}@{domain}"

        elif rule.type == "phone":
            return self.faker.phone_number()

        elif rule.type == "integer":
            min_v = rule.min_value or 0
            max_v = rule.max_value or 1000
            return random.randint(int(min_v), int(max_v))

        elif rule.type == "float":
            min_v = rule.min_value or 0.0
            max_v = rule.max_value or 100.0
            return round(random.uniform(float(min_v), float(max_v)), 2)

        elif rule.type == "boolean":
            return random.choice([True, False])

        elif rule.type == "datetime":
            start = datetime(2020, 1, 1)
            end = datetime.now()
            if rule.min_value:
                start = datetime.fromisoformat(str(rule.min_value))
            if rule.max_value:
                end = datetime.fromisoformat(str(rule.max_value))
            delta = end - start
            return start + timedelta(
                seconds=random.randint(0, int(delta.total_seconds()))
            )

        elif rule.type == "enum":
            if not rule.enum_values:
                raise ValueError("enum_values required for enum type")
            if rule.weights:
                return random.choices(rule.enum_values, weights=rule.weights, k=1)[0]
            return random.choice(rule.enum_values)

        elif rule.type == "reference":
            if not rule.ref_table:
                raise ValueError("ref_table required for reference type")
            # 从已生成的父表ID池中随机选取
            ref_ids = self.generated_ids.get(rule.ref_table)
            if not ref_ids:
                raise ValueError(
                    f"Reference table {rule.ref_table} not generated yet (dependency order error)"
                )
            return random.choice(ref_ids)

        else:
            raise ValueError(f"Unsupported field type: {rule.type}")

    def _insert_batch(
        self, table_name: str, records: List[Dict[str, Any]], batch_size: int = 500
    ):
        if not records:
            return

        with self.engine.begin() as conn:  # 自动事务
            for i in range(0, len(records), batch_size):
                batch = records[i : i + batch_size]
                columns = list(batch[0].keys())
                values = [tuple(r[col] for col in columns) for r in batch]

                # PostgreSQL / MySQL 兼容的批量插入
                placeholders = ",".join(
                    ["(" + ",".join(["%s"] * len(columns)) + ")"] * len(batch)
                )
                sql = f"""
                    INSERT INTO {table_name} ({','.join(columns)})
                    VALUES {placeholders}
                    ON CONFLICT DO NOTHING  -- PostgreSQL
                """
                try:
                    conn.execute(text(sql), *sum(values, ()))
                except Exception as e:
                    if "ON CONFLICT" in str(e) and "mysql" in str(self.engine.url):
                        # MySQL 语法回退
                        sql = f"""
                            INSERT IGNORE INTO {table_name} ({','.join(columns)})
                            VALUES {placeholders}
                        """
                        conn.execute(text(sql), *sum(values, ()))
                    else:
                        raise
