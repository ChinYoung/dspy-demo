# schema.py
from typing import Literal, Optional, Dict, List, Union
from pydantic import BaseModel, Field, field_validator, validator


class FieldRule(BaseModel):
    type: Literal[
        "auto_increment",
        "uuid",
        "string",
        "email",
        "phone",
        "integer",
        "float",
        "boolean",
        "datetime",
        "enum",
        "reference",
    ]
    # 通用配置
    unique: bool = False
    nullable: bool = False

    # 类型特定配置
    pattern: Optional[str] = None  # string: "user_{idx}"
    min_value: Optional[Union[int, float]] = None
    max_value: Optional[Union[int, float]] = None
    enum_values: Optional[List[str]] = None
    weights: Optional[List[int]] = None  # 与enum_values对应

    # 关联配置 (type="reference")
    ref_table: Optional[str] = None
    ref_field: Optional[str] = None  # default: primary key of ref_table


class TableRule(BaseModel):
    name: str
    count: int = Field(..., ge=1)
    fields: Dict[str, FieldRule]
    primary_key: str = "id"

    @field_validator("fields")
    def validate_primary_key_exists(cls, v, values):
        if values.get("primary_key") not in v:
            raise ValueError(f"primary_key '{values.get('primary_key')}' not in fields")
        return v


class MockDataSpec(BaseModel):
    tables: List[TableRule]
    database_type: Literal["postgresql", "mysql"] = "postgresql"
