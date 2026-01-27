# plan_executor.py
from email.policy import default
import json
import re
from typing import Any, Dict, List, Set

# plan_executor.py
from typing import Dict, Any, Callable

# dspy_plan.py
import dspy
from mcp import Tool

"""
{
  "steps": [
    {
      "id": "step_users",
      "tool": "generate_users",
      "args": {"n": 10}
    },
    {
      "id": "step_products",
      "tool": "generate_products",
      "args": {"n": 5}
    },
    {
      "id": "step_orders",
      "tool": "generate_orders",
      "args": {
        "user_ids": "@step_users.id_list",
        "product_ids": "@step_products.id_list",
        "n": 20
      }
    }
  ]
}
"""


# 明确标注：函数接受任意 kwargs，返回任意可序列化对象
ToolFunction = Callable[..., Any]


class GenerateDAGPlan(dspy.Signature):
    """
    create a DAG plan in JSON format with provided utils to generate mock data for all the tables and insert the generated mock data into the database.
    - ** respecting foreign key constraints **
    - ** Use provided tools only **
    """

    user_request: str = dspy.InputField(
        default="Generate mock data for all tables based on the retrieved schemas, respecting foreign key constraints, and insert them into the database. Use the available tools only."
    )
    database_schema: str = dspy.InputField(
        desc="Database schema, including tables, columns, data types, and foreign key relationships."
    )
    tool_descriptions: str = dspy.InputField(
        desc="Available tools with their names and descriptions."
    )
    plan: str = dspy.OutputField(
        desc="JSON with 'steps' array. Each step has 'id', 'tool', 'args', 'desc'. Use @<step_id>.<field> to reference results."
    )


def generate_plan(user_request: str, tool_descriptions: str, database_schema: str):
    predictor = dspy.ChainOfThought(GenerateDAGPlan)
    response = predictor(
        user_request=user_request,
        tool_descriptions=tool_descriptions,
        database_schema=database_schema,
    )
    return response.plan


class PlanExecutor:
    def __init__(self, tool_registry: Dict[str, ToolFunction]):
        self.tools = tool_registry
        self.context = {}  # step_id -> result_dict

    def execute_plan(self, plan_json: str) -> Any:
        plan = json.loads(plan_json)
        steps = plan["steps"]

        # 构建依赖图（统一将 step_id 规范为字符串，避免 int/str 混用导致的 KeyError）
        step_map = {str(step["id"]): step for step in steps}
        dependencies = self._build_dependencies(steps)

        # 拓扑排序
        execution_order = self._topological_sort(step_map.keys(), dependencies)

        # 按顺序执行
        for step_id in execution_order:
            step = step_map[step_id]
            resolved_args = self._resolve_args(step["args"])
            result = self.tools[step["tool"]](**resolved_args)
            self.context[step_id] = result
            print(f"✅ Executed {step_id}: {result.get('count', 'done')}")

        return self.context

    def _build_dependencies(self, steps: list) -> Dict[str, Set[str]]:
        """构建 step_id -> {依赖的 step_id} 的映射"""
        deps = {str(step["id"]): set() for step in steps}
        for step in steps:
            for value in self._extract_references(step["args"]):
                if value.startswith("@"):
                    ref_step = value.split(".")[0][1:]  # "@users.id_list" → "users"
                    deps[str(step["id"])].add(str(ref_step))
        return deps

    def _extract_references(self, obj):
        """递归提取所有字符串值中的 @ 引用"""
        refs = []
        if isinstance(obj, dict):
            for v in obj.values():
                refs.extend(self._extract_references(v))
        elif isinstance(obj, list):
            for item in obj:
                refs.extend(self._extract_references(item))
        elif isinstance(obj, str) and obj.startswith("@"):
            refs.append(obj)
        return refs

    def _topological_sort(self, nodes, dependencies):
        """Kahn's algorithm for topological sort"""
        from collections import deque

        indegree = {node: 0 for node in nodes}
        for step, deps in dependencies.items():
            for dep in deps:
                if dep not in indegree:
                    raise ValueError(f"Plan references undefined step id '{dep}'")
                indegree[step] += 1

        queue = deque([n for n in nodes if indegree[n] == 0])
        order = []

        while queue:
            node = queue.popleft()
            order.append(node)
            for other in nodes:
                if node in dependencies[other]:
                    indegree[other] -= 1
                    if indegree[other] == 0:
                        queue.append(other)

        if len(order) != len(nodes):
            raise ValueError("Circular dependency detected")
        return order

    def _resolve_args(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """解析参数中的 @step_id.field 引用"""

        def resolve_value(value):
            if isinstance(value, str) and value.startswith("@"):
                parts = value[1:].split(
                    ".", 1
                )  # "@step_id.field" → ["step_id", "field"]
                step_id = str(parts[0])
                field = parts[1] if len(parts) > 1 else "result"

                if step_id not in self.context:
                    raise ValueError(f"Step {step_id} not executed yet")

                result = self.context[step_id]
                if isinstance(result, dict):
                    if field not in result:
                        raise ValueError(
                            f"Field '{field}' not found in step {step_id} result"
                        )
                    return result[field]
                elif field == "result":
                    return result
                else:
                    raise ValueError(
                        f"Cannot access field '{field}' on non-dict result"
                    )
            return value

        resolved = {}
        for k, v in args.items():
            if isinstance(v, dict):
                resolved[k] = {subk: resolve_value(subv) for subk, subv in v.items()}
            elif isinstance(v, list):
                resolved[k] = [resolve_value(item) for item in v]
            else:
                resolved[k] = resolve_value(v)
        return resolved
