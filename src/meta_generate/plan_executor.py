# plan_executor.py
import json
import logging
import re
from typing import Any, Callable, Dict, List, Set

from pydantic import BaseModel, Field, ValidationError, model_validator

# dspy_plan.py
import dspy
from mcp import Tool

from meta_generate.signatures import GenerateDAGPlan, PlanModel, PlanStep


# 明确标注：函数接受任意 kwargs，返回任意可序列化对象
ToolFunction = Callable[..., Any]


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

    async def execute_plan_async(self, plan_json: Any) -> Any:
        try:
            if isinstance(plan_json, PlanModel):
                plan_obj = plan_json
            elif isinstance(plan_json, (dict, list)):
                plan_obj = PlanModel.model_validate(plan_json)
            else:
                plan_obj = PlanModel.model_validate_json(plan_json)
        except ValidationError as ve:
            logging.error("Invalid plan: %s", ve)
            raise

        steps = plan_obj.steps

        # 构建依赖图（统一将 step_id 规范为字符串，避免 int/str 混用导致的 KeyError）
        step_map = {step.id: step for step in steps}
        dependencies = self._build_dependencies(steps)

        # 拓扑排序
        execution_order = self._topological_sort(step_map.keys(), dependencies)

        # 按顺序执行
        for step_id in execution_order:
            step = step_map[step_id]
            resolved_args = self._resolve_args(step.args)

            tool = self.tools[step.tool]
            if hasattr(tool, "acall"):
                result = await tool.acall(**resolved_args)
            else:
                result = tool(**resolved_args)

            self.context[step_id] = result

            # Some tools return plain strings instead of dicts; be defensive to avoid AttributeError.
            if isinstance(result, dict):
                summary = result.get("count", result.get("status", "done"))
            else:
                summary = str(result)
            print(f"✅ Executed {step_id}: {summary}")

        return self.context

    def execute_plan(self, plan_json: Any) -> Any:
        """Sync wrapper that runs the async execution. Avoid when already in an event loop."""
        import asyncio

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            raise RuntimeError(
                "execute_plan must be awaited in an async context; call execute_plan_async instead."
            )

        return asyncio.run(self.execute_plan_async(plan_json))

    @staticmethod
    def _extract_step_id(ref: str) -> str | None:
        """Return step id from @ref strings, extracting trailing digits when present."""
        if not isinstance(ref, str):
            return None
        at_pos = ref.find("@")
        if at_pos == -1:
            return None
        candidate = ref[at_pos:]
        m = re.match(r"@\{?([^\.\s\[\}]+)\}?", candidate)
        if m:
            token = m.group(1)
            digit_match = re.search(r"(\d+)", token)
            return digit_match.group(1) if digit_match else token
        return None

    def _build_dependencies(self, steps: List[PlanStep]) -> Dict[str, Set[str]]:
        """构建 step_id -> {依赖的 step_id} 的映射"""
        deps = {step.id: set() for step in steps}
        for step in steps:
            for value in self._extract_references(step.args):
                ref_step = self._extract_step_id(value)
                if ref_step:
                    logging.info(f"Step {step.id} depends on step {ref_step}")
                    deps[step.id].add(str(ref_step))
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
                step_id = self._extract_step_id(value)
                field = value.split(".", 1)[1] if "." in value else "result"

                if not step_id or step_id not in self.context:
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
