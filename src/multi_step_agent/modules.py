# modules.py
import json
import logging
import dspy
from fastmcp import Client
from lib import dspy_utils
from lib.custom_lm.lms import Lm_Glm
from mcp_demo.dspy_tools import DSPyAirlineCustomerService
from multi_step_agent.schemas import PlanSignature, SubTaskInput, SummarySignature
from typing import List, Tuple


client = Client("http://127.0.0.1:8999/mcp")


class PlanThenReAct(dspy.Module):
    """
    标准 DSPy Module：使用 TypedPredictor 结构化规划 + ReAct 执行。

    输入: goal (str)
    输出: final_output (str), sub_tasks (List[str]), execution_log (List[str])
    """

    def __init__(self, max_subtasks: int = 6, max_react_iters: int = 5):
        super().__init__()
        dspy_utils.init_dspy(Lm_Glm)
        self.max_subtasks = max_subtasks

    def _build_context_summary(self, completed: List[Tuple[str, str]]) -> str:
        if not completed:
            return ""
        lines = ["先前已完成的任务及结果："]
        for desc, result in completed:
            lines.append(f"任务: {desc}")
            lines.append(f"结果: {result}")
            lines.append("---")
        return "\n".join(lines)

    def _parse_plan_output(self, raw_json: str) -> List[str]:
        """安全解析 JSON 输出"""
        try:
            data = json.loads(raw_json)
            if isinstance(data, list):
                return [
                    item["description"]
                    for item in data
                    if isinstance(item, dict) and "description" in item
                ][: self.max_subtasks]
        except Exception as e:
            print(f"[Plan Parse Error] {e}")
        return []

    async def forward(self, goal: str):
        async with client:
            tools = await client.list_tools()
            logging.info(f"Available tools: {tools}")
            tool_desc = [tool.description for tool in tools]
            logging.info(f"Tool descriptions: {tool_desc}")
            dspy_tools: List[dspy.Tool] = []
            for tool in tools:
                dspy_tools.append(dspy.Tool.from_mcp_tool(client.session, tool))
            # === Step 1: 结构化 Planning ===
            planner = dspy.Predict(PlanSignature)
            subtask_react = dspy.ReAct(SubTaskInput, tools=dspy_tools)
            plan = planner(goal=goal, available_tools=tool_desc)
            logging.info(f"Generated plan: {plan.sub_tasks_json}")
            sub_tasks = self._parse_plan_output(plan.sub_tasks_json)
            # 提取描述列表（限制数量）
            sub_task_descs = [task_desc for task_desc in sub_tasks[: self.max_subtasks]]

            # === Step 2: 逐个执行子任务 ===
            completed: List[Tuple[str, str]] = []
            execution_log: List[str] = []

            for desc in sub_task_descs:
                # 构建带上下文的问题
                context_summary = self._build_context_summary(completed)
                task_res = await subtask_react.acall(
                    context_summary=context_summary, sub_task=desc
                )
                result = task_res.result
                # 记录
                completed.append((desc, result))
                execution_log.append(f"✅ {desc} → {result.strip()}")

        # # === Step 3: 返回结构化预测 ===
        summary = dspy.Predict(SummarySignature)
        summary_res = summary(execution_log=execution_log)
        logging.info(f"Final summary: {summary_res.final_output}")
        return summary_res.final_output
