from __future__ import annotations
import dspy


class PlanSignature(dspy.Signature):
    """将复杂目标分解为子任务列表。输出必须是纯 JSON 数组，每个元素为 {"description": "子任务描述"}。"""

    goal = dspy.InputField(desc="用户的原始目标")
    available_tools = dspy.InputField(desc="可用工具列表及其功能")
    sub_tasks_json = dspy.OutputField(
        desc='JSON array of tasks, e.g., [{"description": "读取配置文件"}, {"description": "转换格式"}]'
    )


class SubTaskInput(dspy.Signature):
    """为 ReAct Agent 提供带上下文的子任务输入。"""

    context_summary = dspy.InputField(desc="先前任务的执行结果摘要（可为空）")
    sub_task = dspy.InputField(desc="当前要执行的子任务描述")
    result = dspy.OutputField(desc="子任务的答案或结果")


class SummarySignature(dspy.Signature):
    """总结所有子任务的执行日志，生成最终输出。"""

    execution_log = dspy.InputField(desc="所有子任务的执行日志列表")
    final_output = dspy.OutputField(desc="总结后的最终输出结果")
