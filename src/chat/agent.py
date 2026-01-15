from email.policy import default
from gc import collect
from typing import List, Dict
import dspy
from dspy import Module
import dspy.signatures

from lib import dspy_utils
from lib.custom_lm.lms import Lm_Glm
from fastmcp import Client
import logging


client = Client("http://127.0.0.1:8999/mcp")


class AssessCompleteness(dspy.Signature):
    """You are an assistant. Your task is to
    1. assess whether the information and tools provided by the user is sufficient to execute the task.
    2. collect information for task execution as necessary and possible.
    If the information is sufficient, set is_complete to True.
    If not, ask a clear and concise question to obtain the missing information.
    """

    user_request: str = dspy.InputField(desc="The user's original request")
    conversation_history: str = dspy.InputField(
        desc="The conversation history with the user"
    )

    question_to_user: str = dspy.OutputField(
        desc="If current information is insufficient for task execution, ask a clear question for user to provide more info",
        default="",
    )
    collected_info: str = dspy.OutputField(
        desc="The information collected from the user so far", default=""
    )
    is_complete: bool = dspy.OutputField(
        desc="True if current information is sufficient, else False", default=False
    )


class ExecuteTask(dspy.Signature):
    """you are an assistant agent. Your task is to execute the user's request based on the provided information."""

    user_request: str = dspy.InputField(desc="The user's original request")
    conversation_history: str = dspy.InputField(
        desc="The conversation history with the user"
    )
    result: str = dspy.OutputField()


class InteractiveAgent(Module):
    def format_conversation(
        self,
        conversation: List[Dict[str, str]],
    ) -> str:
        formatted = []
        for turn in conversation:
            formatted.append(
                f"agent: {turn['agent']}, user: {turn['user']}, collected_info: {turn['collected_info']}"
            )
        return "\n".join(formatted)

    async def forward(self, task_goal: str):
        dspy_utils.init_dspy(Lm_Glm)
        dspy_tools = []
        logging.info(f"Starting interactive agent for task: {task_goal}")
        async with client:
            dspy_tools = await dspy_utils.list_tools(client)
            available_tools = str(dspy_tools)
            logging.info(f"Available tools: {available_tools}")
            conversation = []

            assess = dspy.ReAct(AssessCompleteness, tools=dspy_tools)
            execute = dspy.ReAct(ExecuteTask, tools=dspy_tools)

            while True:
                user_inputs_str = (
                    self.format_conversation(conversation)
                    if conversation
                    else "(first run, no conversation history yet)"
                )
                # Step 1: 判断信息是否完整
                assessment = await assess.acall(
                    user_request=task_goal,
                    conversation_history=user_inputs_str,
                )

                if not assessment.is_complete:
                    print(f"🤖 Agent: {assessment.question_to_user}")
                    user_reply = input("👤 You: ")
                    conversation.append(
                        {
                            "agent": assessment.question_to_user,
                            "user": user_reply,
                            "collected_info": assessment.collected_info,
                        }
                    )
                else:
                    # Step 3: 信息完整，规划工具调用参数
                    plan = await execute.acall(
                        user_request=task_goal, conversation_history=user_inputs_str
                    )
                    print(f"🤖 Agent: Task completed. Result: {plan.result}")
                    return plan
