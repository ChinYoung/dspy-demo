import logging

import asyncio
from dotenv import load_dotenv
import dspy
from sqlalchemy import desc
from lib.custom_lm.lms import Lm_Glm
from lib.dspy_utils import list_tools, load_env_variable
from lib.utils import parse_args
from fastmcp import Client

from meta_generate.plan_executor import  generate_plan
from meta_generate.signatures import CodeGenerateRequest
import re

from meta_generate.utils import _execute_generated_func, generate_with_mock_func, generate_mock_function, insert_mock_data

logging.basicConfig(level=logging.INFO)
mcp_client = Client("http://127.0.0.1:8999/mcp")


async def generate_mata():
    async with mcp_client:
        dspy.configure(lm=Lm_Glm)
        dspy.configure(show_guidelines=True)
        args = parse_args()

        tools = await list_tools(mcp_client)
        all_tools = tools 
        logging.info(f"User request: {args.user_request}")

        ask = dspy.ReAct(CodeGenerateRequest, tools=all_tools)
        res = ask(
            user_requirements=args.user_request,
            description="do not send generated mock data to LM, only generate and execute the mock function code",
        )
        return res.generated_code


async def main():
    res = await generate_mata()
    # extract first fenced code block (optionally with language)
    m = re.search(r"```(?:\w+\s*)?\n(.*?)```", res, re.S)
    raw_code = m.group(1).strip() if m else res.strip()

    logging.info("Generated Code:")
    logging.info(raw_code)

    mock_res = generate_with_mock_func(raw_code, 10)
    logging.info(f"Generated {len(mock_res)} mock records.")


def run():
    load_env_variable()
    asyncio.run(exe_plan())


async def exe_plan():
    async with mcp_client:
        dspy.configure(lm=Lm_Glm)
        dspy.configure(show_guidelines=True)
        args = parse_args()

        tools = await list_tools(mcp_client)
        all_tools = tools 
        TOOL_REGISTRY = {tool.name: tool for tool in all_tools}
        TOOL_REGISTRY.update({
            "generate_mock_function": generate_mock_function,
            "generate_with_mock_func": generate_with_mock_func,
            "insert_mock_data": insert_mock_data
        })
        logging.info(TOOL_REGISTRY)

        plan_json = generate_plan(args.user_request, tools)
        print("Plan:\n", plan_json)

        # executor = PlanExecutor(TOOL_REGISTRY)
        # final_context = executor.execute_plan(plan_json)
        # logging.info("Final Execution Context:")
