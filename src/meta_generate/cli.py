import logging

import asyncio
from typing import List
import dspy
from lib.custom_lm.lms import Lm_Glm
from lib.dspy_utils import list_tools, load_env_variable
from lib.utils import parse_args
from fastmcp import Client

from meta_generate.plan_executor import PlanExecutor, generate_plan

from meta_generate.signatures import GetTableSchemas
from meta_generate.utils import (
    generate_mock_function,
    insert_mock_data,
)

logging.basicConfig(level=logging.INFO)
mcp_client = Client("http://127.0.0.1:8999/mcp")


def build_tool_desc(tool_registry: dict[str | None, str | None]) -> str:
    desc_lines = ["Available tools:"]
    for name, desc in tool_registry.items():
        if name is not None and desc is not None:
            desc_lines.append(f"- {name}: {desc}")
        else:
            raise ValueError("tool name and desc is required")
    return "\n".join(desc_lines)


async def exe_plan():
    # Allow sync tool calls to execute async implementations (e.g., insert_mock_data)
    mcp_client = Client("http://127.0.0.1:8999/mcp")
    async with mcp_client:
        tools = await list_tools(mcp_client)

        # fetch schemas
        action = dspy.ReAct(GetTableSchemas, tools=tools)
        schema_res = await action.acall()
        logging.info("Schema Retrieval Result:\n%s", schema_res)

        # generate plan
        TOOL_DESC = {tool.name: tool.desc for tool in tools}
        TOOL_DESC.update(
            {
                "generate_mock_function": generate_mock_function.__doc__,
                "insert_mock_data": insert_mock_data.__doc__,
            }
        )
        logging.info(TOOL_DESC)
        tool_desc = build_tool_desc(TOOL_DESC)
        # args = parse_args()
        user_request = "Generate mock data for all tables based on the retrieved schemas, respecting foreign key constraints, and insert them into the database. Use the available tools only."
        plan_json = generate_plan(user_request, tool_desc, schema_res.schemas)
        print("Plan:\n", plan_json)

        TOOL_REGISTRY = {tool.name: tool for tool in tools if tool.name is not None}
        TOOL_REGISTRY["generate_mock_function"] = dspy.Tool(generate_mock_function)
        TOOL_REGISTRY["insert_mock_data"] = dspy.Tool(insert_mock_data)
        executor = PlanExecutor(TOOL_REGISTRY)
        final_context = executor.execute_plan(plan_json)
        logging.info(f"Final Execution Context: {final_context}")


def run():
    load_env_variable()
    dspy.settings.allow_tool_async_sync_conversion = True
    dspy.configure(lm=Lm_Glm)
    dspy.configure(show_guidelines=True)
    asyncio.run(exe_plan())
