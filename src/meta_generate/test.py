import logging
import json
import dspy
from fastmcp import Client
from lib.custom_lm.lms import Lm_Glm
from lib.dspy_utils import list_tools, init_dspy
from meta_generate.plan_executor import PlanExecutor
from meta_generate.utils import (
    generate_mock_function,
    insert_mock_data,
)
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


async def run_plan():
    mcp_client = Client("http://127.0.0.1:8999/mcp")
    SCRIPT_DIR = Path(__file__).parent.resolve()
    logging.info(f"Loading generated plan from {SCRIPT_DIR / 'generated.json'}")
    dspy.configure(show_guidelines=True)
    dspy.configure(lm=Lm_Glm)
    async with mcp_client:
        with open(SCRIPT_DIR / "generated.json", "r", encoding="utf-8") as f:
            generated_plan_json = f.read()
            tools = await list_tools(mcp_client)
            TOOL_REGISTRY = {tool.name: tool for tool in tools if tool.name is not None}
            TOOL_REGISTRY["generate_mock_function"] = dspy.Tool(generate_mock_function)
            TOOL_REGISTRY["insert_mock_data"] = dspy.Tool(insert_mock_data)
            executor = PlanExecutor(TOOL_REGISTRY)
            final_context = executor.execute_plan(generated_plan_json)
            logging.info(f"Final Execution Context: {final_context}")


def run():
    import asyncio

    asyncio.run(run_plan())
