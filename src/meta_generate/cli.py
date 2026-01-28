import json
import logging

import asyncio
from pathlib import Path
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

        # Parse schemas to extract foreign key information
        try:
            schemas_dict = json.loads(schema_res.schemas)
        except (json.JSONDecodeError, AttributeError):
            schemas_dict = {}

        # Build foreign key column mapping for each table
        fk_columns_map = {}
        for table_name, table_info in schemas_dict.items():
            if isinstance(table_info, dict):
                fk_columns = table_info.get("foreign_keys", {})
                fk_columns_map[table_name] = fk_columns

        # generate plan
        TOOL_DESC = {tool.name: tool.desc for tool in tools}
        TOOL_DESC.update(
            {
                "generate_mock_function": (
                    "Generate a mock data function for a table. "
                    "Parameters: table_name (str), schema (dict), fk_deps (list), fk_columns (dict), n_example (int). "
                    "Returns: dict with 'code' (generated function code) and 'table' (table name). "
                    "fk_columns should map foreign key column names to their referenced tables, e.g., {'user_id': 'users'}."
                ),
                "insert_mock_data": (
                    "Generate mock data using the generated function and insert into the database. "
                    "Parameters: code (str), tablename (str), n (int), **fk_ids (keyword args for foreign key IDs). "
                    "Returns: dict with 'records', 'id_list' (for downstream reference), 'count', and 'status'. "
                    "For tables with foreign keys, pass the IDs from parent tables, e.g., category_ids=[1,2,3]."
                ),
            }
        )
        logging.info(TOOL_DESC)
        tool_desc = build_tool_desc(TOOL_DESC)
        # args = parse_args()
        user_request = "Generate mock data for all tables based on the retrieved schemas, respecting foreign key constraints, and insert them into the database. Use the available tools only"
        plan = generate_plan(user_request, tool_desc, schema_res.schemas)
        plan_json = (
            plan.model_dump_json(indent=2)
            if hasattr(plan, "model_dump_json")
            else json.dumps(plan, indent=2)
        )
        print("Plan:\n", plan_json)
        # save to ./generated.json
        SCRIPT_DIR = Path(__file__).parent.resolve()
        with open(SCRIPT_DIR / "generated.json", "w", encoding="utf-8") as f:
            f.write(plan_json)

        # TOOL_REGISTRY = {tool.name: tool for tool in tools if tool.name is not None}
        # TOOL_REGISTRY["generate_mock_function"] = dspy.Tool(generate_mock_function)
        # TOOL_REGISTRY["insert_mock_data"] = dspy.Tool(insert_mock_data)
        # executor = PlanExecutor(TOOL_REGISTRY)
        # final_context = executor.execute_plan(plan_json)
        # logging.info(f"Final Execution Context: {final_context}")


def run():
    load_env_variable()
    dspy.settings.allow_tool_async_sync_conversion = True
    dspy.configure(lm=Lm_Glm)
    dspy.configure(show_guidelines=True)
    asyncio.run(exe_plan())
