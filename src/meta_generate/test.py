import logging

import dspy
from fastmcp import Client
from lib.dspy_utils import list_tools
from meta_generate.plan_executor import PlanExecutor
from meta_generate.utils import (
    _generate_with_mock_func,
    generate_mock_function,
    insert_mock_data,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

generated_plan_json = """
{
  "steps": [
    {
      "id": 1,
      "tool": "generate_mock_function",
      "args": {
        "table_name": "categories"
      },
      "desc": "Generate mock data function for categories"
    },
    {
      "id": 2,
      "tool": "insert_mock_data",
      "args": {
        "code": "@1.code",
        "tablename": "categories",
        "n": 10
      },
      "desc": "Insert 10 records into categories"
    },
    {
      "id": 3,
      "tool": "generate_mock_function",
      "args": {
        "table_name": "users"
      },
      "desc": "Generate mock data function for users"
    },
    {
      "id": 4,
      "tool": "insert_mock_data",
      "args": {
        "code": "@3.code",
        "tablename": "users",
        "n": 10
      },
      "desc": "Insert 10 records into users"
    },
    {
      "id": 5,
      "tool": "generate_mock_function",
      "args": {
        "table_name": "products"
      },
      "desc": "Generate mock data function for products (depends on categories)"
    },
    {
      "id": 6,
      "tool": "insert_mock_data",
      "args": {
        "code": "@5.code",
        "tablename": "products",
        "n": 20
      },
      "desc": "Insert 20 records into products (references categories)"
    },
    {
      "id": 7,
      "tool": "generate_mock_function",
      "args": {
        "table_name": "user_addresses"
      },
      "desc": "Generate mock data function for user_addresses (depends on users)"
    },
    {
      "id": 8,
      "tool": "insert_mock_data",
      "args": {
        "code": "@7.code",
        "tablename": "user_addresses",
        "n": 10
      },
      "desc": "Insert 10 records into user_addresses (references users)"
    },
    {
      "id": 9,
      "tool": "generate_mock_function",
      "args": {
        "table_name": "orders"
      },
      "desc": "Generate mock data function for orders (depends on users)"
    },
    {
      "id": 10,
      "tool": "insert_mock_data",
      "args": {
        "code": "@9.code",
        "tablename": "orders",
        "n": 15
      },
      "desc": "Insert 15 records into orders (references users)"
    },
    {
      "id": 11,
      "tool": "generate_mock_function",
      "args": {
        "table_name": "order_items"
      },
      "desc": "Generate mock data function for order_items (depends on orders and products)"
    },
    {
      "id": 12,
      "tool": "insert_mock_data",
      "args": {
        "code": "@11.code",
        "tablename": "order_items",
        "n": 30
      },
      "desc": "Insert 30 records into order_items (references orders and products)"
    }
  ]
}
"""


async def run_plan():
    mcp_client = Client("http://127.0.0.1:8999/mcp")
    async with mcp_client:
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
