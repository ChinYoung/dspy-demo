import argparse
import logging
import asyncio

import dspy
from lib.custom_lm.lms import Lm_Glm
from lib import dspy_utils
from fastmcp import Client, FastMCP

from lib.utils import parse_args


class DSPyAirlineCustomerService(dspy.Signature):
    """You are a life assistant agent. You are expect to list today's hot news and provide suggestions"""

    user_request: str = dspy.InputField()
    process_result: str = dspy.OutputField(desc=("今日热点, 使用中文回答"))


client = Client("http://127.0.0.1:8999/mcp")


def run():
    asyncio.run(run_async())


async def run_async():
    args = parse_args()
    logging.info(f"User request: {args.user_request}")
    res = await resolve_user_request(args.user_request)
    logging.info(f"Process result: {res.process_result}")


async def resolve_user_request(user_request: str):
    dspy_utils.init_dspy(Lm_Glm)
    async with client:
        tools = await client.list_tools()
        logging.info(f"Available tools: {tools}")
        dspy_tools = []
        for tool in tools:
            dspy_tools.append(dspy.Tool.from_mcp_tool(client.session, tool))

        # Create the agent
        react = dspy.ReAct(DSPyAirlineCustomerService, tools=dspy_tools)
        res = await react.acall(user_request=user_request)
        return res
