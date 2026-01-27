from typing import List
import dspy
import logging
from fastmcp import Client


logging.basicConfig(level=logging.INFO)


def create_lm(model: str = "ollama/qwen2.5:3b-instruct") -> dspy.LM:
    logging.info("Creating local Ollama LLM instance...")
    lm = dspy.LM(
        model=model,
        api_base="http://192.168.50.185:11434",
    )
    return lm


def init_dspy(lm=None):
    if lm is None:
        lm = create_lm()
    logging.info("Initializing dspy with local Ollama LLM...")
    logging.info("Configuring dspy...")
    dspy.configure(lm=lm, logging=True)
    # Allow calling async MCP-backed tools from sync code paths safely.
    # This prevents "coroutine ... was never awaited" warnings when MCP tools
    # are converted to async functions by `dspy.Tool.from_mcp_tool` and later
    # invoked in synchronous contexts.
    try:
        dspy.settings.allow_tool_async_sync_conversion = True
    except Exception:
        logging.warning(
            "Failed to set allow_tool_async_sync_conversion on dspy.settings"
        )
    logging.info("dspy configuration complete.")


async def list_tools(client: Client) -> List[dspy.Tool]:
    dspy_tools: List[dspy.Tool] = []
    tools = await client.list_tools()
    for tool in tools:
        logging.info(f"Found tool: {tool.name} - {tool.description}")
        dspy_tools.append(dspy.Tool.from_mcp_tool(client.session, tool))
    return dspy_tools


def load_env_variable():
    import os
    from dotenv import load_dotenv

    load_dotenv()
    api_key = os.getenv("ZAI_API_KEY")
    if api_key:
        logging.info("ZAI_API_KEY loaded successfully.")
    else:
        logging.warning("ZAI_API_KEY not found in environment variables.")
    return api_key
