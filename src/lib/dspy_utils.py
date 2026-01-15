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
    logging.info("dspy configuration complete.")


async def list_tools(client: Client):
    dspy_tools = []
    async with client:
        tools = await client.list_tools()
        for tool in tools:
            logging.info(f"Found tool: {tool.name} - {tool.description}")
            dspy_tools.append(dspy.Tool.from_mcp_tool(client.session, tool))
    return dspy_tools
