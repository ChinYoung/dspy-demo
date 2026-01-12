import asyncio
from lib.custom_lm.lms import Lm_Glm
from lib.dspy_utils import init_dspy
from mcp_demo.dspy_tools import resolve_user_request


if __name__ == "__main__":
    init_dspy(Lm_Glm)
    user_request = (
        "please help me book a flight from SFO to JFK on 09/01/2025, my name is Adam"
    )
    asyncio.run(resolve_user_request(user_request))
