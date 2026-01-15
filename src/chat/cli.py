import asyncio
import logging
from chat.agent import InteractiveAgent
from lib.utils import parse_args


def run():
    args = parse_args()
    logging.info(f"User request: {args.user_request}")
    agent = InteractiveAgent()
    asyncio.run(agent.forward(task_goal=args.user_request))
