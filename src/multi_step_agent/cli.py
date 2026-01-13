import logging
import asyncio
from lib.utils import parse_args
from multi_step_agent.modules import PlanThenReAct


def run():
    args = parse_args()
    logging.info(f"User request: {args.user_request}")
    planner = PlanThenReAct()
    # planner.forward is async, so run it via the event loop
    result = asyncio.run(planner.forward(goal=args.user_request))
    logging.info(f"Plan result: {result}")
