import dspy
from typing import Optional
import logging
from optimizer import setup_optimizer


logging.basicConfig(level=logging.INFO)


def create_lm():
    logging.info("Creating local Ollama LLM instance...")
    lm = dspy.LM(
        model="ollama/qwen2.5:3b-instruct",
        api_base="http://192.168.50.185:11434",
    )
    return lm


def init_dspy(lm: Optional[dspy.LM] = None):
    logging.info("Initializing dspy with local Ollama LLM...")
    logging.info("Configuring dspy...")
    if lm is None:
        lm = create_lm()

    dspy.configure(
        lm=lm,
        logging=True,
    )
    logging.info("dspy configuration complete.")
    logging.info("dspy configuration complete.")


def main():
    lm = create_lm()
    init_dspy(lm)
    optimized = setup_optimizer(lm)

    response = optimized(text="What does a pending cash withdrawal mean?")
    logging.info(response)


if __name__ == "__main__":
    main()
