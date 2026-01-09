import dspy
import logging
from draft_article import DraftArticle
from optimizer import setup_optimizer
from simple_qa import ask_simple_qa


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
