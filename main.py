import dspy
import logging
from draft_article import DraftArticle

from simple_qa import ask_simple_qa


logging.basicConfig(level=logging.INFO)


def init_dspy():
    logging.info("Initializing dspy with local Ollama LLM...")
    lm = dspy.LM(
        model="ollama/qwen2.5:3b-instruct",
        api_base="http://192.168.50.104:11434",
    )
    logging.info("Configuring dspy...")
    dspy.configure(lm=lm, logging=True)
    logging.info("dspy configuration complete.")


def main():
    init_dspy()
    draft_article = DraftArticle()
    topic = "python的模块系统简介"
    logging.info(f"Drafting article on topic: {topic}")
    article = draft_article(topic=topic)
    logging.info("Article drafted successfully.")
    article_content = f"# {article.title}\n\n" + "\n\n".join(article.sections)
    print("Drafted Article:\n")
    print(article_content)


if __name__ == "__main__":
    main()
