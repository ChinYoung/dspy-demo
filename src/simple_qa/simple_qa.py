import dspy
import logging


class SimpleQa(dspy.Signature):
    question = dspy.InputField()
    answer = dspy.OutputField(desc="简洁准确的答案")


def ask_simple_qa(question: str) -> str:
    qa = dspy.Predict(SimpleQa)
    response = qa(question=question)
    answer = response.get("answer")
    if not answer:
        logging.warning("No answer received from the model.")
        return ""
    return answer
