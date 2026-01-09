import dspy
import json
import random
from pathlib import Path
from typing import List, Literal

from openai import api_key

from custom_lm.glm_lm import LmGlm
from gpea_demo.init_dataset import init_dataset
from gpea_demo.metrics import metric
from gpea_demo.predictions import metric_with_feedback
from gpea_demo.signatures import (
    FacilitySupportAnalyzerCategories,
    FacilitySupportAnalyzerSentiment,
    FacilitySupportAnalyzerUrgency,
)
from init_dspy import create_lm, init_dspy
from dspy import GEPA


class FacilitySupportAnalyzerMM(dspy.Module):
    def __init__(self):
        self.urgency_module = dspy.ChainOfThought(FacilitySupportAnalyzerUrgency)
        self.sentiment_module = dspy.ChainOfThought(FacilitySupportAnalyzerSentiment)
        self.categories_module = dspy.ChainOfThought(FacilitySupportAnalyzerCategories)

    def forward(self, message: str):
        urgency = self.urgency_module(message=message)
        sentiment = self.sentiment_module(message=message)
        categories = self.categories_module(message=message)

        return dspy.Prediction(
            urgency=urgency.urgency,
            sentiment=sentiment.sentiment,
            categories=categories.categories,
        )


def main():
    init_dspy()
    train_set, val_set, test_set = init_dataset()
    program = FacilitySupportAnalyzerMM()
    evaluate = dspy.Evaluate(
        devset=test_set,
        metric=metric,
        num_threads=32,
        display_table=True,
        display_progress=True,
    )
    # evaluate(program)
    reflect_lm = LmGlm()
    optimizer = GEPA(
        metric=metric_with_feedback,
        auto="light",  # <-- We will use a light budget for this tutorial. However, we typically recommend using auto="heavy" for optimized performance!
        num_threads=8,
        track_stats=True,
        use_merge=False,
        reflection_lm=reflect_lm,
    )
    optimized_program = optimizer.compile(
        program,
        trainset=train_set,
        valset=val_set,
    )
    evaluate(optimized_program)


if __name__ == "__main__":
    main()
