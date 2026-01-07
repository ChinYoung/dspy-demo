import dspy
import json
import random
from pathlib import Path
from typing import List, Literal


def init_dataset():
    # Load from the local dataset to avoid network dependency
    data_path = Path(__file__).parent.parent / "assets" / "dataset.json"
    dataset = json.loads(data_path.read_text())
    dspy_dataset = [
        dspy.Example(
            {
                "message": d["fields"]["input"],
                "answer": d["answer"],
            }
        ).with_inputs("message")
        for d in dataset
    ]
    random.Random(0).shuffle(dspy_dataset)
    train_set = dspy_dataset[: int(len(dspy_dataset) * 0.33)]
    val_set = dspy_dataset[
        int(len(dspy_dataset) * 0.33) : int(len(dspy_dataset) * 0.66)
    ]
    test_set = dspy_dataset[int(len(dspy_dataset) * 0.66) :]
    return train_set, val_set, test_set


if __name__ == "__main__":
    train_set, val_set, test_set = init_dataset()
    print(f"Train set size: {len(train_set)}")
    print(f"Validation set size: {len(val_set)}")
    print(f"Test set size: {len(test_set)}")
