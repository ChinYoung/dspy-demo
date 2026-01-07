import dspy
import random
from typing import Any
from datasets import load_dataset
from dspy.datasets import DataLoader
from itertools import islice
from dspy.teleprompt import LabeledFewShot
import logging

logging.basicConfig(level=logging.INFO)


def setup_optimizer(lm: dspy.LM):
    # Load the Banking77 dataset.
    ds = load_dataset("PolyAI/banking77", split="train", trust_remote_code=True)

    # Safely obtain the label names; load_dataset can return different container types
    # depending on typing/runtime, so try multiple access patterns and raise a clear error.
    try:
        label_names = ds.features["label"].names  # type: ignore[attr-defined]
    except Exception:
        try:
            label_names = ds["train"].features["label"].names  # type: ignore[index]
        except Exception as e:
            raise RuntimeError(
                "Failed to load dataset features for 'PolyAI/banking77'"
            ) from e

    CLASSES: list[str] = list(label_names)
    kwargs: dict[str, Any] = {
        "fields": ("text", "label"),
        "input_keys": ("text",),
        "split": "train",
        "trust_remote_code": True,
    }

    # Load the first 2000 examples from the dataset, and assign a hint to each *training* example.
    def _label_index(x: Any) -> int:
        if isinstance(x, dict):
            return x["label"]
        if hasattr(x, "label"):
            return getattr(x, "label")
        raise TypeError("Cannot extract 'label' from example")

    trainset: list[dspy.Example] = []
    hf_loaded = DataLoader().from_huggingface(dataset_name="PolyAI/banking77", **kwargs)
    for x in islice(hf_loaded, 2000):
        label_idx = _label_index(x)
        trainset.append(
            dspy.Example(
                x, hint=CLASSES[label_idx], label=CLASSES[label_idx]
            ).with_inputs("text", "hint")
        )

    random.Random(0).shuffle(trainset)

    # keep the signature string separate to avoid passing a positional argument
    signature_spec: str = "text, hint -> label"
    signature = dspy.Signature(signature_spec).with_updated_fields(
        "label", type={"values": tuple(CLASSES)}
    )
    classify = dspy.ChainOfThought(signature=signature)
    classify.set_lm(lm)

    optimizer = dspy.BootstrapFewShot(
        metric=(lambda x, y, trace=None: x.label == y.label)
    )
    optimized = optimizer.compile(classify, trainset=trainset)
    return optimized
