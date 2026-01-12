import json


def score_urgency(gold_urgency, pred_urgency):
    """
    Compute score for the urgency module.
    """
    score = 1.0 if gold_urgency == pred_urgency else 0.0
    return score


def score_sentiment(gold_sentiment, pred_sentiment):
    """
    Compute score for the sentiment module.
    """
    score = 1.0 if gold_sentiment == pred_sentiment else 0.0
    return score


def score_categories(gold_categories, pred_categories):
    """
    Compute score for the categories module.
    Uses the same match/mismatch logic as category accuracy in the score.
    """
    correct = 0
    for k, v in gold_categories.items():
        if v and k in pred_categories:
            correct += 1
        elif not v and k not in pred_categories:
            correct += 1
    score = correct / len(gold_categories)
    return score


def metric(example, pred, trace=None, pred_name=None, pred_trace=None):
    """
    Computes a score based on agreement between prediction and gold standard for categories, sentiment, and urgency.
    Returns the score (float).
    """
    # Parse gold standard from example
    gold = json.loads(example["answer"])

    # Compute scores for all modules
    score_urgency_val = score_urgency(gold["urgency"], pred.urgency)
    score_sentiment_val = score_sentiment(gold["sentiment"], pred.sentiment)
    score_categories_val = score_categories(gold["categories"], pred.categories)

    # Overall score: average of the three accuracies
    total = (score_urgency_val + score_sentiment_val + score_categories_val) / 3

    return total
