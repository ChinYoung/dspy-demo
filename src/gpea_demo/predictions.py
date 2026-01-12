import json
import dspy


def feedback_urgency(gold_urgency, pred_urgency):
    """
    Generate feedback for the urgency module.
    """
    score = 1.0 if gold_urgency == pred_urgency else 0.0
    if gold_urgency == pred_urgency:
        feedback = f"You correctly classified the urgency of the message as `{gold_urgency}`. This message is indeed of `{gold_urgency}` urgency."
    else:
        feedback = f"You incorrectly classified the urgency of the message as `{pred_urgency}`. The correct urgency is `{gold_urgency}`. Think about how you could have reasoned to get the correct urgency label."
    return feedback, score


def feedback_sentiment(gold_sentiment, pred_sentiment):
    """
    Generate feedback for the sentiment module.
    """
    score = 1.0 if gold_sentiment == pred_sentiment else 0.0
    if gold_sentiment == pred_sentiment:
        feedback = f"You correctly classified the sentiment of the message as `{gold_sentiment}`. This message is indeed `{gold_sentiment}`."
    else:
        feedback = f"You incorrectly classified the sentiment of the message as `{pred_sentiment}`. The correct sentiment is `{gold_sentiment}`. Think about how you could have reasoned to get the correct sentiment label."
    return feedback, score


def feedback_categories(gold_categories, pred_categories):
    """
    Generate feedback for the categories module.
    Uses the same match/mismatch logic as category accuracy in the score.
    """
    correctly_included = [
        k for k, v in gold_categories.items() if v and k in pred_categories
    ]
    incorrectly_included = [
        k for k, v in gold_categories.items() if not v and k in pred_categories
    ]
    incorrectly_excluded = [
        k for k, v in gold_categories.items() if v and k not in pred_categories
    ]
    correctly_excluded = [
        k for k, v in gold_categories.items() if not v and k not in pred_categories
    ]  # For completeness in accuracy check

    # Recompute category accuracy (matches score logic)
    score = (len(correctly_included) + len(correctly_excluded)) / len(gold_categories)

    if score == 1.0:
        fb_text = f"The category classification is perfect. You correctly identified that the message falls under the following categories: `{repr(correctly_included)}`."
    else:
        fb_text = f"The category classification is not perfect. You correctly identified that the message falls under the following categories: `{repr(correctly_included)}`.\n"
        if incorrectly_included:
            fb_text += f"However, you incorrectly identified that the message falls under the following categories: `{repr(incorrectly_included)}`. The message DOES NOT fall under these categories.\n"
        if incorrectly_excluded:
            prefix = "Additionally, " if incorrectly_included else "However, "
            fb_text += f"{prefix}you didn't identify the following categories that the message actually falls under: `{repr(incorrectly_excluded)}`.\n"
        fb_text += "Think about how you could have reasoned to get the correct category labels."
    return fb_text, score


def metric_with_feedback(example, pred, trace=None, pred_name=None, pred_trace=None):
    """
    Computes a score based on agreement between prediction and gold standard for categories, sentiment, and urgency.
    Optionally provides feedback text for a specific predictor module, using the same comparison logic as the score.
    Returns a dspy.Prediction with score (float) and feedback (str).
    """
    # Parse gold standard from example
    gold = json.loads(example["answer"])

    # Compute feedback and scores for all modules
    fb_urgency, score_urgency = feedback_urgency(gold["urgency"], pred.urgency)
    fb_sentiment, score_sentiment = feedback_sentiment(
        gold["sentiment"], pred.sentiment
    )
    fb_categories, score_categories = feedback_categories(
        gold["categories"], pred.categories
    )

    # Overall score: average of the three accuracies
    total = (score_urgency + score_sentiment + score_categories) / 3

    if pred_name is None:
        return total

    # Ensure feedback is always defined, default to empty string for unknown predictors
    feedback = ""
    if pred_name == "urgency_module.predict":
        feedback = fb_urgency
    elif pred_name == "sentiment_module.predict":
        feedback = fb_sentiment
    elif pred_name == "categories_module.predict":
        feedback = fb_categories

    return dspy.Prediction(score=total, feedback=feedback)
