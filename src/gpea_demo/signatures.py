import dspy
from typing import List, Literal


class FacilitySupportAnalyzerUrgency(dspy.Signature):
    """
    Read the provided message and determine the urgency.
    """

    message: str = dspy.InputField()
    urgency: Literal["low", "medium", "high"] = dspy.OutputField()


class FacilitySupportAnalyzerSentiment(dspy.Signature):
    """
    Read the provided message and determine the sentiment.
    """

    message: str = dspy.InputField()
    sentiment: Literal["positive", "neutral", "negative"] = dspy.OutputField()


class FacilitySupportAnalyzerCategories(dspy.Signature):
    """
    Read the provided message and determine the set of categories applicable to the message.
    """

    message: str = dspy.InputField()
    categories: List[
        Literal[
            "emergency_repair_services",
            "routine_maintenance_requests",
            "quality_and_safety_concerns",
            "specialized_cleaning_services",
            "general_inquiries",
            "sustainability_and_environmental_practices",
            "training_and_support_requests",
            "cleaning_services_scheduling",
            "customer_feedback_and_complaints",
            "facility_management_issues",
        ]
    ] = dspy.OutputField()
