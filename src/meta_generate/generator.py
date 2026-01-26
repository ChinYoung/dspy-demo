# dspy_generator.py
import dspy


class GenerateMockFunction(dspy.Signature):
    """Generate a mock data function that respects foreign key dependencies."""

    table_name: str = dspy.InputField()
    schema: str = dspy.InputField(desc="JSON-like dict of column:type")
    fk_deps: str = dspy.InputField(
        desc="Comma-separated list of referenced tables, e.g., 'users,products'"
    )
    n_example: int = dspy.InputField()
    code: str = dspy.OutputField(desc="Python function in ```python ... ```")


def generate_mock_function(
    table_name: str, schema: dict, fk_deps: list, n_example: int = 5
) -> str:
    """Generate a mock data function that respects foreign key dependencies."""

    fk_info = ", ".join(fk_deps) if fk_deps else "none"

    predictor = dspy.Predict(GenerateMockFunction)
    response = predictor(
        table_name=table_name, schema=str(schema), fk_deps=fk_info, n_example=n_example
    )

    # 提取代码块
    import re

    match = re.search(r"```python\n(.*?)\n```", response.code, re.DOTALL)
    if not match:
        raise ValueError("No valid code block found")
    return match.group(1).strip()
