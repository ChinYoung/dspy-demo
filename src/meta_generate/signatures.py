import dspy


class CodeGenerateRequest(dspy.Signature):
    """
    You are a code generator that produces mock data for database testing.

    Generate a single Python function named exactly `generate_mock_data` that:
    - Takes one argument: `n: int` (number of records to generate).
    - Returns a list of `n` dictionaries for each table, in a dictionary with table names as keys.
    - You may use `random`, `datetime`, and `timedelta` — but **import them inside the function**.
    - for multiple database, define mock functions separately, call them in the main function and return combined results.
    - Do **not** include any database logic, print statements, or example usage.
    - Do **not** use external libraries (e.g., faker, numpy).
    - Output **only** the function definition, wrapped in ```python ... ```.
    - The function must be immediately executable via `exec()` and callable as `generate_mock_data(n)`.
    """

    user_requirements: str = dspy.InputField(
        desc="User requirements for the code to be generated."
    )
    generated_code: str = dspy.OutputField(
        desc="The generated code based on the requirements."
    )


class GenerateExecutionPlan(dspy.Signature):
    """
    Generate a structured execution plan with explicit step dependencies.
    Each step has an ID, tool name, and arguments.
    Use @<step_id>.<field> to reference results from prior steps.
    """

    user_request: str = dspy.InputField(
        desc="Natural language description of the desired mock data scenario"
    )

    available_tools: str = dspy.InputField(
        desc="""
        List of available tools in format:
        - tool_name(arg1: type, arg2: type) -> {output_field1, output_field2}
        Example:
        - generate_users(n: int) -> {records: list, id_list: list}
        - insert_mock_data(table: str, records: list) -> {status: str, count: int}
        """
    )

    plan: str = dspy.OutputField(
        desc="""
        A JSON string with the following structure:
        {
          "steps": [
            {
              "id": "unique_step_id",
              "tool": "tool_name",
              "args": {
                "arg1": value_or_"@other_step_id.field"
              }
            }
          ]
        }
        Ensure all referenced step IDs exist and form a valid DAG.
        """
    )
