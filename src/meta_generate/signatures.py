import dspy


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


class GenerateMockFunction(dspy.Signature):
    """Generate a mock data function that respects foreign key dependencies."""

    table_name: str = dspy.InputField()
    schema: str = dspy.InputField(desc="JSON-like dict of column:type")
    fk_deps: str = dspy.InputField(
        desc="Comma-separated list of referenced tables, e.g., 'users,products'"
    )
    n_example: int = dspy.InputField()
    code: str = dspy.OutputField(desc="Python function in ```python ... ```")


class GetTableSchemas(dspy.Signature):
    """Retrieve database table schemas."""

    table_names: str = dspy.InputField(
        desc="Comma-separated list of table names to retrieve schemas for, if not provided, retrieve all."
    )
    schemas: str = dspy.OutputField(
        desc="JSON dict mapping table names to their schema dicts"
    )
