import dspy

# plan_executor.py
import json
import logging
import re
from typing import Any, Callable, Dict, List, Set

from pydantic import BaseModel, Field, ValidationError, model_validator

# dspy_plan.py
from mcp import Tool


class GenerateMockFunction(dspy.Signature):
    """Generate a mock data function that respects foreign key dependencies"""

    table_name: str = dspy.InputField()
    schema: str = dspy.InputField(desc="JSON-like dict of column:type")
    fk_deps: str = dspy.InputField(
        desc="List of referenced tables, e.g., 'users,products'"
    )
    fk_columns: str = dspy.InputField(
        desc=("Mapping foreign key column names to their referenced table names.")
    )

    n_example: int = dspy.InputField()
    code: str = dspy.OutputField(
        desc=(
            """
            Return ONLY the raw Python code string.
            Use only Python standard library, do not use third-party libraries.
            Include **kwargs to accept dynamic foreign key parameters.
            Respect the provided schema, foreign key relationships.
            Avoid ID, key, and foreign key collisions, assuming there are existing records.
            """
        ),
    )


class GetTableSchemas(dspy.Signature):
    """Retrieve database table schemas including foreign key information."""

    table_names: str = dspy.InputField(
        desc="Comma-separated list of table names to retrieve schemas for, if not provided, retrieve all."
    )
    schemas: str = dspy.OutputField(
        desc=(
            "JSON dict mapping table names to their schema information. "
            "Each table schema should include: "
            "- 'columns': dict mapping column names to their data types "
            "- 'foreign_keys': dict mapping foreign key column names to their referenced table names "
            'Example: {"products": {"columns": {"id": "integer", "name": "varchar", "category_id": "integer"}, '
            '"foreign_keys": {"category_id": "categories"}}}'
        )
    )


class PlanStep(BaseModel):
    id: str = Field(..., description="Unique step id")
    tool: str = Field(..., description="Tool name to invoke")
    args: Dict[str, Any] = Field(
        default_factory=dict, description="Arguments for the tool"
    )
    desc: str | None = Field(None, description="Human-readable description")

    @model_validator(mode="before")
    @classmethod
    def coerce_id(cls, data):
        if isinstance(data, dict) and "id" in data:
            data = {**data, "id": str(data["id"])}
        return data


class PlanModel(BaseModel):
    steps: List[PlanStep]


class GenerateDAGPlan(dspy.Signature):
    """
    create a DAG plan in JSON format with provided utils to generate mock data for all the tables and insert the generated mock data into the database.
    - ** respecting foreign key constraints **
    - ** Use provided tools only **
    - ** For tables with foreign keys, pass the foreign key IDs from parent tables using @step_id.id_list **
    """

    user_request: str = dspy.InputField(
        default="""
        Generate mock data for all tables based on the retrieved schemas, respecting foreign key constraints, and insert them into the database. Use the available tools only.
        """
    )
    database_schema: str = dspy.InputField(
        desc="Database schema, including tables, columns, data types, and foreign key relationships. Format each table as: 'table_name: {column: type, ...}' and include foreign key info like 'foreign_keys: {column: referenced_table}'."
    )
    tool_descriptions: str = dspy.InputField(
        desc="Available tools with their names and descriptions."
    )
    plan: PlanModel = dspy.OutputField(
        desc="""
        PlanModel with 'steps' array. Each step has 'id', 'tool', 'args', 'desc'.
        Use @step_id.field to reference results, where step_id refers to the 'id' of previous steps and field is the key in the result dict returned by that step.
        IMPORTANT: 
            For insert_mock_data calls on tables with foreign keys, pass the foreign key IDs using the format 'fk_table_ids': '@parent_step.id_list'.
        """
    )
