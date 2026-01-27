# executor.py
import asyncio
import logging
import types
import random  # 若 mock 函数依赖标准库，需显式导入
import datetime as _dt_module
import dspy

from fastmcp import Client

from meta_generate.signatures import GenerateMockFunction  # 允许安全导入 datetime

# dspy_generator.py


def generate_mock_function(
    table_name: str,
    schema: dict | None = None,
    fk_deps: list | None = None,
    n_example: int = 5,
) -> str:
    """Generate a mock data function that respects foreign key dependencies.

    `schema` and `fk_deps` are optional to tolerate plans that omit them; they
    default to empty structures so callers that only provide `table_name` don't
    crash. Supplying real schema/fk info is recommended for higher fidelity.
    """

    schema = schema or {}
    fk_deps = fk_deps or []

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


def _execute_generated_func(code: str, func_name: str = "generate_mock_data", **kwargs):
    """
    执行生成的 mock 函数，返回 records 列表。
    :param code: 生成的函数源码（str）
    :param func_name: 函数名，默认为 'generate_mock_data'
    :param kwargs: 传给函数的参数，如 n=10
    """

    # 仅允许白名单模块的安全导入（用于支持代码中的 import 语句）
    def _safe_import(name, globals=None, locals=None, fromlist=(), level=0):
        allowed = {
            "random": random,
            "datetime": _dt_module,
        }
        if name in allowed:
            return allowed[name]
        raise ImportError(f"Import of '{name}' is not allowed")

    # 构建安全的全局/局部环境（限制内置函数，加入所需内置）
    safe_globals = {
        "__builtins__": {
            "range": range,
            "len": len,
            "int": int,
            "str": str,
            "list": list,
            "dict": dict,
            "float": float,
            "round": round,
            "random": random,  # 显式允许
            "__import__": _safe_import,  # 控制 import 行为
        },
        "__name__": "__mock__",
    }
    local_env = {}

    try:
        exec(code, safe_globals, local_env)
        if func_name not in local_env:
            raise ValueError(f"Function '{func_name}' not found in generated code.")

        func = local_env[func_name]
        if not isinstance(func, types.FunctionType):
            raise TypeError(f"'{func_name}' is not a function.")

        result = func(**kwargs)
        if not isinstance(result, list):
            raise TypeError(
                "Mock function must return a dict mapping table names to lists of records."
            )
        return result

    except Exception as e:
        raise RuntimeError(f"Failed to execute mock function: {e}")


def _generate_with_mock_func(code: str, n: int) -> list[dict]:
    """
    execute the generated mock function and return the records with target length.
    :param code, generated mock function code
    :param n: number of records to generate per table
    """
    records = _execute_generated_func(code, func_name="generate_mock_data", n=n)
    logging.info(records)
    return records


async def _insert_records(records: list[dict], tablename: str) -> dict:
    """Async helper that streams records into the MCP tool."""

    inserted = 0
    failures = []
    mcp_client = Client("http://127.0.0.1:8999/mcp")
    async with mcp_client:
        for idx, record in enumerate(records):
            try:
                await mcp_client.call_tool(
                    "db_insert_record",
                    {
                        "table_name": tablename,
                        "record": record,
                    },
                )
                inserted += 1
            except Exception as exc:  # noqa: BLE001
                logging.error("Insert failed for record %s: %s", idx, exc)
                failures.append({"index": idx, "error": str(exc)})

    status = "success" if not failures else "partial"
    return {"status": status, "count": inserted, "failures": failures}


async def _insert_mock_data_async(records: list[dict], tablename: str) -> dict:
    """
    Insert generated mock records into the database via MCP HTTP tool.

    This calls the MCP tool `db_insert_record(table_name: str, record: dict)`
    exposed by the MCP server. Intended for use when an event loop already
    exists.
    """

    if not records:
        return {"status": "noop", "count": 0, "failures": []}

    return await _insert_records(records, tablename)


def insert_mock_data(code: str, tablename: str, n=10) -> dict:
    """
    Generate records with generate mock function and insert into the database via MCP HTTP tool.
    This calls the MCP tool `db_insert_record(table_name: str, record: dict)`
    exposed by the MCP server. Intended for use in sync contexts.
    :param code: generated mock function code
    :param tablename: target table name
    :param client: MCP HTTP client
    :param n: number of records to generate
    """
    records = _generate_with_mock_func(code, n=n)
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        raise RuntimeError(
            "insert_mock_data must be awaited in an async context; "
            "call insert_mock_data_async instead."
        )

    return asyncio.run(_insert_mock_data_async(records, tablename))
