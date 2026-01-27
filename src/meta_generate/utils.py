# executor.py
import asyncio
import logging
import types
import inspect
import random  # 若 mock 函数依赖标准库，需显式导入
import datetime as _dt_module
import string
import math
import time
import uuid
import dspy
import typing

# Ensure datetime module exposes utcnow for generated code using `datetime.utcnow()`
if not hasattr(_dt_module, "utcnow"):
    _dt_module.utcnow = _dt_module.datetime.utcnow
# Ensure datetime module exposes now for generated code using `datetime.now()`
if not hasattr(_dt_module, "now"):
    _dt_module.now = _dt_module.datetime.now

from fastmcp import Client

from meta_generate.signatures import GenerateMockFunction  # 允许安全导入 datetime

# dspy_generator.py


def generate_mock_function(
    table_name: str,
    schema: dict | None = None,
    fk_deps: list | None = None,
    n_example: int = 5,
) -> dict:
    """Generate a mock data function that respects foreign key dependencies, function name defaults to generate_mock_data.

    `schema` and `fk_deps` are optional to tolerate plans that omit them; they
    default to empty structures so callers that only provide `table_name` don't
    crash. Supplying real schema/fk info is recommended for higher fidelity.
    :param table_name: target table name
    :param schema: optional table schema dict
    :param fk_deps: optional list of foreign key dependency descriptions
    :param n_example: number of example records to generate in the function docstring
    :return: dict with generated function code under 'code' key and table name under 'table' key
    """

    schema = schema or {}
    fk_deps = fk_deps or []

    fk_info = ", ".join(str(dep) for dep in fk_deps) if fk_deps else "none"

    predictor = dspy.Predict(GenerateMockFunction)
    response = predictor(
        table_name=table_name, schema=str(schema), fk_deps=fk_info, n_example=n_example
    )

    # 提取代码块
    import re

    match = re.search(r"```python\n(.*?)\n```", response.code, re.DOTALL)
    if not match:
        raise ValueError("No valid code block found")

    code = match.group(1).strip()
    # Return a dict so downstream plan references like @<step>.code resolve correctly.
    return {"code": code, "table": table_name}


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
            "string": string,
            "math": math,
            "time": time,
            "uuid": uuid,
            "typing": typing,
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
            "string": string,
            "math": math,
            "time": time,
            "uuid": uuid,
            "timedelta": _dt_module.timedelta,
            "__import__": _safe_import,  # 控制 import 行为
        },
        "datetime": _dt_module,
        "string": string,
        "math": math,
        "time": time,
        "uuid": uuid,
        "timedelta": _dt_module.timedelta,
        "typing": typing,
        "__name__": "__mock__",
    }
    local_env = {}

    try:
        exec(code, safe_globals, local_env)
        func = None
        if func_name in local_env:
            func = local_env[func_name]
            if not isinstance(func, types.FunctionType):
                raise TypeError(f"'{func_name}' is not a function.")
        else:
            # Fallback: pick the first function defined in the generated code.
            funcs = [v for v in local_env.values() if isinstance(v, types.FunctionType)]
            if len(funcs) == 1:
                func = funcs[0]
                logging.warning(
                    "Function '%s' not found; using first defined function '%s' as fallback.",
                    func_name,
                    func.__name__,
                )
            else:
                available = [
                    name
                    for name, v in local_env.items()
                    if isinstance(v, types.FunctionType)
                ]
                raise ValueError(
                    f"Function '{func_name}' not found in generated code. Available: {available}"
                )

        # Filter kwargs to what the function accepts to avoid unexpected kw errors
        sig = inspect.signature(func)
        accepts_var_kw = any(p.kind == p.VAR_KEYWORD for p in sig.parameters.values())
        call_kwargs = {}
        for k, v in kwargs.items():
            if k in sig.parameters or accepts_var_kw:
                call_kwargs[k] = v
            else:
                logging.warning(
                    "Dropping unused argument '%s' for generated function %s",
                    k,
                    func.__name__,
                )

        result = func(**call_kwargs)
        if not isinstance(result, list):
            raise TypeError(
                "Mock function must return a dict mapping table names to lists of records."
            )
        return result

    except Exception as e:
        raise RuntimeError(f"Failed to execute mock function: {e}")


def _generate_with_mock_func(code: str, tablename: str, n: int) -> list[dict]:
    """
    execute the generated mock function and return the records with target length.
    :param code, generated mock function code
    :param tablename: table name to pass to the generated function
    :param n: number of records to generate per table
    """
    records = _execute_generated_func(
        code, func_name="generate_mock_data", table_name=tablename, n=n
    )
    logging.info(records)
    return records


async def _insert_records(records: list[dict], tablename: str) -> dict:
    """Async helper that batch inserts records into the MCP tool."""

    if not records:
        return {"status": "noop", "count": 0, "failures": []}

    mcp_client = Client("http://127.0.0.1:8999/mcp")
    async with mcp_client:
        try:
            logging.info(f"Batch inserting {len(records)} records into '{tablename}'")
            await mcp_client.call_tool(
                "db_batch_insert_records",
                {
                    "table_name": tablename,
                    "records": records,
                },
            )
            return {"status": "success", "count": len(records), "failures": []}
        except Exception as exc:  # noqa: BLE001
            logging.error("Batch insert failed for table '%s': %s", tablename, exc)
            return {
                "status": "failed",
                "count": 0,
                "failures": [{"error": str(exc)}],
            }


async def _insert_mock_data_async(records: list[dict], tablename: str) -> dict:
    """
    Insert generated mock records into the database via MCP HTTP tool.

    This calls the MCP tool `db_batch_insert_records(table_name: str, records: List[dict])`
    exposed by the MCP server. Intended for use when an event loop already
    exists.
    """

    if not records:
        return {"status": "noop", "count": 0, "failures": []}

    return await _insert_records(records, tablename)


async def insert_mock_data(code: str, tablename: str, n=10) -> dict:
    """
    Generate records with generated mock function and insert into the database via MCP HTTP tool.

    This is async so it can be called safely from running event loops. When wrapped by
    `dspy.Tool`, the sync call path will execute the coroutine thanks to
    `allow_tool_async_sync_conversion` being enabled in `init_dspy`.
    """
    records = _generate_with_mock_func(code, tablename=tablename, n=n)
    logging.info(f"Inserting {len(records)} records into table '{tablename}'")
    return await _insert_mock_data_async(records, tablename)
