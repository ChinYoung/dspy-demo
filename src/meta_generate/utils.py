# executor.py
import asyncio
import logging
import types
import random  # 若 mock 函数依赖标准库，需显式导入
import datetime as _dt_module

from fastmcp import Client  # 允许安全导入 datetime


def execute_generated_func(code: str, func_name: str = "generate_mock_data", **kwargs):
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


def generate_mock_data(code: str, n: int) -> list[dict]:
    """
    execute the generated mock function and return the records with target length.
    :param code, generated mock function code
    :param n: number of records to generate per table
    """
    records = execute_generated_func(code, func_name="generate_mock_data", n=n)
    logging.info(records)
    return records


async def _insert_records(records: list[dict], tablename: str, client: Client) -> dict:
    """Async helper that streams records into the MCP tool."""

    inserted = 0
    failures = []

    async with client:
        for idx, record in enumerate(records):
            try:
                await client.call_tool(
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


async def insert_mock_data_async(
    records: list[dict], tablename: str, client: Client
) -> dict:
    """
    Insert generated mock records into the database via MCP HTTP tool.

    This calls the MCP tool `db_insert_record(table_name: str, record: dict)`
    exposed by the MCP server. Intended for use when an event loop already
    exists.
    """

    if not records:
        return {"status": "noop", "count": 0, "failures": []}

    return await _insert_records(records, tablename, client)


def insert_mock_data(records: list[dict], tablename: str, client: Client) -> dict:
    """
    Convenience wrapper that runs `insert_mock_data_async` from sync code.

    If an event loop is already running, call and await
    `insert_mock_data_async` directly instead of this wrapper.
    """

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        raise RuntimeError(
            "insert_mock_data must be awaited in an async context; "
            "call insert_mock_data_async instead."
        )

    return asyncio.run(insert_mock_data_async(records, tablename, client))
