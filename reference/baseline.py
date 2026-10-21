"""Запуск: OPENAI_API_KEY=... OPENAI_MODEL=... OPENAI_BASE_URL=... uv run baseline.py task.md
"""

import asyncio
import json
import os
from pathlib import Path
import subprocess
import sys

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.server.fastmcp import FastMCP
from openai import AsyncOpenAI


def read_task(path: Path) -> str:
    return path.read_text(encoding="utf-8")


mcp = FastMCP("python")


@mcp.tool()
def run_python(code: str) -> str:
    """Выполнить Python и вернуть stdout, stderr и код возврата.

    Каждый вызов — новый процесс. Файлы сохраняются, переменные — нет.
    Рабочий каталог совпадает с каталогом Markdown-файла задачи.
    """
    try:
        result = subprocess.run(
            [sys.executable, "-u", "-c", code],
            capture_output=True, text=True, timeout=1800,
            env={"PATH": os.defpath, "PYTHONIOENCODING": "utf-8"},
        )
        return json.dumps({"stdout": result.stdout[-12000:],
                           "stderr": result.stderr[-12000:],
                           "exit_code": result.returncode}, ensure_ascii=False)
    except subprocess.TimeoutExpired:
        return json.dumps({"error": "Превышен лимит выполнения: 1800 секунд"}, ensure_ascii=False)


async def run_agent(task_path: Path):
    task_path = task_path.resolve()
    task = read_task(task_path)
    model = os.environ["OPENAI_MODEL"]
    messages = [
        {"role": "system", "content": "Ты DS-агент. Реши задачу с помощью run_python. "
         "Предложи самую простую ml-модель, которая решает задачу. Относительные пути заданы от каталога задачи."},
        {"role": "user", "content": task},
    ]
    server = StdioServerParameters(
        command=sys.executable,
        args=[str(Path(__file__).resolve()), "--mcp-server"],
        cwd=str(task_path.parent),
    )
    async with AsyncOpenAI() as client, stdio_client(server) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            available = await session.list_tools()
            tools = [{"type": "function", "function": {
                "name": tool.name, "description": tool.description,
                "parameters": tool.inputSchema,
            }} for tool in available.tools]
            for _ in range(100):
                response = await client.chat.completions.create(
                    model=model, messages=messages, tools=tools, parallel_tool_calls=False,
                )
                message = response.choices[0].message
                messages.append(message.model_dump(exclude_none=True))
                if not message.tool_calls:
                    print(message.content or "Агент завершил работу.")
                    return
                for call in message.tool_calls:
                    try:
                        result = await session.call_tool(call.function.name, json.loads(call.function.arguments))
                        output = "\n".join(block.text for block in result.content if block.type == "text")
                    except ValueError as error:
                        output = f"Ошибка аргументов инструмента: {error}"
                    messages.append({"role": "tool", "tool_call_id": call.id, "content": output})
            raise RuntimeError("Достигнут лимит: 100 шагов агента")


if __name__ == "__main__":
    if sys.argv[1:] == ["--mcp-server"]:
        mcp.run(transport="stdio")
    else:
        asyncio.run(run_agent(Path(sys.argv[1] if len(sys.argv) > 1 else "task.md")))
