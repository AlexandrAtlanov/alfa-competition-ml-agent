"""Точка входа на платформе: python solution.py task.md

Интерфейс как в reference/baseline.py: путь к task.md — первый аргумент
(по умолчанию task.md), LLM настраивается через OPENAI_API_KEY /
OPENAI_BASE_URL / OPENAI_MODEL. Результат — submission.csv рядом с task.md.

Деградация: любой сбой агентского цикла -> детерминированный бейзлайн
tools.run_task (без LLM), если сабмит ещё не сформирован.
"""
import os
import sys
import time
import traceback
from pathlib import Path

T0 = time.time()
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def _task_md(argv: list[str]) -> Path:
    """
    Находит путь к task.md по аргументам командной строки.

    Аргументы:
    argv (list[str]): sys.argv; если передан каталог — берётся task.md внутри

    Возвращает:
    Path
    """
    path = Path(argv[1] if len(argv) > 1 else "task.md").resolve()
    return path / "task.md" if path.is_dir() else path


def main(argv: list[str]) -> int:
    """
    Запускает агента на задаче; при сбое — детерминированный бейзлайн.

    Аргументы:
    argv (list[str]): sys.argv

    Возвращает:
    int: код возврата (0 — сабмит сформирован)
    """
    task_md = _task_md(argv)
    submission = task_md.parent / "submission.csv"
    # логи рядом с задачей; задаём до импорта agent.* (логгеры читают LOG_DIR)
    os.environ.setdefault("LOG_DIR", str(task_md.parent / "logs"))

    try:
        from agent import loop

        summary = loop.run(task_md, t0=T0)
        print(f"Готово: {summary}")
    except BaseException:  # noqa: BLE001 — сабмит важнее любого исключения
        traceback.print_exc()

    if not submission.exists():
        print("Сабмита нет — запускаю бейзлайн tools.run_task", file=sys.stderr)
        try:
            from agent import tools

            tools.run_task(str(task_md))
        except Exception:  # noqa: BLE001
            traceback.print_exc()
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
