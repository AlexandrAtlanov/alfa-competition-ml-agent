"""Тест дедлайна: loop укладывается в бюджет времени и всегда сабмитит (капитан).

Кейс платформы: 3 часа на задачу. loop.py читает AGENT_TIME_BUDGET (сек) —
тест подменяет его на маленькое значение (45 с) и проверяет главное:
агент НЕ выходит за бюджет с запасом и всё равно отдаёт валидный сабмит.

Это страховой тест: если кто-то добавит в цикл шаг без проверки времени
(например, ждёт ответ LLM дольше остатка бюджета), этот тест покраснеет
раньше, чем сгорит реальный прогон на платформе.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
import pytest

from agent import tools

ROOT = Path(__file__).resolve().parents[1]
SOLUTION = ROOT / "solution.py"
TASK_DIR = ROOT / "examples" / "simple_task"

pytestmark = pytest.mark.skipif(
    not SOLUTION.exists(),
    reason="solution.py не запушен (зона A); тест включится сам",
)

BUDGET = 45.0          # сек, вместо платформенных 10800
HARD_CEILING = 150.0   # бюджет + запас на медленную машину CI


def test_loop_respects_time_budget(tmp_path):
    work = tmp_path / "task"
    shutil.copytree(TASK_DIR, work)
    for name in ("submission.csv", "metrics.json"):
        (work / name).unlink(missing_ok=True)
    neutral = tmp_path / "elsewhere"
    neutral.mkdir()

    env = {k: v for k, v in os.environ.items()
           if not k.startswith(("OPENAI_", "LLM_", "AGENT_"))}
    env.update({
        "PYTHONIOENCODING": "utf-8",
        "AGENT_TIME_BUDGET": str(BUDGET),
        "AGENT_EXP_TIMEOUT": "20",
    })

    t0 = time.monotonic()
    proc = subprocess.run(
        [sys.executable, "-u", str(SOLUTION), str(work / "task.md")],
        capture_output=True, text=True, timeout=HARD_CEILING + 120,
        cwd=str(neutral), env=env,
    )
    elapsed = time.monotonic() - t0

    # 1) сабмит создан и валиден — при любом раскладе времени
    assert proc.returncode == 0, f"stderr:\n{proc.stderr[-2000:]}"
    sub_path = work / "submission.csv"
    assert sub_path.exists(), "дедлайн наступил, а сабмита нет"
    sub = pd.read_csv(sub_path)
    sample = pd.read_csv(work / "sample_submission.csv")
    assert tools.validate_submission(sub, sample) == []

    # 2) бюджет времени соблюдён (с запасом на медленную машину)
    assert elapsed <= HARD_CEILING, (
        f"цикл вышел за бюджет: {elapsed:.0f} c > {HARD_CEILING:.0f} c; "
        f"какой-то шаг не проверяет остаток времени"
    )
