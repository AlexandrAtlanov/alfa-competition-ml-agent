"""Приёмка solution.py: контракт платформы (зона капитана).

Платформа запускает решение так (см. reference/baseline.py):
    uv run solution.py <task.md>     # argv[1] = ПУТЬ К ФАЙЛУ task.md

Ожидаемое поведение:
    рядом с task.md появляется submission.csv, валидный по
    sample_submission.csv, строки в порядке test.csv, exit code 0.

Тесты запускают solution.py БЕЗ ключей LLM (чистое окружение без
OPENAI_*/LLM_*): проверяем детерминированный путь — ровно то, что
пойдёт в страховой сабмит пятницы. LLM-слой тестирует scripts/smoke_llm.py.

Пока solution.py не запушен (зона A), тесты пропускаются автоматически —
как только файл появится в репо, приёмка включится сама.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
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


def _prepare(tmp_path: Path) -> Path:
    """Копия демо-задачи в чистом каталоге, без артефактов прошлых прогонов."""
    work = tmp_path / "task"
    shutil.copytree(TASK_DIR, work)
    for name in ("submission.csv", "metrics.json"):
        (work / name).unlink(missing_ok=True)
    return work


def _run(task_md: Path, cwd: Path) -> subprocess.CompletedProcess:
    """Запустить solution.py как это сделала бы платформа."""
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(("OPENAI_", "LLM_"))}
    env["PYTHONIOENCODING"] = "utf-8"
    return subprocess.run(
        [sys.executable, "-u", str(SOLUTION), str(task_md)],
        capture_output=True, text=True, timeout=900,
        cwd=str(cwd), env=env,
    )


def test_creates_valid_submission(tmp_path):
    work = _prepare(tmp_path)
    proc = _run(work / "task.md", cwd=work)
    assert proc.returncode == 0, f"stderr:\n{proc.stderr[-2000:]}"
    sub_path = work / "submission.csv"
    assert sub_path.exists(), "submission.csv не создан рядом с task.md"
    sub = pd.read_csv(sub_path)
    sample = pd.read_csv(work / "sample_submission.csv")
    assert tools.validate_submission(sub, sample) == []


def test_submission_next_to_task_md_not_in_cwd(tmp_path):
    """Классический баг: решение пишет сабмит в текущий каталог, а не к задаче.

    Запускаем из нейтрального каталога: сабмит обязан появиться рядом
    с task.md, а не там, откуда запустили (на платформе cwd другой).
    Бонус: нейтральный cwd вне репо => локальный .env не подхватится.
    """
    work = _prepare(tmp_path)
    neutral = tmp_path / "elsewhere"
    neutral.mkdir()
    proc = _run(work / "task.md", cwd=neutral)
    assert (work / "submission.csv").exists(), \
        f"submission не в каталоге задачи. stderr:\n{proc.stderr[-1500:]}"
    assert not list(neutral.glob("submission*")), \
        "submission записан в cwd (откуда запущен), а не рядом с task.md"
