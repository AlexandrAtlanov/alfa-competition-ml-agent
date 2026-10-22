"""Отказоустойчивость solution.py: без LLM сабмит всё равно выходит (капитан).

Главный страховочный контракт решения (AGENTS.md §3): LLM недоступен
или ключ не задан -> деградация к детерминированному бейзлайн-пути,
а не падение. Именно это отличает «просто скрипт» от решения, которое
переживёт сбой API в день сабмита.

Оба теста запускают solution.py из нейтрального каталога вне репо:
python-dotenv ищет .env от текущего каталога, и локальный .env капитана
не должен подсовывать ключ, маскируя настоящую отказоустойчивость.

Пока solution.py не запушен (зона A), тесты пропускаются автоматически.
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
    work = tmp_path / "task"
    shutil.copytree(TASK_DIR, work)
    for name in ("submission.csv", "metrics.json"):
        (work / name).unlink(missing_ok=True)
    return work


def _windows_minimum(env: dict[str, str]) -> dict[str, str]:
    """Добавить системный минимум, без которого Python на Windows не стартует.

    Без SYSTEMROOT Windows-питон не может импортировать asyncio (Winsock/
    IOCP не инициализируются) — падение ДО любого нашего кода. Это свойство
    Windows, а не решения: платформа (Linux) гарантирует нормальное env,
    там эта ветка не выполняется вовсе. Ключи и LLM-переменные не добавляем.
    """
    if os.name == "nt":
        for key in ("SYSTEMROOT", "SYSTEMDRIVE", "COMSPEC", "TEMP", "TMP"):
            value = os.environ.get(key)
            if value:
                env[key] = value
    return env


def _check_submission(work: Path) -> None:
    sub_path = work / "submission.csv"
    assert sub_path.exists(), "submission.csv не создан — деградация не сработала"
    sub = pd.read_csv(sub_path)
    sample = pd.read_csv(work / "sample_submission.csv")
    assert tools.validate_submission(sub, sample) == []


def test_no_llm_key_still_valid_submission(tmp_path):
    """Полное окружение, но без OPENAI_API_KEY/OPENAI_BASE_URL/OPENAI_MODEL."""
    work = _prepare(tmp_path)
    neutral = tmp_path / "elsewhere"
    neutral.mkdir()
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(("OPENAI_", "LLM_", "AGENT_"))}
    env["PYTHONIOENCODING"] = "utf-8"
    proc = subprocess.run(
        [sys.executable, "-u", str(SOLUTION), str(work / "task.md")],
        capture_output=True, text=True, timeout=900,
        cwd=str(neutral), env=env,
    )
    assert proc.returncode == 0, f"stderr:\n{proc.stderr[-2000:]}"
    _check_submission(work)


def test_minimal_env_like_platform_sandbox(tmp_path):
    """Минимальное окружение: PATH + кодировка + Windows-минимум.

    Жёсткий сценарий «только то, что гарантирует платформа»: нет ни ключей,
    ни переменных разработки. На Windows добавляем лишь системный минимум
    (SYSTEMROOT и др.) — без него падает сам Python при import asyncio,
    что не связано с нашим решением (см. _windows_minimum).
    """
    work = _prepare(tmp_path)
    neutral = tmp_path / "elsewhere"
    neutral.mkdir()
    env = _windows_minimum({"PATH": os.defpath, "PYTHONIOENCODING": "utf-8"})
    proc = subprocess.run(
        [sys.executable, "-u", str(SOLUTION), str(work / "task.md")],
        capture_output=True, text=True, timeout=900,
        cwd=str(neutral), env=env,
    )
    assert proc.returncode == 0, f"stderr:\n{proc.stderr[-2000:]}"
    _check_submission(work)
