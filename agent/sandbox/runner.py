"""Запуск сгенерированного Python-кода в отдельном subprocess.

Каждый скрипт исполняется в новом процессе (как в baseline.py): файлы
сохраняются, переменные — нет. Окружение минимальное (PATH и
PYTHONIOENCODING), без ключей и сетевых секретов.

Две важные гарантии:
- при таймауте убивается **вся группа процессов** (дочерние процессы
  joblib/loky и subprocess'ы сгенерированного кода не остаются висеть);
- `run_python` **не пробрасывает исключения** — любая ошибка возвращается
  как `ExecResult` с `exit_code=-1`, чтобы цикл не падал (AGENTS.md §4).
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from ..event_log import get_logger

MAX_OUTPUT_CHARS = 12_000
KILL_GRACE_SECONDS = 5


@dataclass
class ExecResult:
    """Результат одного запуска сгенерированного скрипта."""

    stdout: str
    stderr: str
    exit_code: int
    duration: float
    timed_out: bool = False

    @property
    def ok(self) -> bool:
        return self.exit_code == 0 and not self.timed_out

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, indent=2)


def _clip(value) -> str:
    """Обрезать вывод до последних MAX_OUTPUT_CHARS символов."""
    if value is None:
        return ""
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    return value[-MAX_OUTPUT_CHARS:]


def _sandbox_env(extra: dict[str, str] | None = None) -> dict[str, str]:
    """Минимальное окружение песочницы: без ключей и прокси."""
    env = {
        "PATH": os.pathsep.join([os.defpath, "/usr/sbin"]),
        "PYTHONIOENCODING": "utf-8",
    }
    if extra:
        env.update(extra)
    return env


def _kill_process_group(proc: subprocess.Popen) -> None:
    """Убить процесс и всех его потомков (нам нужна вся группа)."""
    try:
        if os.name == "posix":
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        else:  # pragma: no cover — платформа и dev на POSIX
            proc.kill()
    except OSError:  # ProcessLookupError и прочие гонки
        pass


def _run_python(
    code: str,
    *,
    timeout: float | None,
    cwd: str | Path | None,
    extra_env: dict[str, str] | None,
    phase: str | None,
) -> ExecResult:
    log = get_logger("exec")
    started = time.time()

    proc = subprocess.Popen(
        [sys.executable, "-u", "-c", code],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=str(cwd) if cwd is not None else None,
        env=_sandbox_env(extra_env),
        start_new_session=(os.name == "posix"),
    )

    timed_out = False
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
        exit_code = proc.returncode
    except subprocess.TimeoutExpired:
        timed_out = True
        _kill_process_group(proc)
        try:
            # добираем то, что успело попасть в пайпы до убийства
            stdout, stderr = proc.communicate(timeout=KILL_GRACE_SECONDS)
        except subprocess.TimeoutExpired:
            proc.kill()
            stdout, stderr = "", ""
        exit_code = -1

    result = ExecResult(
        stdout=_clip(stdout),
        stderr=_clip(stderr),
        exit_code=exit_code,
        duration=round(time.time() - started, 3),
        timed_out=timed_out,
    )
    log.log(
        kind="exec",
        phase=phase,
        exit_code=result.exit_code,
        duration=result.duration,
        timed_out=result.timed_out,
        stdout_chars=len(result.stdout),
        stderr_chars=len(result.stderr),
    )
    return result


def run_python(
    code: str,
    *,
    timeout: float | None = 1800,
    cwd: str | Path | None = None,
    env: dict[str, str] | None = None,
    phase: str | None = None,
) -> ExecResult:
    """Выполнить код в отдельном процессе и вернуть результат.

    Args:
        code: исходный код скрипта.
        timeout: предел выполнения в секундах (None — без лимита).
        cwd: рабочий каталог (каталог задачи); None — текущий.
        env: дополнительные переменные окружения.
        phase: фаза цикла для лога (SCOUT/BASELINE/RESEARCH/FINALIZE).

    Никогда не бросает исключений: любая ошибка упаковывается в ExecResult.
    """
    started = time.time()
    try:
        return _run_python(code, timeout=timeout, cwd=cwd,
                           extra_env=env, phase=phase)
    except Exception as exc:  # noqa: BLE001 — песочница не должна ронять цикл
        duration = round(time.time() - started, 3)
        message = f"{type(exc).__name__}: {exc}"
        get_logger("exec").log(
            kind="exec_error",
            phase=phase,
            duration=duration,
            error=message,
        )
        return ExecResult(stdout="", stderr=message, exit_code=-1, duration=duration)
