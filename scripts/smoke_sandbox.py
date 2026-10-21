"""Smoke-тест песочницы: запуск простого скрипта в отдельном subprocess.

Запуск из корня репозитория:
    python scripts/smoke_sandbox.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.sandbox import run_python  # noqa: E402

CODE = """
import sys
print("hello from sandbox", sys.version_info[:2])
print(2 + 2)
"""


def main() -> int:
    result = run_python(CODE, timeout=30)
    print("ok        :", result.ok)
    print("exit_code :", result.exit_code)
    print("duration  :", result.duration)
    print("stdout    :", result.stdout.strip())
    print("stderr    :", result.stderr.strip())
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
