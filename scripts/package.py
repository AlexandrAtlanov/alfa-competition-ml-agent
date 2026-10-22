"""Сборка сабмит-архива для платформы (зона капитана).

Контракт платформы (инструкция организаторов, раздел «Что положить в ZIP»):
- все файлы лежат в КОРНЕ архива, без папок;
- в каталог задачи копируются только корневые .py-файлы и requirements.txt;
- вложенные папки НЕ копируются: необходимый модуль в подпапке недоступен
  при запуске (риск 0 баллов); не-.py файлы (промпты, конфиги) тоже
  не копируются.

Поэтому архив состоит ровно из двух файлов в корне:
- solution.py — АВТОГЕНЕРАТ: bootstrap-заголовок распаковывает встроенный
  сжатый пакет agent/ (только .py и .md из репозитория) в подкаталог
  _agent_rt/ каталога задачи (промежуточные файлы в подкаталогах задачи
  разрешены инструкцией) и затем выполняется исходный код solution.py,
  приложенный в конец файла без изменений;
- requirements.txt — requirements-org.txt под именем, которое ждёт платформа.

Проверки перед сборкой (блокируют сабмит):
    синтаксис всех .py; случайно закоммиченные секреты (sk-/gsk-, ключи в
    коде); отсутствие solution.py / requirements-org.txt / agent/__init__.py.
    После записи: в архиве ровно 2 корневые записи, автогенерат компилируется.

Запуск:
    python scripts/package.py               # соберёт dist/solution.zip
    python scripts/package.py --check       # только проверки, без записи
    python scripts/package.py --out my.zip  # свой путь

Код возврата: 0 — архив готов; 1 — сабмит заблокирован (читай сообщения).
"""
from __future__ import annotations

import argparse
import ast
import base64
import re
import zipfile
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AGENT = ROOT / "agent"
EXCLUDED_PARTS = {"__pycache__", ".pytest_cache", ".git", ".venv", "logs", "dist"}

# Назначение секретного значения или вставленный ключ (sk-…, gsk-…)
SECRET_RE = re.compile(
    r'OPENAI_API_KEY\s*=\s*["\']?[A-Za-z0-9_\-]{8,}|(?:sk|gsk)-[A-Za-z0-9]{20,}'
)


def agent_files() -> list[Path]:
    """Файлы пакета agent: *.py и *.md, без мусора."""
    if not AGENT.exists():
        return []
    out = []
    for p in sorted(AGENT.rglob("*")):
        if not p.is_file():
            continue
        if any(part in EXCLUDED_PARTS for part in p.parts):
            continue
        if p.suffix in {".py", ".md"}:
            out.append(p)
    return out


def scan_problems(files: list[Path]) -> list[str]:
    """Синтаксис всех .py + поиск случайно закоммиченных секретов."""
    problems: list[str] = []
    for p in files:
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            problems.append(f"не читается {p.name}: {exc}")
            continue
        if p.suffix == ".py":
            try:
                ast.parse(text)
            except SyntaxError as exc:
                problems.append(f"не компилируется {p.name}: {exc}")
        for match in SECRET_RE.finditer(text):
            problems.append(f"похоже на секрет в {p.name}: «{match.group(0)[:24]}…»")
    return problems


def build_payload(files: list[Path]) -> dict[str, bytes]:
    """Упаковать agent/** в {относительный posix-путь: zlib+b85(байты)}."""
    payload: dict[str, bytes] = {}
    for p in files:
        rel = p.relative_to(ROOT).as_posix()
        payload[rel] = base64.b85encode(zlib.compress(p.read_bytes(), 9))
    return payload


def build_standalone(payload: dict[str, bytes], original: str) -> str:
    """Сгенерировать самодостаточный solution.py: bootstrap + исходник.

    Bootstrap выполняется до исходного кода: распаковывает _PAYLOAD в
    подкаталог _agent_rt/ рядом с собой и добавляет его в sys.path. Далее
    работает исходный solution.py (appended as-is): его «from agent import
    ...» находит материализованный пакет, промпты .md лежат на месте.
    """
    head = (
        "# -*- coding: utf-8 -*-\n"
        "# solution.py — АВТОГЕНЕРАТ scripts/package.py. Не править руками:\n"
        "# изменения вносятся в решение в репозитории, архив пересобирается\n"
        "# package.py. Платформа копирует в каталог задачи только корневые\n"
        "# .py-файлы и requirements.txt (инструкция организаторов), поэтому\n"
        "# пакет agent/ встроен ниже сжатым блоком и распаковывается при\n"
        "# старте в подкаталог _agent_rt/ (промежуточные файлы в подкаталогах\n"
        "# задачи разрешены).\n"
        "import base64 as _b64\n"
        "import sys as _sys\n"
        "import zlib as _zlib\n"
        "from pathlib import Path as _Path\n\n"
        "_RT_DIR = _Path(__file__).resolve().parent / \"_agent_rt\"\n"
        "_PAYLOAD = "
    )
    body = (
        "\n\n"
        "def _materialize() -> None:\n"
        "    for _rel, _blob in _PAYLOAD.items():\n"
        "        _target = _RT_DIR.joinpath(*_rel.split(\"/\"))\n"
        "        _target.parent.mkdir(parents=True, exist_ok=True)\n"
        "        _target.write_bytes(_zlib.decompress(_b64.b85decode(_blob)))\n"
        "    _p = str(_RT_DIR)\n"
        "    if _p not in _sys.path:\n"
        "        _sys.path.insert(0, _p)\n"
        "\n\n"
        "_materialize()\n"
        "del _b64, _sys, _zlib, _Path, _RT_DIR, _PAYLOAD, _materialize\n"
        "\n"
        "# ---- ниже исходный solution.py из репозитория (без изменений) ----\n"
        "\n"
    )
    return head + repr(payload) + body + original


def main() -> int:
    parser = argparse.ArgumentParser(description="Сборка сабмит-архива")
    parser.add_argument("--out", default="dist/solution.zip", help="путь к ZIP")
    parser.add_argument("--check", action="store_true",
                        help="только проверки, архив не писать")
    args = parser.parse_args()

    problems: list[str] = []
    solution = ROOT / "solution.py"
    req = ROOT / "requirements-org.txt"

    if not solution.exists():
        problems.append("solution.py отсутствует (зона A) — на платформу нечего отправлять")
    if not req.exists():
        problems.append("requirements-org.txt отсутствует")
    if not (AGENT / "__init__.py").exists():
        problems.append("agent/__init__.py отсутствует — пакет не импортируется")

    files = agent_files()
    if not files:
        problems.append("пакет agent/ пуст или не найден")
    scan_list = files[:]
    if solution.exists():
        scan_list.append(solution)
    if req.exists():
        scan_list.append(req)
    problems.extend(scan_problems(scan_list))

    if problems:
        print("БЛОКЕРЫ САБМИТА:")
        for item in problems:
            print(f"  - {item}")
        return 1

    payload = build_payload(files)
    standalone = build_standalone(payload, solution.read_text(encoding="utf-8"))

    # автогенерат обязан компилироваться ещё до записи в архив
    try:
        ast.parse(standalone)
    except SyntaxError as exc:
        print(f"БЛОКЕР САБМИТА: автогенерат solution.py не компилируется: {exc}")
        return 1

    if args.check:
        print("Проверки пройдены. Архив: 2 корневые записи, без папок.")
        print("    solution.py — автогенерат, внутри пакета agent/:")
        for rel in sorted(payload):
            print(f"      + {rel}")
        print("    requirements.txt")
        return 0

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("solution.py", standalone)
        zf.writestr("requirements.txt", req.read_text(encoding="utf-8"))

    with zipfile.ZipFile(out) as zf:
        names = zf.namelist()
    expected = {"solution.py", "requirements.txt"}
    bad = [n for n in names if n not in expected or "/" in n or "\\" in n]
    if set(names) != expected or bad:
        out.unlink(missing_ok=True)
        print(f"ОШИБКА: структура архива не плоская: {names}")
        return 1

    print(f"OK: {out} — {len(names)} файла (плоско, без папок), "
          f"{out.stat().st_size / 1024:.1f} KB")
    print("    solution.py — автогенерат, внутри пакета agent/:")
    for rel in sorted(payload):
        print(f"      + {rel}")
    print("    requirements.txt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
