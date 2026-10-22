"""Сборка сабмит-архива для платформы (зона капитана).

Что кладём в ZIP:
    solution.py       — точка входа (контракт: uv run solution.py task.md)
    agent/**          — код агента: только *.py и agent/prompts/*.md
    requirements.txt  — requirements-org.txt под именем, которое ждёт платформа

Что НЕ должно попасть (проверяется явно):
    .env и секреты, tests/, scripts/, reference/, examples/, logs/,
    __pycache__, .git, README и прочая дев-обвязка.

Запуск:
    python scripts/package.py               # соберёт dist/solution.zip
    python scripts/package.py --check       # только проверки, без записи
    python scripts/package.py --out my.zip  # свой путь

Код возврата: 0 — архив готов; 1 — сабмит заблокирован (читай сообщения).
Пока solution.py не запушен (зона A), скрипт честно скажет об этом —
это фича: он не даст собрать пустой архив.
"""
from __future__ import annotations

import argparse
import ast
import re
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AGENT = ROOT / "agent"
EXCLUDED_PARTS = {"__pycache__", ".pytest_cache", ".git", ".venv", "logs", "dist"}
FORBIDDEN_PREFIXES = (".env", "tests/", "scripts/", "reference/", "examples/")

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

    files = ([solution] if solution.exists() else []) + agent_files()
    if not agent_files():
        problems.append("пакет agent/ пуст или не найден")
    problems.extend(scan_problems(files + [req] if req.exists() else files))

    if problems:
        print("БЛОКЕРЫ САБМИТА:")
        for item in problems:
            print(f"  - {item}")
        return 1

    entries = [(p, p.relative_to(ROOT)) for p in files]
    entries.append((req, Path("requirements.txt")))  # переименование для платформы

    if args.check:
        print(f"Проверки пройдены. В архив вошло бы {len(entries)} файлов:")
        for _, arc in sorted(entries, key=lambda e: str(e[1])):
            print(f"    {arc}")
        return 0

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        for src, arc in entries:
            zf.write(src, str(arc))

    with zipfile.ZipFile(out) as zf:
        names = zf.namelist()
    leaked = [n for n in names
              if n.startswith(FORBIDDEN_PREFIXES) or "__pycache__" in n or n == ".env"]
    if leaked:
        out.unlink(missing_ok=True)
        print(f"ОШИБКА: в архив попало лишнее: {leaked}")
        return 1

    print(f"OK: {out} — {len(names)} файлов, {out.stat().st_size / 1024:.1f} KB")
    for name in sorted(names):
        print(f"    {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
