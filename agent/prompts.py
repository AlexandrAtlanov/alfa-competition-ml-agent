"""Реестр ролей и загрузка текстовых системных промптов.

Промпты живут отдельными файлами в agent/prompts/<role>.md, чтобы их
можно было читать и ревьюить как текст (правит только C). Модуль отдаёт
готовый системный промпт роли: базовый «характер» + инструкция роли.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

PROMPTS_DIR = Path(__file__).parent / "prompts"

BASE_ROLE = "agent_instruction"
# Роли: демо-цикл (hypothesis/code/review) и продовый цикл агента
# (hypothesis_batch — пачка кандидатов, patch — функция признаков).
ROLES = (
    BASE_ROLE,
    "analyze_task",
    "hypothesis",
    "hypothesis_batch",
    "code",
    "patch",
    "review",
    "portfolio",
    "node_edit",
)


@lru_cache(maxsize=None)
def load_instruction(role: str) -> str:
    """Прочитать текст инструкции роли из agent/prompts/<role>.md."""
    path = PROMPTS_DIR / f"{role}.md"
    if not path.exists():
        raise KeyError(
            f"Нет промпта для роли {role!r}: ожидался файл {path}"
        )
    return path.read_text(encoding="utf-8").strip()


def build_system(role: str) -> str:
    """Собрать системный промпт: базовый + роль-специфичная инструкция."""
    base = load_instruction(BASE_ROLE)
    if role == BASE_ROLE:
        return base
    return f"{base}\n\n{load_instruction(role)}"
