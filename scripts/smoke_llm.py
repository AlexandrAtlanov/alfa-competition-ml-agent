"""Smoke-тест LLM-слоя.

Запуск из корня репозитория:
    python scripts/smoke_llm.py

Проверяет: загрузку .env, сборку системного промпта роли, один вызов
модели со структурированным ответом (JSON по схеме роли), учёт токенов
и запись в logs/llm.jsonl.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.llm import describe, get_usage, query  # noqa: E402

PROMPT = (
    "Данные: train.csv (признаки + таргет), test.csv. "
    "Метрика — RMSE. Предложи одну гипотезу для улучшения модели."
)


def main() -> int:
    print("Конфиг:", describe())
    print("\n--- запрос (structured: hypothesis) ---")
    answer = query([{"role": "user", "content": PROMPT}], role="hypothesis")
    if isinstance(answer, dict):
        print(json.dumps(answer, ensure_ascii=False, indent=2))
    else:
        print(answer)
    print("\n--- токены ---")
    print(get_usage().as_dict())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
