"""Jsonl-логирование событий агента в отдельные файлы.

AGENTS.md §4: логи — jsonl, каждое событие содержит фазу/действие/
длительность/токены/метрику. Здесь — низкоуровневый писатель; содержимое
событий формируют вызывающие модули (llm.py, sandbox/exec.py).

Файлы по умолчанию: logs/llm.jsonl и logs/exec.jsonl в текущем каталоге.
Каталог задаётся переменной окружения LOG_DIR.
"""
from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path

DEFAULT_LOG_DIR = "logs"


class JsonlLogger:
    """Пишет по одному json-объекту на строку, потокобезопасно."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._lock = threading.Lock()
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, **fields) -> None:
        """Добавить событие. Поле 'ts' проставляется автоматически."""
        entry = {"ts": time.time(), **fields}
        line = json.dumps(entry, ensure_ascii=False, default=str)
        with self._lock:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")


_loggers: dict[str, JsonlLogger] = {}


def get_logger(name: str = "events") -> JsonlLogger:
    """Вернуть (и закэшировать) логгер по имени: logs/<name>.jsonl."""
    if name not in _loggers:
        base = Path(os.getenv("LOG_DIR", DEFAULT_LOG_DIR))
        _loggers[name] = JsonlLogger(base / f"{name}.jsonl")
    return _loggers[name]
