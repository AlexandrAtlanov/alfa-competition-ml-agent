"""Генерирует простую синтетическую задачу для smoke-прогона цикла.

Запуск:
    python scripts/make_demo_task.py

Создаёт examples/simple_task/ с train.csv, test.csv, sample_submission.csv,
data_dictionary.csv и task.md. Данные детерминированы (random_state=42).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

OUT_DIR = Path(__file__).resolve().parents[1] / "examples" / "simple_task"

TASK_MD = """# Задача: регрессия (синтетика)

Предсказать числовую цель `target` по табличным признакам.

## Файлы

- `train.csv` — обучающая выборка: `id`, `x1`..`x5`, `cat`, `target`.
- `test.csv` — тестовая выборка: те же признаки без `target`.
- `sample_submission.csv` — формат ответа: `id`, `target`.

## Метрика

RMSE (меньше — лучше).

## Требования

- Самодостаточный Python-скрипт, без интернета.
- Читай `train.csv` и `test.csv` из текущего каталога.
- Напечатай итоговую метрику строкой:
  `METRIC_JSON={"metric": "rmse", "value": <float>, "direction": "min"}`
- Сохрани `submission.csv` (колонки `id`, `target`) в текущем каталоге.
"""

DICTIONARY_ROWS = [
    ("id", "id", "int", "", "идентификатор строки"),
    ("x1", "feature", "float", "", "признак 1 (нормальный шум)"),
    ("x2", "feature", "float", "", "признак 2"),
    ("x3", "feature", "float", "", "признак 3 (равномерный)"),
    ("x4", "feature", "int", "", "признак 4 (категория 0..3)"),
    ("x5", "feature", "float", "", "признак 5"),
    ("cat", "feature", "str", "", "категориальный признак a/b/c"),
    ("target", "target", "float", "", "целевая переменная"),
]


def main() -> None:
    rng = np.random.default_rng(42)
    n_train, n_test = 200, 50
    n = n_train + n_test

    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(5, 2, n)
    x3 = rng.uniform(0, 10, n)
    x4 = rng.integers(0, 4, n).astype(float)
    x5 = rng.normal(0, 1, n)
    cat = rng.choice(["a", "b", "c"], size=n, p=[0.5, 0.3, 0.2])

    cat_effect = pd.Series(cat).map({"a": 0.0, "b": 2.0, "c": -1.0}).to_numpy()
    noise = rng.normal(0, 0.5, n)
    target = (
        3 * x1 + 2 * x2 - 1.5 * x3 + 0.8 * x4 + 0.5 * x5 + cat_effect + noise
    )

    frame = pd.DataFrame(
        {
            "id": np.arange(n),
            "x1": x1.round(4),
            "x2": x2.round(4),
            "x3": x3.round(4),
            "x4": x4.astype(int),
            "x5": x5.round(4),
            "cat": cat,
            "target": target.round(4),
        }
    )
    train = frame.iloc[:n_train].reset_index(drop=True)
    test = frame.iloc[n_train:].drop(columns=["target"]).reset_index(drop=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train.to_csv(OUT_DIR / "train.csv", index=False)
    test.to_csv(OUT_DIR / "test.csv", index=False)
    pd.DataFrame({"id": test["id"], "target": 0.0}).to_csv(
        OUT_DIR / "sample_submission.csv", index=False
    )
    (OUT_DIR / "task.md").write_text(TASK_MD, encoding="utf-8")
    pd.DataFrame(
        DICTIONARY_ROWS,
        columns=["feature", "role", "dtype", "unit", "description_ru"],
    ).to_csv(OUT_DIR / "data_dictionary.csv", index=False)

    print(f"Готово: {OUT_DIR}")
    print(f"  train {train.shape}, test {test.shape}")


if __name__ == "__main__":
    main()
