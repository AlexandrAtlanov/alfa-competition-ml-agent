# Задача

Напиши один самодостаточный Python-скрипт, который обучает модель
и сохраняет предсказания для тестовой выборки.

# Правила

- Один файл, без внешних данных и без интернета.
- Читай train.csv и test.csv из текущего каталога.
- Сделай честную валидацию (KFold / StratifiedKFold) и напечатай метрику.
- Сохрани submission.csv рядом с задачей: колонки и типы как в sample_submission.csv.
- Сохрани `metrics.json` рядом с задачей:
  `{"metric": "rmse", "value": 0.123, "direction": "min"}`
- Продублируй метрику строкой в stdout:
  `METRIC_JSON={"metric": "rmse", "value": 0.123, "direction": "min"}`
  (direction — "min" для RMSE, "max" для F1/accuracy).
- Разрешённые библиотеки (только из requirements.txt): pandas, numpy, scikit-learn,
  lightgbm, xgboost, catboost, imbalanced-learn, category-encoders, optuna.
- Не используй предобученные модели.
- Ответ — только один markdown-блок с кодом, без пояснений вне блока.
