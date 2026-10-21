"""tools.py — детерминированные ML-инструменты (зона владения B).

Модель — LightGBM (должен быть в requirements.txt организаторов).

Без LLM и без хардкода под датасет: id-колонка и таргет определяются
по sample_submission.csv, тип задачи — по самому таргету.
Модуль самодостаточен (нет импортов из соседних модулей), поэтому его
можно положить и в корень архива, и в подпапку.

CV: KFold(5, shuffle=True, random_state=42) для регрессии,
StratifiedKFold с тем же seed для классификации. Не менять без ревью B.
"""
import os
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold, StratifiedKFold

SEED = 42
N_FOLDS = 5


# ----------------------------- метрики ------------------------------------
def rmse(y_true, y_pred):
    """Корень из среднего квадрата ошибки."""
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def macro_f1(y_true, y_pred):
    """Macro F1: F1 по каждому классу, среднее с равными весами."""
    return float(f1_score(y_true, y_pred, average="macro"))


# ----------------------------- профиль ------------------------------------
def detect_task(y):
    """Определяет тип задачи по таргету: 'classification' или 'regression'."""
    # любой нечисловой таргет (object / str / category) и bool — классификация
    if not pd.api.types.is_numeric_dtype(y) or pd.api.types.is_bool_dtype(y):
        return "classification"
    if pd.api.types.is_integer_dtype(y) and y.nunique() <= 20:
        return "classification"
    return "regression"


def profile(train, target, id_col):
    """Краткий JSON-совместимый профиль данных (для LLM и логов)."""
    feats = [c for c in train.columns if c not in (target, id_col)]
    y = train[target]
    task = detect_task(y)
    info = {
        "n_rows": int(len(train)),
        "n_features": len(feats),
        "task": task,
        "target": target,
        "n_numeric": int(sum(pd.api.types.is_numeric_dtype(train[c]) for c in feats)),
        "n_categorical": int(sum(not pd.api.types.is_numeric_dtype(train[c]) for c in feats)),
        "top_missing": {c: round(float(train[c].isna().mean()), 3)
                        for c in train[feats].isna().mean().sort_values(ascending=False).index[:10]
                        if train[c].isna().mean() > 0},
    }
    if task == "classification":
        info["class_balance"] = {str(k): round(float(v), 4)
                                 for k, v in y.value_counts(normalize=True).items()}
    else:
        info["target_stats"] = {k: float(v) for k, v in y.describe().items()}
        info["target_skew"] = float(y.skew())
    return info


# ----------------------------- признаки -----------------------------------
def prepare_features(train, test, id_col, target):
    """Простая предобработка: числа как есть, категории -> коды.

    Коды категорий строятся по train+test без таргета (утечки нет).
    Пропуски GBDT обрабатывает сам. Константные колонки удаляются.
    """
    feats = [c for c in train.columns if c not in (target, id_col)]
    both = pd.concat([train[feats], test[feats]], axis=0, ignore_index=True)
    for c in feats:
        if both[c].dtype == bool:
            both[c] = both[c].astype(int)
        elif not pd.api.types.is_numeric_dtype(both[c]):
            both[c] = both[c].astype("category").cat.codes.replace(-1, np.nan)
    keep = [c for c in feats if both[c].nunique(dropna=False) > 1]
    both = both[keep]
    # LightGBM не принимает спецсимволы в именах колонок -> чистим и делаем уникальными
    names, seen = [], set()
    for c in keep:
        base = re.sub(r"[^0-9a-zA-Z_]", "_", str(c))
        s, k = base, 1
        while s in seen:
            s, k = f"{base}_{k}", k + 1
        seen.add(s)
        names.append(s)
    both.columns = names
    n = len(train)
    return both.iloc[:n].reset_index(drop=True), both.iloc[n:].reset_index(drop=True)


# ----------------------------- модели -------------------------------------
def make_model(task, n_classes=None, class_weight=None, params=None):
    """Фабрика модели: LightGBM (LGBMClassifier / LGBMRegressor).

    params переопределяют значения по умолчанию (для гиперпараметров B).
    """
    base = dict(n_estimators=500, learning_rate=0.05, num_leaves=63,
                subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                n_jobs=8, random_state=SEED, verbose=-1)
    base.update(params or {})
    if task == "classification":
        return lgb.LGBMClassifier(class_weight=class_weight, **base)
    return lgb.LGBMRegressor(**base)


def get_splitter(task):
    """Единая схема CV для всех экспериментов (сравнимость скоров)."""
    if task == "classification":
        return StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED)
    return KFold(N_FOLDS, shuffle=True, random_state=SEED)


def train_eval(X, y, X_test, task, params=None, class_weight=None, time_limit=None):
    """Обучает модель по K фолдам.

    Возвращает dict: score (OOF), oof, test_pred (среднее по фолдам),
    duration. Для классификации oof/test_pred — вероятности (n, K),
    y должен быть закодирован в 0..K-1.
    """
    t0 = time.time()
    n_classes = int(y.max()) + 1 if task == "classification" else None
    oof = np.zeros((len(X), n_classes)) if n_classes else np.zeros(len(X))
    test_pred = np.zeros((len(X_test), n_classes)) if n_classes else np.zeros(len(X_test))
    done = 0
    for tr_idx, va_idx in get_splitter(task).split(X, y):
        if time_limit and done > 0 and time.time() - t0 > time_limit:
            break  # не укладываемся — выходим с тем, что есть
        m = make_model(task, n_classes, class_weight, params)
        m.fit(X.iloc[tr_idx], y.iloc[tr_idx])
        if task == "classification":
            oof[va_idx] = m.predict_proba(X.iloc[va_idx])
            test_pred += m.predict_proba(X_test)
        else:
            oof[va_idx] = m.predict(X.iloc[va_idx])
            test_pred += m.predict(X_test)
        done += 1
    test_pred /= max(done, 1)
    if task == "classification":
        score = macro_f1(y, oof.argmax(axis=1))
    else:
        score = rmse(y, oof)
    return {"score": score, "oof": oof, "test_pred": test_pred,
            "folds_done": done, "duration": time.time() - t0}


def optimize_class_weights(oof_proba, y, grid=None, rounds=2):
    """Подбор множителей к вероятностям классов под Macro F1.

    Покоординатный перебор по OOF. Скор слегка оптимистичен (подгонка
    под OOF) — принимать только если прирост заметен.
    """
    grid = grid if grid is not None else np.exp(np.linspace(-1.5, 1.5, 13))
    w = np.ones(oof_proba.shape[1])
    best = macro_f1(y, (oof_proba * w).argmax(axis=1))
    for _ in range(rounds):
        for k in range(len(w)):
            for g in grid:
                w2 = w.copy()
                w2[k] = g
                s = macro_f1(y, (oof_proba * w2).argmax(axis=1))
                if s > best + 1e-9:
                    best, w = s, w2
    return w, best


# ----------------------------- сабмит -------------------------------------
def validate_submission(sub, sample):
    """Проверка формата. Возвращает список ошибок (пусто = ок)."""
    errs = []
    if list(sub.columns) != list(sample.columns):
        return [f"колонки {list(sub.columns)} != {list(sample.columns)}"]
    id_col, target = sample.columns[0], sample.columns[1]
    if len(sub) != len(sample):
        errs.append(f"длина {len(sub)} != {len(sample)}")
    if sub.isna().any().any():
        errs.append("есть пропуски")
    if sub[id_col].duplicated().any():
        errs.append("повторяющиеся id")
    if set(sub[id_col]) != set(sample[id_col]):
        errs.append("набор id не совпадает с sample_submission")
    if pd.api.types.is_numeric_dtype(sub[target]):
        if not np.isfinite(sub[target].to_numpy(dtype=float)).all():
            errs.append("есть inf/NaN в предсказаниях")
    return errs


def make_submission(test_ids, pred, sample, path):
    """Собирает submission в порядке test.csv (так требует условие) и проверяет формат."""
    id_col, target = sample.columns[0], sample.columns[1]
    sub = pd.DataFrame({id_col: test_ids.values, target: pred})
    errs = validate_submission(sub, sample)
    if errs:
        raise ValueError("Невалидный сабмит: " + "; ".join(errs))
    sub.to_csv(path, index=False)
    return sub


# ----------------------------- бейзлайн-путь ------------------------------
def run_baseline(data_dir, out_path, time_limit=None):
    """Полный бейзлайн: данные -> CV -> сабмит. Возвращает dict с логом."""
    train = pd.read_csv(os.path.join(data_dir, "train.csv"))
    test = pd.read_csv(os.path.join(data_dir, "test.csv"))
    sample = pd.read_csv(os.path.join(data_dir, "sample_submission.csv"))
    id_col, target = sample.columns[0], sample.columns[1]

    prof = profile(train, target, id_col)
    task = prof["task"]
    X, X_test = prepare_features(train, test, id_col, target)

    classes = None
    if task == "classification":
        classes = sorted(train[target].unique())
        y = train[target].map({c: i for i, c in enumerate(classes)})
    else:
        y = train[target]

    res = train_eval(X, y, X_test, task, class_weight="balanced" if classes else None,
                     time_limit=time_limit)
    if classes:
        pred = np.array(classes)[res["test_pred"].argmax(axis=1)]
    else:
        pred = res["test_pred"]
    make_submission(test[id_col], pred, sample, out_path)
    return {"task": task, "metric": "macro_f1" if classes else "rmse",
            "cv_score": res["score"], "duration": res["duration"],
            "folds_done": res["folds_done"], "profile": prof}


def find_data_dir(task_dir):
    """Ищет каталог с train.csv + test.csv + sample_submission.csv.

    В кейсе данные лежат во вложенных папках (например
    multiclass/participants/), а пути указаны в task.md; для скрытых задач
    структура может отличаться, поэтому ищем на любой глубине.
    """
    for train_path in sorted(Path(task_dir).rglob("train.csv")):
        d = train_path.parent
        if (d / "test.csv").exists() and (d / "sample_submission.csv").exists():
            return str(d)
    raise FileNotFoundError(f"В {task_dir} не найдены train/test/sample_submission")


def run_task(task_md, time_limit=None):
    """Точка входа для solution.py: python3 solution.py task.md.

    Данные ищет рядом с task.md, submission.csv кладёт в каталог task.md.
    """
    task_dir = Path(task_md).resolve().parent
    return run_baseline(find_data_dir(task_dir),
                        str(task_dir / "submission.csv"), time_limit)
