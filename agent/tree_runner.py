"""Песочница для узла-кода: исполняет build_model/add_features в subprocess.

Узел — это исходник (str), который пользователь/LLM правит. Здесь он
исполняется один раз на свою задачу: 5-фолдовая CV на том же сплиттере,
что и весь каркас, поэтому скоры узлов сравнимы между собой и с бейзлайном.

Контракт исходника описан в agent/portfolio.py. Дополнительно узел может
объявить TARGET_TRANSFORM и POSTPROCESS — это оставляет за кодом узла
решения, которые раньше жили в ExperimentSpec.

Файл держит только строку кода: импортировать его из агента не нужно,
`run_python` передаёт её в новый интерпретатор.
"""
from __future__ import annotations

# Wrapper исполняется в песочнице: новый процесс, пустое окружение,
# конфиг читается из config.json в рабочем каталоге узла.
RUNNER_CODE = r'''
import json, sys, time, traceback
from pathlib import Path

cfg = json.loads(Path("config.json").read_text(encoding="utf-8"))
sys.path.insert(0, cfg["project_root"])

import numpy as np
import pandas as pd
from agent import tools


def score_of(task, y, pred):
    if task == "classification":
        return tools.macro_f1(y, pred.argmax(axis=1))
    return tools.rmse(y, pred)


def build_namespace(source):
    ns = {}
    exec(compile(source, "<node_code>", "exec"), ns)
    if not callable(ns.get("build_model")):
        raise TypeError("в коде узла нет функции build_model(task_type, n_classes, n_features)")
    return ns


def main():
    t0 = time.time()
    task = cfg["task_type"]
    id_col, target = cfg["id_col"], cfg["target"]
    data = Path(cfg["data_dir"])
    train = pd.read_csv(data / "train.csv")
    test = pd.read_csv(data / "test.csv")

    y_raw = train[target].reset_index(drop=True)
    tr = train.drop(columns=[c for c in (target, id_col) if c in train.columns]).reset_index(drop=True)
    te = test.drop(columns=[c for c in (target, id_col) if c in test.columns]).reset_index(drop=True)
    n_tr, n_te = len(tr), len(te)

    ns = build_namespace(cfg["code"])
    add_features = ns.get("add_features")
    if callable(add_features):
        out = add_features(tr.copy(), te.copy())
        if not (isinstance(out, tuple) and len(out) == 2):
            raise TypeError("add_features должна вернуть (train, test)")
        tr, te = out
        if len(tr) != n_tr or len(te) != n_te:
            raise ValueError("add_features изменила число строк")
        if set(tr.columns) != set(te.columns):
            raise ValueError("add_features: разные колонки train/test")
        te = te[list(tr.columns)]

    X, X_test = tools.prepare_features(tr, te, id_col, target)
    X = X.replace([np.inf, -np.inf], np.nan)
    X_test = X_test.replace([np.inf, -np.inf], np.nan)

    if task == "classification":
        classes = sorted(y_raw.unique())
        y = y_raw.map({c: i for i, c in enumerate(classes)})
        y_fit = y
        n_classes = len(classes)
    else:
        classes = None
        y = y_raw.astype(float)
        y_fit = y
        n_classes = None
        if ns.get("TARGET_TRANSFORM") == "log1p":
            if (y <= -1).any():
                raise ValueError("log1p: в таргете есть значения <= -1")
            y_fit = np.log1p(y)

    n_features = int(X.shape[1])
    oof = (np.zeros((len(X), n_classes)) if n_classes else np.zeros(len(X)))
    test_pred = (np.zeros((len(X_test), n_classes)) if n_classes else np.zeros(len(X_test)))
    done = 0
    for tr_idx, va_idx in tools.get_splitter(task).split(X, y_fit):
        if cfg.get("time_limit") and done > 0 and time.time() - t0 > cfg["time_limit"]:
            break
        model = ns["build_model"](task, n_classes, n_features)
        model.fit(X.iloc[tr_idx], y_fit.iloc[tr_idx])
        if task == "classification":
            oof[va_idx] = model.predict_proba(X.iloc[va_idx])
            test_pred += model.predict_proba(X_test)
        else:
            oof[va_idx] = model.predict(X.iloc[va_idx])
            test_pred += model.predict(X_test)
        done += 1
    test_pred /= max(done, 1)

    if task == "regression" and ns.get("TARGET_TRANSFORM") == "log1p":
        oof, test_pred = np.expm1(oof), np.expm1(test_pred)

    folds = list(tools.get_splitter(task).split(X, y_fit))[:done]
    covered = np.concatenate([va for _, va in folds]) if folds else np.arange(0)

    weights = None
    if task == "classification" and ns.get("POSTPROCESS") == "class_weight_opt" and done == tools.N_FOLDS:
        weights, _ = tools.optimize_class_weights(oof[covered], y.iloc[covered])
        oof, test_pred = oof * weights, test_pred * weights

    fold_scores = [score_of(task, y.iloc[va], oof[va]) for _, va in folds]
    result = {
        "status": "ok",
        "score": score_of(task, y.iloc[covered], oof[covered]) if len(covered) else None,
        "fold_scores": fold_scores,
        "folds_done": done,
        "n_features": n_features,
        "class_weights": None if weights is None else [float(w) for w in weights],
        "duration": time.time() - t0,
    }
    np.save("oof.npy", oof)
    np.save("test_pred.npy", test_pred)
    Path("result.json").write_text(json.dumps(result), encoding="utf-8")
    print("RESULT_JSON=" + json.dumps({k: result[k] for k in ("score", "folds_done")}))


try:
    main()
except Exception as exc:
    Path("result.json").write_text(json.dumps({
        "status": "error",
        "error": f"{type(exc).__name__}: {exc}",
        "traceback": traceback.format_exc()[-3000:],
    }), encoding="utf-8")
    traceback.print_exc()
    sys.exit(1)
'''
