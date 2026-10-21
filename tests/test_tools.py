"""Тесты tools.py на игрушечных данных (обе задачи, без скачивания датасетов)."""
import numpy as np
import pandas as pd
import pytest

from agent import tools


def _make_dir(tmp_path, task):
    rng = np.random.default_rng(0)
    n, m = 600, 200
    def feats(k):
        return pd.DataFrame({"a": rng.normal(size=k), "b": rng.normal(size=k),
                             "cat": rng.choice(["x", "y", "z"], size=k)})
    tr, te = feats(n), feats(m)
    if task == "reg":
        tr["target"] = 3 * tr["a"] - tr["b"] + rng.normal(scale=0.3, size=n)
        sample = pd.DataFrame({"id": range(1000, 1000 + m), "target": 0.0})
    else:
        tr["target"] = np.where(tr["a"] > 0.5, "dos", np.where(tr["b"] > 0, "scan", "norm"))
        sample = pd.DataFrame({"id": range(1000, 1000 + m), "target": "norm"})
    tr.insert(0, "id", range(n))
    te.insert(0, "id", range(1000, 1000 + m))
    tr.to_csv(tmp_path / "train.csv", index=False)
    te.to_csv(tmp_path / "test.csv", index=False)
    sample.to_csv(tmp_path / "sample_submission.csv", index=False)
    return sample


def test_regression_beats_mean_and_valid_submission(tmp_path):
    sample = _make_dir(tmp_path, "reg")
    out = tmp_path / "submission.csv"
    log = tools.run_baseline(str(tmp_path), str(out))
    y = pd.read_csv(tmp_path / "train.csv")["target"]
    assert log["task"] == "regression"
    assert log["cv_score"] < tools.rmse(y, np.full(len(y), y.mean()))
    assert tools.validate_submission(pd.read_csv(out), sample) == []


def test_classification_macro_f1_and_labels(tmp_path):
    sample = _make_dir(tmp_path, "cls")
    out = tmp_path / "submission.csv"
    log = tools.run_baseline(str(tmp_path), str(out))
    sub = pd.read_csv(out)
    assert log["task"] == "classification"
    assert log["cv_score"] > 0.8
    assert set(sub["target"]) <= {"dos", "scan", "norm"}
    assert tools.validate_submission(sub, sample) == []


def test_validator_catches_bad_submission():
    sample = pd.DataFrame({"id": [1, 2], "y": [0, 0]})
    assert tools.validate_submission(pd.DataFrame({"id": [1], "y": [0]}), sample)
    assert tools.validate_submission(pd.DataFrame({"id": [1, 2], "y": [0, None]}), sample)


def test_cv_split_is_fixed():
    X = pd.DataFrame({"a": range(50)})
    a = [v.tolist() for _, v in tools.get_splitter("regression").split(X)]
    b = [v.tolist() for _, v in tools.get_splitter("regression").split(X)]
    assert a == b


def test_class_weights_do_not_hurt():
    rng = np.random.default_rng(1)
    y = pd.Series(rng.choice([0, 1, 2], size=300, p=[0.8, 0.15, 0.05]))
    proba = rng.dirichlet(np.ones(3), size=300)
    base = tools.macro_f1(y, proba.argmax(axis=1))
    _, best = tools.optimize_class_weights(proba, y)
    assert best >= base


def test_run_task_finds_nested_data_and_writes_next_to_task_md(tmp_path):
    """Как на платформе: данные в multiclass/participants/, submission рядом с task.md."""
    nested = tmp_path / "multiclass" / "participants"
    nested.mkdir(parents=True)
    sample = _make_dir(nested, "cls")
    (tmp_path / "task.md").write_text("задача", encoding="utf-8")
    tools.run_task(str(tmp_path / "task.md"))
    sub = pd.read_csv(tmp_path / "submission.csv")
    assert tools.validate_submission(sub, sample) == []


def test_submission_keeps_test_csv_order(tmp_path):
    """Порядок строк = порядок test.csv, даже если sample_submission в другом порядке."""
    sample = _make_dir(tmp_path, "reg")
    sample.iloc[::-1].to_csv(tmp_path / "sample_submission.csv", index=False)
    out = tmp_path / "submission.csv"
    tools.run_baseline(str(tmp_path), str(out))
    test_ids = pd.read_csv(tmp_path / "test.csv")["id"].tolist()
    assert pd.read_csv(out)["id"].tolist() == test_ids


def test_feature_names_are_sanitized_and_unique():
    tr = pd.DataFrame({"id": [0, 1, 2], "a b": [1, 2, 3], "a_b": [3, 2, 1], "t": [1, 2, 3]})
    X, _ = tools.prepare_features(tr, tr.drop(columns="t"), "id", "t")
    assert X.columns.is_unique
    assert all(c.replace("_", "").isalnum() for c in X.columns)
