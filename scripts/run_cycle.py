"""Демонстрационный прогон полного агентского цикла.

    hypothesis -> code -> запуск в песочнице -> review -> accept/reject

Это dev-обвязка: production-цикл (`agent/loop.py`) — зона A. Здесь задача —
проверить, что связка llm + prompts + sandbox работает end-to-end и что
процесс видно в терминале.

Запуск:
    python scripts/run_cycle.py --task examples/simple_task --cycles 2
    python scripts/run_cycle.py --task examples/simple_task --dry-run   # без API

--dry-run берёт заготовленные ответы вместо вызова модели, но код исполняется
по-настоящему: так проверяется вся механика без ключа и без трат токенов.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from agent.llm import get_usage, query  # noqa: E402
from agent.sandbox import run_python  # noqa: E402
import console_ui as ui  # noqa: E402


# --- заготовки для --dry-run ------------------------------------------------

DRY_HYPOTHESIS = {
    "hypothesis": "Ordinal-кодирование cat + HistGradientBoostingRegressor",
    "rationale": "cat влияет на target, дерево не умеет строки",
    "expected_effect": "снизить RMSE за счёт учёта категории",
    "change_type": "encoding",
    "atomic": True,
    "test_plan": "5-fold CV RMSE",
}

DRY_CODE = '''import json

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

y = train["target"]
X = train.drop(columns=["target", "id"])
X_test = test.drop(columns=["id"])

cat_cols = [c for c in X.columns if X[c].dtype == object]

pre = ColumnTransformer(
    [("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), cat_cols)],
    remainder="passthrough",
)
model = Pipeline(
    [("pre", pre), ("reg", HistGradientBoostingRegressor(random_state=42, max_iter=200))]
)

cv = KFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=cv, scoring="neg_root_mean_squared_error")
rmse = float(-scores.mean())

model.fit(X, y)
pred = model.predict(X_test)
pd.DataFrame({"id": test["id"], "target": pred}).to_csv("submission.csv", index=False)

metrics = {"metric": "rmse", "value": round(rmse, 5), "direction": "min"}
with open("metrics.json", "w", encoding="utf-8") as handle:
    json.dump(metrics, handle)

print("METRIC_JSON=" + json.dumps(metrics))
'''

DRY_REVIEW = {
    "status": "ok",
    "reason": "Модель обучена, RMSE посчитан на 5-fold CV, submission.csv записан.",
    "next_action": "Попробовать добавить отношение признаков x1/x3.",
    "leak_suspected": False,
    "metric_found": True,
}


def llm_complete(
    role: str, user: str, dry_run: bool, phase: str = "RESEARCH"
) -> str | dict:
    """Вызов модели по роли (промпт и схема — из роли); в dry-run — заготовка."""
    if dry_run:
        return {
            "hypothesis": DRY_HYPOTHESIS,
            "code": DRY_CODE,
            "review": DRY_REVIEW,
        }[role]
    return query([{"role": "user", "content": user}], role=role, phase=phase)


# --- данные и профиль -------------------------------------------------------

def profile_data(task_dir: Path) -> tuple[str, "object", "object"]:
    import pandas as pd

    train = pd.read_csv(task_dir / "train.csv")
    test = pd.read_csv(task_dir / "test.csv")
    lines = [
        f"train.csv: {train.shape[0]} строк × {train.shape[1]} колонок",
        f"test.csv:  {test.shape[0]} строк × {test.shape[1]} колонок",
        "колонки train: " + ", ".join(map(str, train.columns)),
        "типы: " + ", ".join(f"{c}:{t}" for c, t in train.dtypes.astype(str).items()),
        "пропуски в train: " + str(int(train.isna().sum().sum())),
    ]
    return "\n".join(lines), train, test


# --- контексты --------------------------------------------------------------

def as_text(value) -> str:
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, indent=2)
    return str(value)


def hypothesis_text(item: dict) -> str:
    hypothesis = item.get("hypothesis")
    if isinstance(hypothesis, dict):
        return hypothesis.get("hypothesis", as_text(hypothesis))
    return str(hypothesis)


def history_text(history: list[dict], limit: int = 5) -> str:
    if not history:
        return "(пока нет)"
    lines = []
    for item in history[-limit:]:
        metric = item.get("metric")
        value = metric["value"] if metric else "—"
        verdict = "принята" if item.get("accepted") else "отклонена"
        lines.append(f"{item['cycle']}. {hypothesis_text(item)[:120]} → {value} ({verdict})")
    return "\n".join(lines)


def build_hypothesis_ctx(task_md, profile, best, history) -> str:
    best_line = "пока нет"
    if best:
        best_line = f"{best['metric']} = {best['value']} ({best['direction']})"
    return (
        f"## Описание задачи\n{task_md}\n\n"
        f"## Профиль данных\n{profile}\n\n"
        f"## Текущий лучший результат\n{best_line}\n\n"
        f"## Уже проверенные гипотезы\n{history_text(history)}\n\n"
        "Предложи следующую гипотезу."
    )


def build_code_ctx(task_md, profile, hypothesis) -> str:
    return (
        f"## Описание задачи\n{task_md}\n\n"
        f"## Профиль данных\n{profile}\n\n"
        f"## Гипотеза для проверки\n{as_text(hypothesis)}\n\n"
        "Напиши код, который проверяет эту гипотезу. Работай в каталоге, "
        "где лежат train.csv и test.csv."
    )


def build_review_ctx(hypothesis, code, result, metric) -> str:
    metric_line = "не найдена"
    if metric:
        metric_line = f"{metric['metric']} = {metric['value']} ({metric['direction']})"
    return (
        f"## Гипотеза\n{as_text(hypothesis)}\n\n"
        f"## Код\n{code}\n\n"
        f"## Результат запуска\n"
        f"exit_code={result.exit_code}, timed_out={result.timed_out}, "
        f"duration={result.duration}s\n\n"
        f"stdout:\n{result.stdout}\n\n"
        f"stderr:\n{result.stderr}\n\n"
        f"## Найденная метрика\n{metric_line}\n"
    )


# --- метрика ----------------------------------------------------------------

_METRIC_JSON_RE = re.compile(r"METRIC_JSON\s*=\s*(\{[^\n]*\})")


def extract_metric(stdout: str | None, task_dir: Path | None = None) -> dict | None:
    """Метрика: сначала metrics.json, затем METRIC_JSON в stdout, затем regex."""
    if task_dir is not None:
        path = task_dir / "metrics.json"
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(data, dict) and "value" in data:
                    return data
            except (json.JSONDecodeError, OSError):
                pass

    if stdout:
        match = _METRIC_JSON_RE.search(stdout)
        if match:
            try:
                data = json.loads(match.group(1))
                if "value" in data:
                    return data
            except json.JSONDecodeError:
                pass

        fallback = [
            (r"(?i)rmse[^0-9\-]*([0-9]*\.?[0-9]+)", "rmse", "min"),
            (r"(?i)macro[ _-]?f1[^0-9\-]*([0-9]*\.?[0-9]+)", "f1", "max"),
            (r"(?i)accuracy[^0-9\-]*([0-9]*\.?[0-9]+)", "accuracy", "max"),
            (r"(?i)r2[^0-9\-]*([0-9]*\.?[0-9]+)", "r2", "max"),
        ]
        for pattern, name, direction in fallback:
            match = re.search(pattern, stdout)
            if match:
                return {"metric": name, "value": float(match.group(1)), "direction": direction}
    return None


def is_better(metric: dict | None, best: dict | None) -> bool:
    if not metric or metric.get("value") is None:
        return False
    if best is None:
        return True
    if metric.get("direction", "min") == "min":
        return metric["value"] < best["value"]
    return metric["value"] > best["value"]


# --- вспомогательное --------------------------------------------------------

def usage_delta(before) -> dict:
    after = get_usage()
    return {
        "calls": after.calls - before.calls,
        "prompt_tokens": after.prompt_tokens - before.prompt_tokens,
        "completion_tokens": after.completion_tokens - before.completion_tokens,
        "total_tokens": after.total_tokens - before.total_tokens,
    }


def fmt_usage(delta: dict) -> str:
    if delta["calls"] == 0:
        return "0 токенов (dry-run)"
    return f"{delta['total_tokens']} токенов (in {delta['prompt_tokens']} / out {delta['completion_tokens']})"


def cleanup_outputs(task_dir: Path) -> None:
    """Убрать артефакты прошлого цикла, чтобы не прочитать устаревшую метрику."""
    for name in ("metrics.json", "submission.csv"):
        (task_dir / name).unlink(missing_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Демо-прогон цикла AutoResearch")
    parser.add_argument("--task", default="examples/simple_task", help="каталог задачи")
    parser.add_argument("--cycles", type=int, default=1, help="число гипотез")
    parser.add_argument("--timeout", type=int, default=300, help="секунд на запуск кода")
    parser.add_argument("--dry-run", action="store_true", help="без вызова LLM")
    return parser.parse_args()


def run(args: argparse.Namespace) -> int:
    task_dir = Path(args.task).resolve()
    if not (task_dir / "task.md").exists():
        ui.info(f"Нет task.md в {task_dir}")
        return 1

    ui.banner(
        "AutoResearch — демо-цикл",
        f"задача: {task_dir.name} | dry_run={args.dry_run} | cycles={args.cycles} "
        f"| timeout={args.timeout}s",
    )

    task_md = (task_dir / "task.md").read_text(encoding="utf-8")
    profile, _, _ = profile_data(task_dir)
    ui.section("Профиль данных")
    ui.text_panel("Профиль", profile)

    history: list[dict] = []
    best: dict | None = None
    rows: list[tuple] = []

    for cycle in range(1, args.cycles + 1):
        cycle_t0 = time.time()
        cycle_usage0 = get_usage()

        # 1. Гипотеза
        ui.section(f"ЦИКЛ {cycle} · шаг 1: гипотеза")
        u0 = get_usage()
        hypothesis = llm_complete(
            "hypothesis", build_hypothesis_ctx(task_md, profile, best, history), args.dry_run
        )
        ui.text_panel("Гипотеза", as_text(hypothesis))
        ui.info(f"  {fmt_usage(usage_delta(u0))}")

        # 2. Код
        ui.section(f"ЦИКЛ {cycle} · шаг 2: генерация кода")
        u0 = get_usage()
        code = llm_complete("code", build_code_ctx(task_md, profile, hypothesis), args.dry_run)
        ui.code_panel(code)
        ui.info(f"  {fmt_usage(usage_delta(u0))}")

        # 3. Запуск
        ui.section(f"ЦИКЛ {cycle} · шаг 3: запуск в песочнице")
        cleanup_outputs(task_dir)
        result = run_python(code, timeout=args.timeout, cwd=task_dir, phase="RESEARCH")
        ui.exec_panel(result)

        metric = extract_metric(result.stdout, task_dir)
        ui.metric_panel(metric)
        submission = task_dir / "submission.csv"
        ui.info(f"  submission.csv: {'создан' if submission.exists() else 'НЕ создан'}")

        # 4. Разбор
        ui.section(f"ЦИКЛ {cycle} · шаг 4: разбор результата")
        u0 = get_usage()
        review = llm_complete(
            "review", build_review_ctx(hypothesis, code, result, metric), args.dry_run
        )
        ui.text_panel("Разбор", as_text(review))
        ui.info(f"  {fmt_usage(usage_delta(u0))}")

        # 5. Решение (детерминированное, по метрике)
        accepted = is_better(metric, best)
        if accepted:
            best = metric
        ui.decision_panel(accepted, "" if metric else "метрика не найдена")

        cycle_delta = usage_delta(cycle_usage0)
        cycle_dt = time.time() - cycle_t0
        item = {
            "cycle": cycle,
            "hypothesis": hypothesis,
            "metric": metric,
            "accepted": accepted,
            "tokens": cycle_delta["total_tokens"],
            "duration": cycle_dt,
        }
        history.append(item)
        rows.append(
            (
                cycle,
                hypothesis_text(item)[:48] + ("…" if len(hypothesis_text(item)) > 48 else ""),
                metric["value"] if metric else "—",
                "✔" if accepted else "✘",
                f"{cycle_delta['total_tokens']} tok",
                f"{cycle_dt:.1f}s",
            )
        )

    ui.summary(rows, ["цикл", "гипотеза", "метрика", "решение", "токены", "время"])
    if best:
        ui.info(f"\nЛучший результат: {best['metric']} = {best['value']} ({best['direction']})")
    return 0


def main() -> int:
    args = parse_args()
    try:
        return run(args)
    except Exception as exc:  # noqa: BLE001 — демо: показать ошибку и выйти
        ui.info(f"Ошибка: {type(exc).__name__}: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
