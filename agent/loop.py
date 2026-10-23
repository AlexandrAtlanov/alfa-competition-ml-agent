"""Автомат фаз агента: SCOUT -> BASELINE -> RESEARCH -> ENSEMBLE -> FINALIZE.

Принципы:
- сабмит существует с конца BASELINE и перезаписывается атомарно при каждом
  принятом улучшении — принудительная остановка по таймеру ничего не ломает;
- эксперимент = ExperimentSpec (дельта к лучшему) + опциональный патч
  признаков от LLM. Обучение и метрику считает наш wrapper через
  tools.train_eval в песочнице на фиксированной CV, а не код от LLM —
  поэтому скоры сравнимы и метрику нельзя «подогнать»;
- гипотезу выбирает детерминированный ранжир из пула кандидатов: батч от
  LLM + меню проверенных ходов. Без LLM (сбой, нет ключа, кончились токены)
  цикл продолжает работать на меню.

Настройки — из окружения (см. LoopConfig.from_env).
"""
from __future__ import annotations

import ast
import json
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import tools
from .event_log import get_logger
from .registry import (
    CHANGE_TYPES, LEVELS, Candidate, ExperimentRecord, ExperimentSpec, Registry,
)
from .sandbox import run_python

PROJECT_ROOT = Path(__file__).resolve().parents[1]


# ----------------------------- конфиг -------------------------------------
def _env_float(name: str, default: float) -> float:
    """
    Читает число из переменной окружения.

    Аргументы:
    name (str): имя переменной
    default (float): значение, если переменная не задана или пустая

    Возвращает:
    float
    """
    raw = os.getenv(name, "").strip()
    return float(raw) if raw else default


@dataclass(frozen=True)
class LoopConfig:
    """
    Настройки цикла: время, итерации, бюджет токенов, режим dry-run.
    """

    time_budget: float = 10800.0     # 3 ч на задачу (условие кейса)
    safety_margin: float = 900.0     # запас до жёсткого убийства платформой
    max_iters: int = 25
    patience: int = 6                # подряд без улучшения -> стоп
    max_tokens: int | None = None    # бюджет токенов LLM на задачу
    dry_run: bool = False            # заготовки вместо LLM
    exp_timeout_cap: float = 2400.0  # потолок на один эксперимент
    batch_size: int = 4              # кандидатов в одном запросе к LLM
    llm_timeout: float = 180.0
    ensemble: bool = True            # фаза ENSEMBLE после RESEARCH
    ensemble_seeds: int = 2          # перезапусков лучшей спеки с другим seed
    ensemble_top: int = 8            # сколько лучших экспериментов в пуле блендинга
    ensemble_steps: int = 25         # шагов жадного отбора

    @classmethod
    def from_env(cls) -> "LoopConfig":
        """
        Собирает конфиг из переменных окружения AGENT_*.

        Возвращает:
        LoopConfig
        """
        tokens = os.getenv("AGENT_MAX_TOKENS", "").strip()
        return cls(
            time_budget=_env_float("AGENT_TIME_BUDGET", cls.time_budget),
            safety_margin=_env_float("AGENT_SAFETY_MARGIN", cls.safety_margin),
            max_iters=int(_env_float("AGENT_MAX_ITERS", cls.max_iters)),
            patience=int(_env_float("AGENT_PATIENCE", cls.patience)),
            max_tokens=int(tokens) if tokens else None,
            dry_run=os.getenv("AGENT_DRY_RUN", "").strip().lower() in ("1", "true", "yes"),
            exp_timeout_cap=_env_float("AGENT_EXP_TIMEOUT", cls.exp_timeout_cap),
            batch_size=int(_env_float("AGENT_BATCH_SIZE", cls.batch_size)),
            ensemble=os.getenv("AGENT_ENSEMBLE", "1").strip().lower() not in ("0", "false", "no"),
            ensemble_seeds=int(_env_float("AGENT_ENSEMBLE_SEEDS", cls.ensemble_seeds)),
            ensemble_top=int(_env_float("AGENT_ENSEMBLE_TOP", cls.ensemble_top)),
        )


# ----------------------------- контекст задачи ----------------------------
@dataclass
class TaskContext:
    """
    Всё, что известно о задаче после SCOUT; общий вход для всех фаз.
    """

    task_md: str
    task_dir: Path
    data_dir: Path
    work_dir: Path
    id_col: str
    target: str
    task_type: str                   # regression | classification
    metric: str                      # rmse | macro_f1
    direction: str                   # min | max
    classes: list | None
    profile: dict
    columns: dict                    # признак -> dtype (без id и таргета)
    data_dictionary: str
    deadline: float
    sample: pd.DataFrame = field(repr=False, default=None)
    test_ids: pd.Series = field(repr=False, default=None)
    y: np.ndarray = field(repr=False, default=None)      # таргет; классы -> 0..K-1
    folds: list = field(repr=False, default=None)        # те же фолды, что в train_eval

    @property
    def submission_path(self) -> Path:
        """
        Возвращает путь к итоговому сабмиту (рядом с task.md).

        Возвращает:
        Path
        """
        return self.task_dir / "submission.csv"

    def remaining(self) -> float:
        """
        Считает время до дедлайна.

        Возвращает:
        float: секунды (может быть отрицательным)
        """
        return self.deadline - time.time()


def _read_text(path: Path, limit: int) -> str:
    """
    Читает текстовый файл с обрезкой; при ошибке — пустая строка.

    Аргументы:
    path (Path): путь к файлу
    limit (int): максимум символов

    Возвращает:
    str
    """
    try:
        return path.read_text(encoding="utf-8")[:limit]
    except OSError:
        return ""


def scout(task_md: Path, deadline: float) -> TaskContext:
    """
    Детерминированно разбирает задачу: файлы, id/таргет, тип, профиль.

    Аргументы:
    task_md (Path): путь к task.md
    deadline (float): момент time.time(), к которому надо закончить работу

    Возвращает:
    TaskContext
    """
    task_dir = task_md.parent
    data_dir = Path(tools.find_data_dir(task_dir))
    sample = pd.read_csv(data_dir / "sample_submission.csv")
    id_col, target = sample.columns[0], sample.columns[1]
    train = pd.read_csv(data_dir / "train.csv")
    test_ids = pd.read_csv(data_dir / "test.csv", usecols=[id_col])[id_col]
    prof = tools.profile(train, target, id_col)
    task_type = prof["task"]
    classes = sorted(train[target].unique()) if task_type == "classification" else None
    if classes is not None:
        y = train[target].map({c: i for i, c in enumerate(classes)}).to_numpy()
    else:
        y = train[target].astype(float).to_numpy()
    # фолды зависят только от числа строк и y (как в wrapper) — для блендинга OOF
    folds = list(tools.get_splitter(task_type).split(np.zeros((len(y), 1)), y))

    dd_path = data_dir / "data_dictionary.csv"
    if not dd_path.exists():
        found = sorted(task_dir.rglob("data_dictionary.csv"))
        dd_path = found[0] if found else dd_path

    work_dir = task_dir / ".agent_work"
    work_dir.mkdir(parents=True, exist_ok=True)
    return TaskContext(
        task_md=_read_text(task_md, 8000),
        task_dir=task_dir,
        data_dir=data_dir,
        work_dir=work_dir,
        id_col=id_col,
        target=target,
        task_type=task_type,
        metric="macro_f1" if task_type == "classification" else "rmse",
        direction="max" if task_type == "classification" else "min",
        classes=classes,
        profile=prof,
        columns={c: str(t) for c, t in train.dtypes.items() if c not in (id_col, target)},
        data_dictionary=_read_text(dd_path, 6000),
        deadline=deadline,
        sample=sample,
        test_ids=test_ids,
        y=y,
        folds=folds,
    )


# ----------------------------- исполнение спеки ---------------------------
# Wrapper исполняется в песочнице (новый процесс, пустое окружение), читает
# config.json из своего каталога. Таргет и id отрезаются ДО патчей признаков:
# патч видит только признаки, поэтому утечь таргету через него некуда.
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


def main():
    t0 = time.time()
    spec, task = cfg["spec"], cfg["task_type"]
    id_col, target = cfg["id_col"], cfg["target"]
    data = Path(cfg["data_dir"])
    train = pd.read_csv(data / "train.csv")
    test = pd.read_csv(data / "test.csv")

    y_raw = train[target].reset_index(drop=True)
    tr = train.drop(columns=[c for c in (target, id_col) if c in train.columns]).reset_index(drop=True)
    te = test.drop(columns=[c for c in (target, id_col) if c in test.columns]).reset_index(drop=True)
    n_tr, n_te = len(tr), len(te)

    for i, src in enumerate(spec["feature_patches"]):
        ns = {}
        exec(compile(src, f"<feature_patch_{i}>", "exec"), ns)
        out = ns["add_features"](tr.copy(), te.copy())
        if not (isinstance(out, tuple) and len(out) == 2):
            raise TypeError("add_features должна вернуть (train, test)")
        tr, te = out
        if len(tr) != n_tr or len(te) != n_te:
            raise ValueError(f"патч {i} изменил число строк")
        if set(tr.columns) != set(te.columns):
            raise ValueError(f"патч {i}: разные колонки train/test: "
                             f"{sorted(set(tr.columns) ^ set(te.columns))[:10]}")
        te = te[list(tr.columns)]

    X, X_test = tools.prepare_features(tr, te, id_col, target)
    X = X.replace([np.inf, -np.inf], np.nan)
    X_test = X_test.replace([np.inf, -np.inf], np.nan)

    if task == "classification":
        classes = sorted(y_raw.unique())
        y = y_raw.map({c: i for i, c in enumerate(classes)})
        y_fit = y
    else:
        y = y_raw.astype(float)
        y_fit = y
        if spec["target_transform"] == "log1p":
            if (y <= -1).any():
                raise ValueError("log1p: в таргете есть значения <= -1")
            y_fit = np.log1p(y)

    res = tools.train_eval(X, y_fit, X_test, task, params=spec["lgbm_params"] or None,
                           class_weight=spec["class_weight"], time_limit=cfg["time_limit"])
    oof, test_pred, done = res["oof"], res["test_pred"], res["folds_done"]
    if task == "regression" and spec["target_transform"] == "log1p":
        oof, test_pred = np.expm1(oof), np.expm1(test_pred)

    # те же фолды, что в train_eval; при досрочном выходе считаем только готовые
    folds = list(tools.get_splitter(task).split(X, y_fit))[:done]
    covered = np.concatenate([va for _, va in folds])

    weights = None
    if task == "classification" and spec["postprocess"] == "class_weight_opt":
        weights, _ = tools.optimize_class_weights(oof[covered], y.iloc[covered])
        oof, test_pred = oof * weights, test_pred * weights

    fold_scores = [score_of(task, y.iloc[va], oof[va]) for _, va in folds]
    result = {
        "status": "ok",
        "score": score_of(task, y.iloc[covered], oof[covered]),
        "fold_scores": fold_scores,
        "folds_done": done,
        "n_features": int(X.shape[1]),
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


def _sandbox_env() -> dict | None:
    """
    Собирает дополнительные переменные окружения для песочницы.

    Возвращает:
    dict | None
    """
    # на Windows Python без SYSTEMROOT не стартует (локальная разработка);
    # на платформе (Linux) окружение песочницы не расширяем
    if os.name == "nt" and os.getenv("SYSTEMROOT"):
        return {"SYSTEMROOT": os.environ["SYSTEMROOT"]}
    return None


def run_spec(ctx: TaskContext, reg: Registry, spec: ExperimentSpec, *,
             candidate: Candidate | None, parent_id: int | None, phase: str,
             timeout: float) -> ExperimentRecord:
    """
    Запускает спеку в песочнице и собирает запись (без вердикта accept).

    Аргументы:
    ctx (TaskContext): контекст задачи
    reg (Registry): реестр (для номера эксперимента)
    spec (ExperimentSpec): полная спека эксперимента
    candidate (Candidate | None): гипотеза; None для бейзлайна
    parent_id (int | None): номер эксперимента, от которого строится спека
    phase (str): фаза для логов (BASELINE / RESEARCH)
    timeout (float): лимит песочницы, секунды

    Возвращает:
    ExperimentRecord
    """
    exp_id = reg.next_id()
    exp_dir = ctx.work_dir / f"exp_{exp_id:03d}"
    exp_dir.mkdir(parents=True, exist_ok=True)
    for name in ("result.json", "oof.npy", "test_pred.npy"):
        (exp_dir / name).unlink(missing_ok=True)
    config = {
        "project_root": str(PROJECT_ROOT),
        "data_dir": str(ctx.data_dir),
        "id_col": ctx.id_col,
        "target": ctx.target,
        "task_type": ctx.task_type,
        "spec": spec.to_dict(),
        # train_eval выходит между фолдами — оставляем запас до kill песочницы
        "time_limit": max(30.0, timeout * 0.8),
    }
    (exp_dir / "config.json").write_text(json.dumps(config, ensure_ascii=False, default=str),
                                         encoding="utf-8")

    exec_res = run_python(RUNNER_CODE, timeout=timeout, cwd=exp_dir,
                          env=_sandbox_env(), phase=phase)
    rec = ExperimentRecord(
        exp_id=exp_id, parent_id=parent_id, phase=phase,
        candidate=candidate.to_dict() if candidate else None,
        spec=spec.to_dict(), signature=spec.signature(),
        status="error", duration=exec_res.duration,
        artifacts={"dir": str(exp_dir)},
    )
    result = {}
    try:
        result = json.loads((exp_dir / "result.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        pass

    if exec_res.timed_out:
        rec.status, rec.reason = "timeout", f"превышен лимит {timeout:.0f} с"
    elif result.get("status") == "ok":
        rec.status = "ok"
        rec.cv_score = float(result["score"])
        rec.fold_scores = [float(s) for s in result["fold_scores"]]
        rec.artifacts.update(oof=str(exp_dir / "oof.npy"),
                             test_pred=str(exp_dir / "test_pred.npy"))
        if result["folds_done"] < tools.N_FOLDS:
            rec.reason = f"неполная CV: {result['folds_done']}/{tools.N_FOLDS} фолдов"
    else:
        rec.reason = result.get("error") or (exec_res.stderr or "нет result.json")[-600:]
        rec.artifacts["traceback"] = result.get("traceback") or exec_res.stderr[-3000:]
    return rec


def write_submission(ctx: TaskContext, rec: ExperimentRecord) -> None:
    """
    Пишет сабмит из test_pred эксперимента: tmp-файл + атомарная замена.

    Аргументы:
    ctx (TaskContext): контекст задачи
    rec (ExperimentRecord): эксперимент с артефактом test_pred
    """
    pred = np.load(rec.artifacts["test_pred"])
    if ctx.task_type == "classification":
        pred = np.array(ctx.classes)[pred.argmax(axis=1)]
    tmp = ctx.task_dir / "submission.tmp.csv"
    tools.make_submission(ctx.test_ids, pred, ctx.sample, str(tmp))
    os.replace(tmp, ctx.submission_path)


# ----------------------------- вердикт ------------------------------------
def improvement(ctx: TaskContext, old: float, new: float) -> float:
    """
    Вычисляет прирост метрики с учётом направления оптимизации.

    Аргументы:
    ctx (TaskContext): контекст задачи (direction)
    old (float): скор текущего лучшего
    new (float): скор нового эксперимента

    Возвращает:
    float: положительное значение — new лучше old
    """
    return (old - new) if ctx.direction == "min" else (new - old)


def judge(ctx: TaskContext, rec: ExperimentRecord, best: ExperimentRecord) -> tuple[bool, str]:
    """
    Решает, принять ли эксперимент: улучшение больше шума и держится на большинстве фолдов.

    Аргументы:
    ctx (TaskContext): контекст задачи
    rec (ExperimentRecord): новый эксперимент
    best (ExperimentRecord): текущий лучший

    Возвращает:
    tuple[bool, str]: (принят, пояснение)
    """
    if rec.status != "ok":
        return False, rec.reason or rec.status
    if len(rec.fold_scores) < tools.N_FOLDS:
        return False, rec.reason or "неполная CV"
    gain = improvement(ctx, best.cv_score, rec.cv_score)
    min_delta = 0.002 * abs(best.cv_score) if ctx.metric == "rmse" else 0.002
    if (rec.spec.get("postprocess") == "class_weight_opt"
            and best.spec.get("postprocess") != "class_weight_opt"):
        min_delta *= 2  # подбор весов по OOF оптимистичен
    wins = sum(improvement(ctx, b, n) > 0 for b, n in zip(best.fold_scores, rec.fold_scores))
    need = math.ceil(0.6 * len(rec.fold_scores))
    detail = f"Δ={gain:+.5g} (порог {min_delta:.3g}), лучше в {wins}/{len(rec.fold_scores)} фолдах"
    if gain < min_delta:
        return False, "прирост ниже порога: " + detail
    if wins < need:
        return False, "нестабильно по фолдам: " + detail
    return True, detail


# ----------------------------- кандидаты: меню ----------------------------
def menu(ctx: TaskContext, best: ExperimentSpec) -> list[Candidate]:
    """
    Строит детерминированные ходы по профилю данных — фолбэк и «разминка» без LLM.

    Аргументы:
    ctx (TaskContext): контекст задачи
    best (ExperimentSpec): спека текущего лучшего

    Возвращает:
    list[Candidate]
    """
    p = best.lgbm_params
    lr = p.get("learning_rate", 0.05)
    n_est = p.get("n_estimators", 500)
    out = [
        Candidate("Медленнее учим, больше деревьев",
                  "Меньший learning_rate с пропорционально большим числом деревьев обычно "
                  "даёт небольшой стабильный прирост GBDT.",
                  "model", {"lgbm_params": {"learning_rate": round(lr * 0.6, 4),
                                            "n_estimators": int(min(5000, n_est * 2))}},
                  expected_gain="med", cost="high"),
        Candidate("Проще деревья: num_leaves 31",
                  "Меньше листьев — меньше переобучение на шумных табличных данных.",
                  "model", {"lgbm_params": {"num_leaves": 31}}, expected_gain="low", cost="low"),
        Candidate("Сложнее деревья: num_leaves 127",
                  "Больше листьев — больше взаимодействий признаков на крупной выборке.",
                  "model", {"lgbm_params": {"num_leaves": 127}}, expected_gain="low", cost="med"),
        Candidate("Регуляризация листьев: min_child_samples 50",
                  "Более крупные листья сглаживают предсказания и снижают дисперсию.",
                  "model", {"lgbm_params": {"min_child_samples": 50}},
                  expected_gain="low", cost="low"),
        Candidate("Сильнее сэмплирование признаков: colsample 0.5",
                  "Декорреляция деревьев при большом числе признаков.",
                  "model", {"lgbm_params": {"colsample_bytree": 0.5}},
                  expected_gain="low", cost="low"),
    ]
    if ctx.task_type == "regression":
        stats = ctx.profile.get("target_stats", {})
        if ctx.profile.get("target_skew", 0) > 1 and stats.get("min", -1) > -1:
            out.append(Candidate(
                "log1p-преобразование таргета",
                "Таргет скошен вправо: обучение в лог-шкале уменьшает влияние выбросов.",
                "preprocessing", {"target_transform": "log1p"},
                expected_gain="med", cost="low"))
    else:
        flip = None if best.class_weight == "balanced" else "balanced"
        out.append(Candidate(
            f"class_weight={flip}",
            "Баланс классов напрямую влияет на Macro F1: проверяем обратную настройку весов.",
            "preprocessing", {"class_weight": flip}, expected_gain="med", cost="low"))
        if best.postprocess != "class_weight_opt":
            out.append(Candidate(
                "Подбор порогов (весов вероятностей) под Macro F1",
                "Argmax не оптимален для Macro F1 при дисбалансе; веса подбираются по OOF.",
                "other", {"postprocess": "class_weight_opt"}, expected_gain="med", cost="low"))
    return out


# ----------------------------- кандидаты: LLM -----------------------------
# Разрешённые гиперпараметры LightGBM: тип и допустимый диапазон.
PARAM_RANGES = {
    "learning_rate": (float, 0.005, 0.3),
    "n_estimators": (int, 100, 5000),
    "num_leaves": (int, 7, 511),
    "max_depth": (int, -1, 16),
    "min_child_samples": (int, 5, 500),
    "subsample": (float, 0.3, 1.0),
    "colsample_bytree": (float, 0.2, 1.0),
    "reg_alpha": (float, 0.0, 100.0),
    "reg_lambda": (float, 0.0, 100.0),
    "min_split_gain": (float, 0.0, 10.0),
}

CANDIDATES_SCHEMA = {
    "type": "object",
    "properties": {
        "candidates": {
            "type": "array",
            "maxItems": 8,
            "items": {
                "type": "object",
                "properties": {
                    "title": {"type": "string", "minLength": 5},
                    "rationale": {"type": "string"},
                    "change_type": {"type": "string", "enum": list(CHANGE_TYPES)},
                    "delta": {"type": "object"},
                    "feature_idea": {"type": ["string", "null"]},
                    "expected_gain": {"type": "string", "enum": list(LEVELS)},
                    "cost": {"type": "string", "enum": list(LEVELS)},
                },
                "required": ["title", "rationale", "change_type", "delta",
                             "expected_gain", "cost"],
            },
        },
    },
    "required": ["candidates"],
}

BATCH_INSTRUCTION = """\
# Задача

Предложи {k} РАЗНЫХ кандидатов-гипотез для улучшения текущей лучшей модели.
Модель и CV фиксированы каркасом: LightGBM, 5-fold, метрику считает каркас.
Каждый кандидат — ОДНО атомарное изменение относительно текущей лучшей спеки.

# Что можно менять (поле delta)

- "lgbm_params": {{...}} — только ключи: {params}.
- "class_weight": null | "balanced"  (только классификация).
- "target_transform": null | "log1p"  (только регрессия, таргет > -1).
- "postprocess": null | "class_weight_opt"  (только классификация: подбор весов под Macro F1).
- Новые признаки: delta = {{}}, а в "feature_idea" — конкретное описание, какие признаки
  построить из каких колонок (имена колонок — точно как в списке). Код напишет отдельный шаг.

# Правила

- Не повторяй уже проверенные гипотезы из истории; учитывай, что не сработало.
- Признаки должны вычисляться построчно или по train+test без таргета.
- expected_gain / cost: low | med | high (cost — время обучения).
- change_type: feature | encoding | preprocessing | model | validation | ensemble | other.
"""


def _sanitize_delta(ctx: TaskContext, delta: dict) -> dict:
    """
    Оставляет в дельте только понятное каркасу и приводит значения к диапазонам.

    Аргументы:
    ctx (TaskContext): контекст задачи (тип задачи)
    delta (dict): дельта от LLM

    Возвращает:
    dict: очищенная дельта (может быть пустой)
    """
    clean: dict = {}
    params = {}
    for key, value in (delta.get("lgbm_params") or {}).items():
        if key not in PARAM_RANGES or value is None or isinstance(value, bool):
            continue
        kind, lo, hi = PARAM_RANGES[key]
        try:
            value = kind(value)
        except (TypeError, ValueError):
            continue
        params[key] = min(max(value, lo), hi)
    if params:
        clean["lgbm_params"] = params
    if ctx.task_type == "classification":
        if delta.get("class_weight") in (None, "balanced") and "class_weight" in delta:
            clean["class_weight"] = delta["class_weight"]
        if delta.get("postprocess") in (None, "class_weight_opt") and "postprocess" in delta:
            clean["postprocess"] = delta["postprocess"]
    elif delta.get("target_transform") in (None, "log1p") and "target_transform" in delta:
        clean["target_transform"] = delta["target_transform"]
    return clean


def _parse_candidates(ctx: TaskContext, data: dict) -> list[Candidate]:
    """
    Превращает ответ LLM в кандидатов; пустые и неатомарные отбрасывает.

    Аргументы:
    ctx (TaskContext): контекст задачи
    data (dict): ответ по CANDIDATES_SCHEMA

    Возвращает:
    list[Candidate]
    """
    out = []
    for item in data.get("candidates", []):
        delta = _sanitize_delta(ctx, item.get("delta") or {})
        idea = (item.get("feature_idea") or "").strip() or None
        if idea and delta:
            idea = None  # атомарность: признаки — отдельным кандидатом
        if not delta and not idea:
            continue
        out.append(Candidate(
            title=item["title"].strip()[:160],
            rationale=item.get("rationale", "")[:600],
            change_type="feature" if idea else item["change_type"],
            delta=delta, feature_idea=idea,
            expected_gain=item["expected_gain"], cost=item["cost"], source="llm",
        ))
    return out


def _context_block(ctx: TaskContext, reg: Registry) -> str:
    """
    Собирает пользовательский промпт для батча гипотез.

    Аргументы:
    ctx (TaskContext): контекст задачи
    reg (Registry): реестр (лучшая спека и история)

    Возвращает:
    str
    """
    best = reg.best()
    best_spec = ExperimentSpec(**best.spec).summary() if best else {}
    cols = ", ".join(f"{c}:{t}" for c, t in list(ctx.columns.items())[:120])
    return (
        f"## Описание задачи\n{ctx.task_md}\n\n"
        f"## Профиль данных\n{json.dumps(ctx.profile, ensure_ascii=False, default=str)}\n\n"
        f"## Колонки (признаки)\n{cols}\n\n"
        f"## Словарь данных\n{ctx.data_dictionary}\n\n"
        f"## Метрика\n{ctx.metric} ({'меньше' if ctx.direction == 'min' else 'больше'} — лучше)\n\n"
        f"## Текущая лучшая спека\n{json.dumps(best_spec, ensure_ascii=False)}\n"
        f"CV = {best.cv_score if best else '—'}\n\n"
        f"## История экспериментов\n{reg.history_text()}\n"
    )


class LLMGate:
    """
    Доступность LLM: ключ, бюджет токенов, сбои подряд.
    """

    def __init__(self, cfg: LoopConfig, log):
        """
        Проверяет, можно ли вообще обращаться к LLM.

        Аргументы:
        cfg (LoopConfig): настройки цикла
        log (JsonlLogger): логгер loop.jsonl
        """
        self.cfg, self.log = cfg, log
        self.failures = 0
        if cfg.dry_run:
            self.enabled = True
            return
        try:
            from . import llm  # noqa: F401 — импорт подгружает .env (локальная разработка)
        except Exception as exc:  # noqa: BLE001
            self.enabled = False
            log.log(kind="llm_disabled", reason=f"llm.py не импортируется: {exc}"[:300])
            return
        self.enabled = bool(os.getenv("OPENAI_API_KEY", "").strip())
        if not self.enabled:
            log.log(kind="llm_disabled", reason="нет OPENAI_API_KEY")

    def tokens(self) -> int:
        """
        Возвращает, сколько токенов уже потрачено.

        Возвращает:
        int
        """
        if self.cfg.dry_run:
            return 0
        try:
            from . import llm
            return llm.get_usage().total_tokens
        except Exception:  # noqa: BLE001
            return 0

    def available(self) -> bool:
        """
        Проверяет, можно ли сейчас звать LLM (с учётом бюджета токенов).

        Возвращает:
        bool
        """
        if not self.enabled:
            return False
        if self.cfg.max_tokens and self.tokens() >= self.cfg.max_tokens:
            self.enabled = False
            self.log.log(kind="llm_disabled", reason="исчерпан бюджет токенов")
        return self.enabled

    def ok(self) -> None:
        """
        Сбрасывает счётчик сбоев после успешного вызова.
        """
        self.failures = 0

    def fail(self, where: str, exc: Exception) -> None:
        """
        Учитывает сбой; после двух подряд отключает LLM.

        Аргументы:
        where (str): где упало (propose / feature_patch)
        exc (Exception): исключение
        """
        self.failures += 1
        self.log.log(kind="llm_failure", where=where, error=f"{type(exc).__name__}: {exc}"[:500],
                     failures=self.failures)
        if self.failures >= 2:
            self.enabled = False
            self.log.log(kind="llm_disabled", reason="2 сбоя подряд")


def _llm_query(system_extra: str, user: str, *, schema: dict | None, role: str,
               timeout: float):
    """
    Вызывает LLM: системный промпт agent_instruction + инструкция шага.

    Аргументы:
    system_extra (str): инструкция шага (батч гипотез / патч признаков)
    user (str): пользовательское сообщение
    schema (dict | None): JSON-схема ответа; None — текстовый ответ
    role (str): роль для логов
    timeout (float): лимит ожидания ответа, секунды

    Возвращает:
    dict | str
    """
    from . import llm
    from .prompts import build_system

    messages = [
        {"role": "system", "content": build_system("agent_instruction") + "\n\n" + system_extra},
        {"role": "user", "content": user},
    ]
    return llm.query(messages, schema=schema, role=role, phase="RESEARCH", timeout=timeout)


DRY_CANDIDATES = {"candidates": [
    {"title": "Счётчик пропусков и статистики по строке",
     "rationale": "Пропуски часто информативны; агрегаты по строке дают дереву новые срезы.",
     "change_type": "feature", "delta": {},
     "feature_idea": "row_nan_count = число пропусков в строке; row_num_mean = среднее "
                     "числовых признаков по строке",
     "expected_gain": "med", "cost": "low"},
    {"title": "Меньше листьев и больше деревьев",
     "rationale": "Проверка регуляризации через структуру деревьев.",
     "change_type": "model",
     "delta": {"lgbm_params": {"num_leaves": 31, "n_estimators": 800}},
     "expected_gain": "med", "cost": "med"},
]}

DRY_PATCH = '''import numpy as np
import pandas as pd


def add_features(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    for df in (train, test):
        num = df.select_dtypes("number")
        df["row_nan_count"] = df.isna().sum(axis=1)
        df["row_num_mean"] = num.mean(axis=1)
    return train, test
'''


def propose_llm(ctx: TaskContext, reg: Registry, cfg: LoopConfig, gate: LLMGate) -> list[Candidate]:
    """
    Запрашивает у LLM батч кандидатов (пусто при сбое — цикл уйдёт на меню).

    Аргументы:
    ctx (TaskContext): контекст задачи
    reg (Registry): реестр (история для промпта)
    cfg (LoopConfig): настройки цикла
    gate (LLMGate): учёт сбоев LLM

    Возвращает:
    list[Candidate]
    """
    if cfg.dry_run:
        return _parse_candidates(ctx, DRY_CANDIDATES)
    instruction = BATCH_INSTRUCTION.format(k=cfg.batch_size, params=", ".join(PARAM_RANGES))
    try:
        data = _llm_query(instruction, _context_block(ctx, reg), schema=CANDIDATES_SCHEMA,
                          role="hypothesis_batch",
                          timeout=min(cfg.llm_timeout, max(30.0, ctx.remaining() / 4)))
        gate.ok()
        return _parse_candidates(ctx, data)
    except Exception as exc:  # noqa: BLE001 — любой сбой LLM не роняет цикл
        gate.fail("propose", exc)
        return []


# ----------------------------- патч признаков -----------------------------
PATCH_INSTRUCTION = """\
# Задача

Напиши функцию построения признаков по описанию идеи.

# Контракт

```python
def add_features(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
```
- На входе только признаки (без id и таргета), исходные dtypes. Вернуть (train, test)
  с одинаковым набором колонок и тем же числом строк; добавлять колонки, можно удалять.
- Статистики (частоты, агрегаты) считай по train+test вместе, таргет недоступен.
- Импорты: только pandas, numpy, math, re. Никаких файлов, сети, print.
- Деление — с защитой от нуля; категории можно оставить строками (каркас закодирует).
- Ответ — один markdown-блок с кодом.
"""

_ALLOWED_IMPORTS = {"pandas", "numpy", "math", "re"}
_FORBIDDEN_NAMES = {"open", "exec", "eval", "compile", "__import__", "input", "globals",
                    "locals", "vars", "getattr", "setattr", "delattr", "breakpoint"}
_FORBIDDEN_ATTRS = {"to_csv", "to_parquet", "to_pickle", "to_json", "read_csv",
                    "read_parquet", "system", "popen", "remove", "unlink"}


def check_patch(code: str) -> str | None:
    """
    Статически проверяет патч признаков: контракт, импорты, опасные вызовы.

    Аргументы:
    code (str): исходник патча

    Возвращает:
    str | None: описание проблемы или None, если патч корректен
    """
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return f"SyntaxError: {exc}"
    if not any(isinstance(n, ast.FunctionDef) and n.name == "add_features" for n in tree.body):
        return "нет функции add_features на верхнем уровне"
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            bad = [a.name for a in node.names if a.name.split(".")[0] not in _ALLOWED_IMPORTS]
        elif isinstance(node, ast.ImportFrom):
            bad = [node.module] if (node.module or "").split(".")[0] not in _ALLOWED_IMPORTS else []
        else:
            bad = []
        if bad:
            return f"запрещённый импорт: {bad}"
        if isinstance(node, ast.Name) and node.id in _FORBIDDEN_NAMES:
            return f"запрещённое имя: {node.id}"
        if isinstance(node, ast.Attribute) and (node.attr in _FORBIDDEN_ATTRS
                                                or node.attr.startswith("__")):
            return f"запрещённый атрибут: {node.attr}"
    return None


def generate_feature_patch(ctx: TaskContext, cand: Candidate, cfg: LoopConfig, gate: LLMGate,
                           prev_code: str | None = None, prev_error: str | None = None) -> str | None:
    """
    Генерирует код add_features по идее кандидата.

    Аргументы:
    ctx (TaskContext): контекст задачи
    cand (Candidate): кандидат с feature_idea
    cfg (LoopConfig): настройки цикла
    gate (LLMGate): учёт сбоев LLM
    prev_code (str | None): предыдущая версия патча (для починки)
    prev_error (str | None): ошибка предыдущей версии

    Возвращает:
    str | None: код патча или None, если не получилось
    """
    if cfg.dry_run:
        return DRY_PATCH
    if not gate.available():
        return None
    cols = "\n".join(f"- {c}: {t}" for c, t in ctx.columns.items())
    user = (f"## Идея\n{cand.title}\n{cand.feature_idea}\n\n"
            f"## Колонки на входе\n{cols}\n\n## Словарь данных\n{ctx.data_dictionary}\n")
    if prev_code and prev_error:
        user += (f"\n## Предыдущая версия упала — исправь\n```python\n{prev_code}\n```\n"
                 f"Ошибка:\n{prev_error[-1500:]}\n")
    try:
        from .llm import extract_code
        text = _llm_query(PATCH_INSTRUCTION, user, schema=None, role="feature_patch",
                          timeout=min(cfg.llm_timeout, max(30.0, ctx.remaining() / 4)))
        gate.ok()
    except Exception as exc:  # noqa: BLE001
        gate.fail("feature_patch", exc)
        return None
    code = extract_code(text)
    problem = check_patch(code)
    if problem:
        get_logger("loop").log(kind="patch_rejected", title=cand.title, problem=problem)
        return None
    return code


# ----------------------------- выбор гипотезы -----------------------------
GAIN_W = {"low": 1.0, "med": 2.0, "high": 3.0}
COST_W = {"low": 1.0, "med": 1.5, "high": 2.5}


def estimate_time(cand: Candidate, best: ExperimentSpec, reg: Registry) -> float:
    """
    Оценивает длительность: медиана прошлых запусков × стоимость × рост числа деревьев.

    Аргументы:
    cand (Candidate): кандидат
    best (ExperimentSpec): спека текущего лучшего
    reg (Registry): реестр (медианная длительность)

    Возвращает:
    float: секунды
    """
    factor = COST_W.get(cand.cost, 1.5)
    new_n = (cand.delta.get("lgbm_params") or {}).get("n_estimators")
    if new_n:
        factor = max(factor, new_n / best.lgbm_params.get("n_estimators", 500))
    extra = 60.0 if cand.feature_idea else 0.0  # генерация кода
    return reg.median_duration() * factor + extra


def select(cands: list[Candidate], reg: Registry, best: ExperimentSpec,
           remaining: float) -> Candidate | None:
    """
    Выбирает лучшего кандидата детерминированным ранжиром.

    Отсекает проверенные и не укладывающиеся по времени; скор —
    ожидаемый выигрыш / стоимость с поправками на историю.

    Аргументы:
    cands (list[Candidate]): пул кандидатов (меню + LLM)
    reg (Registry): реестр
    best (ExperimentSpec): спека текущего лучшего
    remaining (float): время до дедлайна, секунды

    Возвращает:
    Candidate | None: None, если выбрать нечего
    """
    tried_sigs, tried_keys = reg.tried_signatures(), reg.tried_keys()
    tried_types = reg.tried_change_types()
    scored = []
    seen = set()
    for cand in cands:
        key = cand.key()
        if key in tried_keys or key in seen:
            continue
        seen.add(key)
        if not cand.feature_idea and best.merge(cand.delta).signature() in tried_sigs:
            continue  # такая итоговая спека уже считалась (или дельта пустая)
        if estimate_time(cand, best, reg) > remaining:
            continue
        score = GAIN_W.get(cand.expected_gain, 1.0) / COST_W.get(cand.cost, 1.5)
        if reg.streak_fail(cand.change_type) >= 2:
            score *= 0.3
        if cand.change_type not in tried_types:
            score *= 1.3
        scored.append((score, cand.source == "menu", cand))
    if not scored:
        return None
    scored.sort(key=lambda t: (t[0], t[1]), reverse=True)
    return scored[0][2]


# ----------------------------- фазы ---------------------------------------
def exp_timeout(ctx: TaskContext, cfg: LoopConfig, reg: Registry) -> float:
    """
    Вычисляет лимит песочницы для очередного эксперимента.

    Аргументы:
    ctx (TaskContext): контекст задачи (остаток времени)
    cfg (LoopConfig): настройки цикла (потолок)
    reg (Registry): реестр (медианная длительность)

    Возвращает:
    float: секунды
    """
    budget =max(300.0, 4 * reg.median_duration())
    return max(30.0, min(cfg.exp_timeout_cap, budget, ctx.remaining() - 30))


def run_baseline_phase(ctx: TaskContext, cfg: LoopConfig, reg: Registry, log) -> ExperimentRecord | None:
    """
    Делает гарантированный сабмит in-process и эталонный эксперимент #0 в песочнице.

    Аргументы:
    ctx (TaskContext): контекст задачи
    cfg (LoopConfig): настройки цикла
    reg (Registry): реестр
    log (JsonlLogger): логгер loop.jsonl

    Возвращает:
    ExperimentRecord | None: эталон или None, если он не посчитался
    """
    t = time.time()
    try:
        res = tools.run_baseline(str(ctx.data_dir), str(ctx.submission_path),
                                 time_limit=max(60.0, 0.25 * ctx.remaining()))
        log.log(kind="baseline_inprocess", cv_score=res["cv_score"], metric=res["metric"],
                folds_done=res["folds_done"], duration=round(time.time() - t, 2))
    except Exception as exc:  # noqa: BLE001 — эталон в песочнице ещё может сработать
        log.log(kind="baseline_inprocess_error", error=f"{type(exc).__name__}: {exc}")

    spec = ExperimentSpec(class_weight="balanced" if ctx.task_type == "classification" else None)
    timeout = max(60.0, min(cfg.exp_timeout_cap, 0.4 * ctx.remaining()))
    rec = run_spec(ctx, reg, spec, candidate=None, parent_id=None, phase="BASELINE",
                   timeout=timeout)
    rec.accepted = rec.status == "ok" and len(rec.fold_scores) == tools.N_FOLDS
    rec.reason = rec.reason or "эталон"
    reg.add(rec)
    if rec.accepted:
        if not ctx.submission_path.exists():
            write_submission(ctx, rec)
        return rec
    log.log(kind="baseline_failed", reason=rec.reason[:500])
    return None


def execute_candidate(ctx: TaskContext, cfg: LoopConfig, reg: Registry, gate: LLMGate,
                      cand: Candidate, best: ExperimentRecord, log) -> ExperimentRecord:
    """
    Проводит кандидата по цепочке: (патч) -> спека -> запуск -> вердикт -> реестр.

    Аргументы:
    ctx (TaskContext): контекст задачи
    cfg (LoopConfig): настройки цикла
    reg (Registry): реестр
    gate (LLMGate): учёт сбоев LLM
    cand (Candidate): выбранный кандидат
    best (ExperimentRecord): текущий лучший
    log (JsonlLogger): логгер loop.jsonl

    Возвращает:
    ExperimentRecord: последняя запись (после возможной починки патча)
    """
    best_spec = ExperimentSpec(**best.spec)
    tokens0 = gate.tokens()
    patch = None
    if cand.feature_idea:
        patch = generate_feature_patch(ctx, cand, cfg, gate)
        if patch is None:
            rec = ExperimentRecord(
                exp_id=reg.next_id(), parent_id=best.exp_id, phase="RESEARCH",
                candidate=cand.to_dict(), spec=best.spec, signature="-",
                status="invalid", reason="не удалось получить корректный патч признаков",
                tokens=gate.tokens() - tokens0)
            reg.add(rec)
            return rec

    for attempt in range(2):  # вторая попытка — только починка упавшего патча
        delta = dict(cand.delta)
        if patch:
            delta["feature_patch"] = patch
        spec = best_spec.merge(delta)
        rec = run_spec(ctx, reg, spec, candidate=cand, parent_id=best.exp_id,
                       phase="RESEARCH", timeout=exp_timeout(ctx, cfg, reg))
        rec.accepted, verdict = judge(ctx, rec, best)
        rec.reason = verdict if rec.status == "ok" else rec.reason
        rec.tokens = gate.tokens() - tokens0
        reg.add(rec)
        log.log(kind="experiment", exp_id=rec.exp_id, title=cand.title, source=cand.source,
                change_type=cand.change_type, status=rec.status, cv_score=rec.cv_score,
                best_score=best.cv_score, accepted=rec.accepted, reason=rec.reason[:300],
                duration=rec.duration, tokens=rec.tokens)
        if not (patch and rec.status == "error" and attempt == 0
                and ctx.remaining() > 2 * reg.median_duration()):
            return rec
        error = rec.artifacts.get("traceback") or rec.reason
        patch = generate_feature_patch(ctx, cand, cfg, gate, prev_code=patch, prev_error=error)
        if patch is None:
            return rec
    return rec


def _fresh_batch(ctx: TaskContext, reg: Registry, cfg: LoopConfig, gate: LLMGate,
                 log) -> list[Candidate]:
    """
    Запрашивает батч у LLM и убирает уже проверенные гипотезы.

    Аргументы:
    ctx (TaskContext): контекст задачи
    reg (Registry): реестр
    cfg (LoopConfig): настройки цикла
    gate (LLMGate): учёт сбоев LLM
    log (JsonlLogger): логгер loop.jsonl

    Возвращает:
    list[Candidate]
    """
    batch = propose_llm(ctx, reg, cfg, gate)
    tried = reg.tried_keys()
    fresh = [c for c in batch if c.key() not in tried]
    log.log(kind="llm_batch", n=len(batch), n_fresh=len(fresh),
            titles=[c.title for c in fresh])
    return fresh


def research(ctx: TaskContext, cfg: LoopConfig, reg: Registry, log) -> None:
    """
    Крутит цикл AutoResearch, пока позволяют время, итерации, patience и токены.

    Аргументы:
    ctx (TaskContext): контекст задачи
    cfg (LoopConfig): настройки цикла
    reg (Registry): реестр
    log (JsonlLogger): логгер loop.jsonl
    """
    gate = LLMGate(cfg, log)
    pool: list[Candidate] = []
    iters = no_improve = 0
    while True:
        best = reg.best()
        best_spec = ExperimentSpec(**best.spec)
        # время после резерва под ансамбль
        remaining = ctx.remaining() - ensemble_reserve(cfg, reg)
        if iters >= cfg.max_iters or no_improve >= cfg.patience:
            log.log(kind="research_stop", reason="iters/patience", iters=iters,
                    no_improve=no_improve)
            return
        if remaining < max(60.0, 1.2 * reg.median_duration()):
            log.log(kind="research_stop", reason="время", remaining=round(remaining),
                    ensemble_reserve=round(ensemble_reserve(cfg, reg)))
            return

        tried = reg.tried_keys()
        pool = [c for c in pool if c.key() not in tried]
        asked = False
        if not pool and gate.available():
            pool = _fresh_batch(ctx, reg, cfg, gate, log)
            asked = True

        cand = select(menu(ctx, best_spec) + pool, reg, best_spec, remaining)
        if cand is None and not asked and gate.available():
            pool = _fresh_batch(ctx, reg, cfg, gate, log)
            cand = select(menu(ctx, best_spec) + pool, reg, best_spec, remaining)
        if cand is None:
            log.log(kind="research_stop", reason="нет кандидатов")
            return
        if cand in pool:
            pool.remove(cand)

        log.log(kind="selected", title=cand.title, source=cand.source,
                change_type=cand.change_type, delta=cand.delta, feature_idea=cand.feature_idea,
                expected_gain=cand.expected_gain, cost=cand.cost)
        rec = execute_candidate(ctx, cfg, reg, gate, cand, best, log)
        iters += 1
        if rec.accepted:
            no_improve = 0
            write_submission(ctx, rec)
            log.log(kind="new_best", exp_id=rec.exp_id, cv_score=rec.cv_score)
        else:
            no_improve += 1


# ----------------------------- ансамбль -----------------------------------
# Блендинг идёт по OOF на тех же фолдах, что у всех экспериментов, поэтому
# скор ансамбля сравним со скором одиночной модели и проходит тот же judge.
# greedy_blend — временная реализация зоны A; когда в tools появится
# блендинг от B, достаточно заменить вызов в ensemble_phase.

def ensemble_reserve(cfg: LoopConfig, reg: Registry) -> float:
    """
    Оценивает время, которое RESEARCH оставляет под фазу ENSEMBLE.

    Аргументы:
    cfg (LoopConfig): настройки цикла
    reg (Registry): реестр (медианная длительность)

    Возвращает:
    float: секунды (0, если ансамбль выключен)
    """
    if not cfg.ensemble:
        return 0.0
    return cfg.ensemble_seeds * 1.2 * reg.median_duration() + 30.0


def score_pred(ctx: TaskContext, pred: np.ndarray, idx: np.ndarray | None = None) -> float:
    """
    Считает метрику задачи по OOF-предсказаниям.

    Аргументы:
    ctx (TaskContext): контекст задачи (y, тип задачи)
    pred (np.ndarray): OOF — вероятности (n, K) или значения (n,)
    idx (np.ndarray | None): подмножество строк; None — все

    Возвращает:
    float
    """
    y, p = (ctx.y, pred) if idx is None else (ctx.y[idx], pred[idx])
    if ctx.task_type == "classification":
        return tools.macro_f1(y, p.argmax(axis=1))
    return tools.rmse(y, p)


def fold_scores_of(ctx: TaskContext, pred: np.ndarray) -> list[float]:
    """
    Считает метрику по каждому фолду CV.

    Аргументы:
    ctx (TaskContext): контекст задачи (folds)
    pred (np.ndarray): OOF-предсказания

    Возвращает:
    list[float]
    """
    return [score_pred(ctx, pred, va) for _, va in ctx.folds]


def greedy_blend(ctx: TaskContext, oofs: list[np.ndarray], max_steps: int) -> tuple[np.ndarray, float]:
    """
    Жадно отбирает членов ансамбля с повторами (Caruana) по OOF-скору.

    На каждом шаге добавляет модель, которая сильнее всего улучшает
    среднее; останавливается, когда улучшений нет.

    Аргументы:
    ctx (TaskContext): контекст задачи
    oofs (list[np.ndarray]): OOF-предсказания кандидатов
    max_steps (int): максимум шагов отбора

    Возвращает:
    tuple[np.ndarray, float]: (веса членов, сумма = 1; скор ансамбля)
    """
    counts = np.zeros(len(oofs), dtype=int)
    current, best_score = None, None
    for _ in range(max_steps):
        k = counts.sum()
        step = None
        for i, oof in enumerate(oofs):
            cand = oof if current is None else (current * k + oof) / (k + 1)
            s = score_pred(ctx, cand)
            if step is None or improvement(ctx, step[0], s) > 0:
                step = (s, i, cand)
        s, i, cand = step
        if best_score is not None and improvement(ctx, best_score, s) <= 1e-12:
            break
        counts[i] += 1
        current, best_score = cand, s
    return counts / counts.sum(), best_score


def seed_bagging(ctx: TaskContext, cfg: LoopConfig, reg: Registry, log) -> None:
    """
    Перезапускает лучшую спеку с другими random_state — разнообразие для ансамбля.

    Фолды CV не меняются (их seed фиксирован в tools), меняется только
    случайность бустинга (subsample / colsample).

    Аргументы:
    ctx (TaskContext): контекст задачи
    cfg (LoopConfig): настройки цикла
    reg (Registry): реестр
    log (JsonlLogger): логгер loop.jsonl
    """
    best = reg.best()
    best_spec = ExperimentSpec(**best.spec)
    for i in range(cfg.ensemble_seeds):
        if ctx.remaining() < 1.2 * reg.median_duration() + 30:
            log.log(kind="seed_bagging_stop", reason="время", done=i)
            return
        seed = tools.SEED + 1 + i
        cand = Candidate(f"seed-бэггинг: random_state={seed}",
                         "Та же спека с другим seed бустинга — член ансамбля.",
                         "ensemble", {"lgbm_params": {"random_state": seed}})
        spec = best_spec.merge(cand.delta)
        if spec.signature() in reg.tried_signatures():
            continue
        rec = run_spec(ctx, reg, spec, candidate=cand, parent_id=best.exp_id,
                       phase="ENSEMBLE", timeout=exp_timeout(ctx, cfg, reg))
        rec.reason = rec.reason or "член ансамбля (seed)"
        reg.add(rec)
        log.log(kind="seed_run", exp_id=rec.exp_id, seed=seed, status=rec.status,
                cv_score=rec.cv_score, best_score=best.cv_score, duration=rec.duration)


def ensemble_phase(ctx: TaskContext, cfg: LoopConfig, reg: Registry, log) -> ExperimentRecord | None:
    """
    Собирает ансамбль из лучших экспериментов и принимает его через judge.

    Аргументы:
    ctx (TaskContext): контекст задачи
    cfg (LoopConfig): настройки цикла
    reg (Registry): реестр
    log (JsonlLogger): логгер loop.jsonl

    Возвращает:
    ExperimentRecord | None: запись ансамбля (принятого или нет) или None, если не собирали
    """
    t0 = time.time()
    best = reg.best()
    # сверка фолдов: OOF из песочницы должен давать те же фолд-скоры
    best_oof = np.load(best.artifacts["oof"])
    if not np.allclose(fold_scores_of(ctx, best_oof), best.fold_scores, rtol=1e-6, atol=1e-9):
        log.log(kind="ensemble_skip", reason="фолды цикла не совпали с фолдами песочницы")
        return None

    seed_bagging(ctx, cfg, reg, log)

    pool = [r for r in reg.records
            if r.status == "ok" and len(r.fold_scores) == tools.N_FOLDS
            and r.artifacts.get("oof")]
    reverse = ctx.direction == "max"
    pool = sorted(pool, key=lambda r: r.cv_score, reverse=reverse)[:cfg.ensemble_top]
    if len(pool) < 2:
        log.log(kind="ensemble_skip", reason=f"мало экспериментов: {len(pool)}")
        return None

    oofs = [np.load(r.artifacts["oof"]) for r in pool]
    weights, score = greedy_blend(ctx, oofs, cfg.ensemble_steps)
    members = {pool[i].exp_id: round(float(w), 4) for i, w in enumerate(weights) if w > 0}
    if len(members) < 2:
        log.log(kind="ensemble_skip", reason="блендинг выбрал одну модель",
                members=members, score=score)
        return None

    oof = sum(w * o for w, o in zip(weights, oofs) if w > 0)
    test_pred = sum(w * np.load(r.artifacts["test_pred"])
                    for w, r in zip(weights, pool) if w > 0)
    exp_id = reg.next_id()
    exp_dir = ctx.work_dir / f"exp_{exp_id:03d}"
    exp_dir.mkdir(parents=True, exist_ok=True)
    np.save(exp_dir / "oof.npy", oof)
    np.save(exp_dir / "test_pred.npy", test_pred)

    cand = Candidate("Ансамбль: " + ", ".join(f"#{k}×{v}" for k, v in members.items()),
                     "Жадный блендинг OOF лучших экспериментов и seed-перезапусков.",
                     "ensemble", {})
    rec = ExperimentRecord(
        exp_id=exp_id, parent_id=best.exp_id, phase="ENSEMBLE",
        candidate=cand.to_dict(), spec=best.spec,
        signature="blend:" + ",".join(f"{k}={v}" for k, v in members.items()),
        status="ok", cv_score=float(score), fold_scores=fold_scores_of(ctx, oof),
        duration=round(time.time() - t0, 3),
        artifacts={"dir": str(exp_dir), "oof": str(exp_dir / "oof.npy"),
                   "test_pred": str(exp_dir / "test_pred.npy"), "members": members},
    )
    rec.accepted, rec.reason = judge(ctx, rec, best)
    reg.add(rec)
    log.log(kind="ensemble", exp_id=rec.exp_id, members=members, cv_score=rec.cv_score,
            best_score=best.cv_score, accepted=rec.accepted, reason=rec.reason)
    if rec.accepted:
        write_submission(ctx, rec)
        log.log(kind="new_best", exp_id=rec.exp_id, cv_score=rec.cv_score)
    return rec


def finalize(ctx: TaskContext, reg: Registry, log) -> None:
    """
    Пишет сабмит лучшего эксперимента и проверяет его формат.

    Аргументы:
    ctx (TaskContext): контекст задачи
    reg (Registry): реестр
    log (JsonlLogger): логгер loop.jsonl
    """
    best = reg.best()
    if best is not None and best.artifacts.get("test_pred"):
        write_submission(ctx, best)
    if ctx.submission_path.exists():
        sub = pd.read_csv(ctx.submission_path)
        errors = tools.validate_submission(sub, ctx.sample)
        log.log(kind="finalize", best_exp=best.exp_id if best else None,
                cv_score=best.cv_score if best else None, submission_errors=errors)
        if errors:
            raise ValueError("Невалидный сабмит: " + "; ".join(errors))
    else:
        raise FileNotFoundError("submission.csv не сформирован")


def run(task_md: str | Path, t0: float | None = None, cfg: LoopConfig | None = None) -> dict:
    """
    Выполняет полный прогон одной задачи: SCOUT -> BASELINE -> RESEARCH -> ENSEMBLE -> FINALIZE.

    Аргументы:
    task_md (str | Path): путь к task.md
    t0 (float | None): момент старта (для отсчёта 3 часов); None — сейчас
    cfg (LoopConfig | None): настройки; None — из окружения

    Возвращает:
    dict: краткая сводка прогона
    """
    t0 = t0 or time.time()
    cfg = cfg or LoopConfig.from_env()
    log = get_logger("loop")
    task_md = Path(task_md).resolve()
    margin = min(cfg.safety_margin, 0.25 * cfg.time_budget)
    deadline = t0 + cfg.time_budget - margin

    log.log(kind="phase", phase="SCOUT", config=cfg.__dict__)
    ctx = scout(task_md, deadline)
    log.log(kind="scout", task_type=ctx.task_type, metric=ctx.metric, target=ctx.target,
            id_col=ctx.id_col, n_rows=ctx.profile["n_rows"], n_features=ctx.profile["n_features"],
            data_dir=str(ctx.data_dir))
    reg = Registry(ctx.work_dir / "experiments.json")

    log.log(kind="phase", phase="BASELINE", remaining=round(ctx.remaining()))
    base = run_baseline_phase(ctx, cfg, reg, log)

    if base is not None:
        log.log(kind="phase", phase="RESEARCH", remaining=round(ctx.remaining()),
                baseline_cv=base.cv_score)
        try:
            research(ctx, cfg, reg, log)
        except Exception as exc:  # noqa: BLE001 — лучший результат уже в сабмите
            log.log(kind="research_error", error=f"{type(exc).__name__}: {exc}")

        if cfg.ensemble:
            log.log(kind="phase", phase="ENSEMBLE", remaining=round(ctx.remaining()))
            try:
                ensemble_phase(ctx, cfg, reg, log)
            except Exception as exc:  # noqa: BLE001 — без ансамбля остаётся лучшая модель
                log.log(kind="ensemble_error", error=f"{type(exc).__name__}: {exc}")

    log.log(kind="phase", phase="FINALIZE", remaining=round(ctx.remaining()))
    finalize(ctx, reg, log)
    best = reg.best()
    summary = {
        "task_type": ctx.task_type, "metric": ctx.metric,
        "baseline_cv": base.cv_score if base else None,
        "best_cv": best.cv_score if best else None,
        "best_exp": best.exp_id if best else None,
        "n_experiments": len(reg.records),
        "elapsed": round(time.time() - t0, 1),
    }
    log.log(kind="done", **summary)
    return summary
