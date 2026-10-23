"""Дерево решений агента: портфель -> ветки-код -> правки -> ансамбль листьев.

Схема (соответствует постановке задачи):
1. SCOUT даёт контекст (тип задачи, метрика, профиль).
2. LLM выбирает из портфеля (agent/portfolio.py) семейства, которые стоит
   пробовать; без LLM берутся все, что проходят по размеру выборки.
3. Для каждого семейства создаётся корневой узел с готовым кодом-рецептом и
   оценивается на фиксированной 5-фолдовой CV (agent/tree_runner.py).
4. Лучшие ветки (по CV) продолжают жить: LLM предлагает МИНИМАЛЬНУЮ правку
   кода ветки (SEARCH/REPLACE, с откатом к полному файлу), правка проверяется
   статически и оценивается. Улучшилась — становится новой вершиной ветки,
   нет — ветка остаётся на прежней вершине.
5. Ансамбль: hill climbing по OOF всех успешных узлов; финальный сабмит
   считается взвешенным средним предсказаний листьев.

Все узлы хранятся в nodes.json (рядом с experiments.json) вместе с диффом,
скором и путями к OOF/test_pred — это и есть память агента.
"""
from __future__ import annotations

import ast
import difflib
import json
import os
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import tools
from .event_log import get_logger
from .portfolio import FAMILIES, available_families, family_menu
from .sandbox import run_python
from .tree_runner import RUNNER_CODE

N_FOLDS = tools.N_FOLDS

# Импорты, разрешённые в коде узла: только классический ML и утилиты.
ALLOWED_IMPORTS = {
    "pandas", "numpy", "math", "re", "statistics", "itertools", "collections",
    "functools", "warnings", "sklearn", "lightgbm", "catboost", "xgboost",
}
FORBIDDEN_NAMES = {
    "open", "exec", "eval", "compile", "__import__", "input", "globals",
    "locals", "vars", "getattr", "setattr", "delattr", "breakpoint", "help",
}
FORBIDDEN_ATTRS = {
    "to_csv", "to_parquet", "to_pickle", "to_json", "read_csv", "read_parquet",
    "system", "popen", "remove", "unlink", "rmdir", "makedirs",
}


# ----------------------------- узлы ---------------------------------------
@dataclass
class Node:
    """Один вариант решения: код + результат его оценки."""

    node_id: int
    parent_id: int | None
    family: str
    depth: int
    code: str
    status: str = "pending"          # pending | ok | error | timeout | invalid
    cv_score: float | None = None
    fold_scores: list[float] = field(default_factory=list)
    duration: float = 0.0
    tokens: int = 0
    hypothesis: str = ""             # словами: что и зачем поменяли
    note: str = ""                   # механика правки (блоки / полная замена)
    reason: str = ""                 # вердикт: принята / хуже родителя / ошибка
    diff: str = ""                   # что изменили относительно родителя
    artifacts: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Сериализация для nodes.json."""
        return asdict(self)

    @property
    def ok(self) -> bool:
        """Узел успешно посчитан и годен для ансамбля."""
        return self.status == "ok" and self.cv_score is not None


class Tree:
    """Коллекция узлов одной задачи с атомарной записью на диск."""

    def __init__(self, path: str | Path, direction: str = "min"):
        """
        Аргументы:
        path (str | Path): путь к nodes.json
        direction (str): "min" или "max" — направление метрики задачи
        """
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.direction = direction
        self.nodes: list[Node] = []

    def next_id(self) -> int:
        """Номер следующего узла."""
        return len(self.nodes)

    def add(self, node: Node) -> Node:
        """Добавляет узел и сохраняет дерево."""
        self.nodes.append(node)
        self._dump()
        return node

    def save(self) -> None:
        """Сохранить дерево (публичная обёртка над _dump)."""
        self._dump()

    def by_id(self, node_id: int) -> "Node | None":
        """Найти узел по номеру."""
        for node in self.nodes:
            if node.node_id == node_id:
                return node
        return None

    def _dump(self) -> None:
        """Атомарно перезаписывает nodes.json."""
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps([n.to_dict() for n in self.nodes], ensure_ascii=False,
                                  indent=1, default=str), encoding="utf-8")
        os.replace(tmp, self.path)

    def ok_nodes(self) -> list[Node]:
        """Узлы с посчитанной CV и полными фолдами."""
        return [n for n in self.nodes if n.ok and len(n.fold_scores) == N_FOLDS]

    def best(self) -> Node | None:
        """Лучший узел по направлению метрики (None, если успешных нет)."""
        nodes = self.ok_nodes()
        if not nodes:
            return None
        return max(nodes, key=lambda n: n.cv_score if self.direction == "max" else -n.cv_score)

    def median_duration(self, default: float = 60.0) -> float:
        """Медианная длительность успешных узлов."""
        durations = [n.duration for n in self.nodes if n.status == "ok" and n.duration > 0]
        if not durations:
            return default
        durations.sort()
        return durations[len(durations) // 2]


# ----------------------------- статические проверки ------------------------
def check_code(code: str) -> str | None:
    """
    Проверяет исходник узла до запуска в песочнице.

    Аргументы:
    code (str): исходник узла

    Возвращает:
    str | None: проблема или None, если код корректен
    """
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return f"SyntaxError: {exc}"
    if not any(isinstance(n, ast.FunctionDef) and n.name == "build_model" for n in tree.body):
        return "нет функции build_model на верхнем уровне"
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            bad = [a.name for a in node.names if a.name.split(".")[0] not in ALLOWED_IMPORTS]
        elif isinstance(node, ast.ImportFrom):
            bad = [node.module] if (node.module or "").split(".")[0] not in ALLOWED_IMPORTS else []
        else:
            bad = []
        if bad:
            return f"запрещённый импорт: {bad}"
        if isinstance(node, ast.Name) and node.id in FORBIDDEN_NAMES:
            return f"запрещённое имя: {node.id}"
        if isinstance(node, ast.Attribute) and (node.attr in FORBIDDEN_ATTRS
                                                or node.attr.startswith("__")):
            return f"запрещённый атрибут: {node.attr}"
    return None


# ----------------------------- правки кода --------------------------------
SEARCH_MARK = "<<<<<<< SEARCH"
MIDDLE_MARK = "======="
REPLACE_MARK = ">>>>>>> REPLACE"


HYPOTHESIS_RE = re.compile(
    r"^[ \t]*(?:ГИПОТЕЗА|HYPOTHESIS)[ \t]*[:\-][ \t]*(.*?)(?=\n[ \t]*\n|\n<<<<<<<|\Z)",
    re.IGNORECASE | re.DOTALL | re.MULTILINE,
)


def _split_hypothesis(text: str) -> tuple[str, str]:
    """
    Вынимает из ответа строку HYPOTHESIS: ... (если она есть).

    Аргументы:
    text (str): ответ модели

    Возвращает:
    tuple[str, str]: текст без гипотезы и сама гипотеза
    """
    match = HYPOTHESIS_RE.search(text)
    if not match:
        return text, ""
    hypothesis = " ".join(match.group(1).split())[:800]
    return text[:match.start()] + text[match.end():], hypothesis


def apply_edits(code: str, answer: str) -> tuple[str | None, str, str]:
    """
    Применяет SEARCH/REPLACE блоки из ответа LLM к коду ветки.

    Если блоков нет, но пришёл цельный python-блок — считаем это полной
    заменой файла (страховка для слабых моделей).

    Аргументы:
    code (str): текущий код ветки
    answer (str): ответ модели

    Возвращает:
    tuple[str | None, str, str]: новый код (или None), гипотеза словами,
    механика правки
    """
    text, hypothesis = _split_hypothesis(answer or "")
    if SEARCH_MARK not in text:
        block = _extract_python_block(text)
        if block and "def build_model" in block:
            return block, hypothesis, "полная замена файла"
        return None, hypothesis, "в ответе нет ни SEARCH/REPLACE блоков, ни кода"

    updated = code
    blocks = 0
    for chunk in text.split(SEARCH_MARK)[1:]:
        body = chunk.split(REPLACE_MARK)[0]
        if MIDDLE_MARK not in body:
            return None, hypothesis, "блок без разделителя ====="
        old, new = body.split(MIDDLE_MARK, 1)
        old = old.rstrip("\n")
        new = new.rstrip("\n")
        if not old.strip():
            return None, hypothesis, "пустой SEARCH"
        if updated.count(old) != 1:
            return None, hypothesis, f"SEARCH встречается {updated.count(old)} раз (нужен ровно один)"
        updated = updated.replace(old, new, 1)
        blocks += 1
    return updated, hypothesis, f"применено блоков: {blocks}"


def _extract_python_block(text: str) -> str | None:
    """Вырезает первый ```python-блок из ответа."""
    start = text.find("```")
    if start < 0:
        return None
    end = text.find("```", start + 3)
    if end < 0:
        return None
    body = text[start + 3:end]
    return body.split("\n", 1)[1] if "\n" in body else body


def _diff(before: str, after: str) -> str:
    """Компактный unified diff для журнала и nodes.json."""
    lines = list(difflib.unified_diff(before.splitlines(), after.splitlines(),
                                      fromfile="parent", tofile="child", lineterm="", n=2))
    return "\n".join(lines)[:4000]


# ----------------------------- оценка узла --------------------------------
def evaluate(ctx, node: Node, timeout: float, log) -> Node:
    """
    Запускает код узла в песочнице и заполняет результат.

    Аргументы:
    ctx (TaskContext): контекст задачи
    node (Node): узел с кодом
    timeout (float): лимит песочницы, секунды
    log (JsonlLogger): логгер tree.jsonl (на деле loop.jsonl)

    Возвращает:
    Node: тот же узел с заполненными полями
    """
    node_dir = ctx.work_dir / f"node_{node.node_id:03d}"
    node_dir.mkdir(parents=True, exist_ok=True)
    for name in ("result.json", "oof.npy", "test_pred.npy"):
        (node_dir / name).unlink(missing_ok=True)
    config = {
        "project_root": str(Path(__file__).resolve().parents[1]),
        "data_dir": str(ctx.data_dir),
        "id_col": ctx.id_col,
        "target": ctx.target,
        "task_type": ctx.task_type,
        "code": node.code,
        "time_limit": max(30.0, timeout * 0.8),
    }
    (node_dir / "config.json").write_text(json.dumps(config, ensure_ascii=False, default=str),
                                          encoding="utf-8")

    started = time.time()
    result = run_python(RUNNER_CODE, timeout=timeout, cwd=node_dir, phase="TREE")
    node.duration = round(time.time() - started, 3)
    node.artifacts = {"dir": str(node_dir)}

    payload = {}
    try:
        payload = json.loads((node_dir / "result.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        pass

    if result.timed_out:
        node.status, node.reason = "timeout", f"лимит {timeout:.0f} с"
    elif payload.get("status") == "ok" and payload.get("score") is not None:
        node.status = "ok"
        node.cv_score = float(payload["score"])
        node.fold_scores = [float(s) for s in payload.get("fold_scores") or []]
        node.artifacts.update(oof=str(node_dir / "oof.npy"),
                              test_pred=str(node_dir / "test_pred.npy"))
        if int(payload.get("folds_done") or 0) < N_FOLDS:
            node.reason = f"неполная CV: {payload.get('folds_done')}/{N_FOLDS}"
    else:
        node.status = "error"
        node.reason = (payload.get("error") or (result.stderr or "нет result.json"))[-600:]

    log.log(kind="node", node_id=node.node_id, parent_id=node.parent_id,
            family=node.family, depth=node.depth, status=node.status,
            cv_score=node.cv_score, duration=node.duration,
            hypothesis=node.hypothesis[:300], note=node.note[:120],
            reason=node.reason[:300], folds=len(node.fold_scores))
    return node


# ----------------------------- LLM ----------------------------------------
class Gate:
    """Доступность LLM: ключ, бюджет токенов, сбои подряд."""

    def __init__(self, cfg, log):
        """Проверяет ключ и режим dry-run."""
        self.cfg, self.log = cfg, log
        self.failures = 0
        if getattr(cfg, "dry_run", False):
            self.enabled = False      # dry-run: без LLM, только портфель
            return
        try:
            from . import llm  # noqa: F401 — подгружает .env
        except Exception as exc:  # noqa: BLE001
            self.enabled = False
            log.log(kind="llm_disabled", reason=f"llm.py не импортируется: {exc}"[:300])
            return
        self.enabled = bool(os.getenv("OPENAI_API_KEY", "").strip())
        if not self.enabled:
            log.log(kind="llm_disabled", reason="нет OPENAI_API_KEY")

    def tokens(self) -> int:
        """Израсходовано токенов (0 в dry-run)."""
        if getattr(self.cfg, "dry_run", False):
            return 0
        try:
            from . import llm
            return llm.get_usage().total_tokens
        except Exception:  # noqa: BLE001
            return 0

    def available(self) -> bool:
        """Можно ли звать LLM с учётом бюджета."""
        if not self.enabled:
            return False
        limit = getattr(self.cfg, "max_tokens", None)
        if limit and self.tokens() >= limit:
            self.enabled = False
            self.log.log(kind="llm_disabled", reason="исчерпан бюджет токенов")
        return self.enabled

    def ok(self) -> None:
        """Сбросить счётчик сбоев."""
        self.failures = 0

    def fail(self, where: str, exc: Exception) -> None:
        """Учесть сбой; после двух подряд выключить LLM."""
        self.failures += 1
        self.log.log(kind="llm_failure", where=where, failures=self.failures,
                     error=f"{type(exc).__name__}: {exc}"[:400])
        if self.failures >= 2:
            self.enabled = False
            self.log.log(kind="llm_disabled", reason="2 сбоя подряд")


def choose_families(ctx, cfg, gate, log) -> list[str]:
    """
    Спрашивает LLM, какие семейства портфеля стоит пробовать.

    Аргументы:
    ctx (TaskContext): контекст задачи
    cfg (LoopConfig): настройки (dry_run, portfolio_size)
    gate (Gate): доступность LLM
    log (JsonlLogger): логгер

    Возвращает:
    list[str]: имена семейств; в dry-run или без LLM — все подходящие
    """
    allowed = available_families(int(ctx.profile.get("n_rows") or 0))
    limit = int(getattr(cfg, "portfolio_size", 6))
    fallback = allowed[:limit]
    if not gate.available():
        return fallback
    schema = {
        "type": "object",
        "properties": {
            "families": {"type": "array", "items": {"type": "string", "enum": allowed},
                         "maxItems": limit},
            "reasoning": {"type": "string"},
        },
        "required": ["families", "reasoning"],
    }
    user = (
        f"## Задача\n{ctx.task_md[:3000]}\n\n"
        f"## Данные\n{json.dumps(ctx.profile, ensure_ascii=False, default=str)[:1500]}\n\n"
        f"## Портфель\n{family_menu()}\n\n"
        f"## Ограничение\nВыбери не больше {limit} семейств из списка выше "
        f"(допустимые: {', '.join(allowed)})."
    )
    try:
        from . import llm
        data = llm.query([{"role": "user", "content": user}], role="portfolio",
                         schema=schema, phase="PORTFOLIO",
                         timeout=max(30.0, ctx.remaining() / 6))
    except Exception as exc:  # noqa: BLE001
        gate.fail("portfolio", exc)
        log.log(kind="portfolio_fallback", reason=str(exc)[:200])
        return fallback
    gate.ok()
    chosen = [f for f in data.get("families", []) if f in allowed][:limit]
    if not chosen:
        return fallback
    log.log(kind="portfolio", families=chosen, reasoning=str(data.get("reasoning"))[:400])
    return chosen


def _changed_lines(diff: str, limit: int = 160) -> str:
    """
    Вытаскивает из unified diff только изменённые строки (без заголовков).

    Аргументы:
    diff (str): diff родителя и ребёнка
    limit (int): максимум символов

    Возвращает:
    str: компактное «что поменяли»
    """
    lines = [line for line in diff.splitlines()
             if line[:1] in "+-" and not line.startswith(("+++", "---"))]
    return " ".join(line.strip() for line in lines)[:limit] or "(без изменений)"


def _history_block(ctx, tree: Tree, node: Node, limit: int = 8) -> str:
    """
    Собирает историю попыток для промпта правки.

    Модель должна видеть не только успехи, но и все неудачные правки: именно
    из них берутся следующие гипотезы. Поэтому показываем и другие семейства
    (кто сейчас лучший), и последние попытки своей ветки с реальным диффом.

    Аргументы:
    ctx (TaskContext): контекст задачи
    tree (Tree): дерево
    node (Node): текущая вершина ветки
    limit (int): сколько последних попыток своей ветки показать

    Возвращает:
    str: текст блока для промпта
    """
    best = tree.best()
    best_line = (f"глобальный лучший: #{best.node_id} {best.family} cv={best.cv_score:.5g}"
                 if best else "глобального лучшего пока нет")
    summary = []
    for family in sorted({n.family for n in tree.nodes}):
        scores = [n.cv_score for n in tree.nodes if n.family == family and n.ok]
        if scores:
            pick = max(scores) if ctx.direction == "max" else min(scores)
            summary.append(f"{family}={pick:.5g}")
    families_line = "лучшее по семействам: " + (", ".join(summary) or "—")

    own = [n for n in tree.nodes if n.family == node.family and n.node_id != node.node_id]
    rows = []
    for n in own[-limit:]:
        parent = tree.by_id(n.parent_id) if n.parent_id is not None else None
        delta = ""
        if parent is not None and parent.cv_score is not None and n.cv_score is not None:
            gain = _gain(ctx, parent.cv_score, n.cv_score)
            delta = f", Δ={gain:+.4g}" if abs(gain) > 1e-12 else ""
        verdict = "ПРИНЯТА" if n.reason.startswith("принята") else "отклонена"
        if n.diff:
            what = _changed_lines(n.diff)
        else:
            what = "(корень портфеля)" if n.parent_id is None else "(код не изменился)"
        score = "—" if n.cv_score is None else f"{n.cv_score:.5g}"
        folds = ""
        if (parent is not None and n.fold_scores and parent.fold_scores
                and len(n.fold_scores) == len(parent.fold_scores)):
            wins = sum(1 for b, c in zip(parent.fold_scores, n.fold_scores)
                       if _gain(ctx, b, c) > 0)
            folds = f", лучше в {wins}/{len(n.fold_scores)} фолдах"
        idea = f"«{n.hypothesis[:300]}» " if n.hypothesis else ""
        rows.append(f"- #{n.node_id} {idea}cv={score}{delta}{folds} {verdict}\n"
                    f"  изменение: {what}")
    return "\n".join([best_line, families_line,
                      "попытки в этой ветке:",
                      *(rows or ["(пока не было правок)"])])


def propose_edit(ctx, node: Node, tree: Tree, cfg, gate) -> tuple[str, str, str] | None:
    """
    Просит LLM минимальную правку кода ветки.

    Аргументы:
    ctx (TaskContext): контекст задачи
    node (Node): текущая вершина ветки
    tree (Tree): дерево (история попыток в этой ветке)
    cfg (LoopConfig): настройки
    gate (Gate): доступность LLM

    Возвращает:
    tuple[str, str, str] | None: (новый код, гипотеза словами, механика правки)
    или None, если правку получить не удалось
    """
    if not gate.available():
        return None
    history_lines = _history_block(ctx, tree, node)
    user = (
        f"## Задача\n{ctx.task_md[:2500]}\n\n"
        f"## Данные\n{json.dumps(ctx.profile, ensure_ascii=False, default=str)[:1200]}\n\n"
        f"## Метрика\n{ctx.metric} ({'больше' if ctx.direction == 'max' else 'меньше'} — лучше)\n\n"
        f"## Текущий код ветки «{node.family}» (CV={node.cv_score})\n"
        f"```python\n{node.code}\n```\n\n"
        f"## Что уже пробовали в этой ветке\n{history_lines}\n"
    )
    try:
        from . import llm
        answer = llm.query([{"role": "user", "content": user}], role="node_edit",
                           schema=None, phase="TREE",
                           timeout=min(getattr(cfg, "llm_timeout", 180.0),
                                       max(30.0, ctx.remaining() / 4)))
    except Exception as exc:  # noqa: BLE001
        gate.fail("node_edit", exc)
        return None
    gate.ok()
    if not isinstance(answer, str):
        return None
    new_code, hypothesis, note = apply_edits(node.code, answer)
    if new_code is None:
        get_logger("tree").log(kind="edit_rejected", node_id=node.node_id, problem=note)
        return None
    if new_code.strip() == node.code.strip():
        get_logger("tree").log(kind="edit_rejected", node_id=node.node_id, problem="код не изменился")
        return None
    problem = check_code(new_code)
    if problem:
        get_logger("tree").log(kind="edit_rejected", node_id=node.node_id, problem=problem)
        return None
    return new_code, hypothesis, note


# ----------------------------- ансамбль -----------------------------------
def _true_target(ctx):
    """Таргет train.csv в кодировании раннера (коды классов для классификации)."""
    try:
        raw = pd.read_csv(ctx.data_dir / "train.csv", usecols=[ctx.target])[ctx.target]
    except (OSError, ValueError, KeyError):
        return None
    if ctx.task_type == "classification":
        return raw.map({c: i for i, c in enumerate(sorted(raw.unique()))})
    return raw.astype(float)


def _score_pred(ctx, y, pred) -> float:
    """Метрика задачи по предсказаниям (1-D или вероятностям)."""
    if ctx.task_type == "classification":
        return float(tools.macro_f1(y, pred.argmax(axis=1)))
    return float(tools.rmse(y, pred))


def _gain(ctx, old: float, new: float) -> float:
    """Прирост с учётом направления метрики."""
    return (old - new) if ctx.direction == "min" else (new - old)


def hill_climb(ctx, nodes: list[Node], log, iterations: int = 4) -> dict | None:
    """
    Подбирает веса ансамбля по OOF (hill climbing по сетке шагов).

    Аргументы:
    ctx (TaskContext): контекст задачи
    nodes (list[Node]): успешные узлы с полными фолдами
    log (JsonlLogger): логгер
    iterations (int): сколько проходов по узлам делать

    Возвращает:
    dict | None: {members, weights, score, test_pred, oof} или None
    """
    candidates = [n for n in nodes if n.ok and len(n.fold_scores) == N_FOLDS]
    if not candidates:
        return None
    y = _true_target(ctx)
    if y is None:
        log.log(kind="ensemble_skip", reason="таргет не читается")
        return None
    oofs, tests = {}, {}
    for node in candidates:
        try:
            oofs[node.node_id] = np.load(node.artifacts["oof"])
            tests[node.node_id] = np.load(node.artifacts["test_pred"])
        except (OSError, ValueError, KeyError):
            continue
    candidates = [n for n in candidates if n.node_id in oofs and len(y) == len(oofs[n.node_id])]
    if not candidates:
        log.log(kind="ensemble_skip", reason="нет читаемых артефактов")
        return None

    best = max(candidates, key=lambda n: n.cv_score if ctx.direction == "max" else -n.cv_score)
    weights = {best.node_id: 1.0}
    best_score = best.cv_score
    steps = (0.05, 0.1, 0.15, 0.2, 0.3)
    for _ in range(iterations):
        improved = False
        for node in candidates:
            for step in steps:
                trial = dict(weights)
                trial[node.node_id] = trial.get(node.node_id, 0.0) + step
                total = sum(trial.values())
                blend = sum(w * oofs[nid] for nid, w in trial.items()) / total
                score = _score_pred(ctx, y, blend)
                if _gain(ctx, best_score, score) > 0:
                    weights, best_score, improved = trial, score, True
                    break
        if not improved:
            break

    total = sum(weights.values())
    blended_oof = sum(w * oofs[nid] for nid, w in weights.items()) / total
    blended_test = sum(w * tests[nid] for nid, w in weights.items()) / total
    members = sorted(weights)
    log.log(kind="ensemble", members=members, weights=[round(w / total, 3) for w in weights.values()],
            score=round(float(best_score), 6), best_single=best.cv_score)
    return {"members": members, "weights": {k: w / total for k, w in weights.items()},
            "score": float(best_score), "oof": blended_oof, "test_pred": blended_test}


def _write_submission(ctx, pred) -> None:
    """Пишет submission.csv атомарно (tmp + os.replace)."""
    if ctx.task_type == "classification":
        pred = np.array(ctx.classes)[pred.argmax(axis=1)]
    tmp = ctx.task_dir / "submission.tmp.csv"
    tools.make_submission(ctx.test_ids, pred, ctx.sample, str(tmp))
    os.replace(tmp, ctx.submission_path)


# ----------------------------- поиск --------------------------------------
def _best_first(ctx):
    """
    Ключ сортировки узлов: лучший первым при любом направлении метрики.

    Аргументы:
    ctx (TaskContext): контекст задачи (direction)

    Возвращает:
    callable: ключ для sorted/max
    """
    if ctx.direction == "max":
        return lambda node: -node.cv_score
    return lambda node: node.cv_score


def search(ctx, cfg, log) -> dict:
    """
    Полный цикл: портфель -> ветки -> правки -> ансамбль -> сабмит.

    Аргументы:
    ctx (TaskContext): контекст задачи
    cfg (LoopConfig): настройки
    log (JsonlLogger): логгер

    Возвращает:
    dict: сводка (лучший CV, число узлов, ансамбль)
    """
    tree = Tree(ctx.work_dir / "nodes.json", direction=ctx.direction)
    gate = Gate(cfg, log)

    families = choose_families(ctx, cfg, gate, log)
    log.log(kind="portfolio_start", families=families,
            n_rows=ctx.profile.get("n_rows"), remaining=round(ctx.remaining()))

    # 1) Корневые узлы: по рецепту на семейство.
    for family in families:
        if ctx.remaining() < 90:
            log.log(kind="tree_stop", reason="время на портфель")
            break
        recipe = FAMILIES[family]
        node = Node(node_id=tree.next_id(), parent_id=None, family=family, depth=0,
                    code=recipe.code, reason="корень портфеля")
        timeout = max(45.0, min(float(getattr(cfg, "exp_timeout_cap", 2400.0)),
                                0.35 * ctx.remaining() / max(len(families), 1)))
        tree.add(evaluate(ctx, node, timeout, log))

    roots = tree.ok_nodes()
    if not roots:
        log.log(kind="tree_stop", reason="ни один корень не посчитался")
        return {"families": families, "nodes": len(tree.nodes), "best_cv": None}
    # Ветки идут по очереди, лучшие первыми. branches=0 — брать все корни.
    active = sorted(roots, key=_best_first(ctx))
    limit = int(getattr(cfg, "branches", 0) or 0)
    if limit > 0:
        active = active[:limit]
    log.log(kind="tree_roots", best=[(n.node_id, n.family, n.cv_score) for n in active])

    # 2) Раунды правок: каждая живая ветка пытается улучшить свой код.
    rounds = int(getattr(cfg, "rounds", 6))
    for round_no in range(1, rounds + 1):
        if not gate.available():
            log.log(kind="tree_stop", reason="LLM недоступна")
            break
        if ctx.remaining() < 2.5 * max(tree.median_duration(), 60.0):
            log.log(kind="tree_stop", reason="время на правки")
            break
        improved_any = False
        # По очереди: каждая живая ветка получает ровно одну попытку за раунд,
        # порядок — по текущему качеству ветки (лучшие первыми).
        active.sort(key=_best_first(ctx))
        for index, leaf in enumerate(list(active)):
            if ctx.remaining() < 1.5 * max(tree.median_duration(), 60.0):
                break
            before = gate.tokens()
            proposal = propose_edit(ctx, leaf, tree, cfg, gate)
            if proposal is None:
                continue
            code, hypothesis, note = proposal
            child = Node(node_id=tree.next_id(), parent_id=leaf.node_id, family=leaf.family,
                         depth=leaf.depth + 1, code=code, hypothesis=hypothesis,
                         note=note, reason=note, diff=_diff(leaf.code, code),
                         tokens=gate.tokens() - before)
            timeout = max(45.0, min(float(getattr(cfg, "exp_timeout_cap", 2400.0)),
                                    4.0 * max(tree.median_duration(), 60.0)))
            tree.add(evaluate(ctx, child, timeout, log))
            if child.ok and _gain(ctx, leaf.cv_score, child.cv_score) > 0:
                child.reason = f"принята: cv {child.cv_score:.5g} лучше {leaf.cv_score:.5g}"
                tree.save()
                active[index] = child
                improved_any = True
                log.log(kind="branch_improved", family=leaf.family, node_id=child.node_id,
                        parent_id=leaf.node_id, cv_score=child.cv_score, prev=leaf.cv_score,
                        hypothesis=child.hypothesis[:300])
            else:
                if child.ok and child.cv_score is not None:
                    child.reason = (f"хуже родителя: cv {child.cv_score:.5g} "
                                    f"против {leaf.cv_score:.5g}")
                    tree.save()
                log.log(kind="branch_rejected", family=leaf.family, node_id=child.node_id,
                        cv_score=child.cv_score, prev=leaf.cv_score,
                        hypothesis=child.hypothesis[:300],
                        reason=(child.reason or "нет прироста")[:200])
        log.log(kind="round_done", round=round_no, improved=improved_any,
                nodes=len(tree.nodes))
        # Неудачи раунда — не повод останавливаться: они уходят в контекст
        # следующих правок. Стоп только если явно задан patience_rounds.
        patience = int(getattr(cfg, "patience_rounds", 0) or 0)
        if patience and not improved_any:
            streak = getattr(search, "_fail_streak", 0) + 1
            search._fail_streak = streak
            if streak >= patience:
                log.log(kind="tree_stop", reason=f"{streak} раундов без улучшений")
                break
        else:
            search._fail_streak = 0

    # 3) Ансамбль листьев и сабмит.
    best = tree.best()
    result = hill_climb(ctx, tree.ok_nodes(), log)
    chosen = None
    if result is not None and best is not None:
        if _gain(ctx, best.cv_score, result["score"]) > 0:
            _write_submission(ctx, result["test_pred"])
            chosen = "ensemble"
            log.log(kind="submission", source="ensemble", cv_score=result["score"],
                    members=result["members"])
        else:
            _write_submission(ctx, np.load(best.artifacts["test_pred"]))
            chosen = "best_node"
            log.log(kind="submission", source="best_node", cv_score=best.cv_score,
                    node_id=best.node_id)
    elif best is not None:
        _write_submission(ctx, np.load(best.artifacts["test_pred"]))
        chosen = "best_node"
        log.log(kind="submission", source="best_node", cv_score=best.cv_score,
                node_id=best.node_id)

    summary = {
        "families": families,
        "nodes": len(tree.nodes),
        "best_cv": best.cv_score if best else None,
        "best_node": best.node_id if best else None,
        "ensemble_cv": result["score"] if result else None,
        "ensemble_members": result["members"] if result else [],
        "submission_from": chosen,
        "tokens": gate.tokens(),
    }
    log.log(kind="tree_done", **summary)
    return summary
