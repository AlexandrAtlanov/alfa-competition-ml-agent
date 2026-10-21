"""Реестр экспериментов и контракты записи.

Контракты, общие для всей команды:
- `ExperimentSpec` — полная конфигурация эксперимента. Каждая гипотеза —
  дельта к спеке текущего лучшего (`merge`), поэтому принятые изменения
  накапливаются;
- `Candidate` — гипотеза до запуска (от LLM или из детерминированного меню);
- `ExperimentRecord` — одна строка реестра (формат `experiments.json`).

Реестр хранит историю в памяти и после каждой записи атомарно
перезаписывает experiments.json; каждое событие дублируется в
logs/registry.jsonl.
"""
from __future__ import annotations

import hashlib
import json
import os
import statistics
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .event_log import get_logger

CHANGE_TYPES = (
    "feature", "encoding", "preprocessing", "model", "validation", "ensemble", "other",
)
LEVELS = ("low", "med", "high")
# Ключи дельты, которые понимает merge (всё остальное отбрасывается).
DELTA_KEYS = ("lgbm_params", "class_weight", "target_transform", "postprocess", "feature_patch")


# ----------------------------- спека --------------------------------------
@dataclass
class ExperimentSpec:
    """
    Полная конфигурация эксперимента (то, что исполняет wrapper в песочнице).
    """

    lgbm_params: dict = field(default_factory=dict)  # поверх дефолтов tools.make_model
    class_weight: str | None = None                  # None | "balanced"
    target_transform: str | None = None              # None | "log1p"
    postprocess: str | None = None                   # None | "class_weight_opt"
    feature_patches: list[str] = field(default_factory=list)  # исходники add_features

    def merge(self, delta: dict) -> "ExperimentSpec":
        """
        Собирает новую спеку: текущая + дельта.

        lgbm_params сливаются по ключам (None удаляет ключ);
        feature_patch (str) дописывается в конец списка патчей.

        Аргументы:
        delta (dict): изменения, ключи из DELTA_KEYS

        Возвращает:
        ExperimentSpec
        """
        params = dict(self.lgbm_params)
        for key, value in (delta.get("lgbm_params") or {}).items():
            if value is None:
                params.pop(key, None)
            else:
                params[key] = value
        patches = list(self.feature_patches)
        if delta.get("feature_patch"):
            patches.append(delta["feature_patch"])
        return ExperimentSpec(
            lgbm_params=params,
            class_weight=delta.get("class_weight", self.class_weight),
            target_transform=delta.get("target_transform", self.target_transform),
            postprocess=delta.get("postprocess", self.postprocess),
            feature_patches=patches,
        )

    def signature(self) -> str:
        """
        Вычисляет хэш нормализованной спеки — для дедупа экспериментов.

        Возвращает:
        str: первые 12 символов sha1
        """
        blob = json.dumps(asdict(self), sort_keys=True, ensure_ascii=False, default=str)
        return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:12]

    def to_dict(self) -> dict:
        """
        Сериализует спеку целиком (с исходниками патчей).

        Возвращает:
        dict
        """
        return asdict(self)

    def summary(self) -> dict:
        """
        Собирает компактный вид для промпта и логов (без исходников патчей).

        Возвращает:
        dict
        """
        return {
            "lgbm_params": self.lgbm_params,
            "class_weight": self.class_weight,
            "target_transform": self.target_transform,
            "postprocess": self.postprocess,
            "n_feature_patches": len(self.feature_patches),
        }


# ----------------------------- кандидат -----------------------------------
@dataclass
class Candidate:
    """
    Гипотеза до запуска.
    """

    title: str
    rationale: str
    change_type: str                 # из CHANGE_TYPES
    delta: dict                      # ключи из DELTA_KEYS (кроме feature_patch)
    feature_idea: str | None = None  # описание признаков -> патч генерирует LLM
    expected_gain: str = "med"       # low | med | high
    cost: str = "med"                # low | med | high
    source: str = "menu"             # llm | menu

    def key(self) -> str:
        """
        Вычисляет ключ дедупа до запуска.

        Для признаков исход неизвестен до генерации кода, поэтому ключ —
        по нормализованному описанию; для остального — по дельте.

        Возвращает:
        str
        """
        if self.feature_idea:
            return "feat:" + " ".join(self.title.lower().split())[:80]
        return "delta:" + json.dumps(self.delta, sort_keys=True, default=str)

    def to_dict(self) -> dict:
        """
        Сериализует кандидата.

        Возвращает:
        dict
        """
        return asdict(self)


# ----------------------------- запись -------------------------------------
@dataclass
class ExperimentRecord:
    """
    Одна строка реестра.
    """

    exp_id: int
    parent_id: int | None
    phase: str
    candidate: dict | None           # Candidate.to_dict(); None для бейзлайна
    spec: dict                       # ExperimentSpec.to_dict()
    signature: str
    status: str                      # ok | error | timeout | invalid
    cv_score: float | None = None
    fold_scores: list[float] = field(default_factory=list)
    duration: float = 0.0
    accepted: bool = False
    reason: str = ""
    tokens: int = 0
    artifacts: dict = field(default_factory=dict)  # oof / test_pred / dir

    @property
    def change_type(self) -> str | None:
        """
        Возвращает тип изменения кандидата.

        Возвращает:
        str | None: None для бейзлайна
        """
        return (self.candidate or {}).get("change_type")

    @property
    def title(self) -> str:
        """
        Возвращает заголовок гипотезы.

        Возвращает:
        str: "baseline" для бейзлайна
        """
        return (self.candidate or {}).get("title") or "baseline"

    def to_dict(self) -> dict:
        """
        Сериализует запись для experiments.json.

        Возвращает:
        dict
        """
        return asdict(self)


# ----------------------------- реестр -------------------------------------
class Registry:
    """
    История экспериментов одной задачи.
    """

    def __init__(self, path: str | Path):
        """
        Создаёт пустой реестр.

        Аргументы:
        path (str | Path): путь к experiments.json
        """
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.records: list[ExperimentRecord] = []
        self._log = get_logger("registry")

    def next_id(self) -> int:
        """
        Выдаёт номер следующего эксперимента.

        Возвращает:
        int
        """
        return len(self.records)

    def add(self, rec: ExperimentRecord) -> None:
        """
        Добавляет запись, пишет событие в registry.jsonl и сохраняет experiments.json.

        Аргументы:
        rec (ExperimentRecord): запись эксперимента с вердиктом
        """
        self.records.append(rec)
        event = rec.to_dict()
        event["spec"] = ExperimentSpec(**rec.spec).summary()  # без исходников патчей
        self._log.log(kind="experiment", **event)
        self._dump()

    def _dump(self) -> None:
        """
        Атомарно перезаписывает experiments.json (tmp-файл + os.replace).
        """
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(
            json.dumps([r.to_dict() for r in self.records], ensure_ascii=False,
                       indent=1, default=str),
            encoding="utf-8",
        )
        os.replace(tmp, self.path)

    # --- запросы ---
    def best(self) -> ExperimentRecord | None:
        """
        Находит текущий лучший = последний принятый (принимаем только улучшения).

        Возвращает:
        ExperimentRecord | None: None, если принятых ещё нет
        """
        for rec in reversed(self.records):
            if rec.accepted:
                return rec
        return None

    def tried_signatures(self) -> set[str]:
        """
        Собирает сигнатуры всех запущенных спек.

        Возвращает:
        set[str]
        """
        return {r.signature for r in self.records}

    def tried_keys(self) -> set[str]:
        """
        Собирает ключи всех проверенных кандидатов (Candidate.key).

        Возвращает:
        set[str]
        """
        keys = set()
        for r in self.records:
            if r.candidate:
                keys.add(Candidate(**r.candidate).key())
        return keys

    def recent(self, n: int = 8) -> list[ExperimentRecord]:
        """
        Возвращает последние записи.

        Аргументы:
        n (int): сколько записей взять

        Возвращает:
        list[ExperimentRecord]
        """
        return self.records[-n:]

    def streak_fail(self, change_type: str) -> int:
        """
        Считает, сколько последних экспериментов этого типа подряд не приняты.

        Аргументы:
        change_type (str): тип изменения из CHANGE_TYPES

        Возвращает:
        int
        """
        streak = 0
        for rec in reversed(self.records):
            if rec.change_type != change_type:
                continue
            if rec.accepted:
                break
            streak += 1
        return streak

    def tried_change_types(self) -> set[str]:
        """
        Собирает типы изменений, которые уже пробовали.

        Возвращает:
        set[str]
        """
        return {r.change_type for r in self.records if r.change_type}

    def median_duration(self, default: float = 120.0) -> float:
        """
        Вычисляет медианную длительность успешных экспериментов.

        Аргументы:
        default (float): значение, пока успешных экспериментов нет

        Возвращает:
        float: секунды
        """
        durations = [r.duration for r in self.records if r.status == "ok" and r.duration > 0]
        return statistics.median(durations) if durations else default

    def history_text(self, n: int = 10) -> str:
        """
        Форматирует последние эксперименты строками — для промпта гипотез.

        Аргументы:
        n (int): сколько последних экспериментов показать

        Возвращает:
        str
        """
        if not self.records:
            return "(пока нет)"
        lines = []
        for rec in self.recent(n):
            score = f"{rec.cv_score:.5g}" if rec.cv_score is not None else "—"
            if rec.accepted:
                verdict = "ПРИНЯТА"
            elif rec.status != "ok":
                verdict = f"{rec.status}: {rec.reason[:120]}"
            else:
                verdict = f"отклонена: {rec.reason[:120]}"
            cand = rec.candidate or {}
            what = cand.get("feature_idea") or json.dumps(cand.get("delta", {}), ensure_ascii=False)
            lines.append(
                f"#{rec.exp_id} [{rec.change_type or 'baseline'}] {rec.title[:80]} | "
                f"{what[:160]} -> {score} ({verdict})"
            )
        return "\n".join(lines)
