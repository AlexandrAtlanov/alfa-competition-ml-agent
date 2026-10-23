"""Портфель базовых моделей и их стартовый код.

Идея: агент не «придумывает» модель с нуля каждый раз, а получает набор
готовых рецептов (как портфель AutoGluon), выбирает из них подходящие и
дальше улучшает **код** выбранной ветки. Рецепт — это исходник узла,
который исполняет песочница (см. agent/tree_runner.py).

Контракт исходника узла:
    def build_model(task_type: str, n_classes: int | None, n_features: int)
    # опционально:
    def add_features(train, test) -> tuple[train, test]
    TARGET_TRANSFORM = "log1p" | None      # регрессия, таргет > -1
    POSTPROCESS = "class_weight_opt" | None  # классификация

Только классический ML из requirements организаторов: GBDT, деревья,
линейные, kNN. Никакого интернета и предобученных весов.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Family:
    """Один рецепт портфеля."""

    name: str
    title: str
    kind: str          # gbdt | trees | linear | knn
    cost: str          # low | med | high — ожидаемое время обучения
    code: str          # стартовый исходник узла
    max_rows: int | None = None  # семейство не имеет смысла на больших данных


LGBM = '''import lightgbm as lgb


def build_model(task_type, n_classes, n_features):
    params = dict(n_estimators=600, learning_rate=0.05, num_leaves=63,
                  subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                  n_jobs=8, random_state=42, verbose=-1)
    if task_type == "classification":
        return lgb.LGBMClassifier(class_weight="balanced", **params)
    return lgb.LGBMRegressor(**params)
'''

CATBOOST = '''from catboost import CatBoostClassifier, CatBoostRegressor


def build_model(task_type, n_classes, n_features):
    params = dict(iterations=800, learning_rate=0.06, depth=6, random_seed=42,
                  verbose=0, thread_count=8, allow_writing_files=False)
    if task_type == "classification":
        return CatBoostClassifier(loss_function="MultiClass",
                                  auto_class_weights="Balanced", **params)
    return CatBoostRegressor(**params)
'''

XGBOOST = '''import xgboost as xgb


def build_model(task_type, n_classes, n_features):
    params = dict(n_estimators=700, learning_rate=0.05, max_depth=6,
                  subsample=0.8, colsample_bytree=0.8, tree_method="hist",
                  n_jobs=8, random_state=42)
    if task_type == "classification":
        return xgb.XGBClassifier(**params)
    return xgb.XGBRegressor(**params)
'''

HIST = '''from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor


def build_model(task_type, n_classes, n_features):
    params = dict(max_iter=400, learning_rate=0.06, max_leaf_nodes=63,
                  early_stopping=False, random_state=42)
    if task_type == "classification":
        return HistGradientBoostingClassifier(**params)
    return HistGradientBoostingRegressor(**params)
'''

EXTRATREES = '''from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


def build_model(task_type, n_classes, n_features):
    if task_type == "classification":
        est = ExtraTreesClassifier(n_estimators=400, n_jobs=8, random_state=42)
    else:
        est = ExtraTreesRegressor(n_estimators=400, n_jobs=8, random_state=42)
    return Pipeline([("impute", SimpleImputer(strategy="median")),
                     ("model", est)])
'''

RIDGE = '''from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_model(task_type, n_classes, n_features):
    if task_type == "classification":
        est = LogisticRegression(max_iter=2000, class_weight="balanced")
    else:
        est = Ridge(alpha=1.0)
    return Pipeline([("impute", SimpleImputer(strategy="median")),
                     ("scale", StandardScaler()),
                     ("model", est)])
'''

KNN = '''from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_model(task_type, n_classes, n_features):
    if task_type == "classification":
        est = KNeighborsClassifier(n_neighbors=25, n_jobs=8)
    else:
        est = KNeighborsRegressor(n_neighbors=25, n_jobs=8)
    return Pipeline([("impute", SimpleImputer(strategy="median")),
                     ("scale", StandardScaler()),
                     ("model", est)])
'''

FAMILIES: dict[str, Family] = {
    "lgbm": Family("lgbm", "LightGBM", "gbdt", "med", LGBM),
    "catboost": Family("catboost", "CatBoost", "gbdt", "high", CATBOOST),
    "xgboost": Family("xgboost", "XGBoost", "gbdt", "high", XGBOOST),
    "hist": Family("hist", "HistGradientBoosting (sklearn)", "gbdt", "med", HIST),
    "extratrees": Family("extratrees", "ExtraTrees", "trees", "med", EXTRATREES),
    "ridge": Family("ridge", "Ridge / LogisticRegression", "linear", "low", RIDGE),
    "knn": Family("knn", "kNN", "knn", "low", KNN, max_rows=20_000),
}


def available_families(n_rows: int) -> list[str]:
    """
    Возвращает семейства, осмысленные для такого размера выборки.

    Аргументы:
    n_rows (int): число строк train

    Возвращает:
    list[str]: имена семейств в стабильном порядке
    """
    return [name for name, family in FAMILIES.items()
            if family.max_rows is None or n_rows <= family.max_rows]


def family_menu() -> str:
    """Строка для промпта: имя — описание — стоимость — когда уместно."""
    lines = []
    for family in FAMILIES.values():
        lines.append(f"- {family.name}: {family.title}; cost={family.cost}; "
                     f"kind={family.kind}")
    return "\n".join(lines)
