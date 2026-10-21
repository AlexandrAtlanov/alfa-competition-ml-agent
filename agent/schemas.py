"""JSON-схемы структурированных ответов по ролям.

Разделение:
- `analyze_task`, `hypothesis`, `review` — возвращают JSON (схема ниже);
- `code` — возвращает текст (markdown-блок), схема не нужна.

Валидатор — минимальный, без внешних зависимостей: поддерживает только
те ключевые слова, которые используются в схемах (type, enum, required,
properties, items, maxItems, minLength, minimum, maximum). Этого достаточно,
а лишняя зависимость в проде не нужна.
"""
from __future__ import annotations

# --- схемы ------------------------------------------------------------------

ANALYZE_TASK_SCHEMA = {
    "type": "object",
    "properties": {
        "task_type": {
            "type": "string",
            "enum": ["regression", "classification", "other"],
        },
        "target": {"type": "string"},
        "id_column": {"type": ["string", "null"]},
        "metric": {"type": "string"},
        "metric_direction": {"type": "string", "enum": ["min", "max"]},
        "n_classes": {"type": ["integer", "null"]},
        "notes": {"type": "array", "items": {"type": "string"}, "maxItems": 5},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
    },
    "required": ["task_type", "target", "metric", "metric_direction"],
}

HYPOTHESIS_SCHEMA = {
    "type": "object",
    "properties": {
        "hypothesis": {"type": "string", "minLength": 10},
        "rationale": {"type": "string"},
        "expected_effect": {"type": "string"},
        "change_type": {
            "type": "string",
            "enum": [
                "feature",
                "encoding",
                "preprocessing",
                "model",
                "validation",
                "ensemble",
                "other",
            ],
        },
        "atomic": {"type": "boolean"},
        "test_plan": {"type": "string"},
    },
    "required": ["hypothesis", "rationale", "change_type", "atomic"],
}

REVIEW_SCHEMA = {
    "type": "object",
    "properties": {
        "status": {
            "type": "string",
            "enum": ["ok", "bug", "timeout", "no_metric", "error"],
        },
        "reason": {"type": "string"},
        "next_action": {"type": ["string", "null"]},
        "leak_suspected": {"type": "boolean"},
        "metric_found": {"type": "boolean"},
    },
    "required": ["status", "reason", "leak_suspected"],
}

SCHEMAS: dict[str, dict] = {
    "analyze_task": ANALYZE_TASK_SCHEMA,
    "hypothesis": HYPOTHESIS_SCHEMA,
    "review": REVIEW_SCHEMA,
}

# Роли, для которых llm.py ожидает текст, а не JSON.
CODE_ROLE = "code"


def schema_for(role: str) -> dict | None:
    """Вернуть схему роли или None, если роль текстовая."""
    return SCHEMAS.get(role)


# --- мини-валидатор ---------------------------------------------------------

_TYPE_MAP = {
    "string": str,
    "boolean": bool,
    "array": list,
    "object": dict,
    "null": type(None),
}


def _matches_type(value, type_spec) -> bool:
    names = type_spec if isinstance(type_spec, list) else [type_spec]
    for name in names:
        if name in ("number", "integer"):
            if isinstance(value, bool):
                continue
            if name == "integer":
                if isinstance(value, int):
                    return True
            elif isinstance(value, (int, float)):
                return True
            continue
        expected = _TYPE_MAP.get(name)
        if expected is not None and isinstance(value, expected):
            return True
    return False


def validate(data, schema: dict, path: str = "$") -> list[str]:
    """Проверить данные по подмножеству JSON Schema. Вернуть список ошибок."""
    errors: list[str] = []

    if "type" in schema and not _matches_type(data, schema["type"]):
        errors.append(f"{path}: ожидался тип {schema['type']}")
        return errors

    if "enum" in schema and data not in schema["enum"]:
        errors.append(f"{path}: значение {data!r} не из {schema['enum']}")

    if isinstance(data, dict):
        for name in schema.get("required", []):
            if name not in data:
                errors.append(f"{path}: нет обязательного поля {name!r}")
        for name, subschema in schema.get("properties", {}).items():
            if name in data:
                errors.extend(validate(data[name], subschema, f"{path}.{name}"))

    if isinstance(data, list) and "items" in schema:
        if "maxItems" in schema and len(data) > schema["maxItems"]:
            errors.append(f"{path}: элементов больше {schema['maxItems']}")
        for index, item in enumerate(data):
            errors.extend(validate(item, schema["items"], f"{path}[{index}]"))

    if isinstance(data, str) and "minLength" in schema:
        if len(data) < schema["minLength"]:
            errors.append(f"{path}: короче {schema['minLength']} символов")

    if isinstance(data, (int, float)) and not isinstance(data, bool):
        if "minimum" in schema and data < schema["minimum"]:
            errors.append(f"{path}: меньше {schema['minimum']}")
        if "maximum" in schema and data > schema["maximum"]:
            errors.append(f"{path}: больше {schema['maximum']}")

    return errors
