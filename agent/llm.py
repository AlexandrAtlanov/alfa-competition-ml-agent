"""Единственная точка вызова LLM (OpenAI-совместимый API).

Правила проекта (AGENTS.md §3–4):
- все обращения к модели идут только через этот модуль;
- ключ/адрес/модель читаются из окружения (OPENAI_API_KEY, OPENAI_BASE_URL,
  OPENAI_MODEL) — в проде их подставляет платформа;
- в локальной разработке значения подхватываются из .env;
- ретраи только на транзиентные ошибки, с экспоненциальной паузой;
- каждый вызов учитывается по токенам и пишется в jsonl-лог.

Структурированный вывод (по ролям):
- `analyze_task`, `hypothesis`, `review` -> dict, валидируется по схеме
  из agent/schemas.py (response_format=json_object + локальная проверка);
- `code` -> str: из ответа вырезается markdown-блок с кодом.

Синхронная реализация. При необходимости async-вариант добавляется
отдельной функцией, не меняя контракт query().
"""
from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass

import httpx

from .event_log import get_logger
from .prompts import build_system
from .schemas import CODE_ROLE, schema_for, validate

# --- Локальное окружение ---------------------------------------------------
# .env читаем только в разработке; уже заданные переменные не перетираем.
try:
    from dotenv import load_dotenv

    load_dotenv(override=False)
except ImportError:  # pragma: no cover — в проде .env не нужен
    pass

try:
    import openai
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "Не найден пакет openai. Для локального запуска: "
        "pip install -r requirements-dev.txt"
    ) from exc


DEFAULT_MODEL = "qwen/qwen3.8-27b"
_AUTO = object()  # «schema не передана» — взять схему роли по умолчанию


class StructuredOutputError(RuntimeError):
    """Модель не вернула валидный JSON по схеме после всех попыток."""


def _parse_timeout(raw: str | None, default: float | None) -> float | None:
    """Разобрать таймаут: пусто/0/none/off -> None (без лимита)."""
    if raw is None or raw.strip() == "":
        return default
    value = raw.strip().lower()
    if value in ("0", "none", "off", "no"):
        return None
    return float(value)


@dataclass(frozen=True)
class LLMConfig:
    """Настройки клиента, собранные из переменных окружения."""

    api_key: str
    base_url: str | None
    model: str
    timeout: float | None
    code_timeout: float | None
    max_retries: int
    temperature: float
    max_tokens: int | None

    @classmethod
    def from_env(cls) -> "LLMConfig":
        """Собрать конфиг из окружения; бросить понятную ошибку без ключа."""
        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY не задан. Заполни .env локально "
                "(или передай переменную окружения при запуске)."
            )
        max_tokens_raw = os.getenv("LLM_MAX_TOKENS", "").strip()
        return cls(
            api_key=api_key,
            base_url=os.getenv("OPENAI_BASE_URL") or None,
            model=os.getenv("OPENAI_MODEL", DEFAULT_MODEL),
            timeout=_parse_timeout(os.getenv("LLM_TIMEOUT"), 120.0),
            code_timeout=_parse_timeout(os.getenv("LLM_CODE_TIMEOUT"), None),
            max_retries=int(os.getenv("LLM_MAX_RETRIES", "5")),
            temperature=float(os.getenv("LLM_TEMPERATURE", "0.2")),
            max_tokens=int(max_tokens_raw) if max_tokens_raw else None,
        )


@dataclass
class TokenUsage:
    """Накопленный расход токенов."""

    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    def as_dict(self) -> dict:
        return {
            "calls": self.calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
        }


_usage = TokenUsage()
_client_cache: dict[tuple[str | None, str], "openai.OpenAI"] = {}


def get_usage() -> TokenUsage:
    """Копия текущего расхода токенов."""
    return TokenUsage(**_usage.as_dict())


def reset_usage() -> None:
    """Обнулить счётчик (например, в начале новой задачи)."""
    global _usage
    _usage = TokenUsage()


def _get_client(cfg: LLMConfig) -> "openai.OpenAI":
    key = (cfg.base_url, cfg.api_key)
    client = _client_cache.get(key)
    if client is None:
        client = openai.OpenAI(
            api_key=cfg.api_key,
            base_url=cfg.base_url,
            timeout=cfg.timeout,
            max_retries=0,  # ретраи делает наш _chat_with_retries()
        )
        _client_cache[key] = client
    return client


def _is_retryable(exc: Exception) -> bool:
    """Транзиентные ошибки: лимиты, сеть, таймауты, 5xx."""
    if isinstance(
        exc,
        (
            openai.RateLimitError,
            openai.APIConnectionError,
            openai.APITimeoutError,
            openai.InternalServerError,
        ),
    ):
        return True
    if isinstance(exc, openai.APIStatusError):
        return exc.status_code in (429, 500, 502, 503, 504)
    return False


def _is_unsupported_response_format(exc: Exception) -> bool:
    """Провайдер не умеет response_format=json_object (частый случай у локальных)."""
    text = str(exc).lower()
    return "response_format" in text or "json_object" in text


def _backoff_delay(attempt: int) -> float:
    """Экспоненциальная пауза: 1.5, 2.25, 3.4, ... с потолком 60 с."""
    return min(1.5 * (1.5 ** attempt), 60.0)


def _chat_with_retries(
    client, kwargs: dict, cfg: LLMConfig, log, phase: str | None = None
):
    """Один вызов с сетевыми ретраями. Возвращает (response, номер_попытки)."""
    last_error: Exception | None = None
    for attempt in range(cfg.max_retries + 1):
        started = time.time()
        try:
            return client.chat.completions.create(**kwargs), attempt
        except Exception as exc:  # noqa: BLE001 — классифицируем ниже
            duration = time.time() - started
            retryable = _is_retryable(exc)
            log.log(
                kind="llm_error",
                phase=phase,
                attempt=attempt,
                model=cfg.model,
                duration=round(duration, 3),
                retryable=retryable,
                error=f"{type(exc).__name__}: {exc}",
            )
            if not retryable or attempt >= cfg.max_retries:
                last_error = exc
                break
            time.sleep(_backoff_delay(attempt))
    raise last_error  # type: ignore[misc]


def _accumulate_usage(response) -> tuple[int, int, int]:
    """Учесть токены ответа и вернуть (prompt, completion, total)."""
    usage = getattr(response, "usage", None)
    prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
    completion_tokens = getattr(usage, "completion_tokens", 0) or 0
    total_tokens = getattr(
        usage, "total_tokens", prompt_tokens + completion_tokens
    ) or 0
    _usage.calls += 1
    _usage.prompt_tokens += prompt_tokens
    _usage.completion_tokens += completion_tokens
    _usage.total_tokens += total_tokens
    return prompt_tokens, completion_tokens, total_tokens


def _with_schema_instruction(messages: list[dict], schema: dict | None) -> list[dict]:
    """Добавить в system-сообщение требование вернуть JSON по схеме."""
    result = [dict(message) for message in messages]
    if schema is None:
        return result
    instruction = (
        "Верни ответ строго как JSON-объект, соответствующий схеме ниже. "
        "Никакого markdown, никаких пояснений вокруг.\n"
        "Схема:\n" + json.dumps(schema, ensure_ascii=False, indent=2)
    )
    if result and result[0].get("role") == "system":
        result[0]["content"] = result[0]["content"].rstrip() + "\n\n" + instruction
    else:
        result.insert(0, {"role": "system", "content": instruction})
    return result


def _strip_fences(text: str) -> str:
    text = (text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json|JSON)?\s*", "", text)
        text = re.sub(r"\s*```\s*$", "", text)
    return text.strip()


def _parse_and_validate(text: str, schema: dict) -> tuple[dict | None, str]:
    """Распарсить JSON из ответа и проверить схему. Вернуть (data, ошибка)."""
    stripped = _strip_fences(text)
    candidates = [stripped]
    match = re.search(r"\{.*\}", stripped, re.DOTALL)
    if match:
        candidates.append(match.group(0))

    for candidate in candidates:
        try:
            data = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        errors = validate(data, schema)
        if not errors:
            return data, ""
        return None, "; ".join(errors)
    return None, "валидный JSON не найден"


_CODE_FENCE_RE = re.compile(r"```(?:python|py)?\s*\n?(.*?)```", re.DOTALL)


def extract_code(text: str) -> str:
    """Вырезать python-код из markdown-блока; если блока нет — вернуть текст."""
    text = (text or "").strip()
    match = _CODE_FENCE_RE.search(text)
    if match:
        return match.group(1).strip()
    return text


def query(
    messages: list[dict],
    *,
    temperature: float | None = None,
    max_tokens: int | None = None,
    cfg: LLMConfig | None = None,
    schema: dict | None = None,
    role: str | None = None,
    phase: str | None = None,
    timeout=_AUTO,
) -> str | dict:
    """Низкоуровневый вызов чата.

    Если задан `schema` — возвращает dict, провалидированный по схеме
    (с одним corrective-retry). Иначе — текст ответа.

    `timeout=None` — ждать ответ без лимита (нужно для генерации кода).
    По умолчанию для роли `code` берётся `LLM_CODE_TIMEOUT` (пусто = без лимита),
    для остальных ролей — `LLM_TIMEOUT`.
    """
    cfg = cfg or LLMConfig.from_env()
    if timeout is _AUTO:
        timeout = cfg.code_timeout if role == CODE_ROLE else cfg.timeout
    client = _get_client(cfg)
    log = get_logger("llm")

    base_messages = _with_schema_instruction(messages, schema)
    base_kwargs: dict = {
        "model": cfg.model,
        "temperature": cfg.temperature if temperature is None else temperature,
        "timeout": httpx.Timeout(None) if timeout is None else timeout,
    }
    resolved_max_tokens = cfg.max_tokens if max_tokens is None else max_tokens
    if resolved_max_tokens is not None:
        base_kwargs["max_tokens"] = resolved_max_tokens
    if schema is not None:
        base_kwargs["response_format"] = {"type": "json_object"}

    parse_attempts = 2 if schema is not None else 1
    current_messages = base_messages
    last_text = ""
    last_problem = ""

    for parse_attempt in range(parse_attempts):
        call_kwargs = dict(base_kwargs)
        call_kwargs["messages"] = current_messages
        started = time.time()
        try:
            response, _ = _chat_with_retries(client, call_kwargs, cfg, log, phase=phase)
        except Exception as exc:
            # локальные модели часто не поддерживают response_format — пробуем без него
            if (
                schema is not None
                and "response_format" in call_kwargs
                and _is_unsupported_response_format(exc)
            ):
                call_kwargs.pop("response_format", None)
                response, _ = _chat_with_retries(client, call_kwargs, cfg, log, phase=phase)
            else:
                raise

        duration = time.time() - started
        text = response.choices[0].message.content or ""
        prompt_tokens, completion_tokens, total_tokens = _accumulate_usage(response)
        log.log(
            kind="llm_response",
            role=role,
            phase=phase,
            attempt=parse_attempt,
            model=cfg.model,
            duration=round(duration, 3),
            structured=schema is not None,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            response_chars=len(text),
        )

        if schema is None:
            return text

        parsed, problem = _parse_and_validate(text, schema)
        if parsed is not None:
            return parsed

        last_text, last_problem = text, problem
        log.log(
            kind="llm_parse_error",
            role=role,
            phase=phase,
            attempt=parse_attempt,
            problem=problem,
        )
        current_messages = base_messages + [
            {"role": "assistant", "content": text},
            {
                "role": "user",
                "content": (
                    f"Ответ не соответствует схеме: {problem}. "
                    "Верни ТОЛЬКО валидный JSON без markdown и пояснений."
                ),
            },
        ]

    raise StructuredOutputError(
        f"Не удалось получить валидный JSON для роли {role!r}: {last_problem}. "
        f"Ответ: {last_text[:400]!r}"
    )


def complete(
    role: str,
    user: str,
    *,
    schema=_AUTO,
    phase: str | None = None,
    **kwargs,
) -> str | dict:
    """Высокоуровневый вызов: системный промпт роли + пользовательский текст.

    Схема выбирается по роли автоматически (см. agent/schemas.py).
    Для роли `code` из ответа вырезается markdown-блок с кодом.
    """
    if schema is _AUTO:
        schema = schema_for(role)

    system = build_system(role)
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    result = query(messages, schema=schema, role=role, phase=phase, **kwargs)

    if role == CODE_ROLE and schema is None and isinstance(result, str):
        return extract_code(result)
    return result


def describe() -> dict:
    """Краткая сводка конфига для отладки (ключ замаскирован)."""
    cfg = LLMConfig.from_env()
    return {
        "model": cfg.model,
        "base_url": cfg.base_url,
        "timeout": cfg.timeout,
        "code_timeout": cfg.code_timeout,
        "max_retries": cfg.max_retries,
        "temperature": cfg.temperature,
        "max_tokens": cfg.max_tokens,
        "api_key": (cfg.api_key[:6] + "…") if cfg.api_key else None,
    }
