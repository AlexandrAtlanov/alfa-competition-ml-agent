# alfa-ml-agent

Решение хакатона Альфа-Банка "AI-агент для разработки ML-моделей". Агент,
без человека решающий произвольные ML-задачи end-to-end - от чтения условия
до файла предсказаний. Проверяеются две публичные задачи (регрессия / RMSE;
многоклассовая классификация / Macro F1) и две скрытые, которые агент увидит
впервые.

Архитектура: детерминированный каркас (фазы, CV, реестр
экспериментов, сабмит) + LLM только для выбора гипотез и генерации кода
признаков. При любом сбое деградация к бейзлайн-пути без LLM. Скор
нормируется от бейзлайна организаторов. Задача минимум — стабильно
превзойти его.

## Структура

```
solution.py               # точка входа: интерфейс baseline, таймер, деградация
agent/
  loop.py                 # автомат фаз: SCOUT -> BASELINE -> RESEARCH -> FINALIZE
  llm.py                  # единственная обертка API: роли, ретраи (429), бюджет
  tools.py                # детерминированные инструменты: profile, train_eval, submission
  registry.py             # реестр экспериментов (experiments.json)
  prompts/
    agent_instruction.md  # системный промпт (маркеры AGENT_PROMPT_START/END)
  sandbox/
    runner.py             # исполнение LLM-кода: subprocess + таймаут
tests/                    # smoke / deadline / fault / format — владелец капитан
```

## Быстрый старт

```bash
git clone <адрес> && cd alfa-ml-agent
python -m pip install -r requirements.txt
python -m pytest tests/ -v
python solution.py data/<задача>
```

## Локальный тестовый цикл

Проверка полного цикла на простой синтетической задаче `examples/simple_task`:

```bash
python -m pip install -r requirements-dev.txt

# заполнить .env: OpenAI-совместимый бэкенд
# (DeepSeek / Groq / Ollama / прокси организаторов)
cp .env.example .env

# сгенерировать тестовую задачу
python scripts/make_demo_task.py

# прогон без LLM: песочница, метрика и accept/reject работают по-настоящему
AGENT_DRY_RUN=1 python solution.py examples/simple_task/task.md

# реальный прогон (нужен .env)
python solution.py examples/simple_task/task.md
```

- `AGENT_DRY_RUN=1` — заготовки вместо вызова модели, код исполняется по-настоящему;
- бюджет: `AGENT_TIME_BUDGET` (3 ч по умолчанию), `AGENT_MAX_ITERS`, `AGENT_PATIENCE`;
- логи: `logs/llm.jsonl`, `logs/loop.jsonl`, `logs/exec.jsonl`;
- тесты: `python -m pytest tests/ -q`.

### Демо-цикл по шагам (`scripts/run_cycle.py`)

Облегчённая dev-обвязка без полного автомата фаз: hypothesis -> code ->
запуск в песочнице -> review -> accept/reject. Полезна, чтобы видеть каждый
шаг в терминале и проверить связку llm + prompts + sandbox.

```bash
# без LLM: заготовки вместо ответов модели, код исполняется по-настоящему
python scripts/run_cycle.py --task examples/simple_task --cycles 2 --dry-run

# реальный прогон (нужен .env)
python scripts/run_cycle.py --task examples/simple_task --cycles 2
```

- `--cycles N` — сколько гипотез проверить (по умолчанию 1);
- `--timeout S` — лимит на исполнение сгенерированного кода (по умолчанию 300 с);
- `--dry-run` — не звать модель, но прогнать всю механику.

## LLM-слой

Единственная точка вызова — `agent.llm.query()`. Роль (`role`) задаёт системный
промпт `agent/prompts/<role>.md`, схему ответа и таймаут; для роли `code` из
ответа вырезается markdown-блок. Отдельной обёртки `complete()` нет.

Ретраи: транзиентные ошибки — экспоненциальная пауза; 429 — отдельные
`LLM_RATE_LIMIT_RETRIES` / `LLM_RATE_LIMIT_WAIT` (по умолчанию 3 × 30 с).

`requirements-dev.txt` — только для локальной разработки; в ZIP для платформы
идёт `requirements-org.txt` (переименовывается в `requirements.txt`).
