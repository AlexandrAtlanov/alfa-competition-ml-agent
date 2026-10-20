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
  llm.py                  # единственная обертка API Qwen: ретраи, бюджет, кэш
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
