"""Красивый вывод прогресса цикла в терминал.

Использует rich, если он установлен; иначе деградирует к простому тексту.
Это dev-обвязка (не production), поэтому живёт в scripts/.
"""
from __future__ import annotations

from typing import Iterable, Sequence

try:  # rich желателен, но не обязателен
    from rich.console import Console
    from rich.panel import Panel
    from rich.rule import Rule
    from rich.syntax import Syntax
    from rich.table import Table

    _RICH = True
except ImportError:  # pragma: no cover
    _RICH = False

_console = Console() if _RICH else None


def banner(title: str, subtitle: str = "") -> None:
    if _RICH:
        body = f"[bold]{title}[/]"
        if subtitle:
            body += f"\n[dim]{subtitle}[/]"
        _console.print(Panel(body, border_style="cyan", expand=True))
    else:
        print("=" * 78)
        print(title)
        if subtitle:
            print(subtitle)
        print("=" * 78)


def section(title: str) -> None:
    if _RICH:
        _console.print(Rule(f"[bold yellow]{title}[/]"))
    else:
        print(f"\n--- {title} ---")


def text_panel(label: str, value: str, *, max_chars: int = 2500) -> None:
    value = (value or "(пусто)").strip()
    if len(value) > max_chars:
        value = value[:max_chars] + "\n… (обрезано)"
    if _RICH:
        _console.print(Panel(value, title=f"[bold]{label}[/]", border_style="dim"))
    else:
        print(f"\n[{label}]\n{value}")


def code_panel(code: str, *, max_lines: int = 150) -> None:
    code = (code or "").strip() or "# (пусто)"
    lines = code.splitlines()
    if len(lines) > max_lines:
        code = "\n".join(lines[:max_lines]) + f"\n… (ещё {len(lines) - max_lines} строк)"
    if _RICH:
        _console.print(
            Panel(
                Syntax(code, "python", line_numbers=False, word_wrap=True),
                title="[bold]Сгенерированный код[/]",
                border_style="blue",
            )
        )
    else:
        print("\n[КОД]")
        print(code)


def exec_panel(result) -> None:
    ok = getattr(result, "ok", False)
    style = "green" if ok else "red"
    status = "OK" if ok else "FAIL"
    head = (
        f"[{style}]{status}[/]  exit_code={result.exit_code}  "
        f"duration={result.duration}s  timed_out={result.timed_out}"
    )
    if _RICH:
        _console.print(head)
        if result.stdout:
            _console.print(
                Panel(result.stdout.strip(), title="stdout", border_style="dim")
            )
        if result.stderr:
            _console.print(
                Panel(result.stderr.strip(), title="stderr", border_style="red")
            )
    else:
        print(f"\n[ЗАПУСК] {status} exit={result.exit_code} dur={result.duration}s "
              f"timeout={result.timed_out}")
        if result.stdout:
            print("stdout:\n" + result.stdout)
        if result.stderr:
            print("stderr:\n" + result.stderr)


def metric_panel(metric: dict | None) -> None:
    if metric is None:
        msg = "[yellow]Метрика не найдена в выводе[/]"
        (_console.print(msg) if _RICH else print("Метрика не найдена в выводе"))
        return
    text = (
        f"{metric.get('metric', '?')} = "
        f"[bold green]{metric.get('value')}[/] "
        f"({metric.get('direction', '?')})"
    )
    if _RICH:
        _console.print(Panel(text, title="Метрика", border_style="green"))
    else:
        print(f"[МЕТРИКА] {metric.get('metric')} = {metric.get('value')} "
              f"({metric.get('direction')})")


def decision_panel(accepted: bool, reason: str = "") -> None:
    if _RICH:
        if accepted:
            _console.print(f"[bold green]✔ ГИПОТЕЗА ПРИНЯТА[/] {reason}")
        else:
            _console.print(f"[bold red]✘ ГИПОТЕЗА ОТКЛОНЕНА[/] {reason}")
    else:
        print(("ПРИНЯТА" if accepted else "ОТКЛОНЕНА") + f" {reason}")


def summary(rows: Sequence[Sequence], headers: Sequence[str]) -> None:
    if _RICH:
        table = Table(title="Итог цикла", show_lines=False)
        for header in headers:
            table.add_column(header, overflow="fold")
        for row in rows:
            table.add_row(*[str(cell) for cell in row])
        _console.print(table)
    else:
        print("\n[ИТОГ]")
        print(" | ".join(headers))
        for row in rows:
            print(" | ".join(str(cell) for cell in row))


def info(message: str) -> None:
    (_console.print(message) if _RICH else print(message))
