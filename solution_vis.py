"""Визуализация прогона solution.py: описание запуска и дерево решений.

Запускает **тот же** solution.py отдельным процессом (как платформа) и
параллельно читает журнал агента, собирая из событий дерево узлов: кто
родитель, какое семейство, какой CV, улучшила ли правка ветку. Ничего в
агента не подмешивает — что нарисовано, то и было.

Экран состоит из двух частей:
- описание запуска: задача, тип, метрика, фаза, сколько узлов, лучший узел,
  ансамбль и откуда взят сабмит;
- дерево: корни портфеля и их потомки с вердиктами.

Запуск:
    python solution_vis.py examples/simple_task/task.md
    python solution_vis.py examples/simple_task/task.md --plain   # дерево текстом
    python solution_vis.py examples/simple_task --interval 0.5

Код возврата — код возврата solution.py.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SOLUTION = ROOT / "solution.py"

try:  # rich желателен, но не обязателен
    from rich.console import Console, Group
    from rich.live import Live
    from rich.panel import Panel
    from rich.text import Text
    from rich.tree import Tree as RichTree

    _RICH = True
except ImportError:  # pragma: no cover
    _RICH = False

LEGEND = "▲ улучшила ветку · ▽ хуже родителя · ✖ ошибка · ★ лучший узел"


# ----------------------------- состояние ----------------------------------
class Node:
    """Узел дерева в том виде, в каком он нужен для отрисовки."""

    def __init__(self, node_id: int, parent_id: int | None, family: str) -> None:
        self.node_id = node_id
        self.parent_id = parent_id
        self.family = family
        self.cv_score: float | None = None
        self.status = "pending"
        self.hypothesis = ""
        self.verdict = ""      # improved | rejected
        self.depth = 0


class State:
    """Всё, что показывает экран."""

    def __init__(self, task_md: Path) -> None:
        self.task_md = task_md
        self.phase = "запуск"
        self.remaining: float | None = None
        self.task_type: str | None = None
        self.metric: str | None = None
        self.nodes: dict[int, Node] = {}
        self.ensemble_cv: float | None = None
        self.ensemble_members: list[int] = []
        self.submission_from: str | None = None
        self.submission_cv: float | None = None
        self.finished = False

    @property
    def title(self) -> str:
        """Короткое имя задачи для шапки."""
        return self.task_md.name if self.task_md.is_file() else str(self.task_md)

    @property
    def direction(self) -> str:
        """Направление метрики: min для RMSE, max для F1."""
        return "max" if self.metric == "macro_f1" else "min"

    def best_node(self) -> Node | None:
        """Лучший узел по текущей метрике."""
        scored = [n for n in self.nodes.values() if n.cv_score is not None]
        if not scored:
            return None
        reverse = self.direction == "max"
        return sorted(scored, key=lambda n: n.cv_score, reverse=reverse)[0]


def _fmt(value) -> str:
    """Число для печати: 5 значащих цифр или '—'."""
    if value is None:
        return "—"
    if isinstance(value, (int, float)):
        return f"{value:.5g}"
    return str(value)


# ----------------------------- журнал -> состояние ------------------------
def apply_event(state: State, event: dict) -> None:
    """
    Обновляет состояние по событию журнала агента.

    Аргументы:
    state (State): состояние экрана
    event (dict): одна строка loop.jsonl
    """
    kind = event.get("kind")
    if kind == "phase":
        state.phase = str(event.get("phase"))
        remaining = event.get("remaining")
        if isinstance(remaining, (int, float)):
            state.remaining = float(remaining)
    elif kind == "scout":
        state.task_type = event.get("task_type")
        state.metric = event.get("metric")
    elif kind == "node":
        node_id = int(event.get("node_id") or 0)
        node = state.nodes.get(node_id) or Node(node_id, event.get("parent_id"),
                                                str(event.get("family") or "?"))
        node.parent_id = event.get("parent_id")
        node.family = str(event.get("family") or node.family)
        node.cv_score = event.get("cv_score")
        node.status = str(event.get("status") or "pending")
        node.hypothesis = str(event.get("hypothesis") or "")
        node.depth = int(event.get("depth") or 0)
        state.nodes[node_id] = node
    elif kind == "branch_improved":
        node = state.nodes.get(int(event.get("node_id") or 0))
        if node:
            node.verdict = "improved"
    elif kind == "branch_rejected":
        node = state.nodes.get(int(event.get("node_id") or 0))
        if node:
            node.verdict = "rejected"
    elif kind == "ensemble":
        state.ensemble_cv = event.get("score")
        state.ensemble_members = list(event.get("members") or [])
    elif kind == "submission":
        state.submission_from = event.get("source")
        state.submission_cv = event.get("cv_score")
    elif kind == "done":
        state.finished = True


def _children(state: State, parent_id: int | None) -> list[Node]:
    """Дети узла (для корней — parent_id None), по возрастанию номера."""
    kids = [n for n in state.nodes.values() if n.parent_id == parent_id]
    return sorted(kids, key=lambda n: n.node_id)


def _roots(state: State) -> list[Node]:
    """Корни: узлы без родителя или с неизвестным родителем."""
    known = set(state.nodes)
    return sorted([n for n in state.nodes.values()
                   if n.parent_id is None or n.parent_id not in known],
                  key=lambda n: n.node_id)


def _label(state: State, node: Node) -> str:
    """Текст подписи узла для дерева."""
    best = state.best_node()
    mark = " ★" if best is not None and best.node_id == node.node_id else ""
    if node.status == "error":
        symbol = " ✖"
    elif node.verdict == "improved":
        symbol = " ▲"
    elif node.verdict == "rejected":
        symbol = " ▽"
    else:
        symbol = ""
    idea = f"  «{node.hypothesis[:44]}»" if node.hypothesis else ""
    return f"#{node.node_id} {node.family}  cv={_fmt(node.cv_score)}{symbol}{mark}{idea}"


def _node_style(node: Node, best_id: int | None) -> str:
    """Цвет подписи узла."""
    if node.status == "error":
        return "red"
    if best_id is not None and node.node_id == best_id:
        return "bold green"
    if node.verdict == "improved":
        return "green"
    if node.verdict == "rejected":
        return "dim"
    return ""


# ----------------------------- отрисовка ----------------------------------
def _description_lines(state: State) -> list[tuple[str, str]]:
    """Строки описания запуска: (текст, стиль rich)."""
    best = state.best_node()
    lines = [
        (f"{state.title}  ·  {state.task_type or '?'}  ·  {state.metric or '?'}", "bold"),
        (f"фаза: {state.phase}", "yellow" if not state.finished else "green"),
    ]
    if state.remaining is not None and not state.finished:
        lines.append((f"осталось: {int(state.remaining)} с", "dim"))
    lines.append((f"узлов: {len(state.nodes)}", "dim"))
    lines.append(("лучший узел: " + (f"#{best.node_id} {best.family} cv={_fmt(best.cv_score)}"
                                    if best else "—"), "green"))
    lines.append(("ансамбль: " + (f"{_fmt(state.ensemble_cv)} из {state.ensemble_members}"
                                  if state.ensemble_cv is not None else "ещё не собран"), "cyan"))
    if state.submission_from:
        lines.append((f"сабмит: {state.submission_from} (cv={_fmt(state.submission_cv)})",
                      "bold green"))
    lines.append((LEGEND, "dim"))
    return lines


def render_description(state: State) -> Panel:
    """Панель описания запуска."""
    body = Text()
    for index, (line, style) in enumerate(_description_lines(state)):
        if index:
            body.append("\n")
        body.append(line, style=style)
    return Panel(body, title="описание", border_style="cyan", padding=(0, 1))


def render_tree(state: State) -> RichTree:
    """Дерево узлов (rich.Tree)."""
    best = state.best_node()
    best_id = best.node_id if best else None

    def build(node: Node, tree: RichTree) -> None:
        for child in _children(state, node.node_id):
            branch = tree.add(Text(_label(state, child), style=_node_style(child, best_id)))
            build(child, branch)

    root = RichTree(Text("портфель", style="bold"), guide_style="dim")
    roots = _roots(state)
    if not roots:
        root.add(Text("ждём первые узлы…", style="dim"))
    for node in roots:
        branch = root.add(Text(_label(state, node), style=_node_style(node, best_id)))
        build(node, branch)
    return root


def render_live(state: State):
    """Кадр для rich.Live: описание + дерево."""
    return Group(render_description(state), render_tree(state))


def print_plain(state: State) -> None:
    """Печатает описание и дерево текстом (режим --plain)."""
    print("\n".join(line for line, _ in _description_lines(state)))
    print("\nдерево:")

    def walk(nodes: list[Node], prefix: str = "") -> None:
        for index, node in enumerate(nodes):
            last = index == len(nodes) - 1
            print(f"{prefix}{'└─ ' if last else '├─ '}{_label(state, node)}")
            walk(_children(state, node.node_id), prefix + ("   " if last else "│  "))

    walk(_roots(state))


# ----------------------------- чтение журнала -----------------------------
def _follow(path: Path, start_offset: int, queue: "deque[dict]", stop: threading.Event) -> None:
    """
    Читает loop.jsonl с указанной позиции по мере появления строк.

    Аргументы:
    path (Path): журнал агента
    start_offset (int): позиция старта (размер файла на момент запуска)
    queue (deque): очередь событий
    stop (threading.Event): сигнал остановки
    """
    deadline = time.time() + 30
    while not path.exists():
        if stop.is_set() or time.time() > deadline:
            return
        time.sleep(0.1)
    offset = start_offset
    buffer = ""
    while not stop.is_set():
        try:
            with path.open("r", encoding="utf-8", errors="replace") as handle:
                handle.seek(offset)
                chunk = handle.read()
                offset = handle.tell()
        except OSError:
            time.sleep(0.2)
            continue
        if not chunk:
            time.sleep(0.1)
            continue
        buffer += chunk
        *lines, buffer = buffer.split("\n")
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                queue.append(json.loads(line))
            except json.JSONDecodeError:
                continue


def task_path(argv: list[str]) -> Path:
    """
    Повторяет правило solution.py: argv[1] — task.md или каталог с ним.

    Аргументы:
    argv (list[str]): sys.argv

    Возвращает:
    Path
    """
    raw = Path(argv[1] if len(argv) > 1 else "task.md").resolve()
    return raw / "task.md" if raw.is_dir() else raw


def main(argv: list[str]) -> int:
    """
    Запускает solution.py и рисует дерево его решений.

    Аргументы:
    argv (list[str]): sys.argv

    Возвращает:
    int: код возврата solution.py
    """
    plain = "--plain" in argv
    interval = 0.3
    passthrough: list[str] = []
    skip_next = False
    for position, arg in enumerate(argv[1:], start=1):
        if skip_next:
            skip_next = False
            continue
        if arg == "--plain":
            continue
        if arg == "--interval":
            try:
                interval = float(argv[position + 1])
                skip_next = True
            except (IndexError, ValueError):
                print("--interval ждёт число", file=sys.stderr)
                return 2
            continue
        passthrough.append(arg)

    task_md = task_path([argv[0], *passthrough])
    if not task_md.exists():
        print(f"Не найден task.md: {task_md}", file=sys.stderr)
        return 2

    log_path = task_md.parent / "logs" / "loop.jsonl"
    start_offset = log_path.stat().st_size if log_path.exists() else 0

    state = State(task_md)
    queue: deque = deque()
    stop = threading.Event()
    reader = threading.Thread(target=_follow, args=(log_path, start_offset, queue, stop),
                              daemon=True)
    reader.start()

    proc = subprocess.Popen([sys.executable, "-u", str(SOLUTION), str(task_md)],
                            cwd=str(task_md.parent), env=dict(os.environ),
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    console = Console() if _RICH else None
    try:
        if plain or not _RICH:
            while proc.poll() is None or queue:
                while queue:
                    apply_event(state, queue.popleft())
                time.sleep(interval)
        else:
            with Live(render_live(state), console=console, refresh_per_second=int(1 / interval),
                      screen=False) as live:
                while proc.poll() is None or queue:
                    updated = False
                    while queue:
                        apply_event(state, queue.popleft())
                        updated = True
                    if updated:
                        live.update(render_live(state))
                    time.sleep(interval)
    except KeyboardInterrupt:  # pragma: no cover — ручная остановка
        stop.set()
        proc.terminate()
        print("\nпрервано пользователем", file=sys.stderr)
        return 130
    finally:
        stop.set()

    stdout, stderr = proc.communicate()
    stop.set()
    while queue:
        apply_event(state, queue.popleft())

    if plain or not _RICH:
        print_plain(state)
    if stdout.strip():
        print(stdout.strip())
    if stderr.strip():
        print(stderr.strip(), file=sys.stderr)
    return proc.returncode


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
