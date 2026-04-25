"""Compact probe-health status helpers for benchmark summaries."""

from __future__ import annotations

import math
import sys


ANSI_RESET = "\033[0m"
ANSI_GREEN = "\033[32m"
ANSI_YELLOW = "\033[33m"
ANSI_RED = "\033[31m"


def _use_color() -> bool:
    return bool(getattr(sys.stdout, "isatty", lambda: False)())


def _paint(text: str, color: str) -> str:
    if not _use_color():
        return text
    return f"{color}{text}{ANSI_RESET}"


def _rank(value: int | None) -> float:
    return float("inf") if value is None else float(value)


def signal_status(
    *,
    message_on_fraction: float | None,
    message_diag_fraction: float | None,
    probe_expr_delta: float | None,
) -> str:
    if float(message_on_fraction or 0.0) > 0.0 and float(probe_expr_delta or 0.0) > 0.0:
        return "live"
    if float(message_diag_fraction or 0.0) > 0.0 or float(message_on_fraction or 0.0) > 0.0:
        return "seen"
    return "off"


def geometry_status(*, split_mrr: float | None, neighbor_alignment: float | None) -> str:
    split_mrr = float(split_mrr or 0.0)
    neighbor_alignment = float(neighbor_alignment or 0.0)
    if split_mrr >= 0.15 and neighbor_alignment >= 0.10:
        return "good"
    if split_mrr >= 0.10 and neighbor_alignment >= 0.08:
        return "improving"
    return "weak"


def utility_status(
    *,
    baseline_episode: int | None,
    probe_episode: int | None,
    no_message_episode: int | None,
) -> str:
    probe_rank = _rank(probe_episode)
    baseline_rank = _rank(baseline_episode)
    no_message_rank = _rank(no_message_episode)
    if math.isinf(probe_rank):
        return "worse"
    if probe_rank < baseline_rank and probe_rank < no_message_rank:
        return "beats_both"
    if probe_rank < baseline_rank:
        return "beats_baseline"
    return "worse"


def format_status(label: str, value: str) -> str:
    color = ANSI_RED
    if value in {"good", "live", "beats_both"}:
        color = ANSI_GREEN
    elif value in {"improving", "seen", "beats_baseline"}:
        color = ANSI_YELLOW
    return f"{label}={_paint(value, color)}"
