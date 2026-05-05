"""Compact terminal logging helpers for PPO probe runs."""

from __future__ import annotations

import sys


ANSI_RESET = "\033[0m"
ANSI_GREEN = "\033[32m"
ANSI_YELLOW = "\033[33m"
ANSI_RED = "\033[31m"
ANSI_CYAN = "\033[36m"


def _use_color() -> bool:
    return bool(getattr(sys.stdout, "isatty", lambda: False)())


def _paint(text: str, color: str) -> str:
    if not _use_color():
        return text
    return f"{color}{text}{ANSI_RESET}"


def color_message_mode(mode: str) -> str:
    mode_text = str(mode).upper()
    if mode == "on":
        return _paint(mode_text, ANSI_GREEN)
    if mode == "diag":
        return _paint(mode_text, ANSI_YELLOW)
    return _paint(mode_text, ANSI_RED)


def print_plain_episode_status(
    *,
    run_index: int,
    total_runs: int,
    seed: int,
    variant_label: str,
    episode: int,
    episode_return: float,
    avg10: float,
    best_return: float,
    total_env_steps: int,
    solved_episode: int | None,
    peer_status: str,
    new_best: bool,
) -> None:
    prefix = _paint(f"[{run_index}/{total_runs} {variant_label} s{seed}]", ANSI_CYAN)
    solve_text = "solved" if solved_episode is not None else "running"
    print(
        f"{prefix} ep {episode:04d} | ret {episode_return:6.1f} | avg10 {avg10:6.1f} | "
        f"best {best_return:6.1f} | steps {total_env_steps:6d} | {solve_text} | {peer_status}"
    )
    if new_best:
        print(f"  best -> {best_return:6.1f}")


def print_probe_episode_status(
    *,
    run_index: int,
    total_runs: int,
    seed: int,
    variant_label: str,
    episode: int,
    episode_return: float,
    avg10: float,
    best_return: float,
    total_env_steps: int,
    probe_count: int,
    episode_probe_steps: int,
    message_mode: str,
    message_detail: str,
    solved_episode: int | None,
    peer_status: str,
    new_best: bool,
    show_detail_line: bool,
) -> None:
    prefix = _paint(f"[{run_index}/{total_runs} {variant_label} s{seed}]", ANSI_CYAN)
    solve_text = "solved" if solved_episode is not None else "running"
    print(
        f"{prefix} ep {episode:04d} | ret {episode_return:6.1f} | avg10 {avg10:6.1f} | "
        f"best {best_return:6.1f} | steps {total_env_steps:6d} | probe {probe_count:2d}/{episode_probe_steps:4d} | "
        f"msg {color_message_mode(message_mode)} | {solve_text}"
    )
    if show_detail_line:
        print(f"  note -> {message_detail} | {peer_status}")
    elif new_best:
        print(f"  best -> {best_return:6.1f} | {peer_status}")


def print_belief_episode_status(
    *,
    run_index: int,
    total_runs: int,
    seed: int,
    variant_label: str,
    episode: int,
    episode_return: float,
    avg10: float,
    best_return: float,
    total_env_steps: int,
    probe_count: int,
    episode_probe_steps: int,
    trust: float,
    usage: float,
    usage_label: str,
    refresh_count: int,
    avg50: float,
    target_return: float,
    solved_episode: int | None,
    peer_status: str,
    new_best: bool,
) -> None:
    prefix = _paint(f"[{run_index}/{total_runs} {variant_label} s{seed}]", ANSI_CYAN)
    solve_text = "solved" if solved_episode is not None else "running"
    print(
        f"{prefix} ep {episode:04d} | ret {episode_return:6.1f} | avg10 {avg10:6.1f} | "
        f"best {best_return:6.1f} | steps {total_env_steps:6d} | "
        f"probe {probe_count:2d}/{episode_probe_steps:4d} | trust {trust:5.3f} | "
        f"{usage_label} {usage:5.3f} | refresh {refresh_count:3d} | {solve_text}"
    )
    if new_best:
        print(
            f"  best -> {best_return:6.1f} | avg50 {avg50:6.1f} | "
            f"target {target_return:6.1f} | {peer_status}"
        )


def print_probe_failure(
    *,
    episode: int,
    probe_steps: int,
) -> None:
    print(
        f"{_paint('[probe]', ANSI_RED)} ep {episode:04d} | probe collection failed | steps {probe_steps:4d}"
    )


def print_solve_event(
    *,
    run_index: int,
    total_runs: int,
    seed: int,
    variant_label: str,
    episode: int,
    total_env_steps: int,
    episode_return: float,
    probe_count: int | None = None,
) -> None:
    prefix = _paint(f"[{run_index}/{total_runs} {variant_label} s{seed}]", ANSI_GREEN)
    probe_suffix = "" if probe_count is None else f" | probes {probe_count}"
    print(
        f"{prefix} solved at ep {episode:04d} | steps {total_env_steps} | ret {episode_return:.2f}{probe_suffix}"
    )
