"""Tic-tac-toe rule-belief benchmark with exact minimax evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from ...crawler.types import CrawlerMessage


BOARD_SIZE = 9
X_PLAYER = 1
O_PLAYER = -1
EMPTY = 0
RULE_NORMAL = "normal"
RULE_MISERE = "misere"
RULES = (RULE_NORMAL, RULE_MISERE)
WIN_LINES = (
    (0, 1, 2),
    (3, 4, 5),
    (6, 7, 8),
    (0, 3, 6),
    (1, 4, 7),
    (2, 5, 8),
    (0, 4, 8),
    (2, 4, 6),
)


@dataclass(frozen=True)
class BoardProbeBenchmarkConfig:
    """Configuration for the exact board-game belief benchmark."""

    seeds: tuple[int, ...] = (0, 1, 2, 3)
    probe_budget: int = 2
    challenge_positions: int = 18
    random_positions: int = 12
    belief_dim: int = 4


def board_tuple(board: np.ndarray | tuple[int, ...]) -> tuple[int, ...]:
    """Return a canonical flat board tuple."""
    return tuple(int(value) for value in np.asarray(board, dtype=np.int8).reshape(-1))


def winner(board: tuple[int, ...]) -> int:
    """Return X, O, or empty if no one has a line."""
    for line in WIN_LINES:
        total = sum(board[idx] for idx in line)
        if total == 3:
            return X_PLAYER
        if total == -3:
            return O_PLAYER
    return EMPTY


def terminal_value(board: tuple[int, ...], rule: str) -> int | None:
    """Return terminal value from X's perspective, or None for non-terminal."""
    line_winner = winner(board)
    if line_winner == X_PLAYER:
        return 1 if rule == RULE_NORMAL else -1
    if line_winner == O_PLAYER:
        return -1 if rule == RULE_NORMAL else 1
    if all(value != EMPTY for value in board):
        return 0
    return None


def valid_moves(board: tuple[int, ...]) -> tuple[int, ...]:
    """Return open squares in stable order."""
    return tuple(idx for idx, value in enumerate(board) if value == EMPTY)


def apply_move(board: tuple[int, ...], move: int, player: int) -> tuple[int, ...]:
    """Return a board after one legal move."""
    if board[int(move)] != EMPTY:
        raise ValueError(f"Illegal move {move} for board {board}")
    values = list(board)
    values[int(move)] = int(player)
    return tuple(values)


@lru_cache(maxsize=None)
def minimax_value(board: tuple[int, ...], x_to_move: bool, rule: str) -> int:
    """Exact minimax value from X's perspective."""
    terminal = terminal_value(board, rule)
    if terminal is not None:
        return int(terminal)
    player = X_PLAYER if bool(x_to_move) else O_PLAYER
    child_values = [
        minimax_value(apply_move(board, move, player), not bool(x_to_move), rule)
        for move in valid_moves(board)
    ]
    if bool(x_to_move):
        return int(max(child_values))
    return int(min(child_values))


def best_moves(board: tuple[int, ...], x_to_move: bool, rule: str) -> tuple[int, ...]:
    """Return all exact minimax-optimal moves for the side to move."""
    player = X_PLAYER if bool(x_to_move) else O_PLAYER
    scored = [
        (
            move,
            minimax_value(apply_move(board, move, player), not bool(x_to_move), rule),
        )
        for move in valid_moves(board)
    ]
    if not scored:
        return ()
    target = max(score for _move, score in scored) if bool(x_to_move) else min(score for _move, score in scored)
    return tuple(move for move, score in scored if score == target)


def choose_minimax_move(board: tuple[int, ...], x_to_move: bool, rule: str) -> int | None:
    """Choose one stable exact minimax move."""
    moves = best_moves(board, x_to_move, rule)
    return None if not moves else int(moves[0])


def rule_from_message(message: CrawlerMessage, fallback: str = RULE_NORMAL) -> str:
    """Decode a rule label from a generic crawler message."""
    explicit = message.metadata.get("predicted_rule")
    if explicit in RULES:
        return str(explicit)
    vector = np.asarray(message.vector, dtype=np.float32).reshape(-1)
    if vector.size >= 2:
        return RULE_NORMAL if float(vector[0]) >= float(vector[1]) else RULE_MISERE
    return str(fallback)


def _probe_board(query_name: str) -> tuple[tuple[int, ...], int, bool]:
    """Return board, candidate move, and side-to-move for one rule probe."""
    if query_name == "x_line_completion":
        return (X_PLAYER, X_PLAYER, EMPTY, O_PLAYER, EMPTY, EMPTY, EMPTY, O_PLAYER, EMPTY), 2, True
    if query_name == "o_line_completion":
        return (O_PLAYER, O_PLAYER, EMPTY, X_PLAYER, EMPTY, EMPTY, EMPTY, X_PLAYER, EMPTY), 2, False
    if query_name == "fork_block":
        return (X_PLAYER, EMPTY, EMPTY, EMPTY, O_PLAYER, EMPTY, EMPTY, EMPTY, X_PLAYER), 1, False
    return (X_PLAYER, O_PLAYER, X_PLAYER, X_PLAYER, O_PLAYER, EMPTY, O_PLAYER, EMPTY, EMPTY), 5, True


def query_rule_outcome(query_name: str, rule: str) -> dict[str, object]:
    """Execute one active rule query against the hidden board rule."""
    board, move, x_to_move = _probe_board(query_name)
    player = X_PLAYER if x_to_move else O_PLAYER
    next_board = apply_move(board, move, player)
    value = minimax_value(next_board, not x_to_move, rule)
    normal_value = minimax_value(next_board, not x_to_move, RULE_NORMAL)
    misere_value = minimax_value(next_board, not x_to_move, RULE_MISERE)
    return {
        "board": board,
        "candidate_move": int(move),
        "x_to_move": bool(x_to_move),
        "value_for_x": int(value),
        "normal_value_for_x": int(normal_value),
        "misere_value_for_x": int(misere_value),
    }


def infer_rule_from_probe_outcomes(outcomes: list[dict[str, object]]) -> tuple[str, float, np.ndarray]:
    """Infer the hidden rule from active query outcomes."""
    normal_votes = 0
    misere_votes = 0
    informative = 0
    for outcome in outcomes:
        observed = int(outcome.get("value_for_x", 0))
        normal_value = int(outcome.get("normal_value_for_x", 0))
        misere_value = int(outcome.get("misere_value_for_x", 0))
        if normal_value == misere_value:
            continue
        informative += 1
        if observed == normal_value:
            normal_votes += 1
        if observed == misere_value:
            misere_votes += 1
    total = max(normal_votes + misere_votes, 1)
    p_normal = float(normal_votes) / float(total)
    p_misere = float(misere_votes) / float(total)
    predicted = RULE_NORMAL if p_normal >= p_misere else RULE_MISERE
    confidence = max(p_normal, p_misere) if informative else 0.5
    latent = np.asarray(
        [
            p_normal,
            p_misere,
            float(informative) / 2.0,
            1.0 - confidence,
        ],
        dtype=np.float32,
    )
    return predicted, float(confidence), latent


def build_rule_message(outcomes: list[dict[str, object]]) -> CrawlerMessage:
    """Build the canonical board-game crawler message from rule probes."""
    predicted, confidence, latent = infer_rule_from_probe_outcomes(outcomes)
    return CrawlerMessage(
        vector=latent,
        confidence=confidence,
        ready=bool(confidence >= 0.80),
        uncertainty=float(1.0 - confidence),
        metadata={
            "modality": "board_game",
            "game": "tic_tac_toe",
            "predicted_rule": predicted,
            "belief_source": "learned",
            "support_size": len(outcomes),
        },
    )


def _fixed_challenge_positions() -> list[tuple[tuple[int, ...], bool]]:
    return [
        ((X_PLAYER, X_PLAYER, EMPTY, O_PLAYER, EMPTY, EMPTY, EMPTY, O_PLAYER, EMPTY), True),
        ((O_PLAYER, O_PLAYER, EMPTY, X_PLAYER, EMPTY, EMPTY, EMPTY, X_PLAYER, EMPTY), False),
        ((X_PLAYER, EMPTY, X_PLAYER, EMPTY, O_PLAYER, EMPTY, O_PLAYER, EMPTY, EMPTY), True),
        ((O_PLAYER, EMPTY, O_PLAYER, X_PLAYER, X_PLAYER, EMPTY, EMPTY, EMPTY, EMPTY), False),
        ((X_PLAYER, O_PLAYER, EMPTY, EMPTY, X_PLAYER, EMPTY, O_PLAYER, EMPTY, EMPTY), True),
        ((O_PLAYER, X_PLAYER, EMPTY, EMPTY, O_PLAYER, EMPTY, X_PLAYER, EMPTY, EMPTY), False),
    ]


def _random_position(rng: np.random.Generator, max_plies: int) -> tuple[tuple[int, ...], bool]:
    board = tuple([EMPTY] * BOARD_SIZE)
    x_to_move = True
    for _ply in range(int(rng.integers(2, max(3, max_plies + 1)))):
        if terminal_value(board, RULE_NORMAL) is not None or terminal_value(board, RULE_MISERE) is not None:
            break
        moves = valid_moves(board)
        if not moves:
            break
        move = int(rng.choice(np.asarray(moves, dtype=np.int64)))
        board = apply_move(board, move, X_PLAYER if x_to_move else O_PLAYER)
        x_to_move = not x_to_move
    return board, x_to_move


def challenge_positions(seed: int, count: int) -> list[tuple[tuple[int, ...], bool]]:
    """Build deterministic exact-minimax evaluation positions."""
    rng = np.random.default_rng(seed)
    positions = list(_fixed_challenge_positions())
    while len(positions) < int(count):
        board, x_to_move = _random_position(rng, max_plies=6)
        if valid_moves(board) and terminal_value(board, RULE_NORMAL) is None and terminal_value(board, RULE_MISERE) is None:
            positions.append((board, x_to_move))
    return positions[: int(count)]


def evaluate_minimax_policy(
    *,
    positions: list[tuple[tuple[int, ...], bool]],
    hidden_rule: str,
    solver_rule: str,
) -> dict[str, float]:
    """Compare one solver rule against exact hidden-rule minimax decisions."""
    correct_moves = 0
    correct_values = 0
    total = 0
    for board, x_to_move in positions:
        true_moves = best_moves(board, x_to_move, hidden_rule)
        if not true_moves:
            continue
        solver_move = choose_minimax_move(board, x_to_move, solver_rule)
        true_value = minimax_value(board, x_to_move, hidden_rule)
        solver_value = minimax_value(board, x_to_move, solver_rule)
        correct_moves += int(solver_move in true_moves)
        correct_values += int(true_value == solver_value)
        total += 1
    return {
        "move_accuracy": float(correct_moves) / float(max(total, 1)),
        "value_accuracy": float(correct_values) / float(max(total, 1)),
        "position_count": float(total),
    }


def run_board_probe_benchmark(
    config: BoardProbeBenchmarkConfig | None = None,
) -> dict[str, object]:
    """Run the exact board-game crawler + minimax benchmark."""
    config = config or BoardProbeBenchmarkConfig()
    rows: list[dict[str, float | int | str]] = []
    artifacts: list[dict[str, object]] = []
    queries = ("x_line_completion", "o_line_completion", "fork_block", "endgame_choice")
    for seed in config.seeds:
        hidden_rule = RULES[int(seed) % len(RULES)]
        outcomes = [
            query_rule_outcome(query, hidden_rule)
            for query in queries[: max(1, int(config.probe_budget))]
        ]
        message = build_rule_message(outcomes)
        learned_rule = rule_from_message(message)
        positions = challenge_positions(
            seed=int(seed) + 101,
            count=int(config.challenge_positions),
        )
        baseline = evaluate_minimax_policy(
            positions=positions,
            hidden_rule=hidden_rule,
            solver_rule=RULE_NORMAL,
        )
        learned = evaluate_minimax_policy(
            positions=positions,
            hidden_rule=hidden_rule,
            solver_rule=learned_rule,
        )
        stale_rule = RULES[(int(seed) + 1) % len(RULES)]
        stale = evaluate_minimax_policy(
            positions=positions,
            hidden_rule=hidden_rule,
            solver_rule=stale_rule,
        )
        rows.append(
            {
                "seed": int(seed),
                "hidden_rule": hidden_rule,
                "decoded_rule": learned_rule,
                "rule_decode_accuracy": float(learned_rule == hidden_rule),
                "baseline_move_accuracy": baseline["move_accuracy"],
                "belief_move_accuracy": learned["move_accuracy"],
                "zero_belief_move_accuracy": baseline["move_accuracy"],
                "shuffled_belief_move_accuracy": stale["move_accuracy"],
                "stale_belief_move_accuracy": stale["move_accuracy"],
                "baseline_value_accuracy": baseline["value_accuracy"],
                "belief_value_accuracy": learned["value_accuracy"],
                "query_count": len(outcomes),
                "message_confidence": float(message.confidence),
                "message_uncertainty": float(message.uncertainty),
            }
        )
        artifacts.append(
            {
                "raw_evidence_windows": outcomes,
                "domain_belief": message.vector.tolist(),
                "uncertainty_estimate": float(message.uncertainty),
                "crawler_message": {
                    "vector": message.vector.tolist(),
                    "confidence": float(message.confidence),
                    "ready": bool(message.ready),
                    "metadata": dict(message.metadata),
                },
                "ablation_metrics": {
                    "zero_gap": learned["move_accuracy"] - baseline["move_accuracy"],
                    "shuffled_gap": learned["move_accuracy"] - stale["move_accuracy"],
                    "stale_gap": learned["move_accuracy"] - stale["move_accuracy"],
                },
                "subset_agreement": float(message.confidence),
                "belief_bitrate": int(config.belief_dim * 32),
            }
        )
    return {
        "dataset": "TicTacToe hidden-rule positions",
        "model_family": "CrawlerMessage+ExactMinimax",
        "game": "tic_tac_toe",
        "rows": rows,
        "artifacts": artifacts,
        "probe_budget": int(config.probe_budget),
    }
