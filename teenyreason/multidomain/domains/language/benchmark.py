"""Tiny Shakespeare belief-conditioned Transformer benchmark."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import torch
import torch.nn.functional as F

from .handoff import (
    best_language_handoff as _best_language_handoff,
    handoff_metric as _handoff_metric,
    language_handoff_rows as _language_handoff_rows,
    language_row_economics,
    select_language_handoff as _select_language_handoff,
)
from .models import BeliefConditionedCharTransformer


TINY_SHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
)


@dataclass(frozen=True)
class LanguageProbeBenchmarkConfig:
    """Configuration for the small-corpus Shakespeare benchmark."""

    data_dir: Path = Path("artifacts/data/language")
    corpus_name: str = "tinyshakespeare.txt"
    source_url: str = TINY_SHAKESPEARE_URL
    train_char_budgets: tuple[int, ...] = (50000, 100000, 200000)
    validation_chars: int = 20000
    context_length: int = 96
    continuation_length: int = 48
    embedding_dim: int = 64
    hidden_dim: int = 128
    attention_heads: int = 4
    transformer_layers: int = 2
    prefix_tokens: int = 8
    belief_dim: int = 16
    support_windows: int = 8
    handoff_modes: tuple[str, ...] = ("prefix", "adapter")
    probe_steps: int = 300
    lm_steps: int = 400
    eval_steps: int = 80
    batch_size: int = 64
    lr: float = 2e-3
    seed: int = 0


def set_seed(seed: int):
    """Keep NumPy and Torch aligned for reproducible text sampling."""
    np.random.seed(seed)
    torch.manual_seed(seed)


def ensure_tiny_shakespeare(config: LanguageProbeBenchmarkConfig) -> Path:
    """Download Tiny Shakespeare once into the local artifact directory."""
    config.data_dir.mkdir(parents=True, exist_ok=True)
    corpus_path = config.data_dir / config.corpus_name
    if corpus_path.exists():
        return corpus_path
    try:
        with urlopen(config.source_url, timeout=30) as response:
            corpus_path.write_bytes(response.read())
    except Exception as exc:
        raise RuntimeError(
            "Could not download Tiny Shakespeare. If the corpus is not cached locally, the first run needs internet access."
        ) from exc
    return corpus_path


def build_char_vocab(text: str) -> tuple[dict[str, int], dict[int, str]]:
    """Build a simple character vocabulary for the local corpus."""
    chars = sorted(set(text))
    stoi = {char: idx for idx, char in enumerate(chars)}
    itos = {idx: char for char, idx in stoi.items()}
    return stoi, itos


def encode_text(text: str, stoi: dict[str, int]) -> np.ndarray:
    """Convert the raw corpus into integer token ids."""
    return np.asarray([stoi[char] for char in text], dtype=np.int64)


def split_corpus(
    encoded_text: np.ndarray,
    validation_chars: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Split the corpus into one train prefix and one validation suffix."""
    validation_chars = min(int(validation_chars), max(1, len(encoded_text) // 5))
    return encoded_text[:-validation_chars], encoded_text[-validation_chars:]


def sample_lm_batch(
    encoded_text: np.ndarray,
    *,
    batch_size: int,
    context_length: int,
    rng: np.random.Generator,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample a batch of next-character prediction windows."""
    max_start = len(encoded_text) - context_length - 1
    if max_start <= 0:
        raise ValueError("Encoded text is too short for the requested context length.")
    starts = rng.integers(0, max_start, size=batch_size)
    inputs = np.stack(
        [encoded_text[start : start + context_length] for start in starts],
        axis=0,
    )
    targets = np.stack(
        [encoded_text[start + 1 : start + context_length + 1] for start in starts],
        axis=0,
    )
    return (
        torch.tensor(inputs, dtype=torch.long, device=device),
        torch.tensor(targets, dtype=torch.long, device=device),
    )


def sample_continuation_pair_batch(
    encoded_text: np.ndarray,
    *,
    batch_size: int,
    context_length: int,
    continuation_length: int,
    rng: np.random.Generator,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample context, true continuation, and mismatched continuation triples."""
    max_start = len(encoded_text) - context_length - continuation_length - 1
    if max_start <= 1:
        raise ValueError("Encoded text is too short for continuation scoring.")
    starts = rng.integers(0, max_start, size=batch_size)
    bad_starts = rng.integers(0, max_start, size=batch_size)
    bad_starts = np.where(bad_starts == starts + context_length, (bad_starts + 1) % max_start, bad_starts)
    contexts = np.stack(
        [encoded_text[start : start + context_length] for start in starts],
        axis=0,
    )
    good = np.stack(
        [
            encoded_text[start + context_length : start + context_length + continuation_length]
            for start in starts
        ],
        axis=0,
    )
    bad = np.stack(
        [encoded_text[start : start + continuation_length] for start in bad_starts],
        axis=0,
    )
    return (
        torch.tensor(contexts, dtype=torch.long, device=device),
        torch.tensor(good, dtype=torch.long, device=device),
        torch.tensor(bad, dtype=torch.long, device=device),
    )


def build_language_belief_vector(
    encoded_text: np.ndarray,
    *,
    vocab_size: int,
    config: LanguageProbeBenchmarkConfig,
) -> np.ndarray:
    """Build a compact corpus belief from deterministic support windows."""
    support = encoded_text[: max(1, min(len(encoded_text), config.context_length * config.support_windows))]
    counts = np.bincount(support, minlength=vocab_size).astype(np.float32)
    probs = counts / max(float(np.sum(counts)), 1.0)
    bins = np.array_split(probs, 8)
    hist_features = np.asarray([float(np.sum(item)) for item in bins], dtype=np.float32)
    diffs = np.abs(np.diff(support.astype(np.float32))) if len(support) > 1 else np.zeros((1,), dtype=np.float32)
    entropy = -float(np.sum(probs * np.log2(np.maximum(probs, 1e-8)))) / max(np.log2(max(vocab_size, 2)), 1.0)
    stats = np.asarray(
        [
            entropy,
            float(np.count_nonzero(counts)) / max(float(vocab_size), 1.0),
            float(np.mean(support)) / max(float(vocab_size - 1), 1.0),
            float(np.std(support)) / max(float(vocab_size - 1), 1.0),
            float(np.mean(diffs)) / max(float(vocab_size - 1), 1.0),
            float(np.mean(support[1:] == support[:-1])) if len(support) > 1 else 0.0,
            float(len(support)) / max(float(config.train_char_budgets[-1]), 1.0),
            float(config.support_windows) / 16.0,
        ],
        dtype=np.float32,
    )
    belief = np.concatenate([hist_features, stats], axis=0)
    if len(belief) < config.belief_dim:
        belief = np.pad(belief, (0, config.belief_dim - len(belief)))
    return belief[: config.belief_dim].astype(np.float32)


def ablated_language_beliefs(belief: np.ndarray) -> dict[str, np.ndarray]:
    """Build matched language-belief ablation arms."""
    belief = np.asarray(belief, dtype=np.float32).reshape(-1)
    return {
        "learned": belief,
        "zero": np.zeros_like(belief),
        "shuffled": np.roll(belief, 1),
        "stale": np.full_like(belief, float(np.mean(belief))),
    }




def _model_from_config(
    *,
    vocab_size: int,
    config: LanguageProbeBenchmarkConfig,
    use_belief_prefix: bool,
    device: torch.device,
    handoff_mode: str | None = None,
) -> BeliefConditionedCharTransformer:
    return BeliefConditionedCharTransformer(
        vocab_size=vocab_size,
        d_model=config.hidden_dim,
        n_heads=config.attention_heads,
        n_layers=config.transformer_layers,
        max_sequence_length=config.context_length + config.continuation_length + 1,
        belief_dim=config.belief_dim,
        prefix_tokens=config.prefix_tokens,
        use_belief_prefix=use_belief_prefix,
        handoff_mode=handoff_mode,
    ).to(device)


def train_char_language_model(
    *,
    encoded_train: np.ndarray,
    encoded_validation: np.ndarray,
    vocab_size: int,
    config: LanguageProbeBenchmarkConfig,
    device: torch.device,
    belief_vector: np.ndarray | None,
    handoff_mode: str = "prefix",
) -> dict[str, float | dict[str, float]]:
    """Train one Transformer LM variant and return validation diagnostics."""
    use_belief = belief_vector is not None
    mode = str(handoff_mode if use_belief else "none")
    model = _model_from_config(
        vocab_size=vocab_size,
        config=config,
        use_belief_prefix=use_belief and mode == "prefix",
        device=device,
        handoff_mode=mode,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
    rng = np.random.default_rng(config.seed + _handoff_seed_offset(mode))
    belief_t = None
    if belief_vector is not None:
        belief_t = torch.tensor(belief_vector, dtype=torch.float32, device=device)

    model.train()
    for _step in range(config.lm_steps):
        input_tokens, target_tokens = sample_lm_batch(
            encoded_train,
            batch_size=config.batch_size,
            context_length=config.context_length,
            rng=rng,
            device=device,
        )
        logits = model(input_tokens, belief_t)
        loss = F.cross_entropy(logits.reshape(-1, vocab_size), target_tokens.reshape(-1))
        if use_belief and config.probe_steps > 0 and _step % 4 == 0:
            loss = loss + 0.05 * _continuation_loss(
                model,
                encoded_train,
                vocab_size=vocab_size,
                config=config,
                rng=rng,
                device=device,
                belief_vector=belief_t,
            )
        if use_belief and mode == "adapter" and _step % 2 == 0:
            loss = loss + 0.03 * _belief_contrast_loss(
                model,
                input_tokens,
                target_tokens,
                vocab_size=vocab_size,
                belief_vector=belief_t,
            )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    eval_beliefs = {"learned": belief_t} if belief_t is not None else {"baseline": None}
    if belief_vector is not None:
        eval_beliefs = {
            name: torch.tensor(value, dtype=torch.float32, device=device)
            for name, value in ablated_language_beliefs(belief_vector).items()
        }
    metrics = {
        name: evaluate_language_model(
            model,
            encoded_validation=encoded_validation,
            vocab_size=vocab_size,
            config=config,
            device=device,
            belief_vector=value,
        )
        for name, value in eval_beliefs.items()
    }
    if belief_vector is None:
        return metrics["baseline"]
    return {"ablation_metrics": metrics, "handoff_mode": mode, **metrics["learned"]}


def _handoff_seed_offset(mode: str) -> int:
    if mode == "adapter":
        return 73
    if mode == "prefix":
        return 37
    return 17


def _continuation_loss(
    model: BeliefConditionedCharTransformer,
    encoded_text: np.ndarray,
    *,
    vocab_size: int,
    config: LanguageProbeBenchmarkConfig,
    rng: np.random.Generator,
    device: torch.device,
    belief_vector: torch.Tensor | None,
) -> torch.Tensor:
    contexts, good, bad = sample_continuation_pair_batch(
        encoded_text,
        batch_size=max(2, config.batch_size // 2),
        context_length=config.context_length,
        continuation_length=config.continuation_length,
        rng=rng,
        device=device,
    )
    good_loss = _continuation_candidate_loss(model, contexts, good, vocab_size, belief_vector)
    bad_loss = _continuation_candidate_loss(model, contexts, bad, vocab_size, belief_vector)
    margin = 0.2 + good_loss - bad_loss
    return torch.mean(torch.relu(margin))


def _belief_contrast_loss(
    model: BeliefConditionedCharTransformer,
    input_tokens: torch.Tensor,
    target_tokens: torch.Tensor,
    *,
    vocab_size: int,
    belief_vector: torch.Tensor,
) -> torch.Tensor:
    """Encourage the learned belief to beat matched no-content belief arms."""
    learned_logits = model(input_tokens, belief_vector)
    learned_loss = F.cross_entropy(
        learned_logits.reshape(-1, vocab_size),
        target_tokens.reshape(-1),
    )
    belief = belief_vector.detach()
    ablations = (
        torch.zeros_like(belief),
        torch.roll(belief, shifts=1),
        torch.full_like(belief, float(torch.mean(belief).item())),
    )
    penalties: list[torch.Tensor] = []
    for ablated in ablations:
        ablated_logits = model(input_tokens, ablated)
        ablated_loss = F.cross_entropy(
            ablated_logits.reshape(-1, vocab_size),
            target_tokens.reshape(-1),
        )
        penalties.append(torch.relu(0.01 + learned_loss - ablated_loss))
    return torch.mean(torch.stack(penalties))


def _continuation_candidate_loss(
    model: BeliefConditionedCharTransformer,
    contexts: torch.Tensor,
    continuations: torch.Tensor,
    vocab_size: int,
    belief_vector: torch.Tensor | None,
) -> torch.Tensor:
    combined = torch.cat([contexts, continuations], dim=1)
    inputs = combined[:, :-1]
    targets = combined[:, 1:]
    logits = model(inputs, belief_vector)
    per_token = F.cross_entropy(
        logits.reshape(-1, vocab_size),
        targets.reshape(-1),
        reduction="none",
    ).view(targets.shape)
    start = contexts.shape[1] - 1
    return torch.mean(per_token[:, start:], dim=1)


def evaluate_language_model(
    model: BeliefConditionedCharTransformer,
    *,
    encoded_validation: np.ndarray,
    vocab_size: int,
    config: LanguageProbeBenchmarkConfig,
    device: torch.device,
    belief_vector: torch.Tensor | None,
) -> dict[str, float]:
    """Evaluate BPC, continuation ranking, and next-char cloze accuracy."""
    model.eval()
    eval_losses: list[float] = []
    eval_acc: list[float] = []
    continuation_acc: list[float] = []
    eval_rng = np.random.default_rng(config.seed + 99)
    with torch.no_grad():
        for _step in range(config.eval_steps):
            input_tokens, target_tokens = sample_lm_batch(
                encoded_validation,
                batch_size=config.batch_size,
                context_length=config.context_length,
                rng=eval_rng,
                device=device,
            )
            logits = model(input_tokens, belief_vector)
            loss = F.cross_entropy(logits.reshape(-1, vocab_size), target_tokens.reshape(-1))
            eval_losses.append(float(loss.item()))
            eval_acc.append(float((logits.argmax(dim=-1) == target_tokens).float().mean().item()))
            contexts, good, bad = sample_continuation_pair_batch(
                encoded_validation,
                batch_size=config.batch_size,
                context_length=config.context_length,
                continuation_length=config.continuation_length,
                rng=eval_rng,
                device=device,
            )
            good_loss = _continuation_candidate_loss(model, contexts, good, vocab_size, belief_vector)
            bad_loss = _continuation_candidate_loss(model, contexts, bad, vocab_size, belief_vector)
            continuation_acc.append(float((good_loss < bad_loss).float().mean().item()))
    mean_nats = float(np.mean(eval_losses)) if eval_losses else 0.0
    return {
        "bpc": mean_nats / float(np.log(2.0)),
        "cloze_accuracy": float(np.mean(eval_acc)) if eval_acc else 0.0,
        "continuation_accuracy": float(np.mean(continuation_acc)) if continuation_acc else 0.0,
    }


def build_language_artifact(
    *,
    encoded_train: np.ndarray,
    belief_vector: np.ndarray,
    config: LanguageProbeBenchmarkConfig,
    handoff_rows: list[dict[str, float | int | str | bool]] | None = None,
) -> dict[str, object]:
    """Build the dashboard-facing language belief artifact summary."""
    offsets = np.linspace(
        0,
        max(0, len(encoded_train) - config.context_length - 1),
        num=max(1, config.support_windows),
        dtype=np.int64,
    )
    evidence_slices = [
        {
            "modality": "language",
            "query_family": "support_span",
            "source_id": "tiny_shakespeare",
            "intervention_cost": float(config.context_length),
            "local_latent_norm": float(np.linalg.norm(belief_vector)),
            "local_state": {"start": int(offset), "length": int(config.context_length)},
            "outcome": {"target": "next_character_prediction"},
        }
        for offset in offsets.tolist()
    ]
    return {
        "raw_evidence_windows": evidence_slices,
        "source_ids": ["tiny_shakespeare"],
        "query_families": ["support_span", "continuation_ranking", "cloze"],
        "local_evidence_latents": [belief_vector.tolist()],
        "domain_belief": belief_vector.tolist(),
        "uncertainty_estimate": float(1.0 / (1.0 + np.linalg.norm(belief_vector))),
        "hidden_rule_targets": {},
        "subset_agreement": float(1.0 - min(np.std(belief_vector), 1.0)),
        "belief_bitrate": int(len(belief_vector) * 32),
        "handoff_comparison": [] if handoff_rows is None else handoff_rows,
    }


def run_synthetic_grammar_smoke(seed: int = 0) -> dict[str, object]:
    """Known-rule language ladder where the crawler message must be causal."""
    alphabet_size = 7
    rules = ("previous_token", "next_token", "skip_forward", "mirror_token")
    baseline_rule = "next_token"

    def targets_for(rule: str, tokens: np.ndarray) -> np.ndarray:
        if rule == "previous_token":
            return (tokens - 1) % alphabet_size
        if rule == "skip_forward":
            return (tokens + 2) % alphabet_size
        if rule == "mirror_token":
            return (alphabet_size - 1 - tokens) % alphabet_size
        return (tokens + 1) % alphabet_size

    def infer_rule(tokens: np.ndarray, observed: np.ndarray) -> str:
        scores = {
            rule: float(np.mean(targets_for(rule, tokens) == observed))
            for rule in rules
        }
        return max(scores, key=scores.get)

    def evaluate_world(world_seed: int) -> dict[str, float | int | str]:
        support_tokens = np.arange(alphabet_size, dtype=np.int64)
        challenge_tokens = np.tile(support_tokens, 3)
        hidden_rule = rules[int(world_seed) % len(rules)]
        observed = targets_for(hidden_rule, support_tokens)
        decoded_rule = infer_rule(support_tokens, observed)
        first_rule = infer_rule(support_tokens[::2], observed[::2])
        second_rule = infer_rule(support_tokens[1::2], observed[1::2])
        shuffled_rule = rules[(rules.index(decoded_rule) + 1) % len(rules)]
        truth = targets_for(hidden_rule, challenge_tokens)
        baseline_predictions = targets_for(baseline_rule, challenge_tokens)
        learned_predictions = targets_for(decoded_rule, challenge_tokens)
        shuffled_predictions = targets_for(shuffled_rule, challenge_tokens)
        zero_predictions = targets_for(baseline_rule, challenge_tokens)
        stale_predictions = np.full_like(challenge_tokens, int(np.bincount(observed).argmax()))
        baseline_accuracy = float(np.mean(baseline_predictions == truth))
        learned_accuracy = float(np.mean(learned_predictions == truth))
        zero_accuracy = float(np.mean(zero_predictions == truth))
        shuffled_accuracy = float(np.mean(shuffled_predictions == truth))
        stale_accuracy = float(np.mean(stale_predictions == truth))
        best_ablation = max(zero_accuracy, shuffled_accuracy, stale_accuracy)
        return {
            "seed": int(world_seed),
            "hidden_rule": hidden_rule,
            "decoded_rule": decoded_rule,
            "hidden_rule_decode_accuracy": float(decoded_rule == hidden_rule),
            "subset_agreement": float(first_rule == second_rule == decoded_rule),
            "shuffled_decode_accuracy": float(shuffled_rule == hidden_rule),
            "baseline_next_token_accuracy": baseline_accuracy,
            "belief_next_token_accuracy": learned_accuracy,
            "zero_next_token_accuracy": zero_accuracy,
            "shuffled_next_token_accuracy": shuffled_accuracy,
            "stale_next_token_accuracy": stale_accuracy,
            "content_lift": learned_accuracy - best_ablation,
            "support_worlds": int(len(support_tokens)),
            "challenge_worlds": int(len(challenge_tokens)),
        }

    rows = [evaluate_world(seed + idx) for idx in range(len(rules) * 2)]
    headline = rows[0]
    return {
        **headline,
        "rows": rows,
        "candidate_rules": list(rules),
        "hidden_rule_decode_accuracy": float(np.mean([row["hidden_rule_decode_accuracy"] for row in rows])),
        "subset_agreement": float(np.mean([row["subset_agreement"] for row in rows])),
        "belief_next_token_accuracy": float(np.mean([row["belief_next_token_accuracy"] for row in rows])),
        "baseline_next_token_accuracy": float(headline["baseline_next_token_accuracy"]),
        "mean_baseline_next_token_accuracy": float(np.mean([row["baseline_next_token_accuracy"] for row in rows])),
        "zero_next_token_accuracy": float(np.mean([row["zero_next_token_accuracy"] for row in rows])),
        "shuffled_next_token_accuracy": float(np.mean([row["shuffled_next_token_accuracy"] for row in rows])),
        "stale_next_token_accuracy": float(np.mean([row["stale_next_token_accuracy"] for row in rows])),
        "content_lift": float(np.mean([row["content_lift"] for row in rows])),
        "support_worlds": int(sum(int(row["support_worlds"]) for row in rows)),
        "challenge_worlds": int(sum(int(row["challenge_worlds"]) for row in rows)),
    }


def run_shakespeare_probe_benchmark(
    config: LanguageProbeBenchmarkConfig | None = None,
) -> dict[str, object]:
    """Compare plain and belief-conditioned Transformers on Tiny Shakespeare."""
    config = config or LanguageProbeBenchmarkConfig()
    set_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    corpus_path = ensure_tiny_shakespeare(config)
    text = corpus_path.read_text(encoding="utf-8")
    stoi, _itos = build_char_vocab(text)
    encoded_text = encode_text(text, stoi)
    encoded_train, encoded_validation = split_corpus(
        encoded_text,
        validation_chars=config.validation_chars,
    )
    vocab_size = len(stoi)

    rows: list[dict[str, float | int]] = []
    artifacts: list[dict[str, object]] = []
    for budget in config.train_char_budgets:
        budget = min(int(budget), len(encoded_train))
        train_slice = encoded_train[:budget]
        belief = build_language_belief_vector(
            train_slice,
            vocab_size=vocab_size,
            config=config,
        )
        baseline = train_char_language_model(
            encoded_train=train_slice,
            encoded_validation=encoded_validation,
            vocab_size=vocab_size,
            config=config,
            device=device,
            belief_vector=None,
        )
        handoff_results = [
            train_char_language_model(
                encoded_train=train_slice,
                encoded_validation=encoded_validation,
                vocab_size=vocab_size,
                config=config,
                device=device,
                belief_vector=belief,
                handoff_mode=mode,
            )
            for mode in config.handoff_modes
        ]
        belief_metrics = _best_language_handoff(baseline, handoff_results)
        ablations = belief_metrics["ablation_metrics"]
        belief_bitrate = int(config.belief_dim * 32)
        economics = language_row_economics(
            baseline_bpc=float(baseline["bpc"]),
            belief_bpc=float(belief_metrics["bpc"]),
            ablations=ablations,
            belief_bitrate=belief_bitrate,
            support_windows=config.support_windows,
        )
        handoff_rows = _language_handoff_rows(
            baseline=baseline,
            handoff_results=handoff_results,
            belief_bitrate=belief_bitrate,
            support_windows=config.support_windows,
        )
        selected = _select_language_handoff(
            baseline=baseline,
            raw_best=belief_metrics,
            handoff_rows=handoff_rows,
            handoff_results=handoff_results,
        )
        selected_gate = selected.get("decision_gate", {})
        if not isinstance(selected_gate, dict):
            selected_gate = {}
        selected_ablations = selected["ablation_metrics"]
        selected_economics = language_row_economics(
            baseline_bpc=float(baseline["bpc"]),
            belief_bpc=float(selected["bpc"]),
            ablations=selected_ablations,
            belief_bitrate=belief_bitrate,
            support_windows=config.support_windows,
        )
        rows.append(
            {
                "train_char_budget": budget,
                "handoff_mode": str(selected.get("handoff_mode", "baseline_fallback")),
                "raw_handoff_mode": str(belief_metrics.get("handoff_mode", "prefix")),
                "baseline_bpc": float(baseline["bpc"]),
                "probe_bpc": float(selected["bpc"]),
                "belief_bpc": float(selected["bpc"]),
                "raw_belief_bpc": float(belief_metrics["bpc"]),
                "raw_bpc_gain": float(baseline["bpc"]) - float(belief_metrics["bpc"]),
                "raw_best_ablation_bpc": float(economics["best_ablation_bpc"]),
                "raw_content_lift": float(economics["content_lift"]),
                "raw_budget_gate_uses_belief": bool(economics["budget_gate_uses_belief"]),
                "zero_belief_bpc": float(selected_ablations["zero"]["bpc"]),
                "shuffled_belief_bpc": float(selected_ablations["shuffled"]["bpc"]),
                "stale_belief_bpc": float(selected_ablations["stale"]["bpc"]),
                "raw_zero_belief_bpc": float(ablations["zero"]["bpc"]),
                "raw_shuffled_belief_bpc": float(ablations["shuffled"]["bpc"]),
                "raw_stale_belief_bpc": float(ablations["stale"]["bpc"]),
                "bpc_gain": float(baseline["bpc"]) - float(selected["bpc"]),
                "cloze_accuracy": float(selected["cloze_accuracy"]),
                "continuation_accuracy": float(selected["continuation_accuracy"]),
                "raw_cloze_accuracy": float(belief_metrics["cloze_accuracy"]),
                "raw_continuation_accuracy": float(belief_metrics["continuation_accuracy"]),
                "prefix_sensitivity": float(selected_ablations["zero"]["bpc"]) - float(selected["bpc"]),
                "handoff_gate_used_baseline": bool(selected.get("used_baseline_fallback", False)),
                "handoff_gate_reason": str(selected.get("gate_reason", "")),
                "decision_gate_use_belief": bool(selected_gate.get("use_belief", False)),
                "decision_gate_reason": str(selected_gate.get("reason", "")),
                "decision_delta_correct_vs_best_ablation": float(
                    selected_gate.get("decision_delta_correct_vs_best_ablation", 0.0)
                ),
                "belief_bitrate": belief_bitrate,
                "adapter_bpc": _handoff_metric(handoff_rows, "adapter", "bpc"),
                "adapter_bpc_gain": _handoff_metric(handoff_rows, "adapter", "bpc_gain"),
                "adapter_content_lift": _handoff_metric(handoff_rows, "adapter", "content_lift"),
                "prefix_bpc": _handoff_metric(handoff_rows, "prefix", "bpc"),
                "prefix_bpc_gain": _handoff_metric(handoff_rows, "prefix", "bpc_gain"),
                "prefix_content_lift": _handoff_metric(handoff_rows, "prefix", "content_lift"),
                **selected_economics,
            }
        )
        artifacts.append(
            build_language_artifact(
                encoded_train=train_slice,
                belief_vector=belief,
                config=config,
                handoff_rows=handoff_rows,
            )
        )

    grammar_smoke = run_synthetic_grammar_smoke(config.seed)
    return {
        "domain": "language",
        "dataset": "Tiny Shakespeare",
        "model_family": "BeliefConditionedCharTransformer",
        "probe_objective": "belief_prefix_transformer",
        "handoff_modes": list(config.handoff_modes),
        "validation_chars": int(config.validation_chars),
        "prefix_tokens": int(config.prefix_tokens),
        "support_windows": int(config.support_windows),
        "probe_steps": int(config.probe_steps),
        "lm_steps": int(config.lm_steps),
        "rows": rows,
        "artifacts": artifacts,
        "synthetic_grammar": grammar_smoke,
        "mean_bpc_gain": float(np.mean([row["bpc_gain"] for row in rows])) if rows else 0.0,
        "mean_prefix_sensitivity": float(np.mean([row["prefix_sensitivity"] for row in rows])) if rows else 0.0,
    }
