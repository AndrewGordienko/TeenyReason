"""Tiny Shakespeare sample-efficiency benchmark for probe-pretraining vs plain LMs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import torch
import torch.nn as nn


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
    probe_steps: int = 300
    lm_steps: int = 400
    eval_steps: int = 80
    batch_size: int = 64
    lr: float = 2e-3
    seed: int = 0


class CharBackbone(nn.Module):
    """Shared embedding plus GRU stack used by both probe and LM heads."""

    def __init__(self, vocab_size: int, embedding_dim: int, hidden_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.GRU(embedding_dim, hidden_dim, batch_first=True)

    def encode(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        embedded = self.embedding(tokens)
        outputs, hidden = self.rnn(embedded)
        return outputs, hidden[-1]


class ContinuationProbeModel(nn.Module):
    """Binary continuation-discrimination probe over local text windows."""

    def __init__(self, vocab_size: int, embedding_dim: int, hidden_dim: int):
        super().__init__()
        self.backbone = CharBackbone(vocab_size, embedding_dim, hidden_dim)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, context_tokens: torch.Tensor, continuation_tokens: torch.Tensor) -> torch.Tensor:
        _context_outputs, context_state = self.backbone.encode(context_tokens)
        _continuation_outputs, continuation_state = self.backbone.encode(continuation_tokens)
        features = torch.cat([context_state, continuation_state], dim=-1)
        return self.head(features)


class CharLanguageModel(nn.Module):
    """Small character-level language model used for the baseline and probe variant."""

    def __init__(self, vocab_size: int, embedding_dim: int, hidden_dim: int):
        super().__init__()
        self.backbone = CharBackbone(vocab_size, embedding_dim, hidden_dim)
        self.head = nn.Linear(hidden_dim, vocab_size)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        outputs, _hidden = self.backbone.encode(tokens)
        return self.head(outputs)


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
    starts = rng.integers(0, max_start, size=batch_size)
    inputs = np.stack([encoded_text[start:start + context_length] for start in starts], axis=0)
    targets = np.stack([encoded_text[start + 1:start + context_length + 1] for start in starts], axis=0)
    return (
        torch.tensor(inputs, dtype=torch.long, device=device),
        torch.tensor(targets, dtype=torch.long, device=device),
    )


def sample_continuation_probe_batch(
    encoded_text: np.ndarray,
    *,
    batch_size: int,
    context_length: int,
    continuation_length: int,
    rng: np.random.Generator,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample one balanced continuation-discrimination batch."""
    max_start = len(encoded_text) - context_length - continuation_length - 1
    if max_start <= 1:
        raise ValueError("Encoded text is too short for the requested context and continuation lengths.")

    contexts = np.zeros((batch_size, context_length), dtype=np.int64)
    continuations = np.zeros((batch_size, continuation_length), dtype=np.int64)
    labels = np.zeros((batch_size,), dtype=np.int64)

    for row_idx in range(batch_size):
        start = int(rng.integers(0, max_start))
        contexts[row_idx] = encoded_text[start:start + context_length]
        labels[row_idx] = row_idx % 2
        if labels[row_idx] == 1:
            continuation_start = start + context_length
        else:
            continuation_start = int(rng.integers(0, max_start))
            if continuation_start == start + context_length:
                continuation_start = (continuation_start + continuation_length) % max_start
        continuations[row_idx] = encoded_text[
            continuation_start:continuation_start + continuation_length
        ]

    permutation = rng.permutation(batch_size)
    return (
        torch.tensor(contexts[permutation], dtype=torch.long, device=device),
        torch.tensor(continuations[permutation], dtype=torch.long, device=device),
        torch.tensor(labels[permutation], dtype=torch.long, device=device),
    )


def train_continuation_probe(
    *,
    encoded_text: np.ndarray,
    vocab_size: int,
    config: LanguageProbeBenchmarkConfig,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Pretrain the shared backbone on continuation discrimination."""
    model = ContinuationProbeModel(
        vocab_size=vocab_size,
        embedding_dim=config.embedding_dim,
        hidden_dim=config.hidden_dim,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    loss_fn = nn.CrossEntropyLoss()
    rng = np.random.default_rng(config.seed)

    model.train()
    for _step in range(config.probe_steps):
        context_tokens, continuation_tokens, labels = sample_continuation_probe_batch(
            encoded_text,
            batch_size=config.batch_size,
            context_length=config.context_length,
            continuation_length=config.continuation_length,
            rng=rng,
            device=device,
        )
        logits = model(context_tokens, continuation_tokens)
        loss = loss_fn(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return {
        key: value.detach().cpu().clone()
        for key, value in model.backbone.state_dict().items()
    }


def train_char_language_model(
    *,
    encoded_train: np.ndarray,
    encoded_validation: np.ndarray,
    vocab_size: int,
    config: LanguageProbeBenchmarkConfig,
    device: torch.device,
    init_backbone_state: dict[str, torch.Tensor] | None,
) -> float:
    """Train one LM variant and return validation bits-per-character."""
    model = CharLanguageModel(
        vocab_size=vocab_size,
        embedding_dim=config.embedding_dim,
        hidden_dim=config.hidden_dim,
    ).to(device)
    if init_backbone_state is not None:
        model.backbone.load_state_dict(init_backbone_state)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    loss_fn = nn.CrossEntropyLoss()
    rng = np.random.default_rng(config.seed + 17)

    model.train()
    for _step in range(config.lm_steps):
        input_tokens, target_tokens = sample_lm_batch(
            encoded_train,
            batch_size=config.batch_size,
            context_length=config.context_length,
            rng=rng,
            device=device,
        )
        logits = model(input_tokens)
        loss = loss_fn(logits.reshape(-1, vocab_size), target_tokens.reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()
    eval_losses: list[float] = []
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
            logits = model(input_tokens)
            loss = loss_fn(logits.reshape(-1, vocab_size), target_tokens.reshape(-1))
            eval_losses.append(float(loss.item()))
    mean_nats = float(np.mean(eval_losses)) if eval_losses else 0.0
    return mean_nats / float(np.log(2.0))


def run_shakespeare_probe_benchmark(
    config: LanguageProbeBenchmarkConfig | None = None,
) -> dict[str, object]:
    """Compare a plain small LM to a probe-pretrained LM on Tiny Shakespeare."""
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
    for budget in config.train_char_budgets:
        budget = min(int(budget), len(encoded_train))
        train_slice = encoded_train[:budget]
        probe_state = train_continuation_probe(
            encoded_text=train_slice,
            vocab_size=vocab_size,
            config=config,
            device=device,
        )
        baseline_bpc = train_char_language_model(
            encoded_train=train_slice,
            encoded_validation=encoded_validation,
            vocab_size=vocab_size,
            config=config,
            device=device,
            init_backbone_state=None,
        )
        probe_bpc = train_char_language_model(
            encoded_train=train_slice,
            encoded_validation=encoded_validation,
            vocab_size=vocab_size,
            config=config,
            device=device,
            init_backbone_state=probe_state,
        )
        rows.append(
            {
                "train_char_budget": budget,
                "baseline_bpc": baseline_bpc,
                "probe_bpc": probe_bpc,
                "bpc_gain": baseline_bpc - probe_bpc,
            }
        )

    return {
        "domain": "language",
        "dataset": "Tiny Shakespeare",
        "probe_objective": "continuation_discrimination",
        "validation_chars": int(config.validation_chars),
        "probe_steps": int(config.probe_steps),
        "lm_steps": int(config.lm_steps),
        "rows": rows,
        "mean_bpc_gain": float(np.mean([row["bpc_gain"] for row in rows])) if rows else 0.0,
    }
