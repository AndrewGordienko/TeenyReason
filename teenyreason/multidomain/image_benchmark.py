"""MNIST sample-efficiency benchmark for probe-pretraining vs plain CNNs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision.datasets import MNIST


@dataclass(frozen=True)
class ImageProbeBenchmarkConfig:
    """Configuration for the MNIST probe-pretraining benchmark."""

    data_dir: Path = Path("artifacts/data/mnist")
    label_budgets: tuple[int, ...] = (256, 1024, 4096)
    unlabeled_budget: int = 20000
    test_budget: int = 5000
    probe_epochs: int = 3
    finetune_epochs: int = 4
    batch_size: int = 128
    lr: float = 1e-3
    feature_dim: int = 64
    seed: int = 0


class MNISTEncoder(nn.Module):
    """Small CNN backbone shared by the probe head and the classifier."""

    def __init__(self, feature_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, feature_dim),
            nn.ReLU(),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.net(images)


class RotationProbeModel(nn.Module):
    """Self-supervised probe head that predicts which rotation was applied."""

    def __init__(self, feature_dim: int):
        super().__init__()
        self.encoder = MNISTEncoder(feature_dim)
        self.head = nn.Linear(feature_dim, 4)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(images))


class MNISTClassifier(nn.Module):
    """Small digit classifier used for both baseline and probe variants."""

    def __init__(self, feature_dim: int):
        super().__init__()
        self.encoder = MNISTEncoder(feature_dim)
        self.head = nn.Linear(feature_dim, 10)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(images))


def set_seed(seed: int):
    """Keep NumPy and Torch aligned for reproducible subsets and batches."""
    np.random.seed(seed)
    torch.manual_seed(seed)


def stratified_subset_indices(labels: torch.Tensor, budget: int, seed: int) -> np.ndarray:
    """Sample a roughly class-balanced labeled subset."""
    labels_np = labels.detach().cpu().numpy().astype(np.int64)
    unique_labels = sorted(int(label) for label in np.unique(labels_np))
    rng = np.random.default_rng(seed)
    per_class = max(1, int(budget) // max(len(unique_labels), 1))
    chosen: list[int] = []
    leftovers: list[int] = []

    for label in unique_labels:
        label_indices = np.flatnonzero(labels_np == label)
        rng.shuffle(label_indices)
        chosen.extend(label_indices[:per_class].tolist())
        leftovers.extend(label_indices[per_class:].tolist())

    if len(chosen) < int(budget):
        rng.shuffle(leftovers)
        chosen.extend(leftovers[: int(budget) - len(chosen)])

    chosen = chosen[: int(budget)]
    rng.shuffle(chosen)
    return np.asarray(chosen, dtype=np.int64)


def sample_subset_indices(size: int, budget: int, seed: int) -> np.ndarray:
    """Sample one deterministic subset without replacement."""
    rng = np.random.default_rng(seed)
    budget = min(int(budget), int(size))
    return np.asarray(rng.choice(size, size=budget, replace=False), dtype=np.int64)


def load_mnist_tensors(data_dir: Path) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Load MNIST into plain tensors so the benchmark can sample subsets cheaply."""
    data_dir.mkdir(parents=True, exist_ok=True)
    try:
        train = MNIST(root=str(data_dir), train=True, download=True)
        test = MNIST(root=str(data_dir), train=False, download=True)
    except Exception as exc:
        raise RuntimeError(
            "Could not load MNIST. If the dataset is not cached locally, the first run needs internet access."
        ) from exc

    train_images = train.data.unsqueeze(1).float() / 255.0
    train_labels = train.targets.long()
    test_images = test.data.unsqueeze(1).float() / 255.0
    test_labels = test.targets.long()
    return train_images, train_labels, test_images, test_labels


def build_loader(
    images: torch.Tensor,
    labels: torch.Tensor | None,
    indices: np.ndarray,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    """Build one plain DataLoader from a tensor subset."""
    index_t = torch.tensor(indices, dtype=torch.long)
    subset_images = images.index_select(0, index_t)
    if labels is None:
        dataset = TensorDataset(subset_images)
    else:
        subset_labels = labels.index_select(0, index_t)
        dataset = TensorDataset(subset_images, subset_labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def apply_random_rotations(images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Rotate each image by one of four quarter turns and return the labels."""
    batch_size = int(images.shape[0])
    rotation_labels = torch.randint(0, 4, (batch_size,), device=images.device)
    rotated = torch.empty_like(images)
    for rotation_idx in range(4):
        mask = rotation_labels == rotation_idx
        if torch.any(mask):
            rotated[mask] = torch.rot90(images[mask], k=rotation_idx, dims=(-2, -1))
    return rotated, rotation_labels


def train_rotation_probe(
    *,
    train_images: torch.Tensor,
    unlabeled_indices: np.ndarray,
    config: ImageProbeBenchmarkConfig,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Pretrain the encoder on rotation prediction using unlabeled images."""
    model = RotationProbeModel(feature_dim=config.feature_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    loss_fn = nn.CrossEntropyLoss()
    loader = build_loader(
        images=train_images,
        labels=None,
        indices=unlabeled_indices,
        batch_size=config.batch_size,
        shuffle=True,
    )

    model.train()
    for _epoch in range(config.probe_epochs):
        for (images,) in loader:
            images = images.to(device)
            rotated, rotation_labels = apply_random_rotations(images)
            logits = model(rotated)
            loss = loss_fn(logits, rotation_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return {
        key: value.detach().cpu().clone()
        for key, value in model.encoder.state_dict().items()
    }


def train_classifier(
    *,
    train_images: torch.Tensor,
    train_labels: torch.Tensor,
    test_images: torch.Tensor,
    test_labels: torch.Tensor,
    labeled_indices: np.ndarray,
    init_encoder_state: dict[str, torch.Tensor] | None,
    config: ImageProbeBenchmarkConfig,
    device: torch.device,
) -> float:
    """Train one classifier variant and return test accuracy."""
    model = MNISTClassifier(feature_dim=config.feature_dim).to(device)
    if init_encoder_state is not None:
        model.encoder.load_state_dict(init_encoder_state)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    loss_fn = nn.CrossEntropyLoss()
    train_loader = build_loader(
        images=train_images,
        labels=train_labels,
        indices=labeled_indices,
        batch_size=config.batch_size,
        shuffle=True,
    )
    test_indices = sample_subset_indices(len(test_images), config.test_budget, config.seed + 99)
    test_loader = build_loader(
        images=test_images,
        labels=test_labels,
        indices=test_indices,
        batch_size=config.batch_size,
        shuffle=False,
    )

    model.train()
    for _epoch in range(config.finetune_epochs):
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            loss = loss_fn(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            predictions = model(images).argmax(dim=-1)
            correct += int((predictions == labels).sum().item())
            total += int(labels.numel())
    return float(correct) / float(max(total, 1))


def run_mnist_probe_benchmark(
    config: ImageProbeBenchmarkConfig | None = None,
) -> dict[str, object]:
    """Compare a plain CNN to a probe-pretrained CNN on low-label MNIST."""
    config = config or ImageProbeBenchmarkConfig()
    set_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_images, train_labels, test_images, test_labels = load_mnist_tensors(config.data_dir)
    unlabeled_indices = sample_subset_indices(
        len(train_images),
        config.unlabeled_budget,
        config.seed,
    )
    encoder_state = train_rotation_probe(
        train_images=train_images,
        unlabeled_indices=unlabeled_indices,
        config=config,
        device=device,
    )

    rows: list[dict[str, float | int]] = []
    for budget in config.label_budgets:
        labeled_indices = stratified_subset_indices(train_labels, budget, config.seed + int(budget))
        baseline_accuracy = train_classifier(
            train_images=train_images,
            train_labels=train_labels,
            test_images=test_images,
            test_labels=test_labels,
            labeled_indices=labeled_indices,
            init_encoder_state=None,
            config=config,
            device=device,
        )
        probe_accuracy = train_classifier(
            train_images=train_images,
            train_labels=train_labels,
            test_images=test_images,
            test_labels=test_labels,
            labeled_indices=labeled_indices,
            init_encoder_state=encoder_state,
            config=config,
            device=device,
        )
        rows.append(
            {
                "label_budget": int(budget),
                "baseline_accuracy": baseline_accuracy,
                "probe_accuracy": probe_accuracy,
                "accuracy_gain": probe_accuracy - baseline_accuracy,
            }
        )

    return {
        "domain": "image",
        "dataset": "MNIST",
        "probe_objective": "rotation_prediction",
        "unlabeled_budget": int(config.unlabeled_budget),
        "test_budget": int(config.test_budget),
        "probe_epochs": int(config.probe_epochs),
        "finetune_epochs": int(config.finetune_epochs),
        "rows": rows,
        "mean_accuracy_gain": float(np.mean([row["accuracy_gain"] for row in rows])) if rows else 0.0,
    }
