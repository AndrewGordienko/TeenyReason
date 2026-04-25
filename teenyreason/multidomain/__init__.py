"""Cross-domain sample-efficiency benchmarks."""

from .image_benchmark import ImageProbeBenchmarkConfig, run_mnist_probe_benchmark
from .language_benchmark import LanguageProbeBenchmarkConfig, run_shakespeare_probe_benchmark

__all__ = [
    "ImageProbeBenchmarkConfig",
    "LanguageProbeBenchmarkConfig",
    "run_mnist_probe_benchmark",
    "run_shakespeare_probe_benchmark",
]
