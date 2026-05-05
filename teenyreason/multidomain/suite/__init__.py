"""Cross-domain suite runner and payload helpers."""

from .config import MultidomainSuiteConfig
from .reporting import build_suite_payload, summarize_rows
from .runner import main, run_multidomain_suite

__all__ = [
    "MultidomainSuiteConfig",
    "build_suite_payload",
    "main",
    "run_multidomain_suite",
    "summarize_rows",
]
