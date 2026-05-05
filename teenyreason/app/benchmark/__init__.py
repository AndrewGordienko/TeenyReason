"""Benchmark harness public API.

The implementation is split across focused modules, but older callers import
the common entrypoints from ``teenyreason.app.benchmark``.
"""

from .runner import print_array_shapes, run_single_seed, run_training_pipeline, set_seed
from .support import (
    apply_system_id_representation_override,
    belief_source_from_mode,
    benchmark_profile_flags,
    classify_probe_run,
    compute_belief_progress_index,
    compute_system_id_progress_index,
    default_seeds_for_profile,
    evaluate_latent_win_gate,
    evaluate_representation_gate,
    matched_eval_summary_dict,
    print_return_summary,
    print_solve_summary,
    probe_strict_usage_status,
    resolve_benchmark_profile,
    solve_eval_episodes_for_profile,
)

__all__ = [
    "apply_system_id_representation_override",
    "belief_source_from_mode",
    "benchmark_profile_flags",
    "classify_probe_run",
    "compute_belief_progress_index",
    "compute_system_id_progress_index",
    "default_seeds_for_profile",
    "evaluate_latent_win_gate",
    "evaluate_representation_gate",
    "matched_eval_summary_dict",
    "print_array_shapes",
    "print_return_summary",
    "print_solve_summary",
    "probe_strict_usage_status",
    "resolve_benchmark_profile",
    "run_single_seed",
    "run_training_pipeline",
    "set_seed",
    "solve_eval_episodes_for_profile",
]
