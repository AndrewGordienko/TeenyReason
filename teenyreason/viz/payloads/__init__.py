"""Dashboard artifact payload builders."""

from .benchmark import build_benchmark_payload
from .common import (
    aggregate_json_counter_rows,
    aggregate_json_list_rows,
    average_json_metric_rows,
    build_support_validity_payload,
    load_array_with_fallback,
    load_benchmark_summary,
    load_optional_json_rows,
    load_optional_string,
    normalize_matched_eval_summary,
    normalize_projection_2d,
    summarize_matched_eval_rows,
)
from .index import (
    build_index_payload,
    list_benchmark_paths,
    load_benchmark_profile_name,
    load_dashboard_context,
    load_main_module_context,
    order_benchmark_paths,
    preferred_benchmark_summary_name,
    preferred_latent_snapshot_name,
)
from .latent import build_latent_payload

__all__ = [
    "aggregate_json_counter_rows",
    "aggregate_json_list_rows",
    "average_json_metric_rows",
    "build_benchmark_payload",
    "build_index_payload",
    "build_latent_payload",
    "build_support_validity_payload",
    "list_benchmark_paths",
    "load_array_with_fallback",
    "load_benchmark_profile_name",
    "load_benchmark_summary",
    "load_dashboard_context",
    "load_main_module_context",
    "load_optional_json_rows",
    "load_optional_string",
    "normalize_matched_eval_summary",
    "normalize_projection_2d",
    "order_benchmark_paths",
    "preferred_benchmark_summary_name",
    "preferred_latent_snapshot_name",
    "summarize_matched_eval_rows",
]
