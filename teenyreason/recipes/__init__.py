"""User-facing crawler recipe compositions."""

from .base import BenchmarkSpec, CrawlerRecipe
from .benchmark import build_benchmark_recipe, build_generic_rl_recipe
from .bipedal import build_bipedal_recipe, register_bipedal_recipe_targets
from .cartpole import build_cartpole_recipe, register_cartpole_recipe_targets
from .language import build_language_recipe
from .mnist import build_mnist_recipe

_DEFAULT_TARGETS_REGISTERED = False


def register_default_recipe_targets() -> None:
    """Register the built-in benchmark target builders once."""
    global _DEFAULT_TARGETS_REGISTERED
    if _DEFAULT_TARGETS_REGISTERED:
        return
    register_cartpole_recipe_targets()
    register_bipedal_recipe_targets()
    _DEFAULT_TARGETS_REGISTERED = True


__all__ = [
    "BenchmarkSpec",
    "CrawlerRecipe",
    "build_benchmark_recipe",
    "build_bipedal_recipe",
    "build_cartpole_recipe",
    "build_generic_rl_recipe",
    "build_language_recipe",
    "build_mnist_recipe",
    "register_default_recipe_targets",
]
