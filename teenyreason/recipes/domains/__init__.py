"""Domain-specific crawler recipes."""

from .bipedal import build_bipedal_recipe, register_bipedal_recipe_targets
from .board import build_board_recipe
from .cartpole import build_cartpole_recipe, register_cartpole_recipe_targets
from .language import build_language_recipe
from .mnist import build_mnist_recipe

__all__ = [
    "build_bipedal_recipe",
    "build_board_recipe",
    "build_cartpole_recipe",
    "build_language_recipe",
    "build_mnist_recipe",
    "register_bipedal_recipe_targets",
    "register_cartpole_recipe_targets",
]
