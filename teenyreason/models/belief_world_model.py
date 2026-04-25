"""Compatibility facade for the belief world-model package."""

from . import belief as _belief
from .belief import *  # noqa: F401,F403

__all__ = _belief.__all__
