"""Compatibility facade for probe data-collection modules."""

from . import data as _data
from .data import *  # noqa: F401,F403

__all__ = _data.__all__
