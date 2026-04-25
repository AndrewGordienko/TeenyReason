"""Compatibility facade for the env-belief package."""

from . import envbelief as _envbelief
from .envbelief import *  # noqa: F401,F403

__all__ = _envbelief.__all__
