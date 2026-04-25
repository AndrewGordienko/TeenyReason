"""Compatibility facade for probe latent modules."""

from . import latent as _latent
from .latent import *  # noqa: F401,F403

__all__ = _latent.__all__
