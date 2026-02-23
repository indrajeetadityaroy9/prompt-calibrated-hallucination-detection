"""Dataset loaders for evaluation."""

from .triviaqa import load_triviaqa
from .squad import load_squad

__all__ = ["load_triviaqa", "load_squad"]
