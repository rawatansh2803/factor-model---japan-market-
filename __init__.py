"""Japanese equity ML factor model package.

Implements a thesis-oriented research pipeline for benchmark-relative,
long-only factor investing using value and momentum features.
"""

from .config import ResearchConfig
from .pipeline import run_pipeline

__all__ = ["ResearchConfig", "run_pipeline"]
