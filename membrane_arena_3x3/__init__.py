# Copyright (c) 2026 Jean-Francois Brisson (Spark AI NLP)
# SPDX-License-Identifier: AGPL-3.0-only
"""SYNTHETIC 3x3 density-matrix membrane lattice with independent shields."""

__version__ = "0.3.0"

from .engine import MembraneArena3x3, run_simulation

__all__ = ["MembraneArena3x3", "run_simulation"]
