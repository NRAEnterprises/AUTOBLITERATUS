"""AUTOBLITERATUS: post-ablation architecture-aware quantization-calibrated
capability tuning for OBLITERATUS.

Every tuning type is a separate module under ``autobliteratus.tuning``.
Every tuning type is independently toggleable at runtime. The orchestrator
in ``autobliteratus.tuning.tuning_orchestrator`` reads the toggle registry
and runs only the tuning types whose toggles are on.
"""

from __future__ import annotations

__version__ = "0.1.0"
__author__ = "AUTOBLITERATUS contributors"
__license__ = "AGPL-3.0-or-later"

__all__ = ["__version__", "__author__", "__license__"]
