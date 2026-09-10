"""Registry of causal estimators."""

from __future__ import annotations

from typing import Any, Callable

import pandas as pd

from . import did, iv, psm, rdd, synthetic_control
from .base import IdentificationError, MethodOutput, MethodSpec

Estimator = Callable[[pd.DataFrame, dict[str, str | None], dict[str, Any]], MethodOutput]

_MODULES = (did, rdd, psm, iv, synthetic_control)

SPECS: dict[str, MethodSpec] = {m.SPEC.id: m.SPEC for m in _MODULES}
ESTIMATORS: dict[str, Estimator] = {m.SPEC.id: m.estimate for m in _MODULES}

# Presentation order: strongest/most common designs first.
ORDER = ("did", "rdd", "iv", "synthetic_control", "psm")


def run(method: str, df: pd.DataFrame, roles: dict, options: dict) -> MethodOutput:
    if method not in ESTIMATORS:
        raise IdentificationError(f"Unknown method '{method}'.")
    return ESTIMATORS[method](df, roles, options)


__all__ = ["SPECS", "ESTIMATORS", "ORDER", "run", "IdentificationError", "MethodOutput", "MethodSpec"]
