"""Re-export PCVRHyFormer from baseline experiment."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_baseline_model_path = Path(__file__).resolve().parent.parent / "baseline" / "model.py"
_spec = importlib.util.spec_from_file_location("baseline_model", _baseline_model_path)
_baseline_module = importlib.util.module_from_spec(_spec)
sys.modules["baseline_model"] = _baseline_module
_spec.loader.exec_module(_baseline_module)

PCVRHyFormer = _baseline_module.PCVRHyFormer
ModelInput = _baseline_module.ModelInput

__all__ = ["PCVRHyFormer", "ModelInput"]
