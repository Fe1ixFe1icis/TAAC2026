"""TokenFormer model for p0p1p2-only ablation."""

import sys
from pathlib import Path
import importlib.util

# Load tokenformer model module directly to avoid circular import
_tokenformer_dir = Path(__file__).resolve().parent.parent / "tokenformer"
_tokenformer_model_path = _tokenformer_dir / "model.py"

_spec = importlib.util.spec_from_file_location(
    "_tokenformer_model_internal",
    _tokenformer_model_path,
)
_tokenformer_module = importlib.util.module_from_spec(_spec)
sys.modules["_tokenformer_model_internal"] = _tokenformer_module
_spec.loader.exec_module(_tokenformer_module)

# Re-export all symbols
ModelInput = _tokenformer_module.ModelInput
TokenFormerModel = _tokenformer_module.TokenFormerModel
PerFieldResSwiGLUTokenizer = _tokenformer_module.PerFieldResSwiGLUTokenizer
RMSNorm = _tokenformer_module.RMSNorm
SwiGLUFFN = _tokenformer_module.SwiGLUFFN
SparsePerTokenMoE = _tokenformer_module.SparsePerTokenMoE

# Import from baseline model
_baseline_path = Path(__file__).resolve().parent.parent / "baseline"
if str(_baseline_path) not in sys.path:
    sys.path.insert(0, str(_baseline_path))
from model import (
    RotaryEmbedding,
    apply_rope_to_tensor,
    rotate_half,
)

__all__ = [
    "ModelInput",
    "TokenFormerModel",
    "PerFieldResSwiGLUTokenizer",
    "RMSNorm",
    "SwiGLUFFN",
    "SparsePerTokenMoE",
    "RotaryEmbedding",
    "apply_rope_to_tensor",
    "rotate_half",
]
