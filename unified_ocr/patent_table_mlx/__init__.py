"""MLX-native PP-StructureV3 components for macOS Apple Silicon (Metal GPU).

Weights come from `plaincompute/ppocr-mlx <https://huggingface.co/plaincompute/ppocr-mlx>`_,
an MLX conversion of the official PaddleX / PP-Structure checkpoints.  Each model
is a structurally faithful MLX port whose parameter names match the converted
safetensors keys, so weights load directly.

Implemented so far:
  * ``SLANeXt`` (SLANeXt_wired / SLANeXt_wireless) — table structure recognition
"""

from __future__ import annotations

from .slanext import SLANeXt, SLANeXtConfig
from .weight_loader import (
    SLANEXT_CHARACTER_DICT,
    build_character_list,
    decode_structure,
    decode_structure_tokens,
    load_mlx_weights,
    load_slanext,
)

__all__ = [
    "SLANeXt",
    "SLANeXtConfig",
    "SLANEXT_CHARACTER_DICT",
    "build_character_list",
    "decode_structure",
    "decode_structure_tokens",
    "load_mlx_weights",
    "load_slanext",
    "PatentTableMLXPipeline",
    "PatentPipelineMLXConfig",
]


def __getattr__(name):
    # Lazy: the pipeline pulls in PaddleX, which is an optional dependency.
    if name in ("PatentTableMLXPipeline", "PatentPipelineMLXConfig"):
        from . import pipeline as _pipeline

        return getattr(_pipeline, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
