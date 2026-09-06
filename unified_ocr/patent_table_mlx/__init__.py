"""MLX-based PP-StructureV3 for macOS Apple Silicon (Metal GPU) acceleration.

这个模块基于 `plaincompute/ppocr-mlx` 的 MLX 权重格式（与 PaddleX 官方权重同源），
实现了 PP-StructureV3 专利表格识别 pipeline 的纯 MLX 推理层，跑在 Metal GPU 上。

架构总览::

    PatentTablePipelineMLX
      ├── PP-DocLayout_plus-L   (版面分析，跳过 / Paddle CPU 后备)
      ├── RT-DETR-L_wired       (表格单元格检测 → ONNX CoreML)
      ├── SLANeXt_wired         (表格结构识别 → MLX native)
      ├── PP-OCRv5_server_det   (文本检测 → ONNX CoreML)
      └── PP-OCRv5_server_rec   (文本识别 → MLX native)

当前实现进度:
  ✅ SLANeXt_wired  — MLX native (SAM-ViT encoder + Attention GRU + autoregressive解码)
  ✅ PP-OCRv5_rec   — MLX native (SVTR encoder + CTC head)
  🔲 RT-DETR-L  — ONNX CoreML (已转换)
  🔲 PP-OCRv5_det — ONNX CoreML (已转换)
  🔲 端到端 pipeline 组装
"""

from __future__ import annotations

from .slanext import SLANeXt, SLANeXtConfig
from .weight_loader import load_paddle_weights_to_mlx

__all__ = [
    "SLANeXt",
    "SLANeXtConfig",
    "load_paddle_weights_to_mlx",
]