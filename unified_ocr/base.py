"""OCR 后端抽象基类。

任何引擎适配层只需实现 `recognize()`，并可选实现 `close()` 释放资源。
后端加载是惰性的：`load()` 在首次识别前调用，避免 CLI 只列引擎时
也拉取数 GB 权重。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .models import OCRResult


@dataclass
class EngineSpec:
    """引擎静态信息，供 `unified-ocr list-engines` 与文档展示。"""

    id: str                    # CLI 中使用的引擎标识，如 "glm-ocr"
    display_name: str          # 展示名，如 "GLM-OCR (mlx-vlm)"
    model_hint: str            # 默认模型路径 / HF id 提示
    accelerator: str           # "MLX (Metal)" / "Metal via llama.cpp" / "CPU"
    license: str = "unknown"   # 模型 / 仓库许可证
    repo: str = ""             # 上游开源仓库 URL
    requires_gguf: bool = False
    requires_mlx_vlm: bool = False


class OCRBackend(ABC):
    """统一引擎适配层接口。"""

    #: 静态引擎元信息，子类必须填充
    spec: EngineSpec

    #: 供 load() 使用的默认模型路径（可被构造参数覆盖）
    default_model: str = ""

    def __init__(self, model: str | None = None, **kwargs: Any) -> None:
        self.model = model or self.default_model
        self.kwargs = kwargs
        self._loaded = False

    # -- 生命周期 ---------------------------------------------------------
    def load(self) -> "OCRBackend":
        """惰性加载模型权重；幂等。子类覆写并在最后调用 super().load()。"""
        self._loaded = True
        return self

    def close(self) -> None:  # pragma: no cover - 可选释放
        """释放模型资源。默认无操作。"""
        self._loaded = False

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    # -- 识别 -------------------------------------------------------------
    @abstractmethod
    def recognize(self, image: str | Path | Any, **options: Any) -> OCRResult:
        """识别一张图像，返回统一 OCRResult。

        Args:
            image: 图像路径，或已解码的 PIL.Image / numpy 数组。
            options: 引擎特有参数（如 prompt、max_tokens），透传给适配层。
        """

    def __enter__(self) -> "OCRBackend":
        self.load()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()