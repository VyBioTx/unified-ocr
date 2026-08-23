"""unified-ocr —— 统合 PaddleOCR-VL / HunyuanOCR / GLM-OCR 的 macOS 统一 OCR 框架。

快速开始::

    from unified_ocr import OCR

    ocr = OCR()
    result = ocr.run("scan.png", engines=["glm-ocr", "paddleocr-vl"])
    for r in result:
        print(r.engine, r.text[:80])
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from . import backends  # noqa: F401  导入即注册所有引擎
from .base import EngineSpec, OCRBackend
from .models import OCRResult
from .registry import (
    create_backend,
    get_backend_class,
    has_engine,
    list_engines,
)

__version__ = "0.1.0"

__all__ = [
    "OCR",
    "OCRBackend",
    "OCRResult",
    "EngineSpec",
    "list_engines",
    "create_backend",
    "get_backend_class",
    "has_engine",
    "__version__",
]


class OCR:
    """统一 OCR 门面：多引擎识别与结果选择。"""

    def __init__(self, default_model_overrides: dict[str, str] | None = None) -> None:
        self._overrides: dict[str, str] = default_model_overrides or {}

    # -- 引擎信息 ----------------------------------------------------------
    @staticmethod
    def engines() -> list[EngineSpec]:
        return list_engines()

    # -- 识别 --------------------------------------------------------------
    def run(
        self,
        image: str | Path | Any,
        engines: Iterable[str] | str = ("glm-ocr", "paddleocr-vl"),
        **engine_options: Any,
    ) -> list[OCRResult]:
        """在指定引擎上依次识别，返回各引擎的 OCRResult 列表。

        Args:
            image: 图像路径或 PIL.Image / numpy 数组。
            engines: 引擎 id 或 id 列表；默认 mlx-vlm 两引擎（无需 GGUF）。
            engine_options: 透传给每个后端构造器的关键字参数
                （如 model=、prompt=、max_tokens=）。按引擎分别传参时，可用
                形如 {"glm-ocr": {...}} 的嵌套 dict（见 ``run_mapped``）。
        """
        if isinstance(engines, str):
            engines = [engines]
        results: list[OCRResult] = []
        for engine_id in engines:
            kwargs = dict(engine_options)
            if self._overrides.get(engine_id):
                kwargs.setdefault("model", self._overrides[engine_id])
            backend = create_backend(engine_id, **kwargs)
            with backend:
                results.append(backend.recognize(image))
        return results

    def run_mapped(
        self,
        image: str | Path | Any,
        engine_configs: dict[str, dict[str, Any]],
    ) -> dict[str, OCRResult]:
        """每个引擎独立配置后识别，返回 {engine_id: OCRResult}。"""
        out: dict[str, OCRResult] = {}
        for engine_id, kwargs in engine_configs.items():
            backend = create_backend(engine_id, **kwargs)
            with backend:
                out[engine_id] = backend.recognize(image)
        return out