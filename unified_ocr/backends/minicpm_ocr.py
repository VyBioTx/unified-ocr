"""MiniCPM-OCR 1.0 后端：mlx-vlm 原生 MLX 推理（Apple Silicon）。

OpenBMB 的 1.3B 文档解析模型（SigLIP 视觉编码 + Qwen3.5 混合注意力主干，
24 层里 6 层全注意力、18 层 GatedDeltaNet）。mlx-vlm 主分支已内置
``minicpm_ocr`` 架构，社区提供现成 MLX 权重：

    bf16  nativ-community/MiniCPM-OCR-1.0-bf16   2.60 GB   ← 默认
    4bit  nativ-community/MiniCPM-OCR-1.0-4bit   2.16 GB

注意默认用 bf16：4bit 只量化了 96 个线性层，视觉塔 / 词嵌入 / 递归线性注意力
仍是 bf16，相对 bf16 只省 0.44 GB；而官方 validation.json 里密集页的字符
相似度只有 0.76~0.83。省内存的收益不值得这个精度损失。

依赖: ``pixi run -e mlx ...``（mlx / mlx-lm / mlx-vlm @ main）。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from ..base import EngineSpec, OCRBackend, generation_text
from ..models import OCRResult
from ..registry import register

log = logging.getLogger(__name__)

#: 上游模型卡给的示例提示词（英文训练语料）。
DEFAULT_PROMPT = "Convert this page to markdown."


@register
class MiniCPMOcrMLXBackend(OCRBackend):
    """MiniCPM-OCR 1.0 的 mlx-vlm 后端。"""

    spec = EngineSpec(
        id="minicpm-ocr",
        display_name="MiniCPM-OCR 1.0 (mlx-vlm)",
        model_hint="nativ-community/MiniCPM-OCR-1.0-bf16",
        accelerator="MLX (Metal)",
        license="Apache-2.0",
        repo="https://github.com/OpenBMB/MiniCPM-OCR",
        requires_mlx_vlm=True,
    )
    default_model = "nativ-community/MiniCPM-OCR-1.0-bf16"

    def __init__(
        self,
        model: str | None = None,
        prompt: str = DEFAULT_PROMPT,
        max_tokens: int = 4096,
        temperature: float = 0.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(model=model, **kwargs)
        self.prompt = prompt
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._model = None
        self._processor = None

    def load(self) -> "MiniCPMOcrMLXBackend":
        if self._loaded:
            return self
        try:
            from mlx_vlm import load  # type: ignore
        except ImportError as exc:  # pragma: no cover - 依赖缺失路径
            raise RuntimeError(
                "缺少 mlx-vlm。请先安装：pixi install -e mlx"
                "（mlx-vlm 需从 git 主分支安装，PyPI 版尚未内置 minicpm_ocr 架构）"
            ) from exc
        log.info("加载 MiniCPM-OCR %s（首次运行会下载权重）", self.model)
        self._model, self._processor = load(self.model)
        return super().load()

    def close(self) -> None:
        self._model = None
        self._processor = None
        super().close()

    def recognize(self, image: str | Path | Any, **options: Any) -> OCRResult:
        """识别一张图，返回统一 OCRResult。

        options 可覆盖 ``prompt`` / ``max_tokens`` / ``temperature``。
        """
        if not self._loaded:
            self.load()
        from mlx_vlm import generate  # type: ignore
        from mlx_vlm.prompt_utils import apply_chat_template  # type: ignore

        prompt = options.pop("prompt", self.prompt)
        max_tokens = int(options.pop("max_tokens", self.max_tokens))
        temperature = float(options.pop("temperature", self.temperature))
        if options:
            log.warning("未能识别的选项被忽略: %s", sorted(options))

        # MiniCPM-OCR 的 chat template 必须在外面套一层：
        # 直接传裸 prompt 时图像占位符数量与视觉特征对不上。
        templated = apply_chat_template(
            self._processor,
            getattr(self._model, "config", None),
            prompt,
            num_images=1,
        )
        out = generate(
            self._model,
            self._processor,
            templated,
            image=[str(image)],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        text = generation_text(out).strip()
        return OCRResult.from_text(
            engine=self.spec.id,
            text=text,
            model=self.model,
            prompt=prompt,
        )
