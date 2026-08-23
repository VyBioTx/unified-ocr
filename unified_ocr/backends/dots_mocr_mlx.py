"""dots.mocr 的 MLX 后端接入点。

现状（2026-08 核对）
--------------------
dots.mocr 的自定义视觉编码器 ``dots.vit``（NaViT 架构）不在 mlx-vlm 0.3.3
已支持的架构列表（qwen2_vl / qwen2_5_vl / glm4v / internvl / llava /
idefics / paligemma / phi3_v / florence2 / gemma3 / mllama / pixtral /
molmo / smolvlm / …）中，GitHub 上也暂无社区 MLX 转换（mlx-community 无
dots.mocr 权重）。因此 mlx-vlm 目前**无法直接加载** dots.mocr 权重。

本后端保留引擎注册与 CLI 集成（``unified-ocr list-engines`` 可见），加载时
按以下顺序尝试，全部失败则给出可操作的升级指引：

1. 若本机安装的 mlx-vlm 已内置 dots 架构（上游新增后），直接用
   ``mlx_vlm.load()`` + ``mlx_vlm.generate()`` 推理；
2. 若权重仓库自带 MLX 自定义模型代码（未来 dots 官方或社区提供），动态
   import 并调用；
3. 否则抛出 RuntimeError，说明需要等待 mlx-vlm 支持 dots.vit 或自行转换。

转换可行性备注
--------------
dots.vit 是 NaViT 风格的视觉编码器（训练时用可变分辨率 patch），对齐
Qwen2.5-1.5B LLM。理论上可按 unified-ocr 中 glm-ocr / paddleocr-vl 的
mlx-vlm 集成方式，将权重转换为 MLX 格式并注册新架构，但工作量在 mlx-vlm
模型库的完整实现量级，需在有 mlx 的机器上迭代验证，超出本环境可验证范围。

依赖::

    pip install "mlx-vlm @ git+https://github.com/Blaizzy/mlx-vlm.git"  # 待支持 dots
"""

from __future__ import annotations

import importlib
import logging
from pathlib import Path
from typing import Any

from ..base import EngineSpec, OCRBackend
from ..models import OCRResult
from ..registry import register

log = logging.getLogger(__name__)

#: 与 transformers 后端共用默认提示词。
DEFAULT_PROMPT = "Extract the text content from this image."

#: mlx-vlm 内已内置的 dots 模型模块名（上游未来支持时按此约定注册）。
_MLX_DOTS_MODULES = ("mlx_vlm.models.dots_mocr", "mlx_vlm.models.dotsmocr")


@register
class DotsMocrMLXBackend(OCRBackend):
    """dots.mocr 的 MLX 后端（架构支持前提，见模块 docstring）。

    注册表 / CLI 完整接入；``load()`` 时按模块 docstring 的优先级探测
    mlx-vlm 对 dots.vit 的支持情况，未支持时给出升级指引。
    """

    spec = EngineSpec(
        id="dots-mocr-mlx",
        display_name="dots.mocr (mlx-vlm / MLX-Metal)",
        model_hint="本地 MLX 权重目录（或 HF id）",
        accelerator="MLX (Metal)",
        license="dots.ocr LICENSE AGREEMENT (custom)",
        repo="https://github.com/rednote-hilab/dots.mocr",
        requires_mlx_vlm=True,
    )

    #: MLX 引擎默认模型路径（本地权重目录）
    default_model = ""

    def __init__(
        self,
        model: str | None = None,
        prompt: str = DEFAULT_PROMPT,
        max_tokens: int = 8192,
        **kwargs: Any,
    ) -> None:
        super().__init__(model=model, **kwargs)
        self.prompt = prompt
        self.max_tokens = max_tokens
        self._model = None
        self._processor = None

    # -- 架构支持探测 -----------------------------------------------------
    def _check_mlx_vlm_support(self) -> tuple[bool, str]:
        """探测 mlx-vlm 是否支持 dots.mocr 架构。

        Returns:
            (supported, detail)：supported 为 True 时 detail 说明如何加载；
            否则 detail 说明缺失原因。
        """
        try:
            from mlx_vlm.utils import load as mlx_load  # noqa: F401
        except ImportError:
            return False, "未安装 mlx-vlm"

        # 1) mlx-vlm 主库内置 dots 架构（上游未来支持）
        for mod_name in _MLX_DOTS_MODULES:
            try:
                importlib.import_module(mod_name)
                return True, f"mlx-vlm 已内置 dots 架构（{mod_name}），可直接 mlx_vlm.load()"
            except ImportError:
                continue

        # 2) 权重目录自带 MLX 自定义模型代码
        model = getattr(self, "model", "") or ""
        if model:
            try:
                importlib.import_module("modeling_dots_mocr")
                return True, "权重目录自带 modeling_dots_mocr（MLX 版），可动态加载"
            except ImportError:
                pass
        return (
            False,
            "mlx-vlm 尚未支持 dots.vit（NaViT 自定义视觉编码器）架构，"
            "也无社区 MLX 权重。需等待上游支持或自行转换（见模块 docstring）。",
        )

    def load(self) -> "DotsMocrMLXBackend":
        if self._loaded:
            return self
        supported, detail = self._check_mlx_vlm_support()
        if not supported:
            raise RuntimeError(
                "dots-mocr-mlx 引擎暂不可用：" + detail
            )
        from mlx_vlm import load, generate  # type: ignore
        log.info("加载 dots.mocr MLX 模型 %s", self.model)
        self._model, self._processor = load(self.model)
        return super().load()

    def close(self) -> None:
        self._model = None
        self._processor = None
        super().close()

    def recognize(self, image: str | Path | Any, **options: Any) -> OCRResult:
        if not self._loaded:
            self.load()
        from mlx_vlm import generate  # type: ignore  (load() 已保证依赖存在)

        prompt = options.pop("prompt", self.prompt)
        max_tokens = options.pop("max_tokens", self.max_tokens)
        if options:
            log.warning("未能识别的选项被忽略: %s", sorted(options))

        text = generate(
            self._model,
            self._processor,
            prompt,
            image=str(image),
            max_tokens=int(max_tokens),
        )
        return OCRResult.from_text(
            engine=self.spec.id,
            text=text,
            model=self.model,
            prompt=prompt,
        )
