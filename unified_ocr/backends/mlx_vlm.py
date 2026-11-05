"""基于 mlx-vlm 的 MLX 后端适配。

覆盖两个引擎：
  * glm-ocr  —— 智谱 GLM-OCR（0.9B encoder-decoder VLM）
  * paddleocr-vl —— PaddleOCR-VL 1.5（文档理解 VLM）

两者都能通过 mlx-vlm 在 Apple Silicon 上以 MLX/Metal 原生推理，
无需转成 GGUF。mlx-vlm 内部走 `mlx_vlm.load()` -> `mlx_vlm.generate()`。

依赖::

    pip install mlx mlx-lm "transformers>=4.45"
    pip install "mlx-vlm @ git+https://github.com/Blaizzy/mlx-vlm.git"  # 必须 git 安装

注意：GLM-OCR 的 mlx 部署（官方 examples/mlx-deploy）要求 mlx-vlm 装于
transformers>=5.0.0rc 的环境；PaddleOCR-VL 架构在 mlx-vlm 主分支已内置。
两个引擎建议使用同一 mlx-vlm 环境（README「环境」一节有说明）。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from ..base import EngineSpec, OCRBackend, generation_text
from ..models import OCRResult
from ..registry import register

log = logging.getLogger(__name__)

#: 默认识别提示词。模型为中文语料训练，中文提示更贴合。
DEFAULT_PROMPT = "请识别图片中的所有文字内容，保留原有排版结构，输出为 Markdown 格式。"


@register
class MLXVLMBackend(OCRBackend):
    """mlx-vlm 通用后端：GLM-OCR 与 PaddleOCR-VL 共用一套加载/生成逻辑。"""

    spec = EngineSpec(
        id="glm-ocr",
        display_name="GLM-OCR (mlx-vlm)",
        model_hint="mlx-community/GLM-OCR-bf16",
        accelerator="MLX (Metal)",
        license="Apache-2.0 (code) / MIT (model)",
        repo="https://github.com/zai-org/GLM-OCR",
        requires_mlx_vlm=True,
    )

    def __init__(self, model: str | None = None, prompt: str = DEFAULT_PROMPT,
                 max_tokens: int = 4096, **kwargs: Any) -> None:
        super().__init__(model=model, **kwargs)
        self.prompt = prompt
        self.max_tokens = max_tokens
        self._model = None
        self._processor = None

    def load(self) -> "MLXVLMBackend":
        if self._loaded:
            return self
        try:
            from mlx_vlm import load  # type: ignore
        except ImportError as exc:  # pragma: no cover - 依赖缺失路径
            raise RuntimeError(
                "缺少 mlx-vlm。请先安装："
                "pip install mlx mlx-lm \"transformers>=4.45\" 且 "
                "pip install \"mlx-vlm @ git+https://github.com/Blaizzy/mlx-vlm.git\""
            ) from exc
        log.info("加载模型 %s（首次运行会下载权重）", self.model)
        self._model, self._processor = load(self.model)
        cfg = getattr(self._model, "config", None)
        iti = getattr(cfg, "image_token_index", None) if cfg is not None else None
        itid = getattr(cfg, "image_token_id", None) if cfg is not None else None
        if cfg is not None and iti is None and itid is not None:
            cfg.image_token_index = itid
            log.info("补全 image_token_index=%s（取自 image_token_id）", itid)
        return super().load()

    def close(self) -> None:
        self._model = None
        self._processor = None
        super().close()

    def recognize(self, image: str | Path | Any, **options: Any) -> OCRResult:
        """调用 mlx-vlm，返回统一 OCRResult。

        options 可覆盖 prompt / max_tokens。
        """
        if not self._loaded:
            self.load()

        prompt = options.pop("prompt", self.prompt)
        max_tokens = options.pop("max_tokens", self.max_tokens)
        if options:
            log.warning("未能识别的选项被忽略: %s", sorted(options))

        # 本地适配：GLM-OCR 需要手工拼 prompt/像素输入，见 _run_generate
        text = self._run_generate(str(image), prompt, int(max_tokens))
        return OCRResult.from_text(
            engine=self.spec.id,
            text=text,
            model=self.model,
            prompt=prompt,
        )
    def _run_generate(self, image_path: str, prompt: str, max_tokens: int) -> str:
        """引擎相关多模态输入构造后推理，返回纯文本。"""
        from mlx_vlm import generate  # type: ignore
        model_type = getattr(getattr(self._model, "config", None), "model_type", "")
        if model_type == "glm_ocr":
            text = self._build_glm_prompt(prompt)
            import numpy as np
            import mlx.core as mx
            proc = self._processor(images=[image_path], text=text, return_tensors="np")
            ids = mx.array(np.asarray(proc["input_ids"], dtype=np.int32))
            pix = mx.array(np.asarray(proc["pixel_values"], dtype=np.float32))
            attn = mx.array(np.asarray(proc["attention_mask"], dtype=np.int32))
            grid = mx.array(np.asarray(proc["image_grid_thw"], dtype=np.int32))
            res = generate(
                self._model,
                self._processor,
                prompt=text,
                input_ids=ids,
                pixel_values=pix,
                mask=attn,
                image_grid_thw=grid,
                max_tokens=max_tokens,
                verbose=False,
            )
        else:
            p = prompt if "<image>" in prompt else "<image>" + prompt
            res = generate(
                self._model,
                self._processor,
                p,
                image=image_path,
                max_tokens=max_tokens,
            )
        # mlx-vlm 新版返回 GenerationResult(.text)，旧版直接返回 str
        return generation_text(res)

    def _build_glm_prompt(self, user_text: str) -> str:
        """GLM-OCR 对话模板：固定 begin/end 图像块包裹占位符 + 用户指令 + assistant 结尾。"""
        user_text = (user_text or "").replace("<image>", "").strip()
        if not user_text:
            user_text = "请识别图片中的所有文字内容，保留原有排版结构，输出为 Markdown 格式。"
        return (
            "[gMASK]<sop><|user|>\n"
            "<|begin_of_image|><|image|><|end_of_image|>\n"
            + user_text
            + "<|assistant|>"
        )



# ---------------------------------------------------------------------------
# PaddleOCR-VL 与 GLM-OCR 的差异化注册
# ---------------------------------------------------------------------------
# 共用 MLXVLMBackend 的加载/生成逻辑，仅 spec 与默认模型不同。
# 直接构造两个独立子类，避免装饰器重复注册同一个类。

def _make_mlx_vlm_engine(
    engine_id: str,
    display_name: str,
    default_model: str,
    repo: str,
    license_: str,
) -> type[MLXVLMBackend]:
    # 注意：类体作用域不解析闭包变量，因此先建空类再赋值属性。
    class _Engine(MLXVLMBackend):
        pass

    _Engine.__name__ = f"{engine_id.replace('-', '_').title()}Backend"
    _Engine.default_model = default_model
    _Engine.spec = EngineSpec(
        id=engine_id,
        display_name=display_name,
        model_hint=default_model,
        accelerator="MLX (Metal)",
        license=license_,
        repo=repo,
        requires_mlx_vlm=True,
    )
    return _Engine


register(_make_mlx_vlm_engine(
    engine_id="paddleocr-vl",
    display_name="PaddleOCR-VL (mlx-vlm)",
    default_model="PaddlePaddle/PaddleOCR-VL",
    repo="https://github.com/PaddlePaddle/PaddleOCR",
    license_="Apache-2.0",
))
