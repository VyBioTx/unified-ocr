"""HunyuanOCR 后端：transformers 原生推理（Apple Silicon 走 PyTorch MPS）。

HunyuanOCR（Tencent-Hunyuan/HunyuanOCR，v1.0 分支）是 1B 轻量 OCR VLM，
官方推荐 transformers 的 ``HunYuanVLForConditionalGeneration`` 原生推理，
无需 GGUF/llama.cpp。权重从 ModelScope 获取后需先运行两个修复脚本
（见 README「HunyuanOCR 权重修复」）。

依赖::

    pip install torch transformers accelerate   # 或 pip install ".[hunyuan]"
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from ..base import EngineSpec, OCRBackend
from ..models import OCRResult
from ..registry import register

log = logging.getLogger(__name__)

DEFAULT_PROMPT = (
    "请提取文档图片中正文的所有信息用 markdown 格式表示，"
    "其中页眉、页脚部分忽略，表格用 HTML 格式表达，"
    "文档中公式用 latex 格式表示，按照阅读顺序组织进行解析。"
)


def clean_repeated_substrings(text: str) -> str:
    """清理长文本末尾的重复子串（官方推荐后处理）。"""
    n = len(text)
    if n < 8000:
        return text
    for length in range(2, n // 10 + 1):
        candidate = text[-length:]
        count = 0
        i = n - length
        while i >= 0 and text[i:i + length] == candidate:
            count += 1
            i -= length
        if count >= 10:
            return text[:n - length * (count - 1)]
    return text


@register
class HunyuanTransformersBackend(OCRBackend):
    """HunyuanOCR 的 transformers 原生后端（MPS）。"""

    spec = EngineSpec(
        id="hunyuanocr",
        display_name="HunyuanOCR-1.0 (transformers)",
        model_hint="Tencent-Hunyuan/HunyuanOCR",
        accelerator="MPS (Metal) via PyTorch",
        license="Tencent Hunyuan Community License",
        repo="https://github.com/Tencent-Hunyuan/HunyuanOCR",
    )

    def __init__(
        self,
        model: str | None = None,
        prompt: str = DEFAULT_PROMPT,
        max_new_tokens: int = 16384,
        **kwargs: Any,
    ) -> None:
        super().__init__(model=model, **kwargs)
        self.prompt = prompt
        self.max_new_tokens = max_new_tokens
        self._model = None
        self._processor = None

    def load(self) -> "HunyuanTransformersBackend":
        if self._loaded:
            return self
        try:
            import torch

            from transformers import AutoProcessor, HunYuanVLForConditionalGeneration
        except ImportError as exc:  # pragma: no cover - 依赖缺失路径
            raise RuntimeError(
                "缺少 transformers 的 HunYuanVL 支持。请安装：pip install \".[hunyuan]\""
                "（torch + transformers>=5 + accelerate）"
            ) from exc
        log.info("加载 HunyuanOCR processor ...")
        self._processor = AutoProcessor.from_pretrained(
            self.model, use_fast=False, trust_remote_code=True
        )
        log.info("加载 HunyuanOCR 模型 ...")
        self._model = HunYuanVLForConditionalGeneration.from_pretrained(
            self.model,
            attn_implementation="eager",
            dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        return super().load()

    def close(self) -> None:
        self._model = None
        self._processor = None
        super().close()

    def recognize(self, image: str | Path | Any, **options: Any) -> OCRResult:
        if not self._loaded:
            self.load()
        assert self._model is not None and self._processor is not None

        prompt = options.pop("prompt", self.prompt)
        max_new_tokens = options.pop("max_tokens", self.max_new_tokens)
        if options:
            log.warning("未能识别的选项被忽略: %s", sorted(options))

        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": str(image)},
                {"type": "text", "text": prompt},
            ],
        }]
        texts = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._processor(
            text=texts, images=str(image), padding=True, return_tensors="pt"
        )
        device = next(self._model.parameters()).device
        inputs = inputs.to(device)
        import torch

        with torch.no_grad():
            generated_ids = self._model.generate(
                **inputs, max_new_tokens=int(max_new_tokens), do_sample=False
            )
        input_ids = inputs.input_ids
        trimmed = [out[len(ids):] for ids, out in zip(input_ids, generated_ids)]
        text = self._processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        text = clean_repeated_substrings(text)
        return OCRResult.from_text(
            engine=self.spec.id,
            text=text,
            model=self.model,
            prompt=prompt,
        )
