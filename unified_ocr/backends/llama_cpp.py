"""基于 llama.cpp（llama-cpp-python）的 HunyuanOCR-1.0 后端。

背景：HunyuanOCR（Tencent-Hunyuan/HunyuanOCR，v1.0 分支）官方不提供
MLX 支持，唯一能在本地（含 Apple Silicon）运行的路径是官方推出的
llama.cpp GGUF 部署。llama.cpp 本身支持 Metal 加速，因此仍可享受
GPU 加速，只是不走 MLX。

requirements::

    pip install llama-cpp-python   # macOS 上默认即编译启用 Metal

需要两个文件（需用户按官方文档准备，见 README「模型准备」）：
  * 语言模型 GGUF（如 HunyuanOCR-1.0-Q4_K_M.gguf）
  * 视觉编码器 mmproj（llama.cpp 多模态投影文件）

注意：HunyuanOCR 的 macOS/llama.cpp 路径属于官方未显式验证的推断路径；
若你的 GLM-OCR / PaddleOCR-VL 已可用，优先使用 mlx-vlm 引擎。
"""

from __future__ import annotations

import base64
import logging
import os
from pathlib import Path
from typing import Any

from ..base import EngineSpec, OCRBackend
from ..models import OCRResult
from ..registry import register

log = logging.getLogger(__name__)

DEFAULT_PROMPT = "识别图片中的所有文字内容，保留原有排版结构，输出为 Markdown 格式。"


def _image_data_url(path: str | Path) -> str:
    """读取图像并编码为 data URL（llama.cpp 视觉输入格式）。"""
    import mimetypes

    p = Path(path)
    mime = mimetypes.guess_type(p.name)[0] or "image/jpeg"
    b64 = base64.b64encode(p.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{b64}"


@register
class LlamaCppHunyuanOCRBackend(OCRBackend):
    """HunyuanOCR-1.0 的 llama.cpp 后端（GGUF + mmproj）。"""

    spec = EngineSpec(
        id="hunyuanocr",
        display_name="HunyuanOCR-1.0 (llama.cpp)",
        model_hint="HunyuanOCR-1.0-*.gguf + mmproj",
        accelerator="Metal via llama.cpp",
        license="Tencent Hunyuan Community License",
        repo="https://github.com/Tencent-Hunyuan/HunyuanOCR",
        requires_gguf=True,
    )

    def __init__(
        self,
        model: str | None = None,
        mmproj: str | None = None,
        prompt: str = DEFAULT_PROMPT,
        max_tokens: int = 4096,
        chat_format: str | None = None,
        n_ctx: int = 8192,
        **kwargs: Any,
    ) -> None:
        super().__init__(model=model, **kwargs)
        self.mmproj = mmproj or os.environ.get("HUNYUANOCR_MMPROJ", "")
        if not self.mmproj:
            raise ValueError(
                "hunyuanocr 后端需要 mmproj（视觉投影）路径："
                "构造参数 mmproj=... 或环境变量 HUNYUANOCR_MMPROJ"
            )
        self.prompt = prompt
        self.max_tokens = max_tokens
        self.chat_format = chat_format
        self.n_ctx = n_ctx
        self._llm = None

    def load(self) -> "LlamaCppHunyuanOCRBackend":
        if self._loaded:
            return self
        try:
            from llama_cpp import Llama  # type: ignore
        except ImportError as exc:  # pragma: no cover - 依赖缺失路径
            raise RuntimeError(
                "缺少 llama-cpp-python。请安装：pip install llama-cpp-python"
                "（macOS 默认启用 Metal 加速）"
            ) from exc
        log.info("加载 GGUF %s（mmproj=%s）", self.model, self.mmproj)
        self._llm = Llama(
            model_path=self.model,
            mmproj=self.mmproj,
            chat_format=self.chat_format,
            n_ctx=self.n_ctx,
            verbose=False,
        )
        return super().load()

    def close(self) -> None:
        self._llm = None
        super().close()

    def recognize(self, image: str | Path | Any, **options: Any) -> OCRResult:
        if not self._loaded:
            self.load()
        assert self._llm is not None

        prompt = options.pop("prompt", self.prompt)
        max_tokens = options.pop("max_tokens", self.max_tokens)
        if options:
            log.warning("未能识别的选项被忽略: %s", sorted(options))

        out = self._llm.create_chat_completion(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": _image_data_url(image)}},
                    ],
                }
            ],
            max_tokens=int(max_tokens),
        )
        text = out["choices"][0]["message"]["content"] or ""
        return OCRResult.from_text(
            engine=self.spec.id,
            text=text,
            model=self.model,
            mmproj=self.mmproj,
            prompt=prompt,
        )