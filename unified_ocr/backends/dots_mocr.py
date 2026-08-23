"""dots.mocr / dots.ocr 后端：transformers 原生推理（Apple Silicon 走 MPS）。

dots.ocr（2026.03 更名 dots.mocr，仓库 rednote-hilab/dots.mocr，权重
`rednote-hilab/dots.mocr` / `rednote-hilab/dots.mocr-svg`）是基于 1.7B LLM
的多语言文档解析 VLM（文档解析、结构化图形转 SVG、网页截图解析、场景文字检测）。

模型架构说明
------------
dots 使用自定义 ``dots.vit`` 视觉编码器（NaViT 架构）对齐 Qwen2.5-1.5B 语言
模型，权重以 ``trust_remote_code=True`` 的自定义代码发布，官方推理路径为：

* vLLM 部署（推荐，GPU）：vllm/vllm-openai 镜像 + OpenAI 兼容 API；
* transformers 原生（``use_hf=True``）：AutoModelForCausalLM + AutoProcessor
  （官方示例默认 flash-attn + CUDA）。

本后端实现 macOS 上的 transformers 原生推理：
* 视觉编码器加载依赖模型仓库内的自定义 modeling 代码（trust_remote_code）；
* ``attn_implementation="eager"`` —— 不依赖 flash-attn（macOS 无 CUDA）；
* 优先 MPS（Apple Silicon Metal），否则 CPU；bf16（MPS）/ fp32（CPU）；
* 图像预处理（smart_resize / fetch_image）与消息组装复用官方 qwen_vl_utils
  流程，与 dots_ocr 仓库 ``demo/demo_hf.py`` 一致。

依赖::

    pip install torch transformers qwen-vl-utils   # 或 pip install ".[dots]"
    # transformers 建议 ==4.56.1（dots 官方锁定版本）

MLX 支持见 ``dots_mocr_mlx`` 引擎（``dots_mocr_mlx.py``）。
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

from ..base import EngineSpec, OCRBackend
from ..models import BBox, OCRBlock, OCRLine, OCRResult
from ..registry import register

log = logging.getLogger(__name__)

#: 默认模型路径 —— 与 dots 官方 tools/download_model.py 的落盘目录一致
#: （脚本下载到仓库根 weights/DotsMOCR，也可 HF id rednote-hilab/dots.mocr）。
DEFAULT_MODEL = "./weights/DotsMOCR"

#: 默认提示词：纯文本 OCR（与框架其它引擎一致，输出整页文本）。
DEFAULT_PROMPT = "Extract the text content from this image."

#: 文档版面解析提示词（dots 旗舰能力）：输出单个 JSON 对象，
#: 元素含 bbox / category / text，配合 layout_json_to_blocks 使用。
PROMPT_LAYOUT_ALL_EN = """Please output the layout information from the PDF image, including each layout element's bbox, its category, and the corresponding text content within the bbox.

1. Bbox format: [x1, y1, x2, y2]

2. Layout Categories: The possible categories are ['Caption', 'Footnote', 'Formula', 'List-item', 'Page-footer', 'Page-header', 'Picture', 'Section-header', 'Table', 'Text', 'Title'].

3. Text Extraction & Formatting Rules:
    - Picture: For the 'Picture' category, the text field should be omitted.
    - Formula: Format its text as LaTeX.
    - Table: Format its text as HTML.
    - All Others (Text, Title, etc.): Format their text as Markdown.

4. Constraints:
    - The output text must be the original text from the image, with no translation.
    - All layout elements must be sorted according to human reading order.

5. Final Output: The entire output must be a single JSON object.
"""

#: dots 官方 layout category → unified-ocr BLOCK_TYPES 映射
_CATEGORY_TO_KIND = {
    "Table": "table",
    "Formula": "formula",
    "Title": "heading",
    "Section-header": "heading",
    "Picture": "figure",
    "List-item": "list",
    "Caption": "text",
    "Text": "text",
    "Page-header": "text",
    "Page-footer": "text",
    "Footnote": "text",
    "Other": "other",
}


def layout_json_to_blocks(
    text: str,
    image_size: Optional[tuple[int, int]] = None,
) -> list[OCRBlock]:
    """把 dots 的版面解析 JSON 输出转成统一 OCRBlock 列表。

    dots 的 ``prompt_layout_all_en`` 输出形如::

        [
          {"bbox": [x1, y1, x2, y2], "category": "Title",
           "text": "实验报告", "score": 0.9},
          ...
        ]

    bbox 为原图像素坐标（dots 官方 parser 会把它换算回原图），这里归一化到
    [0, 1]（相对图像宽高），与 unified-ocr 的 BBox 约定一致。

    Args:
        text: 模型原始输出（单个 JSON 对象）。
        image_size: 原图 (width, height)；提供时 bbox 归一化，否则 bbox=None。

    Returns:
        统一 OCRBlock 列表；解析失败返回空列表。
    """
    try:
        cells = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return []
    if not isinstance(cells, list):
        return []

    blocks: list[OCRBlock] = []
    w, h = image_size if image_size else (None, None)
    for cell in cells:
        if not isinstance(cell, dict):
            continue
        category = str(cell.get("category", "Other"))
        kind = _CATEGORY_TO_KIND.get(category, "other")
        cell_text = cell.get("text", "")
        if isinstance(cell_text, str):
            cell_text = cell_text.strip()

        bbox: Optional[BBox] = None
        raw = cell.get("bbox")
        if raw and w and h and len(raw) == 4:
            x0, y0, x1, y1 = (float(v) for v in raw)
            bbox = BBox(
                x0=max(0.0, min(1.0, x0 / w)),
                y0=max(0.0, min(1.0, y0 / h)),
                x1=max(0.0, min(1.0, x1 / w)),
                y1=max(0.0, min(1.0, y1 / h)),
            )

        blocks.append(
            OCRBlock(
                kind=kind,
                text=cell_text,
                bbox=bbox,
                confidence=cell.get("score"),
                lines=[OCRLine(text=cell_text, bbox=bbox, confidence=cell.get("score"))],
            )
        )
    return blocks


@register
class DotsTransformersBackend(OCRBackend):
    """dots.mocr 的 transformers 原生后端（macOS：MPS / CPU）。

    与 dots 官方 ``demo_hf.py`` 推理流程一致，但移除 flash-attn/CUDA 依赖：
    eager attention + MPS（Apple Silicon）/ CPU 设备。
    """

    spec = EngineSpec(
        id="dots-mocr",
        display_name="dots.mocr (transformers/MPS)",
        model_hint="rednote-hilab/dots.mocr (或 ./weights/DotsMOCR)",
        accelerator="MPS (Metal) via PyTorch",
        license="dots.ocr LICENSE AGREEMENT (custom)",
        repo="https://github.com/rednote-hilab/dots.mocr",
    )

    #: 默认模型路径 —— 与 dots 官方 tools/download_model.py 落盘目录一致
    default_model = DEFAULT_MODEL

    def __init__(
        self,
        model: str | None = None,
        prompt: str = DEFAULT_PROMPT,
        max_new_tokens: int = 8192,
        device: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(model=model, **kwargs)
        self.prompt = prompt
        self.max_new_tokens = max_new_tokens
        self.device = device
        self._model = None
        self._processor = None

    def load(self) -> "DotsTransformersBackend":
        if self._loaded:
            return self
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoProcessor
            from qwen_vl_utils import process_vision_info
        except ImportError as exc:  # pragma: no cover - 依赖缺失路径
            raise RuntimeError(
                "缺少 dots.mocr 推理依赖。请安装：pip install \".[dots]\""
                "（torch + transformers==4.56.1 + qwen-vl-utils）"
            ) from exc

        # 设备：优先 MPS（Apple Silicon），可用 CUDA 时也用；否则 CPU。
        device = self.device
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self._device = device
        # bf16 在 MPS/CUDA 上可用；CPU 用 fp32 保证数值稳定。
        dtype = torch.bfloat16 if device in ("mps", "cuda") else torch.float32
        log.info("加载 dots.mocr processor（%s，trust_remote_code）...", self.model)
        self._processor = AutoProcessor.from_pretrained(
            self.model, trust_remote_code=True, use_fast=True
        )
        log.info("加载 dots.mocr 模型（device=%s, eager attention）...", device)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model,
            attn_implementation="eager",  # 不依赖 flash-attn，macOS 可跑
            torch_dtype=dtype,
            trust_remote_code=True,
        ).to(device)
        self._process_vision_info = process_vision_info
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

        import torch

        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": str(image)},
                {"type": "text", "text": prompt},
            ],
        }]
        text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = self._process_vision_info(messages)
        inputs = self._processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self._device)
        with torch.no_grad():
            generated_ids = self._model.generate(
                **inputs, max_new_tokens=int(max_new_tokens), do_sample=False
            )
        trimmed = [
            out[len(in_ids):]
            for in_ids, out in zip(inputs.input_ids, generated_ids)
        ]
        response = self._processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        result = OCRResult.from_text(
            engine=self.spec.id,
            text=response,
            model=self.model,
            prompt=prompt,
            device=self._device,
        )
        # 版面解析模式：把 JSON 输出转成结构化 blocks。
        if response.strip().startswith("[") or response.strip().startswith("{"):
            blocks = layout_json_to_blocks(response)
            if blocks:
                result.blocks = blocks
        return result
