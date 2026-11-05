"""OCR 结果可信度打分（MiniCPM-OCR Jev，MLX）。

Jev 是非自回归的二分类判别模型：Qwen3.5-2B 去掉 ``lm_head``、换成 ``[2, 2048]``
打分头，输入是「原图块 / 候选文本 / 候选文本渲染图」三元组，输出
``P(consistent)``。它不生成文本，所以不能走 ``generate()``；mlx-vlm 为它提供
独立入口::

    from mlx_vlm.models.minicpm_ocr_jev import score
    score(model, processor, orig_image, render_image, text)["prob_1"]

渲染图上游不提供，本模块用 Pillow 做确定性渲染（固定字体、字号、行距），
同一段文本在任何机器上得到同一张图。但 Jev 的分数绝对值对字体和行距敏感：
**阈值必须在部署环境上用自己的业务样本标定**，不要直接套 0.5。

权重（MLX）::

    bf16  nativ-community/MiniCPM-OCR-Jev-bf16   4.43 GB
    4bit  nativ-community/MiniCPM-OCR-Jev-4bit   1.72 GB   ← 默认

4bit 的 logit 与 bf16 最大概率差 0.073（均值 0.010），所以量化后一定要重新标定。
"""

from __future__ import annotations

import logging
import re
import tempfile
from pathlib import Path
from typing import Any

from .models import OCRResult

log = logging.getLogger(__name__)

#: 有 CJK 字形的字体，按平台优先级排列；都找不到时退回 Pillow 内置位图字体。
_FONT_CANDIDATES = (
    "/System/Library/Fonts/PingFang.ttc",              # macOS
    "/System/Library/Fonts/Supplemental/Songti.ttc",   # macOS
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",  # Linux
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
)


def _load_font(size: int) -> Any:
    """按候选列表加载一个字体；Pillow 缺字体时用内置默认字体。"""
    from PIL import ImageFont

    for path in _FONT_CANDIDATES:
        if Path(path).exists():
            try:
                return ImageFont.truetype(path, size)
            except OSError:  # pragma: no cover - 字体文件损坏
                continue
    log.warning("未找到 TrueType 字体，回退 Pillow 默认字体（CJK 可能显示为方块）")
    return ImageFont.load_default()


def _wrap_line(line: str, font: Any, max_width: float) -> list[str]:
    """按词（ASCII）与字符（CJK）混合折行。"""
    tokens = re.findall(r"\s+|\S+", line)
    out: list[str] = []
    current = ""
    for token in tokens:
        candidate = current + token
        if font.getlength(candidate) <= max_width or not current:
            if font.getlength(token) <= max_width:
                current = candidate
                continue
            # 单个超长 token（长公式 / URL）：按字符硬切
            current = ""
            for ch in token:
                if current and font.getlength(current + ch) > max_width:
                    out.append(current)
                    current = ch
                else:
                    current += ch
            continue
        out.append(current.rstrip())
        current = token.lstrip()
    out.append(current.rstrip())
    return out


def render_candidate(
    text: str,
    *,
    width: int = 1024,
    font_size: int = 16,
    line_height: int | None = None,
    padding: int = 24,
    background: str = "white",
) -> Any:
    """把候选文本渲染成一张图（Pillow，确定性输出）。

    Markdown 表格按纯文本渲染 —— 没有等宽对齐，和浏览器渲染的版式不一致，
    因此分数只能在本渲染器下自比，不能和官方 README 的分数对齐。
    """
    from PIL import Image, ImageDraw

    font = _load_font(font_size)
    step = line_height or int(font_size * 1.6)
    max_text_width = max(64, width - 2 * padding)

    lines: list[str] = []
    for raw in (text or "").splitlines() or [""]:
        if not raw.strip():
            lines.append("")
        else:
            lines.extend(_wrap_line(raw, font, max_text_width))

    height = padding * 2 + step * max(1, len(lines))
    image = Image.new("RGB", (width, height), background)
    draw = ImageDraw.Draw(image)
    y = padding
    for line in lines:
        draw.text((padding, y), line, fill="black", font=font)
        y += step
    return image


class JevScorer:
    """MiniCPM-OCR Jev 打分器（MLX）。"""

    default_model = "nativ-community/MiniCPM-OCR-Jev-4bit"

    def __init__(
        self,
        model: str | None = None,
        *,
        render_width: int = 1024,
        font_size: int = 16,
    ) -> None:
        self.model = model or self.default_model
        self.render_width = render_width
        self.font_size = font_size
        self._model = None
        self._processor = None

    def load(self) -> "JevScorer":
        if self._model is not None:
            return self
        try:
            from mlx_vlm import load  # type: ignore
        except ImportError as exc:  # pragma: no cover - 依赖缺失路径
            raise RuntimeError(
                "缺少 mlx-vlm。请先安装：pixi install -e mlx"
                "（Jev 需要含 minicpm_ocr_jev 移植的 mlx-vlm 主分支）"
            ) from exc
        log.info("加载 MiniCPM-OCR Jev %s", self.model)
        self._model, self._processor = load(self.model)
        return self

    def close(self) -> None:
        self._model = None
        self._processor = None

    def score(self, image: str | Path | Any, text: str, rendered: Any = None) -> dict[str, Any]:
        """给一个（图块, 候选文本）对打分，返回 ``{"prob_consistent": 0..1, ...}``。

        ``image`` 是原图或原图块；``rendered`` 可传入自制渲染图，省略时用
        :func:`render_candidate`。候选文本未经任何清洗地传给模型。
        """
        if not text or not text.strip():
            return {"prob_consistent": None, "reason": "empty text", "model": self.model}
        self.load()
        try:
            from mlx_vlm.models.minicpm_ocr_jev import score as jev_score  # type: ignore
        except ImportError as exc:
            # 旧版 mlx-vlm 没有这个子模块，别让用户去查 "speculative drafters"。
            raise RuntimeError(
                "当前 mlx-vlm 不含 minicpm_ocr_jev 移植。请在 macOS 上执行 "
                "`pixi update mlx-vlm -e mlx`（或 `pixi install -e mlx`）；"
                "需要 mlx-vlm >= 0.7.11。"
            ) from exc

        render = rendered or render_candidate(
            text, width=self.render_width, font_size=self.font_size
        )
        with tempfile.TemporaryDirectory(prefix="jev-") as tmp:
            render_path = Path(tmp) / "candidate.png"
            render.save(render_path)
            out = jev_score(
                self._model, self._processor, str(image), str(render_path), text
            )
        prob = out.get("prob_1", out.get("prob_consistent"))
        return {
            "prob_consistent": None if prob is None else float(prob),
            "model": self.model,
            "renderer": "pillow",
            "render_width": self.render_width,
            "font_size": self.font_size,
        }


def apply_qc(
    result: OCRResult,
    image: str | Path,
    scorer: JevScorer,
    threshold: float | None = None,
) -> dict[str, Any]:
    """给整页结果打分，写入 ``result.metadata["qc"]["jev"]``，并返回该字典。

    只做整页打分：VLM 引擎的 OCRResult 没有逐块 bbox，切块打分需要版面分析
    （见 ``unified_ocr.patent_table_mlx`` 的 regions）。整页分数是粗信号，
    不是逐块复核依据。
    """
    info = scorer.score(image, result.text)
    info["threshold"] = threshold
    info["needs_review"] = (
        None if threshold is None or info["prob_consistent"] is None
        else info["prob_consistent"] < threshold
    )
    info["scope"] = "page"
    result.metadata.setdefault("qc", {})["jev"] = info
    return info
