"""统一 OCR 数据模型。

三个引擎（PaddleOCR-VL / HunyuanOCR / GLM-OCR）输出格式各异，此处定义
统一的规范化结构，供下游消费者（页面渲染、结构化提取、数据库入库等）
无差别使用。

结构层级::

    OCRResult
      ├─ blocks: list[OCRBlock]     (段落 / 表格 / 公式 / 图片等语义块)
      │    └─ lines: list[OCRLine]  (行)
      │         └─ words: list[OCRWord] (词 / 最小片段)
      ├─ text: str                  整页纯文本（横排拼接的快速视图）
      └─ metadata: dict             引擎名 / 版本 / 原始输出等
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from typing import Any, Optional

# 语义块类型 —— 各引擎将语义映射到统一枚举
BLOCK_TYPES = (
    "text",      # 普通文本段落
    "heading",   # 标题
    "table",     # 表格
    "formula",   # 公式
    "figure",    # 图片 / 图表
    "list",      # 列表
    "other",     # 无法归类
)


def _bbox_to_dict(bbox: Optional["BBox"]) -> Optional[dict[str, float]]:
    return bbox.to_dict() if bbox else None


@dataclass
class BBox:
    """归一化边界框，坐标 ∈ [0, 1]（相对图像宽高）。

    原生引擎若输出像素坐标（如 PaddleOCR 检测框），适配层负责归一化。
    """

    x0: float
    y0: float
    x1: float
    y1: float

    def to_dict(self) -> dict[str, float]:
        return {"x0": self.x0, "y0": self.y0, "x1": self.x1, "y1": self.y1}


@dataclass
class OCRWord:
    text: str
    confidence: Optional[float] = None
    bbox: Optional[BBox] = None


@dataclass
class OCRLine:
    text: str
    bbox: Optional[BBox] = None
    confidence: Optional[float] = None
    words: list[OCRWord] = field(default_factory=list)


@dataclass
class OCRBlock:
    kind: str = "text"          # BLOCK_TYPES 之一
    text: str = ""
    bbox: Optional[BBox] = None
    confidence: Optional[float] = None
    lines: list[OCRLine] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return not self.text.strip()


@dataclass
class OCRResult:
    """统一识别结果。"""

    engine: str                  # 引擎标识，如 "glm-ocr"
    text: str = ""               # 整页纯文本
    blocks: list[OCRBlock] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)  # 引擎原始输出 / 版本等

    # ---- 序列化 ---------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return {
            "engine": self.engine,
            "text": self.text,
            "blocks": [
                {
                    "kind": b.kind,
                    "text": b.text,
                    "bbox": _bbox_to_dict(b.bbox),
                    "confidence": b.confidence,
                    "lines": [
                        {
                            "text": ln.text,
                            "bbox": _bbox_to_dict(ln.bbox),
                            "confidence": ln.confidence,
                            "words": [
                                {
                                    "text": w.text,
                                    "bbox": _bbox_to_dict(w.bbox),
                                    "confidence": w.confidence,
                                }
                                for w in ln.words
                            ],
                        }
                        for ln in b.lines
                    ],
                }
                for b in self.blocks
            ],
            "metadata": self.metadata,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)

    def to_markdown(self) -> str:
        """将 blocks 渲染为 Markdown —— VLM 类引擎的常见输出形态。"""
        parts: list[str] = []
        for block in self.blocks:
            if block.is_empty:
                continue
            if block.kind == "table":
                parts.append(block.text)  # 引擎已给 Markdown 表格
            elif block.kind == "formula":
                parts.append(f"$$\n{block.text}\n$$")
            elif block.kind == "heading":
                parts.append(f"## {block.text}")
            else:
                parts.append(block.text)
        joined = "\n\n".join(parts)
        return joined if joined else self.text

    @classmethod
    def from_text(cls, engine: str, text: str, **metadata: Any) -> "OCRResult":
        """最简构造：仅有整页文本，无块级结构。"""
        return cls(
            engine=engine,
            text=text,
            blocks=split_blocks(text),
            metadata=metadata,
        )


# ---------------------------------------------------------------------------
# Markdown → blocks 的轻量启发式切分
# ---------------------------------------------------------------------------
# VLM 类引擎（HunyuanOCR / GLM-OCR / PaddleOCR-VL）的典型输出是带 Markdown
# 结构的整页文本，且没有逐行 bbox。以下启发式将其切成统一 OCRBlock：
#   - ``` 围栏 → formula / other
#   - # 标题 → heading
#   - 含分隔行的 Markdown 表格 → table
# 若引擎能给出结构化输出（如 PaddleOCR-VL 的 JSON 模式），适配层应优先生成
# 结构化 blocks，而不是依赖本函数。

def split_blocks(text: str) -> list[OCRBlock]:
    if not text or not text.strip():
        return []

    blocks: list[OCRBlock] = []
    lines = text.splitlines()
    i = 0
    n = len(lines)

    while i < n:
        stripped = lines[i].strip()

        # --- ``` 围栏块（公式 / 代码） ---
        if stripped.startswith("```"):
            fence_lang = stripped[3:].strip().lower()
            content: list[str] = []
            i += 1
            while i < n and not lines[i].strip().startswith("```"):
                content.append(lines[i])
                i += 1
            i += 1  # 越过闭合围栏
            kind = (
                "formula"
                if any(k in fence_lang for k in ("latex", "math", "formula", "equation", "tex"))
                else "other"
            )
            if content:
                blocks.append(OCRBlock(kind=kind, text="\n".join(content).strip()))
            continue

        # --- 标题 ---
        if stripped.startswith("#") and stripped.startswith("# "):
            blocks.append(OCRBlock(kind="heading", text=stripped.lstrip("#").strip()))
            i += 1
            continue

        # --- Markdown 表格（连续含 | 的行，其中一行是分隔行） ---
        if _is_table_block(lines, i):
            table_lines: list[str] = []
            while i < n and "|" in lines[i] and lines[i].strip():
                table_lines.append(lines[i].strip())
                i += 1
            blocks.append(OCRBlock(kind="table", text="\n".join(table_lines)))
            continue

        # --- 普通文本段落：累积到空行或下一个特殊块 ---
        para: list[str] = []
        while (
            i < n
            and lines[i].strip()
            and not lines[i].strip().startswith("```")
            and not lines[i].strip().startswith("# ")
            and not _is_table_block(lines, i)
        ):
            para.append(lines[i].strip())
            i += 1
        if para:
            blocks.append(OCRBlock(kind="text", text="\n".join(para)))

        # 空行直接跳过
        if i < n and not lines[i].strip():
            i += 1

    return blocks or [OCRBlock(kind="text", text=text)]


def _is_table_block(lines: list[str], i: int) -> bool:
    """判断 lines[i:] 是否从一个 Markdown 表格开始：至少 2 行且第 2 行是分隔行。"""
    if i + 1 >= len(lines):
        return False
    first, second = lines[i].strip(), lines[i + 1].strip()
    if "|" not in first or not second:
        return False
    # 分隔行只含 | - : 与空白
    return set(second) <= {"|", "-", ":", " ", "\t"}