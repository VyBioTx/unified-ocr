"""数据模型与 Markdown 分块逻辑的离线测试。"""

import json

import pytest

from unified_ocr.models import (
    BBox,
    OCRBlock,
    OCRLine,
    OCRResult,
    OCRWord,
    split_blocks,
)


def test_ocr_result_roundtrip():
    result = OCRResult(
        engine="glm-ocr",
        text="hello",
        blocks=[
            OCRBlock(
                kind="text",
                text="hello",
                bbox=BBox(0.0, 0.0, 1.0, 1.0),
                confidence=0.95,
                lines=[
                    OCRLine(
                        text="hello",
                        confidence=0.95,
                        words=[OCRWord(text="hello", confidence=0.95)],
                    )
                ],
            )
        ],
        metadata={"model": "mlx-community/GLM-OCR-bf16"},
    )
    data = json.loads(result.to_json())
    assert data["engine"] == "glm-ocr"
    assert data["blocks"][0]["lines"][0]["words"][0]["text"] == "hello"
    assert data["blocks"][0]["bbox"]["x1"] == 1.0
    assert data["metadata"]["model"].endswith("GLM-OCR-bf16")


def test_from_text_builds_single_block():
    result = OCRResult.from_text("paddleocr-vl", "第一行\n第二行", model="x")
    assert result.text == "第一行\n第二行"
    assert result.blocks and result.blocks[0].text == "第一行\n第二行"
    assert result.metadata["model"] == "x"


SAMPLE_MARKDOWN = """# 实验报告

这是引言段落，包含
跨两行的内容。

| 基因 | 表达量 |
|------|--------|
| TP53 | 1.2 |

```latex
E = mc^2
```

结尾段落。"""


def test_split_blocks_detects_heading_table_formula_paragraph():
    blocks = split_blocks(SAMPLE_MARKDOWN)
    kinds = [b.kind for b in blocks]
    assert kinds == ["heading", "text", "table", "formula", "text"]
    assert blocks[0].text == "实验报告"
    assert blocks[2].text.startswith("| 基因")
    assert "E = mc^2" in blocks[3].text
    assert blocks[4].text == "结尾段落。"


def test_split_blocks_empty_and_plain():
    assert split_blocks("") == []
    assert split_blocks("   \n  ") == []
    blocks = split_blocks("只有一行")
    assert len(blocks) == 1 and blocks[0].kind == "text"


def test_to_markdown_rendering():
    result = OCRResult.from_text("glm-ocr", SAMPLE_MARKDOWN)
    md = result.to_markdown()
    assert "## 实验报告" in md
    assert "| 基因 | 表达量 |" in md
    assert "$$" in md  # formula 围栏