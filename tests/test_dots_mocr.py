"""dots.mocr 后端离线测试：版面 JSON → blocks 转换、MLX 支持探测。

不加载真实模型（无 torch/mlx 依赖）：只验证可离线测的逻辑。
"""

import json

import pytest

from unified_ocr.backends.dots_mocr import (
    DEFAULT_PROMPT,
    PROMPT_LAYOUT_ALL_EN,
    layout_json_to_blocks,
)
from unified_ocr.backends.dots_mocr_mlx import DotsMocrMLXBackend
from unified_ocr.registry import get_backend_class


def test_layout_json_to_blocks_full():
    cells = [
        {"bbox": [0, 0, 400, 60], "category": "Title", "text": "实验报告", "score": 0.99},
        {"bbox": [0, 80, 200, 120], "category": "Table", "text": "<table>...</table>", "score": 0.9},
        {"bbox": [0, 130, 300, 160], "category": "Formula", "text": "E = mc^2", "score": 0.8},
        {"bbox": [10, 200, 390, 260], "category": "Text", "text": "正文段落。", "score": 0.95},
        {"bbox": [400, 0, 800, 400], "category": "Picture"},
    ]
    text = json.dumps(cells, ensure_ascii=False)
    blocks = layout_json_to_blocks(text, image_size=(800, 400))

    assert [b.kind for b in blocks] == ["heading", "table", "formula", "text", "figure"]
    assert blocks[0].text == "实验报告"
    assert blocks[0].bbox is not None
    # bbox 归一化到 [0,1]：x1=400/800=0.5, y1=60/400=0.15
    assert abs(blocks[0].bbox.x1 - 0.5) < 1e-6
    assert abs(blocks[0].bbox.y1 - 0.15) < 1e-6
    assert abs(blocks[0].bbox.x0 - 0.0) < 1e-6
    assert blocks[1].kind == "table"
    assert blocks[2].kind == "formula"
    assert blocks[4].kind == "figure"
    assert blocks[4].text == ""  # Picture 无文本
    assert blocks[4].confidence is None


def test_layout_json_to_blocks_no_image_size():
    cells = [{"bbox": [0, 0, 100, 50], "category": "Text", "text": "x"}]
    blocks = layout_json_to_blocks(json.dumps(cells), image_size=None)
    assert len(blocks) == 1
    assert blocks[0].bbox is None  # 无原图尺寸时无法归一化


def test_layout_json_to_blocks_invalid():
    assert layout_json_to_blocks("not json", image_size=(100, 100)) == []
    assert layout_json_to_blocks('[{"category": "Text"}]', image_size=(100, 100))[0].text == ""
    assert layout_json_to_blocks('{"not": "a list"}', image_size=(100, 100)) == []


def test_prompt_constants():
    assert "bbox" in PROMPT_LAYOUT_ALL_EN
    assert "category" in PROMPT_LAYOUT_ALL_EN
    assert DEFAULT_PROMPT.strip().endswith("image.")


def test_mlx_backend_registered_and_probe_graceful():
    cls = get_backend_class("dots-mocr-mlx")
    assert cls is DotsMocrMLXBackend
    backend = cls(model="rednote-hilab/dots.mocr")
    supported, detail = backend._check_mlx_vlm_support()
    # 本环境无 mlx-vlm：应当优雅地返回 (False, 说明)，而不是抛异常
    if not supported:
        assert isinstance(detail, str) and detail
        with pytest.raises(RuntimeError, match="dots-mocr-mlx 引擎暂不可用"):
            backend.load()


def test_transformers_backend_spec():
    cls = get_backend_class("dots-mocr")
    assert cls.spec.id == "dots-mocr"
    assert cls.spec.accelerator.startswith("MPS")
    assert "rednote-hilab/dots.mocr" in cls.spec.model_hint
    # 默认模型路径与 dots 官方下载脚本一致
    assert "weights" in cls.default_model
