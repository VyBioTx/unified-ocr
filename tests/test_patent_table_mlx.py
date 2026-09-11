"""Offline tests for the MLX-native PP-StructureV3 components.

The MLX import is optional (``pytest.importorskip``), so the core test suite
still runs in environments without MLX. Weight-loading / forward tests are
skipped automatically when the ``models/ppocr-mlx`` checkout is absent.
"""

import json
from pathlib import Path

import pytest

# The whole package requires MLX; skip the module cleanly where MLX is absent.
pytest.importorskip("mlx.core")

from unified_ocr.patent_table_mlx.weight_loader import (
    BEG_STR,
    END_STR,
    SLANEXT_CHARACTER_DICT,
    build_character_list,
    decode_structure,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
TABLE_WIRED_DIR = REPO_ROOT / "models" / "ppocr-mlx" / "table_wired"

# Minimal SLANeXt_wired config (mirrors models/ppocr-mlx/table_wired/config.json)
SLANEXT_WIRED_CONFIG = {
    "model_type": "slanext",
    "vision_config": {
        "image_size": 512, "output_channels": 256, "num_channels": 3,
        "patch_size": 16, "hidden_act": "gelu", "layer_norm_eps": 1e-6,
        "qkv_bias": True, "use_abs_pos": True, "use_rel_pos": True,
        "window_size": 14, "hidden_size": 768, "num_hidden_layers": 12,
        "num_attention_heads": 12, "global_attn_indexes": [2, 5, 8, 11],
        "mlp_dim": 3072,
    },
    "post_conv_in_channels": 256,
    "post_conv_out_channels": 512,
    "out_channels": 50,
    "hidden_size": 512,
    "max_text_length": 500,
    "loc_reg_num": 8,
}


def test_character_list_length():
    char = build_character_list()
    assert len(char) == 50
    assert char[0] == BEG_STR
    assert char[-1] == END_STR
    # PaddleX removes bare "<td>" and appends "<td></td>"
    assert "<td>" not in char
    assert "<td></td>" in char


def test_decode_structure_stops_at_eos():
    char = build_character_list()
    end_idx = char.index(END_STR)
    # <tbody> <tr> <td > </td> EOS <tr>  -> wrapper added, trailing <tr> ignored
    ids = [char.index("<tbody>"), char.index("<tr>"), char.index("<td"),
           char.index(">"), char.index("</td>"), end_idx, char.index("<tr>")]
    out = decode_structure(ids, char)
    assert out == "<html><body><table><tbody><tr><td></td></table></body></html>"


def test_decode_structure_character_dict_size():
    assert len(SLANEXT_CHARACTER_DICT) == 48


def test_config_from_dict():
    from unified_ocr.patent_table_mlx.slanext import SLANeXtConfig

    cfg = SLANeXtConfig.from_config_dict(SLANEXT_WIRED_CONFIG)
    assert cfg.hidden_size == 768            # vision encoder
    assert cfg.head_hidden_size == 512       # SLA head
    assert cfg.post_conv_out_channels == 512
    assert cfg.out_channels == 50
    assert cfg.window_size == 14
    assert cfg.global_attn_indexes == (2, 5, 8, 11)


def test_slanext_forward_shape():
    mx = pytest.importorskip("mlx.core")
    from unified_ocr.patent_table_mlx.slanext import SLANeXt, SLANeXtConfig

    model = SLANeXt(SLANeXtConfig.from_config_dict(SLANEXT_WIRED_CONFIG))
    probs = model(mx.zeros((1, 3, 512, 512)))
    mx.eval(probs)
    assert probs.shape[0] == 1
    assert probs.shape[2] == 50
    assert 1 <= probs.shape[1] <= 501


@pytest.mark.skipif(not TABLE_WIRED_DIR.exists(), reason="ppocr-mlx weights not checked out")
def test_load_real_weights():
    pytest.importorskip("mlx.core")
    from unified_ocr.patent_table_mlx import load_slanext

    model = load_slanext(TABLE_WIRED_DIR)  # strict=True: raises on any key mismatch
    assert model is not None


@pytest.mark.skipif(not TABLE_WIRED_DIR.exists(), reason="ppocr-mlx weights not checked out")
def test_real_weights_forward():
    mx = pytest.importorskip("mlx.core")
    from unified_ocr.patent_table_mlx import decode_structure, load_slanext

    model = load_slanext(TABLE_WIRED_DIR)
    probs = model(mx.zeros((1, 3, 512, 512)))
    mx.eval(probs)
    ids = [int(v) for v in mx.argmax(probs[0], axis=-1)]
    html = decode_structure(ids)
    assert html.startswith("<html><body><table>")
    assert html.endswith("</table></body></html>")


# ---------------------------------------------------------------------------
# Full-document (layout) mode helpers — pure functions, no model weights.
# ---------------------------------------------------------------------------

def test_reading_order_single_column():
    from unified_ocr.patent_table_mlx.pipeline import reading_order

    regs = [
        {"label": "text", "box": [10, 100, 200, 150]},
        {"label": "text", "box": [10, 10, 200, 50]},
    ]
    ordered = reading_order(regs, width=1000)
    assert [r["box"][1] for r in ordered] == [10, 100]


def test_reading_order_two_columns():
    from unified_ocr.patent_table_mlx.pipeline import reading_order

    regs = [
        {"label": "text", "box": [10, 100, 180, 150]},   # left col, 2nd
        {"label": "text", "box": [600, 100, 980, 150]},  # right col, 2nd
        {"label": "text", "box": [10, 10, 180, 60]},     # left col, 1st
        {"label": "text", "box": [600, 10, 980, 60]},    # right col, 1st
    ]
    ordered = reading_order(regs, width=1000)
    # left column fully, then right column
    assert [r["box"][0] for r in ordered] == [10, 10, 600, 600]
    assert [r["box"][1] for r in ordered] == [10, 100, 10, 100]


def test_join_lines_space_vs_newline():
    from unified_ocr.patent_table_mlx.pipeline import _join_lines

    same_para = [([0, 10, 50, 20], "Hello"), ([0, 22, 50, 32], "world")]
    assert _join_lines(same_para) == "Hello world"
    new_para = [([0, 10, 50, 20], "Hello"), ([0, 200, 50, 210], "world")]
    assert _join_lines(new_para) == "Hello\nworld"


def test_format_region_markdown_labels():
    from unified_ocr.patent_table_mlx.pipeline import format_region_markdown

    assert format_region_markdown("doc_title", "Patent") == "# Patent"
    assert format_region_markdown("paragraph_title", "Claims") == "## Claims"
    assert format_region_markdown("figure_title", "Fig. 1") == "**Fig. 1**"
    assert format_region_markdown("image", "") == "[image]"
    assert format_region_markdown("text", "a paragraph") == "a paragraph"
    assert format_region_markdown("text", "") == ""


def test_document_markdown_groups_by_page():
    from unified_ocr.patent_table_mlx.pipeline import (
        PatentTableMLXPipeline,
        RegionResult,
    )

    regions = [
        RegionResult(page_index=1, label="doc_title", markdown="# T"),
        RegionResult(page_index=1, label="text", markdown="body"),
        RegionResult(page_index=2, label="text", markdown="page two"),
    ]
    md = PatentTableMLXPipeline.document_markdown(regions)
    assert "## 第 1 页" in md and "# T" in md and "body" in md
    assert "## 第 2 页" in md and "page two" in md
    # non-header mode just joins the fragments
    flat = PatentTableMLXPipeline.document_markdown(regions, page_headers=False)
    assert "## 第 1 页" not in flat and "page two" in flat


def test_texts_in_box_selects_centers():
    from unified_ocr.patent_table_mlx.pipeline import PatentTableMLXPipeline

    ocr_pairs = [
        ([10, 10, 30, 20], "inside"),
        ([10, 200, 30, 210], "outside"),
        ([50, 50, 70, 60], "inside2"),
    ]
    hits = PatentTableMLXPipeline._texts_in_box(ocr_pairs, [0, 0, 100, 100])
    assert [t for _, t in hits] == ["inside", "inside2"]


def test_rec_model_name_language_switch():
    from unified_ocr.patent_table_mlx.pipeline import (
        PatentPipelineMLXConfig,
        PatentTableMLXPipeline,
    )

    en = PatentTableMLXPipeline(PatentPipelineMLXConfig(rec_lang="en"))
    assert en.rec_model_name() == "en_PP-OCRv4_mobile_rec"
    ch = PatentTableMLXPipeline(PatentPipelineMLXConfig(rec_lang="ch"))
    assert ch.rec_model_name() == "PP-OCRv5_server_rec"
    # explicit model wins only for the default language
    custom = PatentTableMLXPipeline(
        PatentPipelineMLXConfig(rec_lang="en", rec_model="my_rec")
    )
    assert custom.rec_model_name() == "my_rec"
