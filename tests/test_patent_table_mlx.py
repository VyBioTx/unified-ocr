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
