"""Load ``plaincompute/ppocr-mlx`` MLX safetensors into the MLX SLANeXt model.

The ppocr-mlx checkpoints already use HF/transformers-style parameter names, and
:mod:`unified_ocr.patent_table_mlx.slanext` mirrors those names exactly, so
loading is a straight key passthrough (no shape-matching heuristics).

Typical use::

    from unified_ocr.patent_table_mlx import load_slanext
    model = load_slanext("models/ppocr-mlx/table_wired")
    probs = model(mx.array(img[None]))          # [1, seq, 50]
    structure = decode_structure(mx.argmax(probs[0], axis=-1).tolist())
"""

from __future__ import annotations

import json
from pathlib import Path

import mlx.core as mx

from .slanext import SLANeXt, SLANeXtConfig

# PaddleX SLANeXt_wired / _wireless TableLabelDecode character_dict (48 symbols).
SLANEXT_CHARACTER_DICT = [
    "<thead>", "</thead>", "<tbody>", "</tbody>", "<tr>", "</tr>",
    "<td>", "<td", ">", "</td>",
    ' colspan="2"', ' colspan="3"', ' colspan="4"', ' colspan="5"',
    ' colspan="6"', ' colspan="7"', ' colspan="8"', ' colspan="9"',
    ' colspan="10"', ' colspan="11"', ' colspan="12"', ' colspan="13"',
    ' colspan="14"', ' colspan="15"', ' colspan="16"', ' colspan="17"',
    ' colspan="18"', ' colspan="19"', ' colspan="20"',
    ' rowspan="2"', ' rowspan="3"', ' rowspan="4"', ' rowspan="5"',
    ' rowspan="6"', ' rowspan="7"', ' rowspan="8"', ' rowspan="9"',
    ' rowspan="10"', ' rowspan="11"', ' rowspan="12"', ' rowspan="13"',
    ' rowspan="14"', ' rowspan="15"', ' rowspan="16"', ' rowspan="17"',
    ' rowspan="18"', ' rowspan="19"', ' rowspan="20"',
]

BEG_STR = "sos"
END_STR = "eos"


# ppocr-mlx stores the backbone post conv as ``backbone.post_conv.weight`` while
# the MLX module nests an ``nn.Conv2d`` under ``PostConv.conv`` — remap the key.
_KEY_REMAP = {
    "backbone.post_conv.": "backbone.post_conv.conv.",
}


def _remap_key(key: str) -> str:
    for src, dst in _KEY_REMAP.items():
        if key.startswith(src) and not key.startswith(dst):
            return dst + key[len(src):]
    return key


def load_mlx_weights(model_dir: str | Path) -> list[tuple[str, mx.array]]:
    """Return ``[(key, mx.array), ...]`` from ``<model_dir>/model.mlx.safetensors``."""
    path = Path(model_dir) / "model.mlx.safetensors"
    if not path.exists():
        raise FileNotFoundError(f"MLX weights not found: {path}")
    weights = dict(mx.load(str(path)))
    weights.pop("__metadata__", None)
    return [(_remap_key(k), v) for k, v in weights.items()]


def load_slanext(model_dir: str | Path, strict: bool = True) -> SLANeXt:
    """Instantiate SLANeXt from a ppocr-mlx folder (``config.json`` + weights)."""
    model_dir = Path(model_dir)
    cfg_path = model_dir / "config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"config.json not found: {cfg_path}")
    with open(cfg_path) as f:
        config = SLANeXtConfig.from_config_dict(json.load(f))
    model = SLANeXt(config)
    model.load_weights(load_mlx_weights(model_dir), strict=strict)
    return model


# ---------------------------------------------------------------------------
# Decoding
# ---------------------------------------------------------------------------

def build_character_list(
    dict_character: list[str] | None = None,
    merge_no_span_structure: bool = True,
) -> list[str]:
    """Replicate PaddleX ``TableLabelDecode`` character list construction."""
    dc = list(dict_character if dict_character is not None else SLANEXT_CHARACTER_DICT)
    if merge_no_span_structure:
        if "<td></td>" not in dc:
            dc.append("<td></td>")
        if "<td>" in dc:
            dc.remove("<td>")
    return [BEG_STR] + dc + [END_STR]


def decode_structure_tokens(
    token_ids: list[int],
    character: list[str] | None = None,
    with_wrapper: bool = True,
) -> list[str]:
    """Decode argmax token ids into the list of structure token strings.

    Same stopping/ignoring rules as :func:`decode_structure`, but returns the
    individual tokens (PaddleX ``table_structure_result`` format).  With
    ``with_wrapper=True`` the standard ``<html><body><table>`` prefix and
    ``</table></body></html>`` suffix are added.
    """
    char = character if character is not None else build_character_list()
    end_idx = char.index(END_STR)
    beg_idx = char.index(BEG_STR)

    tokens: list[str] = []
    for i, cid in enumerate(token_ids):
        cid = int(cid)
        if i > 0 and cid == end_idx:
            break
        if cid in (beg_idx, end_idx):
            continue
        tokens.append(char[cid])

    if with_wrapper:
        return ["<html>", "<body>", "<table>"] + tokens + ["</table>", "</body>", "</html>"]
    return tokens


def decode_structure(
    token_ids: list[int],
    character: list[str] | None = None,
    with_wrapper: bool = True,
) -> str:
    """Decode argmax token ids into the table structure string.

    Mirrors PaddleX ``TableLabelDecode.decode``: skip the BOS token, stop at the
    first EOS after position 0, and ignore BOS/EOS ids.  Optionally wrap in
    ``<html><body><table> ... </table></body></html>``.
    """
    char = character if character is not None else build_character_list()
    end_idx = char.index(END_STR)
    beg_idx = char.index(BEG_STR)

    pieces: list[str] = []
    for i, cid in enumerate(token_ids):
        cid = int(cid)
        if i > 0 and cid == end_idx:
            break
        if cid in (beg_idx, end_idx):
            continue
        pieces.append(char[cid])

    structure = "".join(pieces)
    if with_wrapper:
        return "<html><body><table>" + structure + "</table></body></html>"
    return structure
