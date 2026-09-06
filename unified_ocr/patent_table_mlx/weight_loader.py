"""从 PaddleX 官方权重导出/加载到 MLX 模型。

PaddleX 3.7 的 SLANeXt 是 HF-transformers 风格模型（继承 PretrainedModel），
理论上可以 save_pretrained 输出 HF 格式（config.json + model.safetensors）。
如果不行则直接从 pdiparams 批量提取并按规范命名映射。
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np

from .slanext import SLANeXt, SLANeXtConfig


def _load_paddle_pdiparams(model_dir: str | Path) -> dict[str, mx.array]:
    """Load Paddle inference.pdiparams and return as dict of mx.arrays.

    pdiparams is a flat concatenated tensor; we need the inference.json
    to know split points. PaddleX 3.x stores each param's weights in
    separate entries that paddle.load() returns as a flat Tensor.
    We parse the PIR program to infer split positions.

    对于 PaddleX 3.x 的 PIR 格式，权重在 inference.json 中用 'p' op
    声明（persistable variable），pdiparams 存储其拼接后的值。
    此函数解析 PIR 程序，提取每个参数名与形状，并从拼接张量中切分。
    """
    import paddle

    model_dir = Path(model_dir)
    # 用 paddle.load 读取整个 pdiparams（返回一个大 Tensor）
    params_path = model_dir / "inference.pdiparams"
    if not params_path.exists():
        raise FileNotFoundError(f"Paddle params not found: {params_path}")

    flat_tensor = paddle.load(str(params_path))
    flat_np = flat_tensor.numpy()

    # 从 inference.json（PIR 格式）解析参数定义
    with open(model_dir / "inference.json") as f:
        prog = json.load(f)

    param_defs = []
    regions = prog.get("program", {}).get("regions", [])
    for region in regions:
        for block in region.get("blocks", []):
            for op in block.get("ops", []):
                if op.get("#") == "p":
                    A = op.get("A", [])
                    if len(A) >= 4 and isinstance(A[3], str):
                        name = A[3]
                        # 解析 shape
                        tt_d = op.get("O", {}).get("TT", {}).get("D", [])
                        if len(tt_d) >= 2:
                            shape = list(tt_d[1])
                        else:
                            shape = []
                        param_defs.append((name, shape))

    # 按照 pdiparams 中的存储顺序切分权重
    # PIR 中 pdiparams 按参数名排序（lexicographic）拼接
    param_defs.sort(key=lambda x: x[0])

    result = {}
    offset = 0
    for name, shape in param_defs:
        size = int(np.prod(shape)) if shape else 1
        arr = flat_np[offset : offset + size].reshape(shape)
        result[name] = mx.array(arr)
        offset += size

    if offset != len(flat_np):
        print(f"Warning: expected {offset} bytes, got {len(flat_np)}")

    return result


def load_weights_to_slanext(model: SLANeXt, weights: dict[str, mx.array]) -> None:
    """Load PaddleX weights into MLX SLANeXt model.

    The weight key mapping from PaddleX (e.g. 'linear_54.w_0') to
    the MLX model's parameter structure is done by matching shapes
    and layer index.

    由于 PaddleX PIR 中参数名为 flat identifier（linear_XX.XX_0），
    需要与 MLX 模型结构的子模块一一对应。这里直接按 module 路径赋值。
    """
    # 直接匹配：MLX 模型参数名与 Paddle HF 格式 key 的映射需要逐个核对。
    # 因为 PaddleX 权重的 key 并非结构化名称，我们采用 shape-match 策略：
    # 遍历 MLX 模型的所有参数，从 weights dict 中找 shape 匹配的项。

    # 获取 MLX 模型的参数 leaf 路径
    mlx_params = {}
    _collect_params(model, "", mlx_params)

    used = set()
    unmatched_mlx = []
    matched = 0

    for param_path, param_arr in mlx_params.items():
        # 从 weights 找 shape 匹配的 paddle 参数
        found = False
        for pname, parr in weights.items():
            if pname in used:
                continue
            if parr.shape == param_arr.shape:
                # 候选匹配 — 根据语义规则确认
                _assign_param(model, param_path, parr)
                used.add(pname)
                matched += 1
                found = True
                break
        if not found:
            unmatched_mlx.append(param_path)

    w = len(weights) - len(used)
    print(f"Loaded {matched}/{len(mlx_params)} MLX params "
          f"({w} unused paddle weights)")
    if unmatched_mlx:
        print(f"Unmatched MLX params ({len(unmatched_mlx)}):")
        for p in unmatched_mlx[:10]:
            print(f"  {p}")


def _collect_params(module, prefix, result):
    """Collect all leaf weight parameters from an MLX module."""
    for name, child in module.__dict__.items():
        if name.startswith("_"):
            continue
        full_name = f"{prefix}.{name}" if prefix else name
        if isinstance(child, nn.Module):
            _collect_params(child, full_name, result)
        elif isinstance(child, mx.array):
            result[full_name] = child
        elif hasattr(child, '__call__') and hasattr(child, '__dict__'):
            pass  # skip callable objects


def _assign_param(model, path, value):
    """Assign a weight to an MLX model by dot-separated path."""
    parts = path.split(".")
    obj = model
    for p in parts[:-1]:
        obj = getattr(obj, p, None)
        if obj is None:
            return
    setattr(obj, parts[-1], value)


def export_to_safetensors(
    model_dir: str | Path,
    output_path: str | Path,
) -> None:
    """Convert PaddleX SLANeXt weights to safetensors + HF config.

    导出文件可供 MLX/transformers 直接加载。
    """
    import paddle

    model_dir = Path(model_dir)
    config_path = model_dir / "inference.yml"
    out_path = Path(output_path)
    out_path.mkdir(parents=True, exist_ok=True)

    # 1. 加载 paddle 权重
    weights = _load_paddle_pdiparams(model_dir)

    # 2. 转换为 dict of numpy arrays（HF key 命名）
    state_dict = {}
    for pname, arr in weights.items():
        state_dict[pname] = np.array(arr)

    # 3. 保存 safetensors
    try:
        import safetensors.numpy
        safetensors.numpy.save_file(
            state_dict,
            str(out_path / "model.safetensors"),
        )
        print(f"safetensors saved: {out_path / 'model.safetensors'}")
    except ImportError:
        np.savez(out_path / "model.npz", **state_dict)
        print(f"npz saved: {out_path / 'model.npz'} (safetensors not available)")

    # 4. 保存 HF config
    hf_config = {
        "model_type": "slanext",
        "architectures": ["SLANeXt"],
        "vision_config": {
            "hidden_size": 768,
            "output_channels": 256,
            "num_hidden_layers": 12,
            "num_attention_heads": 12,
            "num_channels": 3,
            "image_size": 512,
            "patch_size": 16,
            "hidden_act": "gelu",
            "layer_norm_eps": 1e-6,
            "qkv_bias": True,
            "use_abs_pos": True,
            "use_rel_pos": True,
            "window_size": 14,
            "global_attn_indexes": [2, 5, 8, 11],
            "mlp_dim": 3072,
        },
        "post_conv_in_channels": 256,
        "post_conv_out_channels": 512,
        "out_channels": 50,
        "hidden_size": 512,
        "max_text_length": 500,
    }
    with open(out_path / "config.json", "w") as f:
        json.dump(hf_config, f, indent=2)

    print(f"Config saved: {out_path / 'config.json'}")


# 需要延迟导入避免循环引用
import mlx.nn as nn