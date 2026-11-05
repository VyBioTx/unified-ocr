"""SLANeXt table structure recognition — faithful MLX port.

Ported 1:1 from the PaddleX 3.7 reference implementation
(``paddlex/inference/models/table_structure_recognition/modeling/slanext.py``,
which is itself aligned with the HF ``transformers`` checkpoint), so that the
``plaincompute/ppocr-mlx`` converted weights (``model.mlx.safetensors``) can be
loaded directly by key.

Module/parameter names mirror the HF safetensors keys exactly, e.g.::

    backbone.vision_tower.layers.0.attn.qkv.weight
    backbone.vision_tower.neck.layer_norm1.weight
    backbone.post_conv.weight
    head.structure_attention_cell.rnn.weight_ih
    head.structure_generator.fc1.weight

so :func:`unified_ocr.patent_table_mlx.weight_loader.load_mlx_weights` is a
plain key passthrough.

Architecture: GotOcr2 (SAM-ViT) vision encoder + stride-2 post conv backbone
+ GRU-attention SLA head (autoregressive structure decoding).

Notes vs. the earlier hand-written draft (``slanext.py`` pre-2026-09-11):
  * the generator MLP has **no** activation between fc1 and fc2;
  * the GRU is a Paddle-style GRUCell (gate order r, z, c; h=(h-c)*z+c);
  * decomposed relative-position bias is computed with the exact Paddle
    einsum/broadcast layout.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import mlx.core as mx
import mlx.nn as nn


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class SLANeXtConfig:
    """SLANeXt model configuration (defaults = SLANeXt_wired)."""
    # Vision encoder (SAM-ViT / GotOcr2)
    image_size: int = 512
    patch_size: int = 16
    num_channels: int = 3
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    mlp_dim: int = 3072
    hidden_act: str = "gelu"
    layer_norm_eps: float = 1e-6
    qkv_bias: bool = True
    use_abs_pos: bool = True
    use_rel_pos: bool = True
    window_size: int = 14
    global_attn_indexes: tuple = field(default_factory=lambda: (2, 5, 8, 11))
    output_channels: int = 256  # neck output channels

    # Post conv (backbone downsampler)
    post_conv_in_channels: int = 256
    post_conv_out_channels: int = 512

    # SLA head
    head_hidden_size: int = 512  # top-level ``hidden_size`` (distinct from vision)
    out_channels: int = 50      # vocabulary size for structure tokens
    max_text_length: int = 500
    loc_reg_num: int = 8

    @classmethod
    def from_config_dict(cls, cfg: dict) -> "SLANeXtConfig":
        """Build from a ppocr-mlx ``table_wired/config.json`` dict."""
        vc = cfg.get("vision_config", {})
        kw: dict = {}
        if vc:
            kw.update(
                image_size=vc.get("image_size", 512),
                patch_size=vc.get("patch_size", 16),
                num_channels=vc.get("num_channels", 3),
                hidden_size=vc.get("hidden_size", 768),
                num_hidden_layers=vc.get("num_hidden_layers", 12),
                num_attention_heads=vc.get("num_attention_heads", 12),
                mlp_dim=vc.get("mlp_dim", 3072),
                hidden_act=vc.get("hidden_act", "gelu"),
                layer_norm_eps=vc.get("layer_norm_eps", 1e-6),
                qkv_bias=vc.get("qkv_bias", True),
                use_abs_pos=vc.get("use_abs_pos", True),
                use_rel_pos=vc.get("use_rel_pos", True),
                window_size=vc.get("window_size", 14),
                global_attn_indexes=tuple(vc.get("global_attn_indexes", (2, 5, 8, 11))),
                output_channels=vc.get("output_channels", 256),
            )
        kw.update(
            post_conv_in_channels=cfg.get("post_conv_in_channels", 256),
            post_conv_out_channels=cfg.get("post_conv_out_channels", 512),
            head_hidden_size=cfg.get("hidden_size", 512),
            out_channels=cfg.get("out_channels", 50),
            max_text_length=cfg.get("max_text_length", 500),
            loc_reg_num=cfg.get("loc_reg_num", 8),
        )
        return cls(**kw)


# ---------------------------------------------------------------------------
# Relative position helpers
# ---------------------------------------------------------------------------

def _linear_interp_1d(rel_pos: mx.array, target_len: int) -> mx.array:
    """Paddle ``F.interpolate(mode='linear', align_corners=False)`` on axis 0.

    (Only exercised when the input resolution differs from the configured one;
    at 512x512 the shapes already match, so no interpolation happens.)
    """
    old_len = rel_pos.shape[0]
    if old_len == target_len:
        return rel_pos
    scale = old_len / target_len
    idx = (mx.arange(target_len, dtype=mx.float32) + 0.5) * scale - 0.5
    idx = mx.clip(idx, 0.0, float(old_len - 1))
    lo = mx.floor(idx).astype(mx.int64)
    hi = mx.minimum(lo + 1, old_len - 1)
    frac = (idx - lo.astype(mx.float32))[:, None]
    return rel_pos[lo] * (1 - frac) + rel_pos[hi] * frac


def get_rel_pos(q_size: int, k_size: int, rel_pos: mx.array) -> mx.array:
    """Relative position bias along one axis, shape [q_size, k_size, head_dim]."""
    max_rel_dist = int(2 * max(q_size, k_size) - 1)
    if rel_pos.shape[0] != max_rel_dist:
        rel_pos = _linear_interp_1d(rel_pos, max_rel_dist)

    q_coords = mx.arange(q_size, dtype=mx.float32)[:, None] * max(k_size / q_size, 1.0)
    k_coords = mx.arange(k_size, dtype=mx.float32)[None, :] * max(q_size / k_size, 1.0)
    relative_coords = (q_coords - k_coords) + (k_size - 1) * max(q_size / k_size, 1.0)
    return rel_pos[relative_coords.astype(mx.int64)]


# ---------------------------------------------------------------------------
# Vision encoder (GotOcr2)
# ---------------------------------------------------------------------------

class MLPBlock(nn.Module):
    def __init__(self, config: SLANeXtConfig):
        super().__init__()
        self.lin1 = nn.Linear(config.hidden_size, config.mlp_dim)
        self.lin2 = nn.Linear(config.mlp_dim, config.hidden_size)
        self.act = nn.GELU()

    def __call__(self, x: mx.array) -> mx.array:
        return self.lin2(self.act(self.lin1(x)))


def window_partition(x: mx.array, window_size: int):
    """[B, H, W, C] -> ([-1, ws, ws, C], (Hp, Wp)) with zero padding."""
    B, H, W, C = x.shape
    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    if pad_h > 0 or pad_w > 0:
        x = mx.pad(x, ((0, 0), (0, pad_h), (0, pad_w), (0, 0)))
    Hp, Wp = H + pad_h, W + pad_w
    x = x.reshape(B, Hp // window_size, window_size, Wp // window_size, window_size, C)
    windows = x.transpose(0, 1, 3, 2, 4, 5).reshape(-1, window_size, window_size, C)
    return windows, (Hp, Wp)


def window_unpartition(windows: mx.array, window_size: int, pad_hw, hw) -> mx.array:
    Hp, Wp = pad_hw
    H, W = hw
    B = windows.shape[0] // (Hp * Wp // window_size // window_size)
    x = windows.reshape(B, Hp // window_size, Wp // window_size, window_size, window_size, -1)
    x = x.transpose(0, 1, 3, 2, 4, 5).reshape(B, Hp, Wp, -1)
    if Hp > H or Wp > W:
        x = x[:, :H, :W, :]
    return x


class VisionAttention(nn.Module):
    """Multi-head attention with optional windowing + relative position bias."""

    def __init__(self, config: SLANeXtConfig, window_size: int):
        super().__init__()
        input_size = (
            (config.image_size // config.patch_size, config.image_size // config.patch_size)
            if window_size == 0
            else (window_size, window_size)
        )
        self.num_attention_heads = config.num_attention_heads
        head_dim = config.hidden_size // config.num_attention_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(config.hidden_size, config.hidden_size * 3, bias=config.qkv_bias)
        self.proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.use_rel_pos = config.use_rel_pos
        if self.use_rel_pos:
            self.rel_pos_h = mx.zeros((2 * input_size[0] - 1, head_dim))
            self.rel_pos_w = mx.zeros((2 * input_size[1] - 1, head_dim))

    def _decomposed_rel_pos(self, q, rel_pos_h, rel_pos_w, q_size, k_size):
        q_h, q_w = q_size
        k_h, k_w = k_size
        Rh = get_rel_pos(q_h, k_h, rel_pos_h)   # [q_h, k_h, d]
        Rw = get_rel_pos(q_w, k_w, rel_pos_w)   # [q_w, k_w, d]

        B, heads, _, dim = q.shape
        r_q = q.reshape(B, heads, q_h, q_w, dim)
        # rel_h: (B, heads, q_h, q_w, k_h); rel_w: (B, heads, q_h, q_w, k_w)
        rel_h = mx.einsum("bnqwd,qkd->bnqwk", r_q, Rh)
        rel_w = mx.einsum("bnqwd,wkd->bnqwk", r_q, Rw)
        # broadcast to (B, heads, q_h, q_w, k_h, k_w) — matches Paddle's
        # ``rel_h[..., None] + rel_w[..., None, :]`` axis placement.
        rel_h = rel_h.reshape(B, heads, q_h, q_w, k_h, 1)
        rel_w = rel_w.reshape(B, heads, q_h, q_w, 1, k_w)
        return rel_h + rel_w

    def __call__(self, x: mx.array) -> mx.array:
        B, H, W, _ = x.shape
        N = H * W

        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_attention_heads, -1)
            .transpose(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B, heads, N, d]

        attn = (q * self.scale) @ k.transpose(0, 1, 3, 2)  # [B, heads, N, N]

        if self.use_rel_pos:
            decomposed = self._decomposed_rel_pos(
                q, self.rel_pos_h, self.rel_pos_w, (H, W), (H, W)
            )
            attn = attn.reshape(B, self.num_attention_heads, H, W, N)
            attn = attn + decomposed.reshape(B, self.num_attention_heads, H, W, N)
            attn = attn.reshape(B, self.num_attention_heads, N, N)

        attn = mx.softmax(attn, axis=-1)
        out = (attn @ v).transpose(0, 2, 1, 3).reshape(B, H, W, -1)
        return self.proj(out)


class VisionLayer(nn.Module):
    """Pre-norm transformer block with optional window attention."""

    def __init__(self, config: SLANeXtConfig, window_size: int):
        super().__init__()
        self.window_size = window_size
        self.layer_norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.attn = VisionAttention(config, window_size)
        self.layer_norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = MLPBlock(config)

    def __call__(self, x: mx.array) -> mx.array:
        residual = x
        h = self.layer_norm1(x)
        if self.window_size > 0:
            H, W = h.shape[1], h.shape[2]
            h, pad_hw = window_partition(h, self.window_size)
        h = self.attn(h)
        if self.window_size > 0:
            h = window_unpartition(h, self.window_size, pad_hw, (H, W))
        h = residual + h
        h = h + self.mlp(self.layer_norm2(h))
        return h


class PatchEmbeddings(nn.Module):
    """Image -> patch embeddings via Conv2D (MLX NHWC layout)."""

    def __init__(self, config: SLANeXtConfig):
        super().__init__()
        self.projection = nn.Conv2d(
            config.num_channels, config.hidden_size,
            kernel_size=config.patch_size, stride=config.patch_size,
        )

    def __call__(self, x: mx.array) -> mx.array:
        # x: [B, H, W, C] (NHWC) -> [B, H/p, W/p, hidden]
        return self.projection(x)


class VisionNeck(nn.Module):
    def __init__(self, config: SLANeXtConfig):
        super().__init__()
        self.conv1 = nn.Conv2d(config.hidden_size, config.output_channels, kernel_size=1, bias=False)
        self.layer_norm1 = nn.LayerNorm(config.output_channels, eps=config.layer_norm_eps)
        self.conv2 = nn.Conv2d(config.output_channels, config.output_channels, kernel_size=3, padding=1, bias=False)
        self.layer_norm2 = nn.LayerNorm(config.output_channels, eps=config.layer_norm_eps)

    def __call__(self, x: mx.array) -> mx.array:
        # NHWC throughout
        x = self.layer_norm1(self.conv1(x))
        x = self.layer_norm2(self.conv2(x))
        return x


class VisionEncoder(nn.Module):
    """GotOcr2VisionEncoder — SAM-ViT based encoder (NHWC)."""

    def __init__(self, config: SLANeXtConfig):
        super().__init__()
        self.config = config
        self.patch_embed = PatchEmbeddings(config)
        self.pos_embed = None
        if config.use_abs_pos:
            self.pos_embed = mx.zeros(
                (1, config.image_size // config.patch_size,
                 config.image_size // config.patch_size, config.hidden_size)
            )
        self.layers = [
            VisionLayer(
                config,
                window_size=(config.window_size if i not in config.global_attn_indexes else 0),
            )
            for i in range(config.num_hidden_layers)
        ]
        self.neck = VisionNeck(config)

    def __call__(self, x: mx.array) -> mx.array:
        h = self.patch_embed(x)  # [B, H/p, W/p, hidden]
        if self.pos_embed is not None:
            h = h + self.pos_embed
        for layer in self.layers:
            h = layer(h)
        return self.neck(h)  # [B, H, W, output_channels]


class PostConv(nn.Module):
    """Stride-2 conv downsampler -> flattened [B, H*W, out_channels]."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.conv(x)  # [B, H/2, W/2, C]
        B, H, W, C = x.shape
        return x.reshape(B, H * W, C)


# ---------------------------------------------------------------------------
# SLA head (attention GRU decoder)
# ---------------------------------------------------------------------------

class PaddleGRUCell(nn.Module):
    """Paddle ``nn.GRUCell`` equivalent (gate order r, z, c).

    h = (pre_hidden - c) * z + c, with
        r = sigmoid(x_r + h_r)
        z = sigmoid(x_z + h_z)
        c = tanh(x_c + r * h_c)
    """

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.weight_ih = mx.zeros((3 * hidden_size, input_size))
        self.weight_hh = mx.zeros((3 * hidden_size, hidden_size))
        self.bias_ih = mx.zeros((3 * hidden_size,))
        self.bias_hh = mx.zeros((3 * hidden_size,))
        self.hidden_size = hidden_size

    def __call__(self, x: mx.array, h: mx.array) -> mx.array:
        x_gates = x @ self.weight_ih.T + self.bias_ih
        h_gates = h @ self.weight_hh.T + self.bias_hh
        x_r, x_z, x_c = mx.split(x_gates, 3, axis=1)
        h_r, h_z, h_c = mx.split(h_gates, 3, axis=1)
        r = mx.sigmoid(x_r + h_r)
        z = mx.sigmoid(x_z + h_z)
        c = mx.tanh(x_c + r * h_c)
        return (h - c) * z + c


class AttentionGRUCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_embeddings: int):
        super().__init__()
        self.input_to_hidden = nn.Linear(input_size, hidden_size, bias=False)
        self.hidden_to_hidden = nn.Linear(hidden_size, hidden_size)
        self.score = nn.Linear(hidden_size, 1, bias=False)
        self.rnn = PaddleGRUCell(input_size + num_embeddings, hidden_size)

    def __call__(self, prev_hidden, batch_hidden, char_onehots):
        batch_hidden_proj = self.input_to_hidden(batch_hidden)
        prev_hidden_proj = self.hidden_to_hidden(prev_hidden)[:, None, :]
        scores = mx.tanh(batch_hidden_proj + prev_hidden_proj)
        scores = self.score(scores)                      # [B, seq, 1]
        attn_weights = mx.softmax(scores, axis=1)        # [B, seq, 1]
        context = (attn_weights.transpose(0, 2, 1) @ batch_hidden).squeeze(1)
        concat = mx.concatenate([context, char_onehots], axis=1)
        hidden = self.rnn(concat, prev_hidden)
        return hidden, attn_weights


class StructureGenerator(nn.Module):
    """Two-layer MLP, no activation (matches Paddle SLANeXtMLP)."""

    def __init__(self, hidden_size: int, out_channels: int):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, out_channels)

    def __call__(self, x: mx.array) -> mx.array:
        return self.fc2(self.fc1(x))


class SLAHead(nn.Module):
    def __init__(self, config: SLANeXtConfig):
        super().__init__()
        self.config = config
        self.structure_attention_cell = AttentionGRUCell(
            config.post_conv_out_channels, config.head_hidden_size, config.out_channels
        )
        self.structure_generator = StructureGenerator(config.head_hidden_size, config.out_channels)

    def __call__(self, hidden_states: mx.array) -> mx.array:
        B = hidden_states.shape[0]
        features = mx.zeros((B, self.config.head_hidden_size))
        predicted = mx.zeros((B,), dtype=mx.int64)
        done = mx.zeros((B,), dtype=mx.bool_)

        preds_list = []
        for _ in range(self.config.max_text_length + 1):
            onehot = (predicted[:, None] == mx.arange(self.config.out_channels)[None, :]).astype(mx.float32)
            features, _ = self.structure_attention_cell(
                features, hidden_states.astype(mx.float32), onehot
            )
            step = self.structure_generator(features)
            predicted = mx.argmax(step, axis=1).astype(mx.int64)
            preds_list.append(step)
            done = done | (predicted == self.config.out_channels - 1)
            if bool(mx.all(done)):
                break

        structure_preds = mx.stack(preds_list, axis=1)
        return mx.softmax(structure_preds, axis=-1)


# ---------------------------------------------------------------------------
# Backbone + full model
# ---------------------------------------------------------------------------

class Backbone(nn.Module):
    def __init__(self, config: SLANeXtConfig):
        super().__init__()
        self.vision_tower = VisionEncoder(config)
        self.post_conv = PostConv(config.post_conv_in_channels, config.post_conv_out_channels)

    def __call__(self, x: mx.array) -> mx.array:
        h = self.vision_tower(x)     # NHWC [B,H,W,256]
        return self.post_conv(h)     # [B, H*W, 512]


class SLANeXt(nn.Module):
    """SLANeXt table structure recognition (structure head only).

    Usage::

        model = SLANeXt(SLANeXtConfig.from_config_dict(json.load(open(cfg))))
        model.load_weights(load_mlx_weights("table_wired"))
        probs = model(mx.array(img[None]))   # [B, seq_len, out_channels]
    """

    def __init__(self, config: SLANeXtConfig | None = None):
        super().__init__()
        self.config = config or SLANeXtConfig()
        self.backbone = Backbone(self.config)
        self.head = SLAHead(self.config)

    def __call__(self, x: mx.array) -> mx.array:
        # x: [B, C, H, W] NCHW -> NHWC
        if x.shape[1] == 1:
            x = mx.broadcast_to(x, (x.shape[0], 3, x.shape[2], x.shape[3]))
        x = x.transpose(0, 2, 3, 1)
        features = self.backbone(x)
        return self.head(features)
