"""SLANeXt table structure recognition — pure MLX implementation.

Architecture (from PaddleX 3.7 slanext.py)::

    SLANeXt
      ├── backbone.vision_tower (GotOcr2VisionEncoder)
      │     ├── patch_embed: Conv2D(3→768, k=16, s=16) → [B, 32, 32, 768]
      │     ├── pos_embed: learnable [1, 32, 32, 768]
      │     ├── 12× GotOcr2VisionLayer:
      │     │     ├── LayerNorm (NHWC)
      │     │     ├── WindowAttention (qkv→proj + rel_pos_h/w)
      │     │     └── MLP (768→3072→768)
      │     └── neck: Conv2D(768→256)→LN→Conv2D(256→256)→LN  → [B, 256, H, W]
      ├── post_conv: Conv2D(256→512, k=3, s=2) → [B, 512, 16, 16] → [B, 256, 512]
      └── head
            ├── attention_cell (GRU + attention)
            └── generator (MLP 512→512→50)

Config defaults:
    hidden_size=768, num_hidden_layers=12, num_attention_heads=12,
    image_size=512, patch_size=16, mlp_dim=3072,
    window_size=14, global_attn_indexes=[2,5,8,11],
    post_conv_out_channels=512, out_channels=50, max_text_length=500
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import mlx.core as mx
import mlx.nn as nn


# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------

@dataclass
class SLANeXtConfig:
    """SLANeXt model configuration (defaults = SLANeXt_wired)."""
    # Vision encoder (SAM-ViT)
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

    # Post conv (backbone downsample)
    post_conv_in_channels: int = 256
    post_conv_out_channels: int = 512

    # SLA head
    out_channels: int = 50      # vocabulary size for structure tokens
    max_text_length: int = 500


# ---------------------------------------------------------------------------
# MLP
# ---------------------------------------------------------------------------

class MLPBlock(nn.Module):
    def __init__(self, hidden_size: int, mlp_dim: int):
        super().__init__()
        self.lin1 = nn.Linear(hidden_size, mlp_dim)
        self.lin2 = nn.Linear(mlp_dim, hidden_size)
        self.act = nn.GELU()

    def __call__(self, x: mx.array) -> mx.array:
        return self.lin2(self.act(self.lin1(x)))


# ---------------------------------------------------------------------------
# Windowed Attention with Relative Position Bias
# ---------------------------------------------------------------------------

def get_rel_pos(q_size: int, k_size: int, rel_pos: mx.array) -> mx.array:
    """Relative position bias along one axis — bilinear interpolation.
    
    rel_pos shape: [2*max(q_size,k_size)-1, head_dim]
    Returns: [q_size, k_size, head_dim]
    """
    max_rel_dist = int(2 * max(q_size, k_size) - 1)
    if rel_pos.shape[0] != max_rel_dist:
        # Interpolate rel_pos
        old_len = rel_pos.shape[0]
        # Use simple linear interpolation via reshape → resize
        rel_pos = rel_pos.T  # [head_dim, old_len]
        # Resize to max_rel_dist
        scale = max_rel_dist / old_len
        # Simple 1D interpolation
        x = mx.arange(max_rel_dist, dtype=mx.float32) / scale
        rel_pos = mx.interpolate_one(rel_pos, x)  # uses floor rounding
        rel_pos = rel_pos.T  # [max_rel_dist, head_dim]
    
    # Generate relative coordinates
    coords_h = mx.arange(q_size, dtype=mx.float32)[:, None] - mx.arange(k_size, dtype=mx.float32)[None, :]
    coords_h = mx.clip(coords_h + max_rel_dist // 2, 0, max_rel_dist - 1).astype(mx.int64)
    return rel_pos[coords_h]


# Register custom operator for 1D interpolation if not available
def _interpolate_1d(x: mx.array, indices: mx.array) -> mx.array:
    """Simple linear interpolation for 1D tensor."""
    # x: [C, L], indices: [N], returns [C, N]
    C, L = x.shape
    N = indices.shape[0]
    idx0 = mx.floor(indices).astype(mx.int64)
    idx1 = mx.minimum(idx0 + 1, L - 1)
    frac = indices - idx0
    return x[:, idx0] * (1 - frac) + x[:, idx1] * frac


class WindowAttention(nn.Module):
    """Multi-head attention with optional windowing and relative position."""

    def __init__(self, config: SLANeXtConfig, window_size: int):
        super().__init__()
        self.num_heads = config.num_attention_heads
        head_dim = config.hidden_size // config.num_attention_heads
        self.scale = head_dim ** -0.5
        self.window_size = window_size

        # Determine spatial size for rel_pos init
        input_size = config.image_size // config.patch_size  # 32
        self.spatial_size = input_size if window_size == 0 else window_size

        self.qkv = nn.Linear(config.hidden_size, config.hidden_size * 3, bias=config.qkv_bias)
        self.proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.use_rel_pos = config.use_rel_pos

        if self.use_rel_pos:
            rel_dim = 2 * self.spatial_size - 1
            self.rel_pos_h = mx.zeros((rel_dim, head_dim))
            self.rel_pos_w = mx.zeros((rel_dim, head_dim))

    def _get_decomposed_rel_pos(
        self, attn: mx.array, q: mx.array,
        rel_pos_h: mx.array, rel_pos_w: mx.array,
        q_size: tuple, k_size: tuple,
    ) -> mx.array:
        """Add decomposed relative position bias to attention scores."""
        q_h, q_w = q_size
        k_h, k_w = k_size
        Rh = get_rel_pos(q_h, k_h, rel_pos_h)  # [q_h, k_h, head_dim]
        Rw = get_rel_pos(q_w, k_w, rel_pos_w)  # [q_w, k_w, head_dim]

        # q: [B, num_heads, H*W, head_dim] → reshape for position interaction
        B, nH, HW, d = q.shape
        q = q.reshape(B, nH, q_h, q_w, d)
        rel_h = mx.einsum("bnhwd,qkd->bnhqkw", q, Rh)
        rel_w = mx.einsum("bnhwd,kwd->bnhqkw", q, Rw)
        # rel_h shape: [B, nH, q_h, k_h, q_w]
        # Sum over k_h dimension to get [B, nH, q_h, q_w]
        attn_h = rel_h.sum(axis=-1)  # [B, nH, q_h, q_w]
        attn_w = rel_w.sum(axis=-2)  # [B, nH, q_h, q_w]
        return attn_h + attn_w

    def __call__(self, x: mx.array) -> mx.array:
        B, H, W, C = x.shape
        N = H * W

        if self.window_size > 0 and (H > self.window_size or W > self.window_size):
            # Window partition
            pad_h = (self.window_size - H % self.window_size) % self.window_size
            pad_w = (self.window_size - W % self.window_size) % self.window_size
            if pad_h > 0 or pad_w > 0:
                x = mx.pad(x, ((0, 0), (0, pad_h), (0, pad_w), (0, 0)))
            Hp, Wp = H + pad_h, W + pad_w
            nW_h = Hp // self.window_size
            nW_w = Wp // self.window_size
            w = self.window_size
            x = x.reshape(B, nW_h, w, nW_w, w, C).transpose(0, 1, 3, 2, 4, 5)
            x = x.reshape(B * nW_h * nW_w, w, w, C)
            # Run attention per window
            x = self._forward_attn(x, w, w)
            # Reverse window partition
            x = x.reshape(B, nW_h, nW_w, w, w, C)
            x = x.transpose(0, 1, 3, 2, 4, 5).reshape(B, Hp, Wp, C)
            x = x[:, :H, :W, :]
        else:
            x = self._forward_attn(x, H, W)

        return x

    def _forward_attn(self, x: mx.array, H: int, W: int) -> mx.array:
        B, _, _, C = x.shape
        N = H * W

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.transpose(2, 0, 3, 1, 4)  # [3, B, nH, N, d]
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q * self.scale) @ k.transpose(0, 1, 3, 2)  # [B, nH, N, N]

        if self.use_rel_pos:
            rel = self._get_decomposed_rel_pos(
                attn, q, self.rel_pos_h, self.rel_pos_w, (H, W), (H, W)
            )
            # Add relative position bias — reshape to match attn dimensions
            attn = attn.reshape(B, self.num_heads, H, W, N)
            attn[:, :, :H, :W, :] = attn[:, :, :H, :W, :] + rel.reshape(
                B, self.num_heads, rel.shape[2], rel.shape[3], 1
            )
            attn = attn.reshape(B, self.num_heads, N, N)

        attn = mx.softmax(attn, axis=-1)
        x = (attn @ v).transpose(0, 2, 1, 3).reshape(B, H, W, C)
        return self.proj(x)


# ---------------------------------------------------------------------------
# Vision Transformer Layer
# ---------------------------------------------------------------------------

class VisionLayer(nn.Module):
    """Transformer block with pre-norm and optional window attention."""

    def __init__(self, config: SLANeXtConfig, layer_idx: int):
        super().__init__()
        window_size = 0 if layer_idx in config.global_attn_indexes else config.window_size
        self.window_size = window_size
        self.layer_norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.attn = WindowAttention(config, window_size)
        self.layer_norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = MLPBlock(config.hidden_size, config.mlp_dim)

    def __call__(self, x: mx.array) -> mx.array:
        # Pre-norm
        x = x + self.attn(self.layer_norm1(x))
        x = x + self.mlp(self.layer_norm2(x))
        return x


# ---------------------------------------------------------------------------
# Patch Embeddings
# ---------------------------------------------------------------------------

class PatchEmbeddings(nn.Module):
    """Image → patch embeddings via Conv2D."""

    def __init__(self, config: SLANeXtConfig):
        super().__init__()
        self.projection = nn.Conv2d(
            config.num_channels, config.hidden_size,
            kernel_size=config.patch_size, stride=config.patch_size,
        )

    def __call__(self, x: mx.array) -> mx.array:
        # x: [B, C, H, W] → [B, C, H/p, W/p] → [B, H/p, W/p, C]
        x = self.projection(x)
        x = x.transpose(0, 2, 3, 1)  # NHWC
        return x


# ---------------------------------------------------------------------------
# Vision Neck
# ---------------------------------------------------------------------------

class VisionNeck(nn.Module):
    """Post-encoder conv projection neck."""

    def __init__(self, config: SLANeXtConfig):
        super().__init__()
        self.conv1 = nn.Conv2d(config.hidden_size, config.output_channels, kernel_size=1, bias=False)
        self.layer_norm1 = nn.LayerNorm(config.output_channels, eps=config.layer_norm_eps)
        self.conv2 = nn.Conv2d(config.output_channels, config.output_channels, kernel_size=3, padding=1, bias=False)
        self.layer_norm2 = nn.LayerNorm(config.output_channels, eps=config.layer_norm_eps)

    def __call__(self, x: mx.array) -> mx.array:
        # x: [B, H, W, C] → [B, C, H, W]
        x = x.transpose(0, 3, 1, 2)
        x = self.conv1(x)
        x = x.transpose(0, 2, 3, 1)  # NHWC for LayerNorm
        x = self.layer_norm1(x)
        x = x.transpose(0, 3, 1, 2)  # NCHW for Conv2d
        x = self.conv2(x)
        x = x.transpose(0, 2, 3, 1)  # NHWC for LayerNorm
        x = self.layer_norm2(x)
        x = x.transpose(0, 3, 1, 2)  # Back to NCHW
        return x


# ---------------------------------------------------------------------------
# Vision Encoder (SAM-ViT)
# ---------------------------------------------------------------------------

class VisionEncoder(nn.Module):
    """GotOcr2VisionEncoder — SAM-ViT based vision encoder."""

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

        self.layers = [VisionLayer(config, i) for i in range(config.num_hidden_layers)]
        self.neck = VisionNeck(config)

    def __call__(self, x: mx.array) -> mx.array:
        # x: [B, C, H, W]
        x = self.patch_embed(x)  # [B, H/p, W/p, C]
        if self.pos_embed is not None:
            x = x + self.pos_embed
        for layer in self.layers:
            x = layer(x)
        x = self.neck(x)  # [B, C, H, W] — NCHW
        return x


# ---------------------------------------------------------------------------
# Post Conv (backbone downsampler)
# ---------------------------------------------------------------------------

class PostConv(nn.Module):
    """Conv2D stride-2 projection after vision encoder."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        # x: [B, C, H, W] NCHW
        x = self.conv(x)  # [B, out, H/2, W/2]
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(0, 2, 1)  # [B, H*W, C]
        return x


# ---------------------------------------------------------------------------
# Attention GRU Cell (Autoregressive Decoder)
# ---------------------------------------------------------------------------

class AttentionGRUCell(nn.Module):
    """Attention-based GRU cell for table structure token decoding."""

    def __init__(self, input_size: int, hidden_size: int, num_embeddings: int):
        super().__init__()
        self.input_to_hidden = nn.Linear(input_size, hidden_size, bias=False)
        self.hidden_to_hidden = nn.Linear(hidden_size, hidden_size)
        self.score = nn.Linear(hidden_size, 1, bias=False)
        self.rnn = nn.GRUCell(input_size + num_embeddings, hidden_size)

    def __call__(
        self, prev_hidden: mx.array, batch_hidden: mx.array, char_onehots: mx.array
    ) -> tuple[mx.array, mx.array]:
        # batch_hidden: [B, seq_len, input_size]
        # prev_hidden: [B, hidden_size]
        # char_onehots: [B, num_embeddings]
        batch_hidden_proj = self.input_to_hidden(batch_hidden)  # [B, seq_len, hidden]
        prev_hidden_proj = self.hidden_to_hidden(prev_hidden)[:, None, :]  # [B, 1, hidden]

        scores = mx.tanh(batch_hidden_proj + prev_hidden_proj)
        scores = self.score(scores)  # [B, seq_len, 1]
        attn_weights = mx.softmax(scores, axis=1)  # [B, seq_len, 1]
        context = (attn_weights.transpose(0, 2, 1) @ batch_hidden).squeeze(1)  # [B, input_size]
        concat = mx.concatenate([context, char_onehots], axis=1)
        hidden = self.rnn(concat, prev_hidden)
        return hidden, attn_weights


# ---------------------------------------------------------------------------
# SLA Head (Structure Locator & Attention Head)
# ---------------------------------------------------------------------------

class SLAHead(nn.Module):
    """Autoregressive SLA head — predicts table structure tokens."""

    def __init__(self, config: SLANeXtConfig):
        super().__init__()
        self.config = config
        self.attention_cell = AttentionGRUCell(
            config.post_conv_out_channels, config.hidden_size, config.out_channels,
        )
        self.generator = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.GELU() if config.hidden_act == 'gelu' else nn.ReLU(),
            nn.Linear(config.hidden_size, config.out_channels),
        )

    def __call__(self, hidden_states: mx.array) -> mx.array:
        """Autoregressive decoding.

        Args:
            hidden_states: [B, seq_len, post_conv_out_channels] (B, 256, 512)
        Returns:
            structure_probs: [B, max_length, out_channels]
        """
        B = hidden_states.shape[0]
        h_dim = self.config.hidden_size
        features = mx.zeros((B, h_dim))
        predicted = mx.zeros((B,), dtype=mx.int64)

        preds_list = []

        for _ in range(self.config.max_text_length + 1):
            onehot = mx.one_hot(predicted, self.config.out_channels).astype(mx.float32)
            features, _ = self.attention_cell(features, hidden_states.astype(mx.float32), onehot)
            logits = self.generator(features)
            predicted = mx.argmax(logits, axis=1).astype(mx.int64)
            preds_list.append(logits)

            # Check for end token (if all batch entries predict the last token index)
            # The last token index (out_channels - 1) is the EOS token
            if (predicted == self.config.out_channels - 1).all():
                break

        structure_preds = mx.stack(preds_list, axis=1)
        structure_probs = mx.softmax(structure_preds, axis=-1)
        return structure_probs


# ---------------------------------------------------------------------------
# SLANeXt (Full Model)
# ---------------------------------------------------------------------------

class SLANeXt(nn.Module):
    """SLANeXt table structure recognition — pure MLX implementation.

    Usage::

        config = SLANeXtConfig()
        model = SLANeXt(config)
        # Load weights (see weight_loader.py)
        # ...
        image = mx.zeros((1, 3, 512, 512))  # [B, C, H, W]
        probs = model(image)  # [B, seq_len, out_channels]
        token_ids = mx.argmax(probs, axis=-1)
    """

    def __init__(self, config: SLANeXtConfig):
        super().__init__()
        self.config = config
        self.backbone_vision_tower = VisionEncoder(config)
        self.backbone_post_conv = PostConv(
            config.output_channels, config.post_conv_out_channels,
        )
        self.head_attention_cell = AttentionGRUCell(
            config.post_conv_out_channels, config.hidden_size, config.out_channels,
        )
        self.head_generator = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.GELU() if config.hidden_act == 'gelu' else nn.ReLU(),
            nn.Linear(config.hidden_size, config.out_channels),
        )

    def __call__(self, x: mx.array) -> mx.array:
        # x: [B, C, H, W]
        if x.shape[1] == 1:
            x = mx.broadcast_to(x, (x.shape[0], 3, x.shape[2], x.shape[3]))

        features = self.backbone_vision_tower(x)  # [B, 256, 32, 32]
        features = self.backbone_post_conv(features)  # [B, 256, 512]

        # Autoregressive decode
        B = features.shape[0]
        h_dim = self.config.hidden_size
        hidden = mx.zeros((B, h_dim))
        predicted = mx.zeros((B,), dtype=mx.int64)

        preds_list = []
        for _ in range(self.config.max_text_length + 1):
            onehot = mx.one_hot(predicted, self.config.out_channels).astype(mx.float32)
            hidden, _ = self.head_attention_cell(hidden, features.astype(mx.float32), onehot)
            logits = self.head_generator(hidden)
            predicted = mx.argmax(logits, axis=1).astype(mx.int64)
            preds_list.append(logits)

            if (predicted == self.config.out_channels - 1).all():
                break

        return mx.stack(preds_list, axis=1)  # [B, seq_len, out_channels]