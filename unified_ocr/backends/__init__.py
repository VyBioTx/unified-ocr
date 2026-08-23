"""后端适配层包：导入即完成引擎注册。"""

from . import (  # noqa: F401   (注册副作用)
    dots_mocr,
    dots_mocr_mlx,
    hunyuan_transformers,
    mlx_vlm,
)

__all__ = [
    "dots_mocr",
    "dots_mocr_mlx",
    "hunyuan_transformers",
    "mlx_vlm",
]
