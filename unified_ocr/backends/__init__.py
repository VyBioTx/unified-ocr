"""后端适配层包：导入即完成引擎注册。"""

from . import hunyuan_transformers, mlx_vlm  # noqa: F401   (注册副作用)

__all__ = ["hunyuan_transformers", "mlx_vlm"]