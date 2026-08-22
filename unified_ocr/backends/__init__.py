"""后端适配层包：导入即完成引擎注册。"""

from . import llama_cpp, mlx_vlm  # noqa: F401   (注册副作用)

__all__ = ["llama_cpp", "mlx_vlm"]