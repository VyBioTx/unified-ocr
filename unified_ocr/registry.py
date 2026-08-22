"""引擎注册表。

引擎按 id 注册与查找，CLI 与 Python API 通过注册表解耦具体后端。
"""

from __future__ import annotations

from typing import Type

from .base import EngineSpec, OCRBackend

_REGISTRY: dict[str, Type[OCRBackend]] = {}


def register(cls: Type[OCRBackend]) -> Type[OCRBackend]:
    """类装饰器：按 cls.spec.id 注册后端。"""
    engine_id = cls.spec.id
    if engine_id in _REGISTRY:
        raise ValueError(f"引擎已被注册: {engine_id}")
    _REGISTRY[engine_id] = cls
    return cls


def get_backend_class(engine_id: str) -> Type[OCRBackend]:
    if engine_id not in _REGISTRY:
        raise KeyError(
            f"未知引擎 '{engine_id}'。可用: {', '.join(sorted(_REGISTRY))}。"
            f"（部分后端依赖可选依赖包，请先安装对应 extras）"
        )
    return _REGISTRY[engine_id]


def create_backend(engine_id: str, **kwargs: object) -> OCRBackend:
    return get_backend_class(engine_id)(**kwargs)


def list_engines() -> list[EngineSpec]:
    return [cls.spec for cls in _REGISTRY.values()]


def has_engine(engine_id: str) -> bool:
    return engine_id in _REGISTRY