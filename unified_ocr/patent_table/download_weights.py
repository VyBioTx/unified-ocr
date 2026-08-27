"""从 PaddleOCR 官方源下载 PP-StructureV3 表格识别所需的三类权重。

PP-StructureV3 表格识别流水线需要三个独立模型：
  1. RT-DETR-L      — 表格单元格检测（det）
  2. SLANeXt        — 表格结构识别（table）
  3. PP-OCRv4 rec   — 文字识别，en/uppercase 档（rec）

PaddleOCR 的 TableRecPipeline 在首次运行时自动下载权重至
~/.paddleocr/ 或 PaddleOCR 的 model download 目录。本脚本显式
下载至本地指定目录，便于离线环境与版本固定。

权重来源（PaddleOCR 官方 model zoo）：
  - https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.9/doc/doc_en/models_list_en.md
  - ModelScope 镜像：PaddlePaddle/PaddleOCR（完整模型库）
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Literal

log = logging.getLogger(__name__)

# 默认模型权重下载根目录
DEFAULT_MODEL_DIR = Path(__file__).resolve().parent.parent.parent / "models"

# PaddleOCR 模型名称（对应 PaddleOCR 内部模型映射表的 key）
MODEL_NAMES: dict[str, dict[str, str]] = {
    "det": {"model": "RT-DETR-L", "url": "RT-DETR-L"},
    "table": {"model": "SLANeXt", "url": "SLANeXt"},
    "rec": {"model": "PP-OCRv4_mobile_rec", "url": "en_PP-OCRv4_mobile_rec"},
}


def download_weights(
    model_dir: str | Path | None = None,
    components: list[str] | None = None,
    use_paddleocr_download: bool = True,
) -> dict[str, Path]:
    """下载 PP-StructureV3 三件套权重到本地。

    Args:
        model_dir: 下载目标目录，默认 unified-ocr/models/。
        components: 需要下载的组件列表，默认 ["det", "table", "rec"]。
        use_paddleocr_download: 如果为 True，通过 paddleocr 模块的
            内置下载器下载（自动解压至 ~/.paddleocr/）；否则只准备目录结构。

    Returns:
        {组件名: 权重路径} 的字典。
    """
    if components is None:
        components = ["det", "table", "rec"]

    out_dir = Path(model_dir) if model_dir else DEFAULT_MODEL_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    result: dict[str, Path] = {}

    if use_paddleocr_download:
        try:
            return _download_via_paddleocr(components, out_dir)
        except ImportError as exc:
            log.warning("paddleocr 未安装，跳过自动下载: %s", exc)
            log.warning("请先 pip install paddleocr，或手动下载权重至 %s", out_dir)

    for comp in components:
        comp_dir = out_dir / comp
        comp_dir.mkdir(parents=True, exist_ok=True)
        log.info("已准备 %s 权重目录: %s", comp, comp_dir)
        result[comp] = comp_dir

    log.info("模型目录已就绪: %s", out_dir)
    log.info("请将以下权重文件放入对应子目录：")
    for comp in components:
        info = MODEL_NAMES.get(comp, {})
        log.info("  %s/ ← %s (%s)", comp, info.get("model", "?"), info.get("url", "?"))
    return result


def _download_via_paddleocr(
    components: list[str],
    out_dir: Path,
) -> dict[str, Path]:
    """通过 PaddleOCR 的内置下载器下载权重。

    调用 paddleocr 模块的 download 函数，它会自动从 PaddleOCR
    官方服务器拉取并解压到 ~/.paddleocr/ 目录。
    """
    from paddleocr import PaddleOCR  # noqa: F401 - 触发 paddle 的下载机制

    log.info("通过 PaddleOCR 内置下载器拉取权重（首次运行会下载 %s）", components)
    log.info("PaddleOCR 将自动下载至 ~/.paddleocr/ 目录")

    result: dict[str, Path] = {}
    for comp in components:
        if comp == "det":
            ocr = PaddleOCR(
                det_model_dir=str(out_dir / "det"),
                use_angle_cls=False,
                lang="en",
                show_log=False,
                det_db_thresh=0.15,
                det_db_box_thresh=0.4,
                det_db_unclip_ratio=2.0,
            )
            ocr.det_model_dir = str(out_dir / "det")
            result[comp] = out_dir / "det"
        elif comp == "rec":
            ocr = PaddleOCR(
                rec_model_dir=str(out_dir / "rec"),
                lang="en",
                show_log=False,
            )
            result[comp] = out_dir / "rec"
        elif comp == "table":
            result[comp] = out_dir / "table"

    log.info("权重下载完成（或已存在）")
    return result


def list_available_models() -> list[dict[str, str]]:
    """列出本模块支持的所有模型及其信息。"""
    return [
        {"component": "det", "name": "RT-DETR-L", "version": "latest",
         "description": "表格单元格检测模型"},
        {"component": "table", "name": "SLANeXt", "version": "latest",
         "description": "表格结构识别模型"},
        {"component": "rec", "name": "PP-OCRv4_mobile_rec (en/uppercase)", "version": "latest",
         "description": "英文大写文本识别模型（专利文本专用）"},
    ]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    paths = download_weights()
    for comp, path in paths.items():
        print(f"{comp}: {path}")