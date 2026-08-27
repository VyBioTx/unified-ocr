"""PP-StructureV3 表格识别流水线封装。

按论文配置组装 TableRecPipeline（PP-StructureV3），
以论文指定的参数运行：
  - det_limit_side_len = 3000
  - det_db_thresh = 0.15
  - det_db_box_thresh = 0.4
  - det_db_unclip_ratio = 1.5 ~ 2.0
  - 语言: en (uppercase)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..models import OCRResult

log = logging.getLogger(__name__)


@dataclass
class PatentTablePipelineConfig:
    """论文指定的 PP-StructureV3 表格识别参数。"""

    # 检测器参数
    det_limit_side_len: int = 3000
    det_db_thresh: float = 0.15
    det_db_box_thresh: float = 0.4
    det_db_unclip_ratio: float = 2.0

    # 表格结构识别参数
    table_model: str | None = None

    # 文字识别参数
    rec_model: str | None = None
    rec_lang: str = "en"

    # 输出控制
    output_dir: str | Path | None = None
    show_log: bool = False
    use_gpu: bool = False

    def to_paddleocr_kwargs(self) -> dict[str, Any]:
        """转换为 PaddleOCR 的 TableRecPipeline 构造参数。"""
        kwargs: dict[str, Any] = {
            "det_limit_side_len": self.det_limit_side_len,
            "det_db_thresh": self.det_db_thresh,
            "det_db_box_thresh": self.det_db_box_thresh,
            "det_db_unclip_ratio": self.det_db_unclip_ratio,
            "lang": self.rec_lang,
            "show_log": self.show_log,
            "use_gpu": self.use_gpu,
        }
        if self.table_model:
            kwargs["table_model_dir"] = self.table_model
        if self.rec_model:
            kwargs["rec_model_dir"] = self.rec_model
        if self.output_dir:
            kwargs["output"] = str(self.output_dir)
        return kwargs


class PatentTablePipeline:
    """专利表格 OCR 流水线。

    封装 PaddleOCR 的 PP-StructureV3 表格识别 (TableRecPipeline)，
    按论文参数配置，输入 PDF 渲染图，输出结构化表格结果。

    用法::

        pipe = PatentTablePipeline(config=PatentTablePipelineConfig())
        tables = pipe.process("patent_page.png")
        for table in tables:
            print(table["html"])
    """

    def __init__(
        self,
        config: PatentTablePipelineConfig | None = None,
    ) -> None:
        self.config = config or PatentTablePipelineConfig()
        self._pipeline: Any = None

    def load(self) -> None:
        """惰性加载 PP-StructureV3 流水线。

        首次调用时下载模型权重（若本地不存在）。
        """
        if self._pipeline is not None:
            return
        try:
            from paddleocr import PPStructureV3, PPStructure
        except ImportError as exc:
            raise ImportError(
                "需要 paddleocr。请安装：pip install paddleocr>=2.9"
            ) from exc

        log.info("初始化 PP-StructureV3 表格识别流水线（参数: limit_side_len=%d, "
                 "db_thresh=%.2f, box_thresh=%.2f, unclip_ratio=%.1f）",
                 self.config.det_limit_side_len,
                 self.config.det_db_thresh,
                 self.config.det_db_box_thresh,
                 self.config.det_db_unclip_ratio)

        kwargs = self.config.to_paddleocr_kwargs()
        try:
            self._pipeline = PPStructureV3(**kwargs)
        except TypeError:
            log.info("PPStructureV3 不可用，回退到 PPStructure (v2)")
            kwargs.pop("det_limit_side_len", None)
            self._pipeline = PPStructure(**kwargs)

    def process(self, image: str | Path) -> list[dict[str, Any]]:
        """对一张图像运行表格识别。

        Args:
            image: 图像路径（PNG/JPEG，建议 3000px+ 分辨率）。

        Returns:
            表格列表，每项为 PPStructure 的输出字典，包含:
              - type: "table"
              - html: 表格的 HTML 表示（含 <table><tr><td>）
              - img: 可选，表格区域截图
              - res: 结构化单元格列表
        """
        self.load()
        log.info("识别表格: %s", image)
        result = self._pipeline.predict(str(image))
        tables = [r for r in result if isinstance(r, dict) and r.get("type") == "table"]
        log.info("检测到 %d 个表格", len(tables))
        return tables

    def close(self) -> None:
        self._pipeline = None


def process_patent_page(
    image_path: str | Path,
    unclip_ratio: float = 2.0,
    **kwargs: Any,
) -> list[dict[str, Any]]:
    """快捷函数：对单张专利页面图像运行表格识别。

    Args:
        image_path: 图像路径。
        unclip_ratio: 检测框展开比例（1.5 或 2.0）。
        kwargs: 其他 PatentTablePipelineConfig 参数。

    Returns:
        表格识别结果列表。
    """
    config = PatentTablePipelineConfig(det_db_unclip_ratio=unclip_ratio, **kwargs)
    pipe = PatentTablePipeline(config)
    try:
        return pipe.process(image_path)
    finally:
        pipe.close()