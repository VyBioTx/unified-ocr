"""专利表格 OCR 子模块：基于 PP-StructureV3 的 siRNA 专利表格抽取专用链。

该模块实现了论文 FENNEC（2026）Methods → Data curation 中描述的
专利表格 OCR 方案，使用 PaddleOCR 的 PP-StructureV3 流水线：

  1. RT-DETR-L 检测表格单元格
  2. SLANeXt 还原表格结构（行列/合并关系）
  3. en PP-OCRv4 mobile rec 识别单元格文本

输出结构化的行列数据，用于 siRNA 序列与敲低效应表的合并。
"""

from __future__ import annotations

from .download_weights import download_weights
from .pipeline import PatentTablePipeline, PatentTablePipelineConfig
from .parser import TableCell, TableRow, TableStructure, parse_html_table
from .qc import QCFilter, QCSpec
from .merge import merge_sequence_knockdown

__all__ = [
    "download_weights",
    "PatentTablePipeline",
    "PatentTablePipelineConfig",
    "TableCell",
    "TableRow",
    "TableStructure",
    "parse_html_table",
    "QCFilter",
    "QCSpec",
    "merge_sequence_knockdown",
]