"""专利表格 OCR 顶层入口脚本。

基于 PP-StructureV3 的专利表格 OCR 抽取流程，用于从专利 PDF
中提取 siRNA 序列表与敲低效应表。

用法::

    # 1. 下载 PP-StructureV3 三件套权重
    python run_patent_table.py download

    # 2. 对专利 PDF 渲染页运行表格识别（输出 HTML + 结构化数据）
    python run_patent_table.py run page.png -o result.json

    # 3. 完整流程：识别 → 解析 → QC 过滤 → 合并
    python run_patent_table.py full page.png -o result.json

    # 4. 列出支持的模型
    python run_patent_table.py list-models
"""

from __future__ import annotations

import sys

from unified_ocr.patent_table.cli import main

if __name__ == "__main__":
    sys.exit(main())