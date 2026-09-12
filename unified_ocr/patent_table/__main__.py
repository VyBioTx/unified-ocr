"""patent_table 模块 CLI 入口：python -m unified_ocr.patent_table <cmd>。"""

from __future__ import annotations

import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main())
