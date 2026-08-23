"""将专利 PDF 渲染为高分辨率页面图片，供 OCR 引擎识别。"""
import sys
from pathlib import Path

import fitz  # PyMuPDF

PDF = Path("attachments/提取自82-CN108251420B-CTGF-瑞博.pdf")
OUT = Path("pages")
DPI = 200


def main() -> int:
    OUT.mkdir(exist_ok=True)
    doc = fitz.open(PDF)
    print(f"pages: {doc.page_count}")
    for i, page in enumerate(doc):
        pix = page.get_pixmap(dpi=DPI)
        out_png = OUT / f"page_{i + 1:02d}.png"
        pix.save(out_png)
        print(f"  -> {out_png}  {pix.width}x{pix.height}")
    doc.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
