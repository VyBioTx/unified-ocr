"""CLI for the assembled MLX PP-StructureV3 patent-table pipeline.

Examples::

    # one page image → JSON + markdown
    python -m unified_ocr.patent_table_mlx.cli run page_02.png -o out/

    # whole PDF (rendered with PyMuPDF, paper det params)
    python -m unified_ocr.patent_table_mlx.cli pdf patent.pdf -o out/ --device cpu
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from .pipeline import PatentPipelineMLXConfig, PatentTableMLXPipeline


def _results_to_dict(results, source: str) -> dict:
    return {
        "source": str(source),
        "config": {
            "layout_model": "PP-DocLayout_plus-L",
            "cell_model": "RT-DETR-L_wired_table_cell_det",
            "det_model": "PP-OCRv5_server_det",
            "rec_model": "en_PP-OCRv4_mobile_rec",
            "structure_model": "SLANeXt_wired (MLX native)",
        },
        "tables": [
            {
                "page_index": t.page_index,
                "table_index": t.table_index,
                "box": [round(v, 2) for v in t.box],
                "html": t.html,
                "markdown": t.to_markdown(),
                "n_cells": len(t.cells),
                "n_ocr": len(t.ocr_texts),
            }
            for t in results
        ],
    }


def _write_outputs(payload: dict, out_dir: Path, stem: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{stem}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    md = [f"# {payload['source']} — PP-StructureV3 (MLX) 表格抽取\n"]
    for t in payload["tables"]:
        md.append(f"\n## 第 {t['page_index']} 页 · 表 {t['table_index'] + 1}\n")
        md.append(t["markdown"] or t["html"])
        md.append("")
    (out_dir / f"{stem}.md").write_text("\n".join(md), encoding="utf-8")


def _build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--device", default="cpu",
                        help="PaddleX device, e.g. cpu / gpu:0")
    common.add_argument("--slanext-dir", default="models/ppocr-mlx/table_wired")

    p = argparse.ArgumentParser(prog="patent-table-mlx")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="run one page image", parents=[common])
    r.add_argument("image")
    r.add_argument("-o", "--output", default="out")

    d = sub.add_parser("pdf", help="run a PDF (rendered at --dpi)", parents=[common])
    d.add_argument("pdf")
    d.add_argument("-o", "--output", default="out")
    d.add_argument("--max-pages", type=int, default=None)
    d.add_argument("--dpi", type=int, default=300)
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    out_dir = Path(args.output)

    cfg = PatentPipelineMLXConfig(
        device=args.device,
        slanext_dir=args.slanext_dir,
        pdf_dpi=getattr(args, "dpi", 300),
    )

    if args.cmd == "run":
        src = Path(args.image)
        stem = src.stem + ".mlx_tables"
    else:
        src = Path(args.pdf)
        stem = src.stem + ".mlx_tables"

    pipe = PatentTableMLXPipeline(cfg)
    try:
        if args.cmd == "run":
            results = pipe.process_image(src)
        else:
            results = pipe.process_pdf(src, max_pages=args.max_pages)
        payload = _results_to_dict(results, src)
        _write_outputs(payload, out_dir, stem)
        sys.stdout.write(
            f"wrote {out_dir / (stem + '.json')} ({len(results)} table(s))\n"
        )
        sys.stdout.flush()
    finally:
        try:
            pipe.close()
        except Exception:
            pass

    # Paddle-on-macOS can raise in C++ static destructors after a long run;
    # all outputs are already flushed, so exit hard to keep the process clean.
    os._exit(0)


if __name__ == "__main__":
    sys.exit(main())
