"""CLI for the assembled MLX PP-StructureV3 pipeline.

Two modes:
  * **table** (default) — extract only the table regions (HTML + Markdown).
  * **layout** — full document: layout analysis over all regions, tables plus
    non-table content (titles / text / captions / formulas / figures) rendered
    to a single Markdown document.

Examples::

    # one page image → tables only
    python -m unified_ocr.patent_table_mlx.cli run page_02.png -o out/

    # whole PDF → full document markdown (tables + text)
    python -m unified_ocr.patent_table_mlx.cli pdf patent.pdf -o out/ --layout

    # one page image → full document
    python -m unified_ocr.patent_table_mlx.cli run page_02.png -o out/ --layout
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from .pipeline import PatentPipelineMLXConfig, PatentTableMLXPipeline


def _config_dict(rec_model: str, rec_lang: str, detected_lang: str | None = None) -> dict:
    return {
        "layout_model": "PP-DocLayout_plus-L",
        "cell_model": "RT-DETR-L_wired_table_cell_det",
        "det_model": "PP-OCRv5_server_det",
        "rec_model": rec_model,
        "rec_lang": rec_lang,
        "detected_lang": detected_lang,
        "structure_model": "SLANeXt_wired (MLX native)",
    }


def _results_to_dict(
    results, source: str, rec_model: str, rec_lang: str,
    detected_lang: str | None = None,
) -> dict:
    return {
        "source": str(source),
        "mode": "table",
        "config": _config_dict(rec_model, rec_lang, detected_lang),
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


def _regions_to_dict(
    results, source: str, rec_model: str, rec_lang: str,
    detected_lang: str | None = None,
) -> dict:
    return {
        "source": str(source),
        "mode": "layout",
        "config": _config_dict(rec_model, rec_lang, detected_lang),
        "regions": [
            {
                "page_index": r.page_index,
                "region_index": r.region_index,
                "label": r.label,
                "box": [round(v, 2) for v in r.box],
                "kind": "table" if r.is_table else "text",
                "markdown": r.markdown,
                "html": r.table.html if r.table is not None else None,
                "n_cells": len(r.table.cells) if r.table is not None else None,
                "n_ocr": len(r.texts),
            }
            for r in results
        ],
    }


def _write_outputs(payload: dict, out_dir: Path, stem: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{stem}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if payload.get("mode") == "layout":
        body = PatentTableMLXPipeline.document_markdown(
            _rehydrate_regions(payload)
        )
        md = f"# {payload['source']} — PP-StructureV3 (MLX) 全文档抽取\n\n{body}\n"
    else:
        md = [f"# {payload['source']} — PP-StructureV3 (MLX) 表格抽取\n"]
        for t in payload["tables"]:
            md.append(f"\n## 第 {t['page_index']} 页 · 表 {t['table_index'] + 1}\n")
            md.append(t["markdown"] or t["html"])
            md.append("")
        md = "\n".join(md)
    (out_dir / f"{stem}.md").write_text(md, encoding="utf-8")


def _rehydrate_regions(payload: dict):
    """Rebuild lightweight RegionResult objects from the JSON payload."""
    from .pipeline import RegionResult

    return [
        RegionResult(
            page_index=r["page_index"],
            region_index=r["region_index"],
            label=r["label"],
            box=r["box"],
            markdown=r["markdown"],
        )
        for r in payload["regions"]
    ]


def _build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--device", default="cpu",
                        help="PaddleX device, e.g. cpu / gpu:0")
    common.add_argument("--slanext-dir", default="models/ppocr-mlx/table_wired")
    common.add_argument("--rec-lang", default="auto",
                        help="recognizer language: auto (default; detect from "
                             "the document) / en (paper default) / ch (Chinese, "
                             "uses PP-OCRv5_server_rec)")
    common.add_argument("--rec-model", default=None,
                        help="explicit recognizer model name (overrides --rec-lang)")
    common.add_argument("--lang-detect-pages", type=int, default=3,
                        help="pages sampled by --rec-lang auto detection")
    common.add_argument("--layout", action="store_true",
                        help="full-document mode: also recognise non-table regions")

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

    cfg_kwargs: dict = {
        "device": args.device,
        "slanext_dir": args.slanext_dir,
        "pdf_dpi": getattr(args, "dpi", 300),
        "rec_lang": args.rec_lang,
        "lang_detect_pages": args.lang_detect_pages,
    }
    if args.rec_model:
        cfg_kwargs["rec_model"] = args.rec_model
    cfg = PatentPipelineMLXConfig(**cfg_kwargs)

    if args.cmd == "run":
        src = Path(args.image)
    else:
        src = Path(args.pdf)
    stem = src.stem + (".mlx_layout" if args.layout else ".mlx_tables")

    pipe = PatentTableMLXPipeline(cfg)
    try:
        if args.layout:
            if args.cmd == "run":
                regions = pipe.process_image_layout(src)
            else:
                regions = pipe.process_pdf_layout(src, max_pages=args.max_pages)
            rec_model, detected = pipe.rec_model_name(), pipe.detected_language
            payload = _regions_to_dict(regions, src, rec_model, args.rec_lang,
                                       detected)
            n = len(payload["regions"])
            n_tables = sum(1 for r in payload["regions"] if r["kind"] == "table")
            _write_outputs(payload, out_dir, stem)
            sys.stdout.write(
                f"wrote {out_dir / (stem + '.json')} "
                f"({n} region(s), {n_tables} table(s), lang={detected})\n"
            )
        else:
            if args.cmd == "run":
                results = pipe.process_image(src)
            else:
                results = pipe.process_pdf(src, max_pages=args.max_pages)
            rec_model, detected = pipe.rec_model_name(), pipe.detected_language
            payload = _results_to_dict(results, src, rec_model, args.rec_lang,
                                       detected)
            _write_outputs(payload, out_dir, stem)
            sys.stdout.write(
                f"wrote {out_dir / (stem + '.json')} "
                f"({len(results)} table(s), lang={detected})\n"
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
