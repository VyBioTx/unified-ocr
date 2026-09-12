"""专利表格 OCR CLI 入口。

用法::

    # 下载权重
    python -m unified_ocr.patent_table download

    # 对单页 PDF 渲染图运行表格识别
    python -m unified_ocr.patent_table run page.png

    # 对 PDF 渲染图运行完整流程（识别 + 解析 + QC + 合并）
    python -m unified_ocr.patent_table full page.png --seq-table seq.html --kd-table kd.html

    # 列出支持的模型
    python -m unified_ocr.patent_table list-models
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

from .download_weights import download_weights, list_available_models
from .pipeline import PatentTablePipeline, PatentTablePipelineConfig
from .parser import (
    extract_knockdown_table,
    extract_sequence_table,
    parse_html_table,
)
from .qc import QCFilter, QCSpec
from .merge import merge_sequence_knockdown

log = logging.getLogger("patent_table")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="patent-table-ocr",
        description="基于 PP-StructureV3 的 siRNA 专利表格 OCR 抽取",
    )
    p.add_argument("-v", "--verbose", action="store_true", help="输出调试日志")
    sub = p.add_subparsers(dest="command", required=True)

    # --- download ---
    sub.add_parser("download", help="下载 PP-StructureV3 三件套权重")

    # --- list-models ---
    sub.add_parser("list-models", help="列出支持的模型")

    # --- run ---
    r = sub.add_parser("run", help="对图像运行 PP-StructureV3，输出整页 markdown + 表格 HTML")
    r.add_argument("image", help="图像路径（PNG/JPEG，建议 3000px+）")
    r.add_argument("-o", "--output", help="输出 JSON 文件路径（默认输出到终端）")
    r.add_argument("--unclip", type=float, default=2.0, choices=[1.5, 2.0],
                   help="det_db_unclip_ratio（1.5 或 2.0）")
    r.add_argument("--device", default=None,
                   help="推理设备：gpu:0 / gpu / cpu（默认 auto）")
    r.add_argument("--side-len", type=int, default=3000,
                   help="text_det_limit_side_len（默认 3000）")
    r.add_argument("--gpu-mem-mb", type=int, default=None,
                   help="paddle GPU 显存上限（MB），显存紧张时设置")
    r.add_argument("--model-dir", help="模型权重目录")

    # --- pdf ---
    pdf = sub.add_parser("pdf", help="对 PDF 逐页运行 PP-StructureV3（GPU 批量）")
    pdf.add_argument("pdf", help="PDF 文件路径")
    pdf.add_argument("-o", "--output-dir", default="patent_pdf_result",
                     help="输出目录（每页一个 JSON + 汇总 summary.json）")
    pdf.add_argument("--dpi", type=int, default=300,
                     help="PDF 渲染 DPI（默认 300）")
    pdf.add_argument("--unclip", type=float, default=2.0, choices=[1.5, 2.0])
    pdf.add_argument("--device", default=None,
                     help="推理设备：gpu:0 / gpu / cpu（默认 auto）")
    pdf.add_argument("--side-len", type=int, default=3000)
    pdf.add_argument("--gpu-mem-mb", type=int, default=None)
    pdf.add_argument("--pages", default=None,
                     help="要处理的页码，逗号分隔 1-based（默认全部）")

    # --- full ---
    f = sub.add_parser("full", help="完整流程：识别 → 解析 → QC → 合并")
    f.add_argument("image", help="专利 PDF 页面图像路径")
    f.add_argument("-o", "--output", default="patent_result.json",
                   help="输出 JSON 文件路径")
    f.add_argument("--unclip", type=float, default=2.0, choices=[1.5, 2.0])
    f.add_argument("--device", default=None,
                   help="推理设备：gpu:0 / gpu / cpu（默认 auto）")
    f.add_argument("--side-len", type=int, default=3000)
    f.add_argument("--gpu-mem-mb", type=int, default=None)
    f.add_argument("--model-dir", help="模型权重目录")
    f.add_argument("--min-seq-len", type=int, default=16,
                   help="QC: 最小序列长度（默认 16）")
    f.add_argument("--min-mod-ratio", type=float, default=0.20,
                   help="QC: 最小修饰比例（默认 0.20）")
    f.add_argument("--max-edit-dist", type=int, default=6,
                   help="QC: 最大编辑距离（默认 6）")

    return p


def _cmd_download() -> int:
    paths = download_weights()
    for comp, path in paths.items():
        print(f"  {comp}: {path}")
    return 0


def _cmd_list_models() -> int:
    models = list_available_models()
    for m in models:
        print(f"  {m['component']:6s}  {m['name']:<35s}  {m['description']}")
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    config = PatentTablePipelineConfig(
        det_db_unclip_ratio=args.unclip,
        det_limit_side_len=args.side_len,
        device=args.device,
        gpu_memory_limit_mb=args.gpu_mem_mb,
    )
    if args.model_dir:
        config.table_model = str(Path(args.model_dir) / "table")
        config.rec_model = str(Path(args.model_dir) / "rec")

    pipe = PatentTablePipeline(config)
    try:
        page = pipe.process_page(args.image)
    finally:
        pipe.close()

    result = {
        "page": str(args.image),
        "seconds": page.seconds,
        "n_tables": len(page.tables),
        "markdown": page.markdown,
        "tables": [
            {
                "index": i,
                "html": html,
                "structure": (parse_html_table(html).to_dicts()
                              if parse_html_table(html) else None),
            }
            for i, html in enumerate(page.tables)
        ],
    }

    if args.output:
        Path(args.output).write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"结果已写入: {args.output}")
    else:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def _cmd_pdf(args: argparse.Namespace) -> int:
    """对 PDF 逐页运行 PP-StructureV3，输出每页 JSON + summary。"""
    import time

    try:
        import fitz  # PyMuPDF
    except ImportError as exc:
        print("需要 pymupdf：pip install pymupdf", file=sys.stderr)
        return 1

    config = PatentTablePipelineConfig(
        det_db_unclip_ratio=args.unclip,
        det_limit_side_len=args.side_len,
        device=args.device,
        gpu_memory_limit_mb=args.gpu_mem_mb,
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 渲染 PDF → 页面图片
    doc = fitz.open(args.pdf)
    n_pages = doc.page_count
    wanted = None
    if args.pages:
        wanted = {int(x) for x in args.pages.split(",") if x.strip()}
    print(f"PDF 共 {n_pages} 页，渲染 DPI={args.dpi} ...", flush=True)
    page_images: list[tuple[int, Path]] = []
    for i in range(n_pages):
        if wanted is not None and (i + 1) not in wanted:
            continue
        pix = doc[i].get_pixmap(dpi=args.dpi)
        png = out_dir / f"page_{i + 1:02d}.png"
        pix.save(png)
        page_images.append((i + 1, png))
        print(f"  渲染 page_{i + 1:02d} ({pix.width}x{pix.height})", flush=True)
    doc.close()

    t0 = time.time()
    print(f"初始化 PP-StructureV3（device={config.device or 'auto'}）...", flush=True)
    pipe = PatentTablePipeline(config)
    summary = []
    try:
        for page_idx, png in page_images:
            t1 = time.time()
            print(f"==> page_{page_idx:02d} ...", flush=True)
            page = pipe.process_page(png, page_index=page_idx)
            out = {
                "page": page_idx,
                "seconds": round(page.seconds, 1),
                "n_tables": len(page.tables),
                "markdown": page.markdown,
                "tables": page.tables,
                "width": page.width,
                "height": page.height,
            }
            (out_dir / f"page_{page_idx:02d}.json").write_text(
                json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            summary.append(out)
            print(f"    page_{page_idx:02d}: {page.seconds:.1f}s, "
                  f"{len(page.tables)} tables, md_len={len(page.markdown)}",
                  flush=True)
    finally:
        pipe.close()

    total = time.time() - t0
    summary_out = {
        "pdf": str(args.pdf),
        "total_seconds": round(total, 1),
        "device": config.device or "auto",
        "pages": summary,
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary_out, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"DONE total={total:.1f}s -> {out_dir}/summary.json", flush=True)
    return 0


def _cmd_full(args: argparse.Namespace) -> int:
    config = PatentTablePipelineConfig(
        det_db_unclip_ratio=args.unclip,
        det_limit_side_len=args.side_len,
        device=args.device,
        gpu_memory_limit_mb=args.gpu_mem_mb,
    )
    if args.model_dir:
        config.table_model = str(Path(args.model_dir) / "table")
        config.rec_model = str(Path(args.model_dir) / "rec")

    pipe = PatentTablePipeline(config)
    try:
        page = pipe.process_page(args.image)
    finally:
        pipe.close()

    tables = [{"html": h} for h in page.tables]
    if not tables:
        print("未检测到表格", file=sys.stderr)
        return 1

    # 识别序列表与敲低效应表
    seq_table = extract_sequence_table(tables)
    kd_table = extract_knockdown_table(tables)

    if seq_table is None:
        print("警告: 未识别到序列表（将尝试使用第一张表）", file=sys.stderr)
        kd_html = tables[0].get("html", "")
        seq_table = parse_html_table(kd_html)

    if kd_table is None:
        print("警告: 未识别到敲低效应表（将尝试使用第二张表）", file=sys.stderr)
        if len(tables) > 1:
            kd_html = tables[1].get("html", "")
            kd_table = parse_html_table(kd_html)
        else:
            kd_table = None

    # 合并
    if seq_table and kd_table:
        merged = merge_sequence_knockdown(seq_table, kd_table)
    elif seq_table:
        qc = QCFilter(spec=QCSpec(
            min_sequence_length=args.min_seq_len,
            min_modification_ratio=args.min_mod_ratio,
            max_edit_distance=args.max_edit_dist,
        ))
        merged = []
        for row in seq_table.rows[1:]:
            from .merge import MergedEntry
            text = " ".join(c.text for c in row.cells)
            merged.append(MergedEntry(sequence=text))
    else:
        print("未提取到任何表格", file=sys.stderr)
        return 1

    # QC 过滤
    qc = QCFilter(spec=QCSpec(
        min_sequence_length=args.min_seq_len,
        min_modification_ratio=args.min_mod_ratio,
        max_edit_distance=args.max_edit_dist,
    ))
    seq_rows = [{"sequence": e.sequence, "guide": e.guide,
                  "passenger": e.passenger} for e in merged]
    filtered = qc.filter_rows(seq_rows)

    output = {
        "num_tables_detected": len(tables),
        "num_sequences_found": len(merged),
        "num_sequences_passed_qc": len(filtered),
        "markdown": page.markdown,
        "sequences": [
            {
                "sequence": m.sequence,
                "guide": m.guide,
                "passenger": m.passenger,
                "modifications": m.modifications,
                "target_gene": m.target_gene,
                "knockdown": m.knockdown_data,
                "qc_passed": m.sequence in {r["sequence"] for r in filtered},
            }
            for m in merged
        ],
    }

    Path(args.output).write_text(
        json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"完成: {output['num_sequences_found']} 条序列, "
          f"{output['num_sequences_passed_qc']} 条通过 QC")
    print(f"结果已写入: {args.output}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    if args.command == "download":
        return _cmd_download()
    if args.command == "list-models":
        return _cmd_list_models()
    if args.command == "run":
        return _cmd_run(args)
    if args.command == "pdf":
        return _cmd_pdf(args)
    if args.command == "full":
        return _cmd_full(args)
    return 1


if __name__ == "__main__":
    sys.exit(main())