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
    r = sub.add_parser("run", help="对图像运行表格识别，输出 HTML")
    r.add_argument("image", help="图像路径（PNG/JPEG，建议 3000px+）")
    r.add_argument("-o", "--output", help="输出 JSON 文件路径（默认输出到终端）")
    r.add_argument("--unclip", type=float, default=2.0, choices=[1.5, 2.0],
                   help="det_db_unclip_ratio（1.5 或 2.0）")
    r.add_argument("--model-dir", help="模型权重目录")

    # --- full ---
    f = sub.add_parser("full", help="完整流程：识别 → 解析 → QC → 合并")
    f.add_argument("image", help="专利 PDF 页面图像路径")
    f.add_argument("-o", "--output", default="patent_result.json",
                   help="输出 JSON 文件路径")
    f.add_argument("--unclip", type=float, default=2.0, choices=[1.5, 2.0])
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
    )
    if args.model_dir:
        config.table_model = str(Path(args.model_dir) / "table")
        config.rec_model = str(Path(args.model_dir) / "rec")

    pipe = PatentTablePipeline(config)
    try:
        tables = pipe.process(args.image)
    finally:
        pipe.close()

    result = []
    for i, table in enumerate(tables):
        html = table.get("html", "")
        ts = parse_html_table(html)
        result.append({
            "index": i,
            "html": html,
            "structure": ts.to_dicts() if ts else None,
        })

    if args.output:
        Path(args.output).write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"结果已写入: {args.output}")
    else:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def _cmd_full(args: argparse.Namespace) -> int:
    config = PatentTablePipelineConfig(
        det_db_unclip_ratio=args.unclip,
    )
    if args.model_dir:
        config.table_model = str(Path(args.model_dir) / "table")
        config.rec_model = str(Path(args.model_dir) / "rec")

    pipe = PatentTablePipeline(config)
    try:
        tables = pipe.process(args.image)
    finally:
        pipe.close()

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
    if args.command == "full":
        return _cmd_full(args)
    return 1


if __name__ == "__main__":
    sys.exit(main())