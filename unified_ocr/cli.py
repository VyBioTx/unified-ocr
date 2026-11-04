"""unified-ocr 命令行入口。

用法示例::

    unified-ocr list-engines
    unified-ocr run scan.png -e glm-ocr
    unified-ocr run scan.png --all -o json
    unified-ocr run scan.png -e hunyuanocr \\
        --model hunyuanocr=~/models/HunyuanOCR
    unified-ocr qc scan.png -e glm-ocr --threshold 0.5
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from typing import Any

from . import __version__
from .models import OCRResult
from .registry import create_backend, has_engine, list_engines

log = logging.getLogger("unified_ocr")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="unified-ocr",
        description="统合 PaddleOCR-VL / HunyuanOCR / GLM-OCR 的 macOS 统一 OCR 框架（MLX）。",
    )
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    p.add_argument("-v", "--verbose", action="store_true", help="输出调试日志")
    sub = p.add_subparsers(dest="command", required=True)

    # --- list-engines -----------------------------------------------------
    ls = sub.add_parser("list-engines", help="列出已注册引擎")
    ls.add_argument("-o", "--output", choices=["text", "json"], default="text")

    # --- run --------------------------------------------------------------
    r = sub.add_parser("run", help="对图像执行 OCR 识别")
    r.add_argument("image", help="图像路径（jpg/png/pdf 页等）")
    eng = r.add_mutually_exclusive_group(required=True)
    eng.add_argument("-e", "--engine", action="append", dest="engines",
                     help="引擎 id，可重复（glm-ocr / paddleocr-vl / hunyuanocr）")
    eng.add_argument("--all", action="store_true", help="运行所有已注册引擎")
    r.add_argument("-o", "--output", choices=["text", "json", "markdown"], default="text")
    r.add_argument("--model", action="append", default=[], metavar="ENGINE=PATH",
                   help="覆盖某引擎的模型路径，如 --model glm-ocr=mlx-community/GLM-OCR-bf16")
    r.add_argument("--prompt", default=None, help="覆盖默认识别提示词")
    r.add_argument("--max-tokens", type=int, default=None)

    # --- qc（OCR + Jev 可信度打分）----------------------------------------
    q = sub.add_parser("qc", help="识别一页并给出 Jev 可信度分数")
    q.add_argument("image", help="图像路径")
    q.add_argument("-e", "--engine", default="glm-ocr", help="识别引擎 id（默认 glm-ocr）")
    q.add_argument("-o", "--output", choices=["json", "text"], default="json")
    q.add_argument("--model", default=None, metavar="ENGINE=PATH",
                   help="覆盖识别引擎的模型路径")
    q.add_argument("--jev-model", default=None, help="Jev 权重路径 / HF id")
    q.add_argument("--threshold", type=float, default=None,
                   help="低于该分数标记 needs_review（需按自己样本标定）")
    return p


def _parse_kv(items: list[str], flag: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise argparse.ArgumentTypeError(f"{flag} 需要 ENGINE=PATH 格式，收到: {item!r}")
        k, v = item.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def _resolve_engines(args: argparse.Namespace) -> list[str]:
    if args.all:
        return [s.id for s in list_engines()]
    return args.engines or []


def _engine_options(args: argparse.Namespace, engine_id: str) -> dict[str, Any]:
    models = _parse_kv(args.model, "--model")
    opts: dict[str, Any] = {}
    if engine_id in models:
        opts["model"] = models[engine_id]
    if args.prompt:
        opts["prompt"] = args.prompt
    if args.max_tokens:
        opts["max_tokens"] = args.max_tokens
    return opts


def _run(args: argparse.Namespace) -> int:
    results: list[OCRResult] = []
    for engine_id in _resolve_engines(args):
        if not has_engine(engine_id):
            print(f"错误: 未知引擎 {engine_id!r}（可用: "
                  f"{', '.join(s.id for s in list_engines())}）", file=sys.stderr)
            return 1
        try:
            backend = create_backend(engine_id, **_engine_options(args, engine_id))
        except ValueError as exc:
            # 构造期失败（如后端参数缺失）
            print(f"跳过 {engine_id}: {exc}", file=sys.stderr)
            continue
        try:
            with backend:
                results.append(backend.recognize(args.image))
        except Exception as exc:  # noqa: BLE001 - CLI 需要兜底报错
            print(f"引擎 {engine_id} 识别失败: {exc}", file=sys.stderr)
            if args.verbose:
                log.exception("引擎 %s 失败", engine_id)
    if not results:
        print("没有引擎成功执行识别。", file=sys.stderr)
        return 1

    if args.output == "json":
        print(json.dumps([r.to_dict() for r in results], ensure_ascii=False, indent=2))
    elif args.output == "markdown":
        for r in results:
            print(f"<!-- engine: {r.engine} -->")
            print(r.to_markdown())
            print()
    else:  # text
        for r in results:
            print(f"=== {r.engine} ===")
            print(r.text)
            print()
    return 0


def _qc(args: argparse.Namespace) -> int:
    """单页 OCR + Jev 打分；整页分数写入 result.metadata['qc']['jev']。"""
    from .qc import JevScorer, apply_qc

    opts: dict[str, Any] = {}
    if args.model:
        models = _parse_kv([args.model], "--model")
        if args.engine in models:
            opts["model"] = models[args.engine]
    try:
        backend = create_backend(args.engine, **opts)
        with backend:
            result = backend.recognize(args.image)
    except Exception as exc:  # noqa: BLE001 - CLI 需要兜底报错
        print(f"引擎 {args.engine} 识别失败: {exc}", file=sys.stderr)
        return 1

    info = apply_qc(result, args.image, JevScorer(args.jev_model), args.threshold)
    if args.output == "text":
        prob = info["prob_consistent"]
        print(f"engine={result.engine} jev={prob if prob is None else f'{prob:.5f}'} "
              f"threshold={info['threshold']} needs_review={info['needs_review']}")
    else:
        print(result.to_json())
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    if args.command == "list-engines":
        specs = list_engines()
        if args.output == "json":
            print(json.dumps(
                [{"id": s.id, "display_name": s.display_name, "model_hint": s.model_hint,
                  "accelerator": s.accelerator, "license": s.license, "repo": s.repo}
                 for s in specs],
                ensure_ascii=False, indent=2))
        else:
            for s in specs:
                print(f"{s.id:<14} {s.display_name:<30} {s.accelerator:<20} {s.model_hint}")
        return 0
    if args.command == "run":
        return _run(args)
    if args.command == "qc":
        return _qc(args)
    return 1  # pragma: no cover - parser required=True 已拦截


if __name__ == "__main__":
    sys.exit(main())
