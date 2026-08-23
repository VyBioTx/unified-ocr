#!/usr/bin/env bash
# 从 ModelScope 下载三个 OCR 引擎的模型权重。
#
# 用法:
#   ./setup_models.sh            # 下载全部三个权重到 models/（总计约 6.4GB）
#
# 依赖: modelscope (pip install modelscope)
set -euo pipefail

cd "$(dirname "$0")"
mkdir -p models

echo "==> 下载 PaddleOCR-VL (2.2GB) ..."
HF_ENDPOINT=https://hf-mirror.com python download_paddle.py

echo "==> 下载 GLM-OCR (2.65GB) ..."
HF_ENDPOINT=https://hf-mirror.com python download_glm.py

echo "==> 下载 HunyuanOCR (2.0GB) ..."
HF_ENDPOINT=https://hf-mirror.com python download_hunyuan.py

echo "==> HunyuanOCR 权重修复（transformers>=5 兼容）..."
python fix_hunyuan_tokens.py
python fix_hunyuan_chat_template.py

echo "==> 完成。三个引擎权重位于 models/："
du -sh models/PaddleOCR-VL models/GLM-OCR models/HunyuanOCR

echo "==> 使用（顺序执行避免 Metal 并发超时）："
echo "  python batch_mlx_ocr.py paddleocr-vl ./models/PaddleOCR-VL result_paddleocr_vl.md"
echo "  python batch_mlx_ocr.py glm-ocr ./models/GLM-OCR result_glm_ocr.md"
echo "  python batch_hunyuan_ocr.py"