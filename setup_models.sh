#!/usr/bin/env bash
# 下载三个引擎的模型权重。
#
# 用法:
#   ./setup_models.sh                      # 下载 MLX 两引擎（glm-ocr / paddleocr-vl）
#   ./setup_models.sh --hunyuan            # 额外准备 HunyuanOCR（GGUF，见下方说明）
#   HF_MODEL_DIR=~/models ./setup_models.sh   # 自定义权重目录
#
# 依赖: huggingface_hub (pip install "huggingface_hub[cli]")
set -euo pipefail

MODEL_DIR="${HF_MODEL_DIR:-$HOME/models/unified-ocr}"
mkdir -p "$MODEL_DIR"

echo "==> 权重目录: $MODEL_DIR"

# GLM-OCR —— 官方 mlx-deploy 文档指定的 MLX 权重（0.9B，8GB 内存可跑）
GLM_MODEL="${GLM_MODEL:-mlx-community/GLM-OCR-bf16}"

# PaddleOCR-VL —— mlx-vlm 可直接加载 HF 权重并自动转换/缓存。
# 如 mlx-vlm 的 paddleocr_vl README 指定了其他 id，用 PADDLE_MODEL 覆盖。
PADDLE_MODEL="${PADDLE_MODEL:-PaddlePaddle/PaddleOCR-VL}"

fetch() {
  local repo="$1"
  echo "==> 下载 $repo"
  python3 -m huggingface_hub download "$repo" --local-dir "$MODEL_DIR/$(basename "$repo")"
}

fetch "$GLM_MODEL"
fetch "$PADDLE_MODEL"

if [[ "${1:-}" == "--hunyuan" ]]; then
  cat <<'EOF'

==> HunyuanOCR-1.0 无 MLX 权重；请按 Tencent-Hunyuan/HunyuanOCR 官方指引准备 GGUF：

  1. 仓库: https://github.com/Tencent-Hunyuan/HunyuanOCR（v1.0 分支）
  2. 方式 A（推荐，若官方发布了 GGUF）: 从仓库 Release / HF 下载
     HunyuanOCR-1.0-*.gguf 与 mmproj-*.gguf；
  3. 方式 B（自转）: 用 llama.cpp 的 convert_hf_to_gguf.py 转换 transformers
     权重为 GGUF（视觉部分导出 mmproj）。

  下载后:
    unified-ocr run scan.png -e hunyuanocr \
        --model hunyuanocr=$MODEL_DIR/.../HunyuanOCR-1.0-Q4_K_M.gguf \
        --mmproj hunyuanocr=$MODEL_DIR/.../mmproj-hunyuanocr-f16.gguf

  注意: 该路径为官方未显式验证 macOS 的推断路径；GLM-OCR / PaddleOCR-VL
        优先使用 MLX 引擎。
EOF
fi

echo "==> 完成。可用 unified-ocr list-engines 确认引擎，unified-ocr run <图像> 识别。"