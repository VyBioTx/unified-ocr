"""用 mlx-vlm 识别指定页面（PaddleOCR-VL 或 GLM-OCR，本地权重路径）。"""
import sys
from pathlib import Path

# mlx-vlm 的 generate 按 "<image>" 切分 prompt 并注入图像 token，必须保留占位符。
PROMPT = "<image>请识别图片中的所有文字内容，保留原有排版结构，输出为 Markdown 格式。"


def main() -> int:
    engine = sys.argv[1]      # paddleocr-vl | glm-ocr
    model_dir = sys.argv[2]   # 本地权重目录
    image_path = sys.argv[3]  # 页面图片
    max_tokens = int(sys.argv[4]) if len(sys.argv) > 4 else 4096

    from mlx_vlm import load, generate

    model, processor = load(model_dir)
    output = generate(
        model,
        processor,
        PROMPT,
        image=image_path,
        max_tokens=max_tokens,
        temperature=0.0,
    )
    print(output)
    return 0


if __name__ == "__main__":
    sys.exit(main())