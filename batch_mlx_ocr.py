"""批量识别 PDF 页面，输出完整 markdown（PaddleOCR-VL / GLM-OCR 走 mlx-vlm）。"""
import subprocess
import sys
from pathlib import Path

PROMPT = "请识别图片中的所有文字内容，保留原有排版结构，输出为 Markdown 格式。"
PAGES = [f"pages/page_{i:02d}.png" for i in range(1, 5)]
MAX_TOKENS = 8192


def main() -> int:
    engine = sys.argv[1]          # paddleocr-vl | glm-ocr
    model_dir = sys.argv[2]       # 本地权重目录
    out_file = sys.argv[3]        # 输出 markdown 路径

    parts = [f"<!-- {engine} OCR 结果，共 {len(PAGES)} 页 -->\n"]
    for page in PAGES:
        print(f"[{engine}] 处理 {page} ...", flush=True)
        r = subprocess.run(
            [
                sys.executable, "-m", "mlx_vlm.generate",
                "--model", model_dir,
                "--image", page,
                "--prompt", PROMPT,
                "--max-tokens", str(MAX_TOKENS),
            ],
            capture_output=True, text=True, timeout=1800,
        )
        if r.returncode != 0:
            print(f"  ERROR ({r.returncode}): {r.stderr[-500:]}", flush=True)
            parts.append(f"\n\n<!-- 第 {page} 页识别失败: {r.stderr[-200:]} -->\n")
            continue
        # 去掉 mlx_vlm 的警告行（stderr），stdout 是生成文本
        text = r.stdout.strip()
        print(f"  输出 {len(text)} 字符", flush=True)
        parts.append(f"\n\n## 第 {Path(page).stem} 页\n\n{text}\n")
        r.stdout  # noqa

    Path(out_file).write_text("".join(parts), encoding="utf-8")
    print(f"完成，已写入 {out_file}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())