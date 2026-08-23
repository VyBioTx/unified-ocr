"""批量用 HunyuanOCR（transformers 原生推理）识别 4 页，输出完整 markdown。

每个页面独立 subprocess 调用 run_hunyuan_ocr.py（避免单进程长会话累积显存），
结果拼接到一个 markdown 文件。表格以 HTML 格式输出（HunyuanOCR 规范）。
"""
import subprocess
import sys
from pathlib import Path

PAGES = [f"pages/page_{i:02d}.png" for i in range(1, 5)]
MODEL_DIR = "./models/HunyuanOCR"
MAX_TOKENS = 8192
OUT_FILE = "result_hunyuanocr.md"


def main() -> int:
    parts = [f"<!-- hunyuanocr OCR 结果，共 {len(PAGES)} 页 -->\n"]
    for page in PAGES:
        print(f"[hunyuanocr] 处理 {page} ...", flush=True)
        r = subprocess.run(
            [
                sys.executable, "run_hunyuan_ocr.py",
                MODEL_DIR, page, str(MAX_TOKENS),
            ],
            capture_output=True, text=True, timeout=2400,
        )
        if r.returncode != 0:
            print(f"  ERROR rc={r.returncode}: {r.stderr[-600:]}", flush=True)
            parts.append(f"\n\n<!-- 第 {page} 页识别失败 -->\n{r.stderr[-300:]}\n")
            continue
        text = r.stdout.strip()
        print(f"  输出 {len(text)} 字符", flush=True)
        parts.append(f"\n\n## 第 {Path(page).stem} 页\n\n{text}\n")

    Path(OUT_FILE).write_text("".join(parts), encoding="utf-8")
    print(f"完成，已写入 {OUT_FILE}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())