"""用 modelscope SDK 列出三个模型文件（打印完整路径+大小）。"""
import sys

from modelscope.hub.api import HubApi

MODELS = [
    "PaddlePaddle/PaddleOCR-VL",
    "Tencent-Hunyuan/HunyuanOCR",
    "zai-org/GLM-OCR",
]

def main() -> int:
    api = HubApi()
    for model_id in MODELS:
        print(f"===== {model_id} =====")
        try:
            files = api.get_model_files(model_id=model_id, recursive=True)
            for f in files:
                print(f"  {f.get('Path')}  ({f.get('Size', 0)})")
        except Exception as exc:
            print(f"  ERROR: {type(exc).__name__}: {exc}")
    return 0

if __name__ == "__main__":
    sys.exit(main())