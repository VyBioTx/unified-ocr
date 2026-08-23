"""探查 ModelScope 上三个 OCR 模型的文件列表与可下载性。"""
import sys

from modelscope.hub.api import HubApi

MODELS = [
    "zai-org/GLM-OCR",
    "PaddlePaddle/PaddleOCR-VL",
    "Tencent-Hunyuan/HunyuanOCR",
]

def main() -> int:
    api = HubApi()
    for model_id in MODELS:
        print(f"=== {model_id} ===")
        try:
            files = api.get_model_files(model_id=model_id)
            names = [f.get("Name", f.get("name", str(f))) for f in files]
            print(f"  count={len(names)}")
            for n in names[:25]:
                print(f"  - {n}")
        except Exception as exc:
            print(f"  ERROR: {type(exc).__name__}: {exc}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
