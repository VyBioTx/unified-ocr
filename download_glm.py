"""从 ModelScope 下载 GLM-OCR（官方 ZhipuAI 组织）权重。"""
import sys

from modelscope import snapshot_download

def main() -> int:
    path = snapshot_download(
        model_id="ZhipuAI/GLM-OCR",
        cache_dir="./models_cache",
        local_dir="./models/GLM-OCR",
    )
    print("downloaded to:", path)
    return 0

if __name__ == "__main__":
    sys.exit(main())