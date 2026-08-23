"""从 ModelScope 下载 PaddleOCR-VL 权重到本地 models/ 目录。"""
import sys

from modelscope import snapshot_download

def main() -> int:
    path = snapshot_download(
        model_id="PaddlePaddle/PaddleOCR-VL",
        cache_dir="./models_cache",
        local_dir="./models/PaddleOCR-VL",
    )
    print("downloaded to:", path)
    return 0

if __name__ == "__main__":
    sys.exit(main())