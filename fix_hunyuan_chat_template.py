"""修复 HunyuanOCR chat_template：图片占位符需 start/end 包裹。

新版 transformers 的 HunYuanVLProcessor.validate_inputs 要求图像占位符
为 <start><token><end> 三段式；ModelScope 权重自带的模板只有 <token>。
"""
import json
from pathlib import Path

TC = Path("models/HunyuanOCR/tokenizer_config.json")


def main() -> int:
    tc = json.loads(TC.read_text())
    tpl = tc.get("chat_template", "")
    old = "<｜hy_place▁holder▁no▁102｜>{% elif"
    new = "<｜hy_place▁holder▁no▁100｜><｜hy_place▁holder▁no▁102｜><｜hy_place▁holder▁no▁101｜>{% elif"
    if old not in tpl:
        # 另一种写法: 图片分支单独一行
        print("未找到期望的模板片段，打印模板以人工核对:")
        print(tpl)
        return 1
    tpl = tpl.replace(old, new, 1)
    tc["chat_template"] = tpl
    TC.write_text(json.dumps(tc, ensure_ascii=False, indent=2))
    print("chat_template 已修复：图像分支加 start/end 包裹")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
