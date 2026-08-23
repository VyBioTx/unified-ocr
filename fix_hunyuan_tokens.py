"""修复 HunyuanOCR 的 tokenizer_config.json，补齐 HunYuanVL 所需特殊 token 映射。

新版 transformers (>=5.x) 的 HunYuanVLProcessor 要求 tokenizer 暴露
image_token / image_start_token / image_end_token 等属性，这些属性由
tokenizer_config.json 的 extra_special_tokens 驱动。
ModelScope 权重未带该字段，这里按 config.json 的 token id 补齐。
"""
import json
from pathlib import Path

CFG = Path("models/HunyuanOCR/config.json")
TC = Path("models/HunyuanOCR/tokenizer_config.json")
TOK = Path("models/HunyuanOCR/tokenizer.json")


def main() -> int:
    cfg = json.loads(CFG.read_text())
    tok = json.loads(TOK.read_text())
    added = {t["id"]: t["content"] for t in tok.get("added_tokens", [])}

    tc = json.loads(TC.read_text())
    if tc.get("extra_special_tokens"):
        print("已存在 extra_special_tokens，跳过")
        return 0

    id_to_text = {
        "image_start_token_id": "image_start_token",
        "image_end_token_id": "image_end_token",
        "image_token_id": "image_token",
        "image_newline_token_id": "image_newline_token",
    }
    extra = {}
    for id_key, tok_attr in id_to_text.items():
        tid = cfg.get(id_key)
        text = added.get(tid)
        if text is None:
            print(f"警告: config {id_key}={tid} 在 tokenizer.json 中无对应 added token")
            continue
        extra[tok_attr] = text
    # pad token 用 tokenizer_config 已有定义
    extra["pad_token"] = tc.get("pad_token") or "<｜hy_▁pad▁｜>"

    tc["extra_special_tokens"] = extra
    TC.write_text(json.dumps(tc, ensure_ascii=False, indent=2))
    print("已写入 extra_special_tokens:", json.dumps(extra, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
