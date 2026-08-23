"""用 transformers 原生推理 HunyuanOCR（ModelScope 权重，torch MPS/CPU）。"""
import sys
import time
from pathlib import Path

import torch
from transformers import AutoProcessor, HunYuanVLForConditionalGeneration


def clean_repeated_substrings(text: str) -> str:
    n = len(text)
    if n < 8000:
        return text
    for length in range(2, n // 10 + 1):
        candidate = text[-length:]
        count = 0
        i = n - length
        while i >= 0 and text[i:i + length] == candidate:
            count += 1
            i -= length
        if count >= 10:
            return text[:n - length * (count - 1)]
    return text


def main() -> int:
    model_dir = sys.argv[1]
    image_path = sys.argv[2]
    max_new = int(sys.argv[3]) if len(sys.argv) > 3 else 16384

    prompt = (
        "请提取文档图片中正文的所有信息用 markdown 格式表示，"
        "其中页眉、页脚部分忽略，表格用 HTML 格式表达，"
        "文档中公式用 latex 格式表示，按照阅读顺序组织进行解析。"
    )

    print("加载 processor ...", file=sys.stderr, flush=True)
    processor = AutoProcessor.from_pretrained(model_dir, use_fast=False, trust_remote_code=True)
    print("加载模型 ...", file=sys.stderr, flush=True)
    t0 = time.time()
    model = HunYuanVLForConditionalGeneration.from_pretrained(
        model_dir,
        attn_implementation="eager",
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    print(f"模型加载完成，耗时 {time.time() - t0:.1f}s", file=sys.stderr, flush=True)

    img = sys.argv[2]
    messages = [{"role": "user", "content": [
        {"type": "image", "image": img},
        {"type": "text", "text": prompt},
    ]}]
    texts = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=texts, images=img, padding=True, return_tensors="pt")

    device = next(model.parameters()).device
    inputs = inputs.to(device)
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=max_new, do_sample=False)

    input_ids = inputs.input_ids
    trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(input_ids, generated_ids)]
    outputs = processor.batch_decode(trimmed, skip_special_tokens=True,
                                     clean_up_tokenization_spaces=False)
    print(clean_repeated_substrings(outputs[0]))
    return 0


if __name__ == "__main__":
    sys.exit(main())