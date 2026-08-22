# unified-ocr

在 **macOS（Apple Silicon）** 上统合三个开源 OCR 模型的统一识别框架：

| 引擎 | 上游项目 | 推理后端 | macOS 可行性 |
|------|----------|----------|--------------|
| `glm-ocr` | [zai-org/GLM-OCR](https://github.com/zai-org/GLM-OCR)（Apache-2.0 / MIT） | **mlx-vlm（MLX/Metal）** | ✅ 官方提供 MLX 部署指南 |
| `paddleocr-vl` | [PaddlePaddle/PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)（Apache-2.0） | **mlx-vlm（MLX/Metal）** | ✅ mlx-vlm 原生集成 |
| `hunyuanocr` | [Tencent-Hunyuan/HunyuanOCR](https://github.com/Tencent-Hunyuan/HunyuanOCR)（v1.0 分支） | **llama.cpp（GGUF + Metal）** | ⚠️ 官方无 MLX；llama.cpp 为可行本地路径 |

统一框架在所有引擎之上提供**同一套 API/CLI 与同一输出结构**（`OCRResult`：
blocks → lines → words + 整页文本 + 引擎元信息），切换或对比引擎只需改一个参数。

## 架构

```
                    unified-ocr (CLI / Python API)
                    统一输出: OCRResult  (text / blocks / metadata)
        ┌───────────────────┼──────────────────────┐
   glm-ocr            paddleocr-vl             hunyuanocr
  (mlx-vlm)           (mlx-vlm)              (llama.cpp GGUF)
   MLX/Metal           MLX/Metal               Metal (llama.cpp)
        └───────────────────┼──────────────────────┘
                    Apple Silicon (Metal GPU)
```

- **mlx-vlm**（[Blaizzy/mlx-vlm](https://github.com/Blaizzy/mlx-vlm)）同时支持
  GLM-OCR 与 PaddleOCR-VL 两个架构，是本框架的 MLX 核心；两者都无需转换
  GGUF，直接加载 HF/MLX 权重即可用 Metal 推理。
- **HunyuanOCR** 无 MLX 支持，官方唯一本地路径是 llama.cpp GGUF 部署；
  llama.cpp 在 macOS 上同样走 Metal 加速。

## 安装

要求：macOS（Apple Silicon）+ Python ≥ 3.10。

```bash
git clone <本仓库> && cd unified-ocr

# 1. 核心包（模型无关）
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e .

# 2. MLX 引擎（glm-ocr / paddleocr-vl）
pip install ".[mlx]"
#   注意：mlx-vlm 必须从 git 安装（PyPI 版尚未内置 GLM-OCR / PaddleOCR-VL 架构）。
#   若 GLM-OCR 官方 mlx-deploy 文档要求 transformers>=5.0.0rc3，请按文档升级：
#   pip install -U "transformers>=5.0.0rc3"

# 3. HunyuanOCR 引擎（可选）
pip install ".[hunyuan]"   # llama-cpp-python，macOS 默认启用 Metal
```

### 模型准备

```bash
./setup_models.sh                # 下载 GLM-OCR、PaddleOCR-VL 权重（~/.models 默认）
./setup_models.sh --hunyuan      # 额外打印 HunyuanOCR GGUF 准备指引
```

HunyuanOCR 没有现成 MLX 权重，需要 GGUF（+ 视觉 mmproj）文件：优先使用
官方发布的 GGUF；否则用 llama.cpp 的 `convert_hf_to_gguf.py` 自转。

## 使用

### CLI

```bash
unified-ocr list-engines

# 单引擎
unified-ocr run scan.png -e glm-ocr -o json
unified-ocr run scan.png -e paddleocr-vl -o markdown

# 全部引擎（便于横向对比）—— hunyuanocr 需先提供 GGUF
unified-ocr run scan.png --all -o json

# HunyuanOCR 指定本地 GGUF + mmproj
unified-ocr run scan.png -e hunyuanocr \
    --model hunyuanocr=~/models/HunyuanOCR-1.0-Q4_K_M.gguf \
    --mmproj hunyuanocr=~/models/mmproj-hunyuanocr-f16.gguf \
    --max-tokens 8192

# 覆盖 MLX 模型路径 / 提示词
unified-ocr run scan.png -e glm-ocr \
    --model glm-ocr=mlx-community/GLM-OCR-bf16 \
    --prompt "请提取表格中的数值并输出为 Markdown 表格"
```

输出格式：`json`（统一 `OCRResult`）、`markdown`（块级渲染）、`text`（纯文本）。

### Python API

```python
from unified_ocr import OCR, list_engines

for e in list_engines():
    print(e.id, "→", e.accelerator)

ocr = OCR()
results = ocr.run("scan.png", engines=["glm-ocr", "paddleocr-vl"])
for r in results:
    print(r.engine, r.text[:100])          # 整页文本
    for b in r.blocks:
        print(" ", b.kind, b.text[:40])    # 语义块（text/table/formula/…）

# 每个引擎独立配置
out = ocr.run_mapped("scan.png", {
    "glm-ocr": {"model": "mlx-community/GLM-OCR-bf16"},
    "hunyuanocr": {"model": "~/models/HunyuanOCR.gguf", "mmproj": "~/models/mmproj.gguf"},
})
```

### 统一输出结构

```json
{
  "engine": "glm-ocr",
  "text": "整页纯文本…",
  "blocks": [
    { "kind": "heading", "text": "实验报告", "bbox": null, "confidence": null,
      "lines": [] },
    { "kind": "table", "text": "| 基因 | 表达量 |…", "bbox": null,
      "confidence": null, "lines": [] }
  ],
  "metadata": { "model": "mlx-community/GLM-OCR-bf16", "prompt": "…" }
}
```

VLM 类引擎的典型输出是整段 Markdown（无逐行 bbox），框架对其做轻量启发式
分块（`split_blocks`）；若某引擎给出结构化输出（如 PaddleOCR-VL 的 JSON
模式），适配层优先透传结构化 `blocks`。

## 已知限制与决策说明

- **HunyuanOCR 的 llama.cpp 路径属推断路径**：官方未显式验证 macOS 运行；
  若 GLM-OCR / PaddleOCR-VL 已满足需求，优先使用 MLX 引擎。
- **PaddleOCR-VL 权重 id**：默认 `PaddlePaddle/PaddleOCR-VL`（HF）；如 mlx-vlm
  文档指定其他 id，用 `--model paddleocr-vl=<id>` 或 `PADDLE_MODEL` 环境变量覆盖。
- 框架只在 `load()` 时拉取权重，`list-engines` 不会下载任何模型。
- `split_blocks` 是启发式实现，对复杂排版（嵌套表格、多栏）可能误分；
  需要精确版面结构时请直接消费各引擎的原始输出（`metadata`）。

## 开发

```bash
pip install -e ".[dev]"
pytest -q          # 离线测试：数据模型 / 注册表 / CLI 流程（不加载真实模型）
```

## 参考资料

- mlx-vlm 模型支持表：https://github.com/Blaizzy/mlx-vlm
- GLM-OCR MLX 部署官方文档：`zai-org/GLM-OCR/examples/mlx-deploy/README.md`
- HunyuanOCR：https://github.com/Tencent-Hunyuan/HunyuanOCR
- PaddleOCR（PaddleOCR-VL 论文 arXiv:2606.03264）：https://github.com/PaddlePaddle/PaddleOCR