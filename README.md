# unified-ocr

在 **macOS（Apple Silicon）** 上统合三个开源 OCR 模型的统一识别框架，已在本机（Apple M4 Pro）用真实模型跑通专利文档识别。

| 引擎 | 上游项目 | 推理后端 | macOS 可行性 |
|------|----------|----------|--------------|
| `glm-ocr` | [zai-org/GLM-OCR](https://github.com/zai-org/GLM-OCR)（代码 Apache-2.0 / 模型 MIT） | **mlx-vlm（MLX/Metal）** | ✅ 官方提供 MLX 部署指南 |
| `paddleocr-vl` | [PaddlePaddle/PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)（Apache-2.0） | **mlx-vlm（MLX/Metal）** | ✅ mlx-vlm 原生集成 |
| `hunyuanocr` | [Tencent-Hunyuan/HunyuanOCR](https://github.com/Tencent-Hunyuan/HunyuanOCR)（v1.0 分支） | **transformers 原生推理（MPS）** | ✅ 1B 参数，PyTorch MPS 可跑 |

统一框架在所有引擎之上提供**同一套 API/CLI 与同一输出结构**（`OCRResult`：
blocks → lines → words + 整页文本 + 引擎元信息），切换或对比引擎只需改一个参数。

## 架构

```
                    unified-ocr (CLI / Python API)
                    统一输出: OCRResult  (text / blocks / metadata)
        ┌───────────────────┼──────────────────────┐
   glm-ocr            paddleocr-vl             hunyuanocr
  (mlx-vlm)           (mlx-vlm)          (transformers 原生)
   MLX/Metal           MLX/Metal                MPS
        └───────────────────┼──────────────────────┘
                    Apple Silicon (Metal GPU)
```

- **mlx-vlm**（[Blaizzy/mlx-vlm](https://github.com/Blaizzy/mlx-vlm)）同时支持
  GLM-OCR 与 PaddleOCR-VL 两个架构，是本框架的 MLX 核心；两者都无需转换
  GGUF，直接加载权重即可用 Metal 推理。
- **HunyuanOCR** 是 1B 轻量 VLM，官方推荐 transformers `HunYuanVLForConditionalGeneration`
  原生推理（Apple Silicon 走 PyTorch MPS），无需 GGUF/llama.cpp。

## 安装

要求：macOS（Apple Silicon）+ Python ≥ 3.10。

```bash
git clone <本仓库> && cd unified-ocr

# 1. 核心包（模型无关）
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e .

# 2. MLX 引擎（glm-ocr / paddleocr-vl）
pip install ".[mlx]"
#   注意：mlx-vlm 必须从 git 安装（PyPI 版尚未内置 GLM-OCR / PaddleOCR-VL 架构），
#   且要求 transformers>=5.0：pip install -U "transformers>=5.0"。

# 3. HunyuanOCR 引擎（transformers 原生，可选）
pip install ".[hunyuan]"   # torch + transformers + accelerate
```

## 模型权重下载

本机无法直连 HuggingFace 时，全部三个权重可从 **ModelScope（阿里）镜像**下载。
GLM-OCR 在 ModelScope 的官方仓库是 `ZhipuAI/GLM-OCR`（非 zai-org）。

```bash
pip install modelscope

python download_paddle.py    # PaddlePaddle/PaddleOCR-VL   → models/PaddleOCR-VL  (2.2GB)
python download_hunyuan.py   # Tencent-Hunyuan/HunyuanOCR  → models/HunyuanOCR     (2.0GB)
python download_glm.py       # ZhipuAI/GLM-OCR            → models/GLM-OCR        (2.65GB)
```

> HuggingFace 直连可用时，等价权重 id：`PaddlePaddle/PaddleOCR-VL`、
> `tencent/HunyuanOCR`、`mlx-community/GLM-OCR-bf16`。

### HunyuanOCR 权重修复

ModelScope 的 HunyuanOCR 权重有两个小问题需修复后才能被 transformers≥5 加载，
已提供脚本：

```bash
python fix_hunyuan_tokens.py          # 补齐 tokenizer_config.json 的 extra_special_tokens
python fix_hunyuan_chat_template.py   # 修复 chat_template 图片占位符（start/end 包裹）
```

## 使用

### 引擎 CLI（命令行快捷识别单张图片）

```bash
# MLX 引擎（PaddleOCR-VL / GLM-OCR）
python run_mlx_ocr.py paddleocr-vl ./models/PaddleOCR-VL ./page.png 4096
python run_mlx_ocr.py glm-ocr          ./models/GLM-OCR      ./page.png 4096

# HunyuanOCR（transformers）
python run_hunyuan_ocr.py ./models/HunyuanOCR ./page.png 4096
```

### PDF 批量识别（专利/文献）

```bash
python render_pdf.py                    # pages/page_01~04.png（200dpi）

# 每引擎整份 PDF → 一个 markdown 文件
python batch_mlx_ocr.py paddleocr-vl ./models/PaddleOCR-VL  ./result_paddleocr_vl.md
python batch_mlx_ocr.py glm-ocr        ./models/GLM-OCR       ./result_glm_ocr.md
python batch_hunyuan_ocr.py            # → result_hunyuanocr.md（表格 HTML 格式）
```

> 三个引擎同时跑会触发 Metal GPU 超时（M4 Pro 实测），需顺序执行。

### 统一框架 Python API / CLI

```python
from unified_ocr import OCR, list_engines

for e in list_engines():
    print(e.id, "→", e.accelerator)

ocr = OCR()
results = ocr.run("scan.png", engines=["glm-ocr", "paddleocr-vl"])
for r in results:
    print(r.engine, r.text[:100])
    for b in r.blocks:
        print(" ", b.kind, b.text[:40])
```

```bash
unified-ocr list-engines
unified-ocr run scan.png -e glm-ocr -o json     # 或 markdown / text
unified-ocr run scan.png --all -o json
```

`--all` 会运行全部三个引擎；HunyuanOCR 用本地模型路径时传 `--model hunyuanocr=<路径>`。

## 统一输出结构

```json
{
  "engine": "glm-ocr",
  "text": "整页纯文本…",
  "blocks": [
    { "kind": "heading", "text": "实验报告", "bbox": null, "confidence": null, "lines": [] },
    { "kind": "table", "text": "| 基因 | 表达量 |…", "bbox": null, "lines": [] }
  ],
  "metadata": { "model": "mlx-community/GLM-OCR-bf16", "prompt": "…" }
}
```

VLM 类引擎的典型输出是整段 Markdown（无逐行 bbox），框架对其做轻量启发式
分块（`unified_ocr/models.py:split_blocks`）；若引擎给出结构化输出，适配层优先生成
结构化 blocks。

## 测试

```bash
pip install -e ".[dev]"
pytest -q    # 离线测试：数据模型 / 注册表 / CLI 流程（不加载真实模型）
```

## 许可与致谢

- 本项目代码：MIT License（见 `LICENSE`）。
- 三个下游 OCR 模型的权重 / 代码版权归各自作者所有，使用请遵守其各自许可
  （GLM-OCR：Apache-2.0/MIT；PaddleOCR：Apache-2.0；HunyuanOCR：Tencent
  Hunyuan Community License）。
- 相关资源：[mlx-vlm](https://github.com/Blaizzy/mlx-vlm)、
  [GLM-OCR MLX 部署文档](https://github.com/zai-org/GLM-OCR/tree/main/examples/mlx-deploy)、
  [HunyuanOCR](https://github.com/Tencent-Hunyuan/HunyuanOCR)、
  [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)。