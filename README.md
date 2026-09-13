# unified-ocr

在 **macOS（Apple Silicon）** 上统合多个开源 OCR 模型的统一识别框架，已在本机（Apple M4 Pro）用真实模型跑通专利文档识别。

| 引擎 | 上游项目 | 推理后端 | macOS 可行性 |
|------|----------|----------|--------------|
| `glm-ocr` | [zai-org/GLM-OCR](https://github.com/zai-org/GLM-OCR)（代码 Apache-2.0 / 模型 MIT） | **mlx-vlm（MLX/Metal）** | ✅ 官方提供 MLX 部署指南 |
| `paddleocr-vl` | [PaddlePaddle/PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)（Apache-2.0） | **mlx-vlm（MLX/Metal）** | ✅ mlx-vlm 原生集成 |
| `hunyuanocr` | [Tencent-Hunyuan/HunyuanOCR](https://github.com/Tencent-Hunyuan/HunyuanOCR)（v1.0 分支） | **transformers 原生推理（MPS）** | ✅ 1B 参数，PyTorch MPS 可跑 |
| `dots-mocr` | [rednote-hilab/dots.mocr](https://github.com/rednote-hilab/dots.mocr)（原 dots.ocr，自定义许可） | **transformers 原生推理（MPS/CPU）** | ✅ 1.7B VLM，eager attention 免 flash-attn，MPS 可跑 |
| `dots-mocr-mlx` | 同上 | **mlx-vlm（MLX/Metal）** | ⚠️ 注册已就绪；dots.vit（NaViT）架构 mlx-vlm 尚未支持，见[下文](#dotsmocr--mlx-状态) |

统一框架在所有引擎之上提供**同一套 API/CLI 与同一输出结构**（`OCRResult`：
blocks → lines → words + 整页文本 + 引擎元信息），切换或对比引擎只需改一个参数。

## 架构

```
                    unified-ocr (CLI / Python API)
                    统一输出: OCRResult  (text / blocks / metadata)
        ┌───────────────────┼──────────────────────┐
   glm-ocr            paddleocr-vl             hunyuanocr          dots-mocr
  (mlx-vlm)           (mlx-vlm)          (transformers 原生)   (transformers 原生)
   MLX/Metal           MLX/Metal                MPS                  MPS/CPU
        └───────────────────┼──────────────────────┘
                    Apple Silicon (Metal GPU)
```

- **mlx-vlm**（[Blaizzy/mlx-vlm](https://github.com/Blaizzy/mlx-vlm)）同时支持
  GLM-OCR 与 PaddleOCR-VL 两个架构，是本框架的 MLX 核心；两者都无需转换
  GGUF，直接加载权重即可用 Metal 推理。
- **HunyuanOCR** 是 1B 轻量 VLM，官方推荐 transformers `HunYuanVLForConditionalGeneration`
  原生推理（Apple Silicon 走 PyTorch MPS），无需 GGUF/llama.cpp。

## 安装

要求：macOS（Apple Silicon，≥ 14.0）+ [pixi](https://pixi.sh)（依赖统一由
`pixi.toml` 管理，`pyproject.toml` 仅保留打包元数据）。

```bash
git clone <本仓库> && cd unified-ocr

# 1. 核心环境（模型无关，可离线 list-engines / 测试）
pixi install
pixi run list-engines

# 2. MLX 引擎（glm-ocr / paddleocr-vl）
pixi install -e mlx
#   注意：mlx-vlm 必须从 git 安装（PyPI 版尚未内置 GLM-OCR / PaddleOCR-VL 架构），
#   且要求 transformers>=5.0。

# 3. HunyuanOCR 引擎（transformers 原生，可选）
pixi install -e hunyuan   # torch + transformers + accelerate

# 4. dots.mocr 引擎（transformers 原生，可选；Apple Silicon 走 MPS/CPU）
pixi install -e dots      # torch + transformers==4.56.1 + qwen-vl-utils

# 5. 开发 / 测试环境
pixi install -e dev

# 6. 工具环境（PDF 渲染 / 模型权重下载，可选）
pixi install -e tools    # pymupdf + modelscope

# 7. MCP server 环境（HTTP streamable，含上传 + 任务队列）
pixi install -e mcp      # mcp + httpx-sse + uvicorn + starlette + pymupdf
                         # （同时包含 mlx / hunyuan 两个模型引擎依赖）
```

> 每个引擎对应 `pixi.toml` 里的一个 feature / environment；引擎依赖均延迟
> import，默认环境只需核心依赖即可离线运行。

## 模型权重下载

本机无法直连 HuggingFace 时，全部三个权重可从 **ModelScope（阿里）镜像**下载。
GLM-OCR 在 ModelScope 的官方仓库是 `ZhipuAI/GLM-OCR`（非 zai-org）。

```bash
# 或使用 pixi（tools 环境含 modelscope）：
pixi run -e tools download-paddle    # PaddlePaddle/PaddleOCR-VL   → models/PaddleOCR-VL  (2.2GB)
pixi run -e tools download-hunyuan   # Tencent-Hunyuan/HunyuanOCR  → models/HunyuanOCR     (2.0GB)
pixi run -e tools download-glm       # ZhipuAI/GLM-OCR            → models/GLM-OCR        (2.65GB)
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

### dots.mocr 权重下载

[rednote-hilab/dots.mocr](https://huggingface.co/rednote-hilab/dots.mocr)（原
dots.ocr，2026.03 更名）权重可从 HF 或 ModelScope 下载，用官方脚本：

```bash
# dots 仓库 tools/download_model.py：下载到 ./weights/DotsMOCR
python tools/download_model.py --type huggingface  # 或 --type modelscope
```

权重含自定义 modeling 代码（`trust_remote_code=True`），需整目录保留。

## 使用

### MCP server（HTTP streamable，给 Claude Code / Codex / 其他 AI 工具调用）

```bash
pixi run -e mcp mcp
# MCP endpoint: http://localhost:8802/mcp
```

环境变量：

- `MCP_HOST` / `MCP_PORT` — 绑定地址与端口（默认 `0.0.0.0:8802`）
- `MAX_OCR_PARALLEL` — 并行 OCR 任务数上限（默认 `1`；内置任务队列，批量识别按此并发执行）
- `MCP_UPLOAD_DIR` / `MCP_RESULT_DIR` — 上传 / 结果目录（默认 `<repo>/data/uploads|results`）
- `MCP_PUBLIC_BASE_URL` — 生成下载链接用的外部地址（默认 `http://localhost:8802`）
- `MCP_MODEL_CONFIG` — 服务端模型配置文件路径（默认 `<repo>/mcp_server/models.json`）

MCP 工具与调用流程（图片 / PDF 均可，输出为**去 bbox 的标准 Markdown**）：

1. `upload_instructions()` — 获取上传地址与格式，同时列出可用模型**名称**
2. `POST /upload`（multipart 字段 `file`）→ 返回 `file_id`
3. `start_ocr_task(file_id, model=..., max_tokens=...)` — 提交识别任务 → 返回 `task_id`
4. `get_task_status(task_id)` — 轮询进度；`done` 后返回 `result.download_url` 下载结果
5. `list_tasks()` / `model_status()` — 任务列表 / 模型加载状态

**`model` 参数**（`start_ocr_task`，默认取配置里的 `default_model`）：

- 只接受**服务端配置中已声明的模型名称**（如 `glm-ocr` / `paddleocr-vl` / `hunyuanocr`）。
- 客户端传路径、HF id 或 `engine=path` 一律拒绝（返回 `error` + `allowed_models`），避免暴露服务器目录结构。
- 模型名称 → 引擎 + 权重路径的映射集中在服务端 `mcp_server/models.json`：

```json
{
  "default_model": "glm-ocr",
  "models": {
    "glm-ocr":      { "engine": "glm-ocr",      "path": "models/GLM-OCR" },
    "paddleocr-vl": { "engine": "paddleocr-vl", "path": "models/PaddleOCR-VL" },
    "hunyuanocr":   { "engine": "hunyuanocr",   "path": "models/HunyuanOCR" }
  }
}
```

`path` 可省略（用后端默认权重）；相对路径在仓库根下存在时按仓库根解析，否则原样透传（如 HF repo id）；也可用 `MCP_MODEL_CONFIG` 指向自定义配置。

每个 model 首次使用时懒加载并缓存（同一进程内切换模型不会重复加载权重）；推理通过单一锁串行化，避免多个大模型并发占用 Metal GPU。

```bash
# 上传示例
curl -F "file=@/path/to/doc.pdf" http://localhost:8802/upload
```

注册到 Claude Code / Cursor / Claude Desktop：

```json
{
  "mcpServers": {
    "unified-ocr": {
      "type": "http",
      "url": "http://localhost:8802/mcp"
    }
  }
}
```

注册到 OpenAI Codex CLI（`~/.codex/config.toml`）：

```toml
[mcp_servers.unified-ocr]
type = "http"
url = "http://localhost:8802/mcp"
```

### 引擎 CLI（命令行快捷识别单张图片）

```bash
# MLX 引擎（PaddleOCR-VL / GLM-OCR）
python run_mlx_ocr.py paddleocr-vl ./models/PaddleOCR-VL ./page.png 4096
python run_mlx_ocr.py glm-ocr          ./models/GLM-OCR      ./page.png 4096

# HunyuanOCR（transformers）
python run_hunyuan_ocr.py ./models/HunyuanOCR ./page.png 4096

# dots.mocr（transformers 原生，MPS/CPU）
unified-ocr run page.png -e dots-mocr --model dots-mocr=./weights/DotsMOCR
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

### 专利表格 OCR（PP-StructureV3 专用链）

`unified_ocr/patent_table` 模块按论文（FENNEC 2026, Methods → Data curation）
配置 **PP-StructureV3** 流水线，从专利 PDF 中抽取 siRNA 序列表与敲低效应表。
流水线由 PaddleX 实现：`PP-DocLayout_plus-L` 版面分析 + `PP-OCRv5` 整页 OCR
+ `RT-DETR-L` 表格单元格检测 + `SLANeXt` 表格结构识别，按论文参数运行
（`det_limit_side_len=3000`、`thresh=0.15`、`box_thresh=0.4`、
`unclip_ratio=1.5~2.0`、`lang=en`）。

> 与论文「只出表格」的差异：`PPStructureV3` 完整输出**整页 markdown**（正文
> 段落 + 表格 HTML），正文来自版面分析标记的 text 块 + 全页 OCR；只想要表格时
> 消费 `PageResult.tables`（来自 `table_res_list`）即可。

**安装（pixi，macOS）**：

```bash
# CPU（macOS Apple Silicon）
pixi run -e patent patent-pdf patent.pdf -o result/
```

**安装（pip）**：

```bash
pip install -e ".[patent]"         # CPU
pip install -e ".[patent-gpu]"     # Linux GPU（paddlepaddle-gpu 3.3.1）

# GPU 运行（Linux + NVIDIA CUDA，实测 RTX 3060 每页 ~2-4s，较 CPU 快约 40 倍）
python -m unified_ocr.patent_table pdf patent.pdf -o result/ \
    --device gpu:0 --dpi 300
```

**CLI 用法**：

```bash
# 单页图像 → 整页 markdown + 表格 HTML（JSON）
python -m unified_ocr.patent_table run page.png -o page.json --device gpu:0

# PDF 逐页批量（GPU 推荐）：每页一个 JSON + summary.json
python -m unified_ocr.patent_table pdf patent.pdf -o result/ \
    --device gpu:0 --dpi 300 --unclip 2.0

# 完整流程：识别 → 解析 → QC 过滤 → 序列×效应合并
python -m unified_ocr.patent_table full page.png -o patent_result.json --device gpu:0
```

**Python API**：

```python
from unified_ocr.patent_table import PatentTablePipeline, PatentTablePipelineConfig

pipe = PatentTablePipeline(PatentTablePipelineConfig(device="gpu:0"))
try:
    page = pipe.process_page("page_02.png")
    print(page.markdown)       # 整页 markdown（正文 + 表格）
    for html in page.tables:   # 每张表的 HTML（含 rowspan/colspan）
        print(html)
finally:
    pipe.close()
```

**依赖说明**：paddlepaddle-gpu 3.3.1 不在 PyPI（PyPI 仅到 2.6.2），
`patent-gpu` 特性通过 Paddle 官方 cu126 源安装
（`https://www.paddlepaddle.org.cn/packages/stable/cu126/`）。
GPU 推理显存约需 8GB（全模型 + 3000px 大图），显存紧张时
传 `--gpu-mem-mb 5500` 或 `PatentTablePipelineConfig(gpu_memory_limit_mb=...)` 限制。

### MLX 原生 SLANeXt（macOS Metal，ppocr-mlx 权重）

除 PaddleX 路径外，`unified_ocr/patent_table_mlx` 提供了 **SLANeXt 表格结构识别**
的纯 MLX 实现，直接加载 `plaincompute/ppocr-mlx` 的 MLX 权重
（`model.mlx.safetensors`，官方 PaddleX 权重的 MLX 转换），在 Apple Silicon
的 Metal GPU 上运行，无需 Paddle/PaddleX。

- 参数名与 ppocr-mlx safetensors 键一一对应，权重按键直接加载（`strict=True`
  会在任何键/形状不匹配时报错）。
- 已在 M4 Pro 上与 PaddleX 官方 `SLANeXt_wired` 逐 token 对齐验证：
  同一输入下 183/183 结构 token 一致，解码出的表格 HTML 与 PaddleX **逐字节相同**。
- 完整流水线所需的其余模块（版面分析、文本检测/识别、表格单元格检测）在
  ppocr-mlx 中对应 `doclayoutv3/`、`det/`、`rec/`、`en_rec/`、
  `table_cell_wired/` 等目录。

```bash
# 下载全部 ppocr-mlx 权重（约 3.8 GB，HuggingFace 可达时）
python -c "from huggingface_hub import snapshot_download; \
    snapshot_download('plaincompute/ppocr-mlx', local_dir='models/ppocr-mlx')"
```

```python
import mlx.core as mx
from unified_ocr.patent_table_mlx import load_slanext, decode_structure
from unified_ocr.patent_table_mlx.preprocess import preprocess_image

model = load_slanext("models/ppocr-mlx/table_wired")
x = preprocess_image("table.png")                 # CHW 3x512x512 (BGR, PaddleX 同款预处理)
probs = model(mx.array(x[None]))                  # [1, seq, 50]
ids = [int(v) for v in mx.argmax(probs[0], axis=-1)]
print(decode_structure(ids))                      # <html><body><table>…</table></body></html>
```

> 依赖：`pixi run -e mlx …`（mlx / mlx-lm / mlx-vlm）。测试：`pixi run -e mlx-test pytest tests/test_patent_table_mlx.py`
> （无 MLX 的环境会自动 skip 该模块）。

#### 全文档模式（表格 + 非表格内容）

`PatentTableMLXPipeline` 除**只出表格**外，还支持**全文档模式**：用同一套版面分析
定位页面上的所有区域，表格区域走上面的表格链，其余区域（`doc_title` /
`paragraph_title` / `text` / `figure_title` / `formula` / `image` / `chart` 等）
经检测+识别后按阅读顺序渲染为 Markdown，最终拼成一整篇文档。

- 阅读顺序：移植 PaddleX `sorted_layout_boxes`，支持单栏/双栏版面。
- 文本合并：同一区域内按行读取，行距小用空格、行距大（新段落）用换行。
- 标签映射：标题→`#`/`##`，图注→粗体，公式/图→占位（`[formula]`/`[image]`）。
- 忽略标签默认与 PP-StructureV3 一致：`number / footnote / header / footer /
  header_image / footer_image / aside_text`（可用 `markdown_ignore_labels` 覆盖）。

```bash
# 整份 PDF → 全文档 Markdown（表格 + 正文/标题/图注）
pixi run -e mlx-patent patent-mlx-layout-pdf patent.pdf -o out/

# 单页图像 → 全文档
pixi run -e mlx-patent patent-mlx-layout-run page_02.png -o out/

# 表格模式仍是默认（向后兼容）
pixi run -e mlx-patent patent-mlx-pdf patent.pdf -o out/
```

```python
from unified_ocr.patent_table_mlx import PatentTableMLXPipeline, PatentPipelineMLXConfig

pipe = PatentTableMLXPipeline(PatentPipelineMLXConfig(device="cpu"))
try:
    regions = pipe.process_image_layout("page_02.png")   # 表格 + 非表格区域
    print(PatentTableMLXPipeline.document_markdown(regions))

    # 也可整份 PDF：
    # regions = pipe.process_pdf_layout("patent.pdf")
finally:
    pipe.close()
```

输出：`*.mlx_layout.json`（每个区域的 `label` / `kind` / `markdown` / `box`，
表格区域额外含 `html` / `n_cells`）与 `*.mlx_layout.md`（拼好的整篇文档）。

> **识别语言（自动判断）**：默认 `--rec-lang auto`。流水线会先判断文档语种，
> 再选择识别模型：英文文档用论文的 `en_PP-OCRv4_mobile_rec`，中文/CJK 文档自动
> 切到多语模型 `PP-OCRv5_server_rec`（英文模型会把中文正文渲染成乱码）。判断顺序：
> 1. 先读 PDF 自带的文本层（数字版 PDF 免模型、瞬时）；
> 2. 扫描件/图片没有文本层时，用多语识别器对前几页做一次 OCR 试探，按字符集判定。
>
> ```bash
> # 默认：自动判断（中文专利 → PP-OCRv5_server_rec，英文专利 → en_PP-OCRv4_mobile_rec）
> pixi run -e mlx-patent patent-mlx-layout-pdf patent.pdf -o out/
>
> # 强制指定语言（跳过自动判断）
> pixi run -e mlx-patent patent-mlx-layout-pdf patent.pdf -o out/ --rec-lang ch
> pixi run -e mlx-patent patent-mlx-layout-pdf patent.pdf -o out/ --rec-lang en
> ```
>
> 判定结果会写进输出 JSON 的 `config.detected_lang`。也可用
> `--rec-model <name>` 直接指定识别模型（此时不再做语言判断），或用
> `--lang-detect-pages N`（默认 3）调整自动判断采样的页数。表格结构识别
> （MLX SLANeXt）与语言无关，不受影响。
>
> Python API：`pipe.resolve_language(source="patent.pdf")` 返回 `"en"` / `"ch"`，
> 也可传 `sample_images=[...]` 对图片做试探。

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

`--all` 会运行全部引擎；本地模型路径用 `--model <engine>=<路径>` 覆盖
（如 `--model dots-mocr=./weights/DotsMOCR`、`--model hunyuanocr=<路径>`）。

### dots.mocr / MLX 状态

`dots-mocr-mlx` 引擎已注册进统一框架（`list-engines` 可见），但**当前不可直接
运行**：dots.mocr 的自定义视觉编码器 `dots.vit`（NaViT 架构）不在 mlx-vlm 0.3.3
支持列表内，GitHub 上也暂无社区 MLX 转换权重。加载时会给出可操作的指引。

- **macOS 可用路径（现成）**：`dots-mocr`（transformers 原生，MPS/CPU）——
  与官方 `demo_hf.py` 同流程，仅去掉 flash-attn/CUDA 依赖（eager attention）。
- **MLX 路径（待上游）**：待 mlx-vlm 内置 dots 架构（或社区发布 MLX 权重）后，
  `dots-mocr-mlx` 后端即自动可用（加载逻辑已按 `mlx_vlm.load/generate` 写就）。
- 版面解析（bbox + category + text 的 JSON 输出）已映射为结构化
  `OCRBlock`（kind 归一化为 heading/table/formula/text/figure/list，bbox 归一化到
  [0,1]），用 `PROMPT_LAYOUT_ALL_EN` 提示词触发。

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

# 或使用 pixi（dev 环境含 pytest）：
pixi run -e dev test
```

## 许可与致谢

- 本项目代码：MIT License（见 `LICENSE`）。
- 四个下游 OCR 模型的权重 / 代码版权归各自作者所有，使用请遵守其各自许可
  （GLM-OCR：Apache-2.0/MIT；PaddleOCR：Apache-2.0；HunyuanOCR：Tencent
  Hunyuan Community License；dots.mocr/dots.ocr：dots 自定义许可协议）。
- 相关资源：[mlx-vlm](https://github.com/Blaizzy/mlx-vlm)、
  [GLM-OCR MLX 部署文档](https://github.com/zai-org/GLM-OCR/tree/main/examples/mlx-deploy)、
  [HunyuanOCR](https://github.com/Tencent-Hunyuan/HunyuanOCR)、
  [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)、
  [dots.mocr](https://github.com/rednote-hilab/dots.mocr)。