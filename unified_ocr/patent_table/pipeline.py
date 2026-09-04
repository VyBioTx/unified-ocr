"""PP-StructureV3 专利表格识别流水线封装（适配 paddleocr 3.7.x）。

按论文（FENNEC 2026, Methods → Data curation）配置组装 PaddleX
PP-StructureV3 流水线，以论文指定参数运行：

  - text_det_limit_side_len = 3000
  - text_det_thresh = 0.15
  - text_det_box_thresh = 0.4
  - text_det_unclip_ratio = 1.5 ~ 2.0
  - lang = en

在 paddleocr >= 3.0 中，PP-StructureV3 由 PaddleX 的
LayoutParsingPipeline 实现：

  1. PP-DocLayout_plus-L   版面分析 → layout_det_res（text/table/title 块）
  2. PP-OCRv5_server_det+rec  整页 OCR → overall_ocr_res（全页文本）
  3. RT-DETR-L + SLANeXt   表格结构识别 → table_res_list（表格 HTML）
  4. 三者合并为 parsing_res_list → markdown（整页正文 + 表格）

注意（与论文"只出表格"的差异）：本 pipeline 直接产出 PaddleX 的完整
markdown（含正文段落 + 表格 HTML）；若只要表格，消费 ``tables`` 返回值
（来自 table_res_list）即可——这正是 patent_table 模块 parser/QC 的做法。
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

# paddle 显存上限（MB）。PP-StructureV3 全模型 + 3000px 大图推理约需 8GB；
# 默认不设上限（自动分配），显存紧张时通过 config 指定。
_DEFAULT_FLAGS_GPU_MEMORY_LIMIT_MB = None


@dataclass
class PatentTablePipelineConfig:
    """论文指定的 PP-StructureV3 专利表格识别参数。"""

    # --- 文本检测参数（论文指定） ---
    det_limit_side_len: int = 3000
    det_db_thresh: float = 0.15
    det_db_box_thresh: float = 0.4
    det_db_unclip_ratio: float = 2.0  # 论文给出 1.5–2.0 区间，默认取 2.0

    # --- 语言 ---
    rec_lang: str = "en"

    # --- 推理设备 ---
    # "gpu:0" / "gpu" / "cpu"。默认 None → 由 PaddleX 自动选择（有 GPU 用 GPU）。
    device: str | None = None

    # --- 开关（默认关闭非表格模块，只保留版面+OCR+表格） ---
    use_doc_orientation_classify: bool = False
    use_doc_unwarping: bool = False
    use_textline_orientation: bool = False
    use_seal_recognition: bool = False
    use_table_recognition: bool = True
    use_formula_recognition: bool = False
    use_chart_recognition: bool = False
    use_region_detection: bool = False
    format_block_content: bool = True

    # --- 显存 / 日志 ---
    gpu_memory_limit_mb: int | None = _DEFAULT_FLAGS_GPU_MEMORY_LIMIT_MB
    show_log: bool = False

    # 兼容旧字段（paddleocr 2.x TableRecPipeline 时代的参数，3.x 忽略）
    table_model: str | None = None
    rec_model: str | None = None
    output_dir: str | Path | None = None
    use_gpu: bool | None = None  # 旧版参数；设置后映射到 device

    def __post_init__(self) -> None:
        # 旧版 use_gpu → device 映射（兼容）
        if self.device is None and self.use_gpu is not None:
            self.device = "gpu:0" if self.use_gpu else "cpu"

    def to_paddleocr_kwargs(self) -> dict[str, Any]:
        """转换为 PPStructureV3 构造参数。"""
        kwargs: dict[str, Any] = {
            "lang": self.rec_lang,
            "text_det_limit_side_len": self.det_limit_side_len,
            "text_det_thresh": self.det_db_thresh,
            "text_det_box_thresh": self.det_db_box_thresh,
            "text_det_unclip_ratio": self.det_db_unclip_ratio,
            "use_doc_orientation_classify": self.use_doc_orientation_classify,
            "use_doc_unwarping": self.use_doc_unwarping,
            "use_textline_orientation": self.use_textline_orientation,
            "use_seal_recognition": self.use_seal_recognition,
            "use_table_recognition": self.use_table_recognition,
            "use_formula_recognition": self.use_formula_recognition,
            "use_chart_recognition": self.use_chart_recognition,
            "use_region_detection": self.use_region_detection,
            "format_block_content": self.format_block_content,
        }
        if self.device:
            kwargs["device"] = self.device
        if self.show_log:
            kwargs["show_log"] = True
        return kwargs


@dataclass
class PageResult:
    """单页 PP-StructureV3 识别结果。

    Attributes:
        page_index: 页码（从 1 开始）。
        markdown: 整页 markdown（正文段落 + 表格 HTML，PaddleX 组装）。
        tables: 表格列表，每项为表格 HTML 字符串（来自 table_res_list）。
        table_objects: 原始表格识别结果对象（SingleTableRecognitionResult）。
        width / height: 输入图像尺寸。
        seconds: 该页推理耗时（秒）。
    """

    page_index: int = 1
    markdown: str = ""
    tables: list[str] = field(default_factory=list)
    table_objects: list[Any] = field(default_factory=list)
    width: int = 0
    height: int = 0
    seconds: float = 0.0


class PatentTablePipeline:
    """专利表格 OCR 流水线（PP-StructureV3，paddleocr 3.x）。

    用法::

        pipe = PatentTablePipeline(config=PatentTablePipelineConfig(
            device="gpu:0", det_db_unclip_ratio=2.0,
        ))
        try:
            page = pipe.process_page("patent_page.png")
            print(page.markdown)          # 整页正文 + 表格
            for html in page.tables:      # 每张表的 HTML
                print(html)
        finally:
            pipe.close()
    """

    def __init__(
        self,
        config: PatentTablePipelineConfig | None = None,
    ) -> None:
        self.config = config or PatentTablePipelineConfig()
        self._pipeline: Any = None

    def load(self) -> None:
        """惰性加载 PP-StructureV3 流水线（首次调用自动下载模型权重）。"""
        if self._pipeline is not None:
            return
        try:
            from paddleocr import PPStructureV3
        except ImportError as exc:
            raise ImportError(
                "需要 paddleocr>=3.0。安装：pip install paddleocr 'paddlex[ocr]' "
                "（GPU 另需 paddlepaddle-gpu，见 README）"
            ) from exc

        # 设置 paddle GPU 显存上限（若指定）
        if self.config.gpu_memory_limit_mb:
            os.environ["FLAGS_gpu_memory_limit_mb"] = str(
                self.config.gpu_memory_limit_mb
            )

        log.info(
            "初始化 PP-StructureV3（device=%s, limit_side_len=%d, "
            "thresh=%.2f, box_thresh=%.2f, unclip=%.1f, lang=%s）",
            self.config.device or "auto",
            self.config.det_limit_side_len,
            self.config.det_db_thresh,
            self.config.det_db_box_thresh,
            self.config.det_db_unclip_ratio,
            self.config.rec_lang,
        )
        self._pipeline = PPStructureV3(**self.config.to_paddleocr_kwargs())

    def _parse_page_result(
        self,
        result: Any,
        page_index: int,
        seconds: float,
    ) -> PageResult:
        """把 PaddleX LayoutParsingResultV2 转换为 PageResult。"""
        item = result[0] if isinstance(result, (list, tuple)) else result

        # 整页 markdown
        md_text = ""
        try:
            md = item.markdown
            if isinstance(md, dict):
                md_text = md.get("markdown_texts", "") or ""
            else:
                md_text = str(md)
        except Exception:
            md_text = ""

        # 表格 HTML 列表（table_res_list → pred_html）
        tables: list[str] = []
        table_objects: list[Any] = []
        try:
            trl = item.get("table_res_list", []) or []
            for t in trl:
                html = ""
                if isinstance(t, dict):
                    html = t.get("pred_html") or t.get("html") or ""
                else:
                    try:
                        html = t.get("pred_html") or t.get("html") or ""
                    except Exception:
                        html = ""
                if html:
                    tables.append(html)
                table_objects.append(t)
        except Exception:
            pass

        width = 0
        height = 0
        try:
            width = int(item.get("width", 0) or 0)
            height = int(item.get("height", 0) or 0)
        except Exception:
            pass

        return PageResult(
            page_index=page_index,
            markdown=md_text,
            tables=tables,
            table_objects=table_objects,
            width=width,
            height=height,
            seconds=seconds,
        )

    def process_page(
        self,
        image: str | Path,
        page_index: int = 1,
    ) -> PageResult:
        """对单张页面图像运行 PP-StructureV3，返回整页 markdown + 表格。"""
        self.load()
        import time

        t0 = time.time()
        log.info("识别页面 %d: %s", page_index, image)
        result = self._pipeline.predict(str(image))
        seconds = time.time() - t0
        return self._parse_page_result(result, page_index, seconds)

    # -- 兼容旧 API --------------------------------------------------------
    def process(self, image: str | Path) -> list[dict[str, Any]]:
        """兼容旧接口：只返回表格列表（[{"html": ..., "res": [...]}]）。"""
        page = self.process_page(image)
        return [{"html": h, "res": []} for h in page.tables]

    def close(self) -> None:
        try:
            if self._pipeline is not None:
                self._pipeline.close()
        except Exception:
            pass
        self._pipeline = None

    def __enter__(self) -> "PatentTablePipeline":
        self.load()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


def process_patent_page(
    image_path: str | Path,
    unclip_ratio: float = 2.0,
    **kwargs: Any,
) -> list[dict[str, Any]]:
    """快捷函数：对单张专利页面图像运行表格识别（兼容旧 API）。

    Args:
        image_path: 图像路径。
        unclip_ratio: 检测框展开比例（1.5 或 2.0）。
        kwargs: 其他 PatentTablePipelineConfig 参数（如 device="gpu:0"）。

    Returns:
        表格识别结果列表（[{"html": ...}]）。
    """
    config = PatentTablePipelineConfig(det_db_unclip_ratio=unclip_ratio, **kwargs)
    pipe = PatentTablePipeline(config)
    try:
        return pipe.process(image_path)
    finally:
        pipe.close()
