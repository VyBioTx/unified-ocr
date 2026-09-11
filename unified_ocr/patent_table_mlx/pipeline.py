"""End-to-end patent-table OCR pipeline (macOS).

Assembles the PP-StructureV3 table-extraction chain described in the paper
(FENNEC 2026, Methods -> Data curation) from individual models, with the
**table-structure recognizer running natively on MLX / Metal**:

    page image
      |-- layout detection ....... PP-DocLayout_plus-L            (PaddleX)
      |-- text detection ......... PP-OCRv5_server_det            (PaddleX, paper params)
      |-- text recognition ....... en PP-OCRv4 mobile rec         (PaddleX)
      `-- per table region:
            |-- cell detection ... RT-DETR-L_wired_table_cell_det (PaddleX)
            |-- structure ........ SLANeXt_wired                  (MLX native)
            `-- HTML assembly .... PaddleX TableLabelDecode matching logic (reused)

Paper parameters are honoured on the text detector:
``limit_side_len=3000`` (``limit_type=min``), ``thresh=0.15``,
``box_thresh=0.4``, ``unclip_ratio=2.0``; ``lang=en`` via the EN recognizer.

The MLX structure path is interchangeable with PaddleX: on the same input the
decoded HTML is byte-identical (see ``tests/test_patent_table_mlx.py``).

Two modes are available:

* **table** — :meth:`PatentTableMLXPipeline.process_image` /
  :meth:`~PatentTableMLXPipeline.process_pdf` emit only the table regions.
* **full-document** — :meth:`PatentTableMLXPipeline.process_image_layout` /
  :meth:`~PatentTableMLXPipeline.process_pdf_layout` additionally recognise all
  non-table regions (titles / text / captions / formulas / figures) in reading
  order and assemble a single Markdown document via
  :meth:`PatentTableMLXPipeline.document_markdown`. Regions nested inside a
  table bbox are skipped to avoid duplicating caption text.
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Layout labels (PP-DocLayout_plus-L) → Markdown presentation
# ---------------------------------------------------------------------------
_TITLE_PREFIX = {
    "doc_title": "# ",
    "paragraph_title": "## ",
    "abstract_title": "## ",
    "reference_title": "## ",
    "content_title": "## ",
}
_CAPTION_LABELS = {
    "figure_title", "table_title", "chart_title", "figure_table_chart_title",
}
_VISION_LABELS = {"image", "figure", "chart", "flowchart", "seal"}
# Formula blocks are image-based; OCR yields little text, so mark them explicitly.
_PLACEHOLDER_LABELS = {"formula"} | _VISION_LABELS

# Same default as PaddleX PP-StructureV3: these labels are not emitted to markdown.
DEFAULT_MARKDOWN_IGNORE = (
    "number", "footnote", "header", "header_image",
    "footer", "footer_image", "aside_text",
)


@dataclass
class PatentPipelineMLXConfig:
    """Models and parameters for :class:`PatentTableMLXPipeline`."""

    layout_model: str = "PP-DocLayout_plus-L"
    cell_model: str = "RT-DETR-L_wired_table_cell_det"
    det_model: str = "PP-OCRv5_server_det"
    rec_model: str = "en_PP-OCRv4_mobile_rec"

    # MLX SLANeXt weights (ppocr-mlx checkout)
    slanext_dir: str = "models/ppocr-mlx/table_wired"

    device: Optional[str] = None  # e.g. "cpu" / "gpu:0"; None → PaddleX default
    table_label: str = "table"
    cell_score_thresh: float = 0.3
    det_score_thresh: float = 0.0

    # --- full-document (layout) mode -------------------------------------
    # Drop layout regions below this confidence.
    layout_score_thresh: float = 0.0
    # Layout labels omitted from the Markdown (same defaults as PP-StructureV3).
    markdown_ignore_labels: tuple[str, ...] = DEFAULT_MARKDOWN_IGNORE

    # Paper text-detection parameters
    det_limit_side_len: int = 3000
    det_limit_type: str = "min"
    det_thresh: float = 0.15
    det_box_thresh: float = 0.4
    det_unclip_ratio: float = 2.0

    # PDF rendering
    pdf_dpi: int = 300


@dataclass
class TableResult:
    """One detected table, with structure + recognised cell content."""

    page_index: int = 1
    table_index: int = 0
    box: list = field(default_factory=list)          # page coords [x1, y1, x2, y2]
    html: str = ""
    structure: list = field(default_factory=list)     # token list incl. wrapper
    cells: list = field(default_factory=list)         # crop-coord cell boxes
    ocr_texts: list = field(default_factory=list)     # matched OCR strings

    def to_markdown(self) -> str:
        """Render the extracted table as GitHub-flavoured markdown."""
        from ..patent_table.parser import parse_html_table

        structure = parse_html_table(self.html)
        if structure is None:
            return self.html
        return structure.to_markdown()


@dataclass
class RegionResult:
    """One layout region of a page in full-document mode.

    Covers both table regions (``table`` set, ``label == "table"``) and
    non-table regions (text/title/caption/formula/figure), each already
    rendered to a Markdown fragment in :attr:`markdown`.
    """

    page_index: int = 1
    region_index: int = 0
    label: str = "text"
    box: list = field(default_factory=list)          # page coords [x1, y1, x2, y2]
    markdown: str = ""
    texts: list = field(default_factory=list)        # matched OCR strings
    table: Optional[TableResult] = None

    @property
    def is_table(self) -> bool:
        return self.table is not None


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

def _read_bgr(path_or_array) -> np.ndarray:
    import cv2

    if isinstance(path_or_array, np.ndarray):
        img = path_or_array
    else:
        img = cv2.imread(str(path_or_array), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"cannot read image: {path_or_array}")
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img


def _crop_quad(img: np.ndarray, quad) -> np.ndarray:
    """Perspective-crop a 4-point quad (PaddleX ``CropByQuadPoints``)."""
    import cv2

    points = np.array(quad, dtype=np.float32).reshape(-1, 2)
    if points.shape[0] != 4:
        x1, y1 = points.min(axis=0)
        x2, y2 = points.max(axis=0)
        return img[int(y1):int(y2) + 1, int(x1):int(x2) + 1]

    rect = cv2.minAreaRect(points)
    box = sorted(list(cv2.boxPoints(rect)), key=lambda p: p[0])
    idx_a, idx_d = (0, 1) if box[1][1] > box[0][1] else (1, 0)
    idx_b, idx_c = (2, 3) if box[3][1] > box[2][1] else (3, 2)
    box = [box[idx_a], box[idx_b], box[idx_c], box[idx_d]]

    w = int(max(np.linalg.norm(box[0] - box[1]), np.linalg.norm(box[2] - box[3])))
    h = int(max(np.linalg.norm(box[0] - box[3]), np.linalg.norm(box[1] - box[2])))
    if w <= 1 or h <= 1:
        x1, y1 = points.min(axis=0)
        x2, y2 = points.max(axis=0)
        return img[int(y1):int(y2) + 1, int(x1):int(x2) + 1]

    dst = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    mat = cv2.getPerspectiveTransform(np.float32(box), dst)
    crop = cv2.warpPerspective(img, mat, (w, h), borderMode=cv2.BORDER_REPLICATE,
                               flags=cv2.INTER_CUBIC)
    if crop.shape[0] * 1.0 / crop.shape[1] >= 1.5:
        crop = np.rot90(crop)
    return crop


def _poly_bbox(poly) -> list:
    pts = np.array(poly, dtype=np.float32).reshape(-1, 2)
    return [float(pts[:, 0].min()), float(pts[:, 1].min()),
            float(pts[:, 0].max()), float(pts[:, 1].max())]


def reading_order(regions: list[dict], width: float) -> list[dict]:
    """Order layout blocks top-to-bottom, left-to-right (single/two-column aware).

    Port of PaddleX ``sorted_layout_boxes``: blocks starting in the left band
    and ending before 3/5 of the width are collected as the left column; blocks
    starting past 2/5 of the width as the right column; a full-width block
    flushes both columns first. This yields correct reading order for the
    common single- and two-column patent layouts.
    """
    n = len(regions)
    if n <= 1:
        return list(regions)

    boxes = sorted(regions, key=lambda r: (r["box"][1], r["box"][0]))
    left: list[dict] = []
    right: list[dict] = []
    out: list[dict] = []
    i = 0
    while i < n:
        b = boxes[i]
        if b["box"][0] < width / 4 and b["box"][2] < 3 * width / 5:
            left.append(b)
            i += 1
        elif b["box"][0] > 2 * width / 5:
            right.append(b)
            i += 1
        else:
            out += left
            out += right
            out.append(b)
            left, right = [], []
            i += 1
    out += sorted(left, key=lambda r: r["box"][1])
    out += sorted(right, key=lambda r: r["box"][1])
    return out


def _join_lines(lines: list[tuple[list, str]]) -> str:
    """Join OCR lines in a region: space normally, newline on a large y-gap."""
    lines = [(b, t) for b, t in lines if t and t.strip()]
    if not lines:
        return ""
    lines.sort(key=lambda p: (p[0][1], p[0][0]))
    heights = [max(1.0, b[3] - b[1]) for b, _ in lines]
    median_h = sorted(heights)[len(heights) // 2]
    parts = [lines[0][1].strip()]
    for (pb, _), (cb, ct) in zip(lines, lines[1:]):
        gap = cb[1] - pb[3]
        sep = "\n" if gap > 0.8 * median_h else " "
        parts.append(sep + ct.strip())
    return "".join(parts).strip()


def format_region_markdown(label: str, text: str) -> str:
    """Render a non-table layout region to a Markdown fragment."""
    text = (text or "").strip()
    if label in _TITLE_PREFIX:
        return f"{_TITLE_PREFIX[label]}{text}" if text else ""
    if label in _CAPTION_LABELS:
        return f"**{text}**" if text else ""
    if label in _PLACEHOLDER_LABELS:
        # Formula / figure regions are image-based; OCR text is usually empty.
        return text if text else f"[{label}]"
    return text


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class PatentTableMLXPipeline:
    """PP-StructureV3-style table extraction with MLX-native structure recognition.

    Usage::

        pipe = PatentTableMLXPipeline(PatentPipelineMLXConfig(device="cpu"))
        try:
            tables = pipe.process_image("page_02.png")
            for t in tables:
                print(t.html)
        finally:
            pipe.close()
    """

    def __init__(self, config: PatentPipelineMLXConfig | None = None) -> None:
        self.config = config or PatentPipelineMLXConfig()
        self._layout = None
        self._cell = None
        self._det = None
        self._rec = None
        self._slanext = None
        self._char = None
        self._mx = None

    # -- lazy model loading -------------------------------------------------
    def load(self) -> None:
        if self._slanext is not None:
            return

        import mlx.core as mx
        from paddlex import create_model

        from . import build_character_list, load_slanext

        self._mx = mx
        self._char = build_character_list()

        common: dict[str, Any] = {}
        if self.config.device:
            common["device"] = self.config.device

        log.info("Loading layout model %s", self.config.layout_model)
        self._layout = create_model(self.config.layout_model, **common)

        log.info("Loading table cell detector %s", self.config.cell_model)
        self._cell = create_model(self.config.cell_model, **common)

        log.info("Loading text detector %s (paper params)", self.config.det_model)
        self._det = create_model(
            self.config.det_model,
            limit_side_len=self.config.det_limit_side_len,
            limit_type=self.config.det_limit_type,
            thresh=self.config.det_thresh,
            box_thresh=self.config.det_box_thresh,
            unclip_ratio=self.config.det_unclip_ratio,
            **common,
        )

        log.info("Loading recognizer %s", self.config.rec_model)
        self._rec = create_model(self.config.rec_model, **common)

        log.info("Loading MLX SLANeXt from %s", self.config.slanext_dir)
        self._slanext = load_slanext(self.config.slanext_dir)

    # -- stage: layout ------------------------------------------------------
    def _detect_regions(self, image) -> list[dict]:
        """Run layout detection once, returning all regions with label + box."""
        inp = image if isinstance(image, np.ndarray) else str(image)
        out = list(self._layout.predict(inp))
        res = out[0].json["res"]
        regions = []
        for b in res.get("boxes", []):
            score = float(b.get("score", 1.0))
            if score < self.config.layout_score_thresh:
                continue
            regions.append({
                "label": b.get("label", ""),
                "box": [float(v) for v in b["coordinate"]],
                "score": score,
            })
        return regions

    def _detect_tables(self, image) -> list[list]:
        boxes = [
            r["box"] for r in self._detect_regions(image)
            if r["label"] == self.config.table_label
        ]
        # reading order (top → bottom)
        boxes.sort(key=lambda b: (b[1], b[0]))
        return boxes

    # -- stage: page OCR ----------------------------------------------------
    def _ocr_page(self, img_bgr: np.ndarray):
        det_out = list(self._det.predict(img_bgr))
        det_res = det_out[0].json["res"]
        polys = det_res.get("dt_polys", []) or []
        scores = det_res.get("dt_scores", []) or []

        texts: list[str] = []
        boxes: list[list] = []
        crops = []
        for poly, score in zip(polys, scores):
            if float(score) < self.config.det_score_thresh:
                continue
            crops.append(_crop_quad(img_bgr, poly))
            boxes.append(_poly_bbox(poly))

        if crops:
            rec_out = list(self._rec.predict(crops))
            texts = []
            for o in rec_out:
                res = o.json["res"]
                t = res.get("rec_text", "")
                if isinstance(t, (list, tuple)):
                    t = t[0] if t else ""
                texts.append(t or "")

        keep = min(len(texts), len(boxes))
        return list(zip(boxes[:keep], texts[:keep]))

    # -- stage: MLX structure ----------------------------------------------
    def _slanext_tokens(self, crop_bgr: np.ndarray) -> list[str]:
        from . import decode_structure_tokens
        from .preprocess import preprocess_image

        x = self._mx.array(preprocess_image(crop_bgr)[None])
        probs = self._slanext(x)
        self._mx.eval(probs)
        ids = [int(v) for v in self._mx.argmax(probs[0], axis=-1)]
        return decode_structure_tokens(ids, self._char, with_wrapper=True)

    # -- stage: HTML assembly (reuses PaddleX matching logic) ---------------
    @staticmethod
    def _assemble_html(tokens, cell_boxes, ocr_pairs) -> str:
        from paddlex.inference.pipelines.table_recognition.table_recognition_post_processing_v2 import (
            find_row_start_index,
            get_html_result,
            map_and_get_max,
            match_table_and_ocr,
            sort_table_cells_boxes,
        )

        if not cell_boxes or not ocr_pairs:
            return "".join(tokens)

        ocr_boxes = [p[0] for p in ocr_pairs]
        ocr_texts = [p[1] for p in ocr_pairs]

        cells_sorted, flag = sort_table_cells_boxes([list(map(float, c)) for c in cell_boxes])
        row_start_index = find_row_start_index(tokens)
        mapped = map_and_get_max(flag, row_start_index)
        mapped.append(len(cells_sorted))
        row_start_index.append(len(cells_sorted))

        matched = match_table_and_ocr(cells_sorted, ocr_boxes, mapped, mapped)

        # A cell can match several OCR boxes (e.g. a sequence wrapped over two
        # lines); concatenate them in reading order instead of detection order.
        for row_matched in matched:
            for key, idxs in row_matched.items():
                if isinstance(idxs, list) and len(idxs) > 1:
                    row_matched[key] = sorted(
                        idxs, key=lambda j: (round(ocr_boxes[j][1] / 10.0),
                                             ocr_boxes[j][0])
                    )

        return get_html_result(matched, ocr_texts, tokens, row_start_index)

    # -- stage: full-document (layout) helpers ------------------------------
    @staticmethod
    def _texts_in_box(ocr_pairs, box) -> list[tuple[list, str]]:
        """Select OCR results whose box center lies inside a layout region."""
        x1, y1, x2, y2 = box
        inside = []
        for ob, text in ocr_pairs:
            cx = (ob[0] + ob[2]) / 2.0
            cy = (ob[1] + ob[3]) / 2.0
            if x1 <= cx <= x2 and y1 <= cy <= y2:
                inside.append((ob, text))
        return inside

    def _extract_table_from_region(
        self,
        crop: np.ndarray,
        ocr_pairs,
        page_index: int,
        table_index: int,
        page_box: list,
    ) -> TableResult:
        """Cell detection + MLX structure + HTML assembly for one table crop."""
        x1, y1 = int(round(page_box[0])), int(round(page_box[1]))

        cell_out = list(self._cell.predict(crop))
        cell_res = cell_out[0].json["res"]
        cell_boxes = [
            [float(v) for v in c["coordinate"]]
            for c in cell_res.get("boxes", [])
            if float(c.get("score", 1.0)) >= self.config.cell_score_thresh
        ]

        tokens = self._slanext_tokens(crop)

        local_pairs = []
        for (ob, text) in ocr_pairs:
            lx1, ly1, lx2, ly2 = ob[0] - x1, ob[1] - y1, ob[2] - x1, ob[3] - y1
            if lx2 <= 0 or ly2 <= 0 or lx1 >= crop.shape[1] or ly1 >= crop.shape[0]:
                continue
            local_pairs.append(([lx1, ly1, lx2, ly2], text))

        html = self._assemble_html(tokens, cell_boxes, local_pairs)
        return TableResult(
            page_index=page_index,
            table_index=table_index,
            box=[float(v) for v in page_box],
            html=html,
            structure=tokens,
            cells=cell_boxes,
            ocr_texts=[t for _, t in local_pairs],
        )

    # -- public API ---------------------------------------------------------
    def process_image(self, image, page_index: int = 1) -> list[TableResult]:
        """Run the table-extraction pipeline on one page image (path or BGR ndarray)."""
        self.load()
        img_bgr = _read_bgr(image)

        table_boxes = self._detect_tables(image)
        log.info("page %d: %d table region(s)", page_index, len(table_boxes))
        if not table_boxes:
            return []

        ocr_pairs = self._ocr_page(img_bgr)
        log.info("page %d: %d OCR text boxes", page_index, len(ocr_pairs))

        results: list[TableResult] = []
        for ti, box in enumerate(table_boxes):
            x1, y1, x2, y2 = [int(round(v)) for v in box]
            x1, y1 = max(0, x1), max(0, y1)
            crop = img_bgr[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            results.append(
                self._extract_table_from_region(
                    crop, ocr_pairs, page_index, ti, box
                )
            )
        return results

    # -- full-document mode (tables + non-table layout regions) -------------
    def process_image_layout(self, image, page_index: int = 1) -> list[RegionResult]:
        """Recognise **all** layout regions of one page, in reading order.

        Tables are extracted with the table pipeline; every other region
        (text, titles, captions, formulas, figures) is OCR'd and rendered to a
        Markdown fragment. Returns one :class:`RegionResult` per kept region.
        """
        self.load()
        img_bgr = _read_bgr(image)
        width = img_bgr.shape[1]

        regions = self._detect_regions(image)
        ocr_pairs = self._ocr_page(img_bgr)
        log.info("page %d: %d layout region(s), %d OCR boxes",
                 page_index, len(regions), len(ocr_pairs))

        ordered = reading_order(regions, width)
        ignore = set(self.config.markdown_ignore_labels)
        table_boxes = [
            r["box"] for r in ordered
            if r["label"] == self.config.table_label
        ]

        def _inside_table(box) -> bool:
            # Captions/detections that fall inside a table bbox are already part
            # of the table HTML; emitting them again would duplicate the text.
            cx = (box[0] + box[2]) / 2.0
            cy = (box[1] + box[3]) / 2.0
            return any(
                tb[0] <= cx <= tb[2] and tb[1] <= cy <= tb[3]
                for tb in table_boxes
            )

        results: list[RegionResult] = []
        table_idx = 0
        for box_info in ordered:
            label = box_info["label"]
            box = box_info["box"]
            if label in ignore:
                continue
            if label != self.config.table_label and _inside_table(box):
                continue

            if label == self.config.table_label:
                x1, y1, x2, y2 = [int(round(v)) for v in box]
                x1, y1 = max(0, x1), max(0, y1)
                crop = img_bgr[y1:y2, x1:x2]
                if crop.size == 0:
                    continue
                table = self._extract_table_from_region(
                    crop, ocr_pairs, page_index, table_idx, box
                )
                table_idx += 1
                results.append(RegionResult(
                    page_index=page_index,
                    region_index=len(results),
                    label=label,
                    box=[float(v) for v in box],
                    markdown=table.to_markdown(),
                    texts=table.ocr_texts,
                    table=table,
                ))
            else:
                pairs = self._texts_in_box(ocr_pairs, box)
                text = _join_lines(pairs)
                md = format_region_markdown(label, text)
                if not md:
                    continue
                results.append(RegionResult(
                    page_index=page_index,
                    region_index=len(results),
                    label=label,
                    box=[float(v) for v in box],
                    markdown=md,
                    texts=[t for _, t in pairs],
                ))
        return results

    @staticmethod
    def document_markdown(regions: list[RegionResult], page_headers: bool = True) -> str:
        """Assemble per-page region fragments into one Markdown document."""
        pages: dict[int, list[RegionResult]] = {}
        for r in regions:
            pages.setdefault(r.page_index, []).append(r)

        parts: list[str] = []
        for pi in sorted(pages):
            blocks = [r.markdown for r in pages[pi] if r.markdown]
            if not blocks:
                continue
            body = "\n\n".join(blocks)
            if page_headers:
                parts.append(f"## 第 {pi} 页\n\n{body}")
            else:
                parts.append(body)
        return "\n\n".join(parts)

    def process_pdf_layout(
        self, pdf_path, max_pages: int | None = None
    ) -> list[RegionResult]:
        """Render a PDF and run :meth:`process_image_layout` on each page."""
        results: list[RegionResult] = []
        for i, img in self._iter_pdf_pages(pdf_path, max_pages):
            try:
                results.extend(self.process_image_layout(img, page_index=i))
            except Exception:
                log.exception("page %d failed; skipping", i)
        return results

    # -- PDF rendering ------------------------------------------------------
    def _iter_pdf_pages(self, pdf_path, max_pages: int | None = None):
        """Yield ``(page_index, BGR ndarray)`` for a PDF rendered at pdf_dpi."""
        import cv2
        import fitz  # PyMuPDF

        doc = fitz.open(str(pdf_path))
        try:
            for i, page in enumerate(doc):
                if max_pages is not None and i >= max_pages:
                    break
                pix = page.get_pixmap(dpi=self.config.pdf_dpi)
                arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                    pix.height, pix.width, pix.n
                )
                if pix.n == 4:
                    img = cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
                else:
                    img = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
                log.info("PDF page %d/%d (%dx%d)", i + 1, doc.page_count,
                         pix.width, pix.height)
                yield i + 1, img
        finally:
            doc.close()

    def process_pdf(self, pdf_path, max_pages: int | None = None) -> list[TableResult]:
        """Render a PDF with PyMuPDF and run :meth:`process_image` on each page."""
        results: list[TableResult] = []
        for i, img in self._iter_pdf_pages(pdf_path, max_pages):
            try:
                results.extend(self.process_image(img, page_index=i))
            except Exception:
                log.exception("page %d failed; skipping", i)
        return results

    def close(self) -> None:
        for model in (self._layout, self._cell, self._det, self._rec):
            try:
                if model is not None:
                    model.close()
            except Exception:
                pass
        self._layout = self._cell = self._det = self._rec = None
        self._slanext = None

    def __enter__(self) -> "PatentTableMLXPipeline":
        self.load()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


__all__ = [
    "PatentTableMLXPipeline",
    "PatentPipelineMLXConfig",
    "TableResult",
    "RegionResult",
]
