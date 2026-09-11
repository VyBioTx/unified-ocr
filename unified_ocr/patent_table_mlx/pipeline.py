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
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np

log = logging.getLogger(__name__)


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
    def _detect_tables(self, image) -> list[list]:
        inp = image if isinstance(image, np.ndarray) else str(image)
        out = list(self._layout.predict(inp))
        res = out[0].json["res"]
        boxes = []
        for b in res.get("boxes", []):
            if b.get("label") == self.config.table_label:
                boxes.append([float(v) for v in b["coordinate"]])
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

    # -- public API ---------------------------------------------------------
    def process_image(self, image, page_index: int = 1) -> list[TableResult]:
        """Run the full pipeline on one page image (path or BGR ndarray)."""
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

            results.append(TableResult(
                page_index=page_index,
                table_index=ti,
                box=[float(v) for v in box],
                html=html,
                structure=tokens,
                cells=cell_boxes,
                ocr_texts=[t for _, t in local_pairs],
            ))
        return results

    def process_pdf(self, pdf_path, max_pages: int | None = None) -> list[TableResult]:
        """Render a PDF with PyMuPDF and run :meth:`process_image` on each page."""
        import fitz  # PyMuPDF

        doc = fitz.open(str(pdf_path))
        results: list[TableResult] = []
        try:
            for i, page in enumerate(doc):
                if max_pages is not None and i >= max_pages:
                    break
                pix = page.get_pixmap(dpi=self.config.pdf_dpi)
                arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                    pix.height, pix.width, pix.n
                )
                import cv2

                if pix.n == 4:
                    img = cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
                else:
                    img = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
                log.info("PDF page %d/%d (%dx%d)", i + 1, doc.page_count,
                         pix.width, pix.height)
                try:
                    results.extend(self.process_image(img, page_index=i + 1))
                except Exception:
                    log.exception("page %d failed; skipping", i + 1)
        finally:
            doc.close()
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


__all__ = ["PatentTableMLXPipeline", "PatentPipelineMLXConfig", "TableResult"]
