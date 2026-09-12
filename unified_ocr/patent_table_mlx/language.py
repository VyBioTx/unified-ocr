"""Language detection for patent documents (Chinese vs. English).

The recognizer choice depends on the document language: the paper used the
English mobile recognizer (``en_PP-OCRv4_mobile_rec``), but it renders CJK text
as mojibake, so a Chinese patent must use a Chinese/multilingual recognizer
(``PP-OCRv5_server_rec``).  Rather than forcing the user to know the language up
front, this module infers it.

Two independent sources are supported, cheapest first:

* :func:`extract_pdf_text` — the embedded text layer of a digital PDF (no model
  needed, instant).  This covers born-digital patents.
* :func:`classify_language` on OCR output — used as a fallback for scanned
  PDFs / page images, where the pipeline OCR-probes a couple of pages with the
  multilingual recognizer and classifies the returned text.

Detection is *script*-based, not statistical: a document is deemed Chinese when
a non-trivial amount of CJK-script characters is present, English otherwise.
This is robust for patents, whose numeric/sequence tables are language-neutral.
"""

from __future__ import annotations

from typing import Iterable, Optional

# CJK ideographs (+ compatibility), kana and hangul all require a non-ASCII,
# multilingual recognizer.  Japanese/Korean patents therefore resolve to the
# same "ch" (multilingual) model, which is the only correct choice for them too.
_CJK_RANGES = (
    (0x3000, 0x303F),   # CJK punctuation
    (0x3040, 0x30FF),   # Hiragana + Katakana
    (0x3400, 0x4DBF),   # CJK Unified Ideographs Extension A
    (0x4E00, 0x9FFF),   # CJK Unified Ideographs
    (0xAC00, 0xD7AF),   # Hangul syllables
    (0xF900, 0xFAFF),   # CJK Compatibility Ideographs
    (0x20000, 0x2A6DF),  # CJK Extension B
)

# Latin letters.  ASCII + Latin-1/Extended so accented names/units (e.g.
# "TGF-Î²", "vivo") still count as Latin script.
_LATIN_RANGES = (
    (0x0041, 0x005A),   # A-Z
    (0x0061, 0x007A),   # a-z
    (0x00C0, 0x024F),   # Latin-1 Supplement + Latin Extended-A/B
    (0xFF21, 0xFF3A),   # fullwidth A-Z
    (0xFF41, 0xFF5A),   # fullwidth a-z
)

# Defaults tuned for patents: any real Chinese text has hundreds of CJK chars,
# while an English OCR result has essentially none.  A handful of stray CJK
# glyphs from recognizer noise will not trip the ratio.
DEFAULT_MIN_CHARS = 20
DEFAULT_MIN_CJK = 10
DEFAULT_CJK_RATIO = 0.08


def _in_ranges(cp: int, ranges) -> bool:
    return any(lo <= cp <= hi for lo, hi in ranges)


def script_counts(text: str) -> tuple[int, int]:
    """Return ``(cjk, latin)`` letter counts for *text*.

    Digits, whitespace, punctuation and symbols are ignored — patents are full
    of language-neutral chemical/sequence tables, and counting those would
    dilute the signal.
    """
    cjk = latin = 0
    for ch in text or "":
        cp = ord(ch)
        if _in_ranges(cp, _CJK_RANGES):
            cjk += 1
        elif _in_ranges(cp, _LATIN_RANGES):
            latin += 1
    return cjk, latin


def classify_language(
    text: str,
    *,
    min_chars: int = DEFAULT_MIN_CHARS,
    min_cjk: int = DEFAULT_MIN_CJK,
    cjk_ratio: float = DEFAULT_CJK_RATIO,
) -> Optional[str]:
    """Classify *text* as ``"ch"`` / ``"en"``, or ``None`` if too little data.

    ``"ch"`` means "needs the multilingual recognizer" (CJK, kana or hangul);
    ``"en"`` means Latin script.  ``None`` means there are too few script
    characters to decide confidently.
    """
    cjk, latin = script_counts(text)
    if cjk + latin < min_chars:
        return None
    if cjk >= min_cjk and cjk / (cjk + latin) >= cjk_ratio:
        return "ch"
    return "en"


def detect_language_from_texts(
    texts: Iterable[str],
    *,
    min_chars: int = DEFAULT_MIN_CHARS,
    min_cjk: int = DEFAULT_MIN_CJK,
    cjk_ratio: float = DEFAULT_CJK_RATIO,
) -> Optional[str]:
    """Classify a collection of OCR strings by their combined script counts."""
    cjk = latin = 0
    for t in texts:
        c, l = script_counts(t or "")
        cjk += c
        latin += l
    if cjk + latin < min_chars:
        return None
    if cjk >= min_cjk and cjk / (cjk + latin) >= cjk_ratio:
        return "ch"
    return "en"


def extract_pdf_text(pdf_path, max_pages: int = 3) -> str:
    """Return the embedded text layer of the first *max_pages* PDF pages.

    Returns an empty string when PyMuPDF is unavailable or the PDF is scanned
    (no text layer), which lets the caller fall back to OCR-based detection.
    """
    try:
        import fitz  # PyMuPDF
    except ImportError:  # pragma: no cover - fitz is an optional dependency
        return ""

    parts: list[str] = []
    doc = None
    try:
        doc = fitz.open(str(pdf_path))
        for i, page in enumerate(doc):
            if i >= max_pages:
                break
            parts.append(page.get_text("text") or "")
    except Exception:  # pragma: no cover - unreadable/encrypted PDF
        return ""
    finally:
        if doc is not None:
            doc.close()
    return "\n".join(parts)


def detect_language_from_pdf(
    pdf_path,
    max_pages: int = 3,
    *,
    min_chars: int = DEFAULT_MIN_CHARS,
    min_cjk: int = DEFAULT_MIN_CJK,
    cjk_ratio: float = DEFAULT_CJK_RATIO,
) -> Optional[str]:
    """Detect the language from a PDF's embedded text layer (may be ``None``)."""
    return classify_language(
        extract_pdf_text(pdf_path, max_pages=max_pages),
        min_chars=min_chars,
        min_cjk=min_cjk,
        cjk_ratio=cjk_ratio,
    )


__all__ = [
    "classify_language",
    "detect_language_from_texts",
    "detect_language_from_pdf",
    "extract_pdf_text",
    "script_counts",
]
