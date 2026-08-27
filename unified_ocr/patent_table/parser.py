"""将 PP-StructureV3 输出的 HTML 表格解析为结构化行列数据。

PP-StructureV3 的表格识别输出以 HTML <table> 形式呈现，
本模块将其解析为可编程操作的行列结构，用于后续的
序列表格与敲低效应表格的合并。
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger(__name__)


@dataclass
class TableCell:
    """表格中的一个单元格。"""

    text: str = ""
    row: int = 0
    col: int = 0
    rowspan: int = 1
    colspan: int = 1
    confidence: float | None = None


@dataclass
class TableRow:
    """表格的一行。"""

    cells: list[TableCell] = field(default_factory=list)

    @property
    def text(self) -> str:
        return " | ".join(c.text for c in self.cells)


@dataclass
class TableStructure:
    """完整的表格结构。"""

    rows: list[TableRow] = field(default_factory=list)
    num_rows: int = 0
    num_cols: int = 0
    source_html: str = ""

    def to_dicts(self) -> list[dict[str, str]]:
        """每行一个字典，列名为列索引。"""
        return [
            {str(c.col): c.text for c in row.cells}
            for row in self.rows
        ]

    def to_markdown(self) -> str:
        """渲染为 Markdown 表格。"""
        if not self.rows:
            return ""
        lines: list[str] = []
        for i, row in enumerate(self.rows):
            cells = [c.text.replace("\n", " ") for c in row.cells]
            lines.append("| " + " | ".join(cells) + " |")
            if i == 0:
                lines.append("| " + " | ".join("---" for _ in row.cells) + " |")
        return "\n".join(lines)

    def find_column(self, keywords: list[str]) -> int | None:
        """按关键词搜索表头，返回匹配的列号。"""
        if not self.rows:
            return None
        header = self.rows[0]
        for col_idx, cell in enumerate(header.cells):
            for kw in keywords:
                if kw.lower() in cell.text.lower():
                    return col_idx
        return None


def parse_html_table(html: str) -> TableStructure | None:
    """将 PP-StructureV3 输出的 HTML 表格解析为 TableStructure。

    Args:
        html: PP-StructureV3 输出的 <table> HTML 字符串。

    Returns:
        TableStructure 对象，解析失败时返回 None。
    """
    if not html or "<table" not in html.lower():
        log.warning("输入不是有效的 HTML 表格")
        return None

    try:
        from lxml import etree
    except ImportError:
        log.warning("lxml 未安装，使用正则表达式回退")
        return _parse_html_fallback(html)

    try:
        root = etree.fromstring(html.encode("utf-8"), parser=etree.HTMLParser())
        table = root.find(".//table")
        if table is None:
            return _parse_html_fallback(html)
        return _parse_via_lxml(table)
    except Exception as exc:
        log.warning("lxml 解析失败，使用正则回退: %s", exc)
        return _parse_html_fallback(html)


def _parse_via_lxml(table: Any) -> TableStructure:
    """使用 lxml 解析 HTML 表格。"""
    from lxml import etree

    rows: list[TableRow] = []
    for tr in table.findall(".//tr"):
        cells: list[TableCell] = []
        for td in tr.findall("td"):
            text = _extract_text(td)
            rowspan = int(td.get("rowspan", 1))
            colspan = int(td.get("colspan", 1))
            cells.append(TableCell(
                text=text, rowspan=rowspan, colspan=colspan,
            ))
        for th in tr.findall("th"):
            text = _extract_text(th)
            rowspan = int(th.get("rowspan", 1))
            colspan = int(th.get("colspan", 1))
            cells.append(TableCell(
                text=text, rowspan=rowspan, colspan=colspan,
            ))
        if cells:
            rows.append(TableRow(cells=cells))

    num_cols = max((len(r.cells) for r in rows), default=0)
    return TableStructure(
        rows=rows, num_rows=len(rows), num_cols=num_cols,
    )


def _extract_text(element: Any) -> str:
    """提取 HTML 元素的纯文本（递归处理子元素）。"""
    text = element.text or ""
    for child in element:
        child_text = _extract_text(child)
        if child_text:
            text += child_text
        tail = child.tail or ""
        if tail:
            text += tail
    return text.strip()


def _parse_html_fallback(html: str) -> TableStructure | None:
    """无 lxml 时的正则回退解析（仅处理简单表格）。"""
    row_pattern = re.compile(r"<tr[^>]*>(.*?)</tr>", re.DOTALL | re.IGNORECASE)
    cell_pattern = re.compile(r"<t[dh][^>]*>(.*?)</t[dh]>", re.DOTALL | re.IGNORECASE)

    rows: list[TableRow] = []
    for tr_match in row_pattern.finditer(html):
        cells: list[TableCell] = []
        for col_idx, td_match in enumerate(cell_pattern.finditer(tr_match.group(1))):
            cell_text = re.sub(r"<[^>]+>", "", td_match.group(1)).strip()
            cells.append(TableCell(text=cell_text, col=col_idx))
        if cells:
            rows.append(TableRow(cells=cells))

    if not rows:
        return None

    num_cols = max((len(r.cells) for r in rows), default=0)
    return TableStructure(
        rows=rows, num_rows=len(rows), num_cols=num_cols,
    )


def extract_sequence_table(tables: list[dict[str, Any]]) -> TableStructure | None:
    """从识别结果中识别并提取 siRNA 序列表。

    通过表头关键词（如序列、SEQ、siRNA、修饰等）匹配。

    Args:
        tables: process() 返回的表格列表。

    Returns:
        匹配到的序列表，未找到时返回 None。
    """
    seq_keywords = ["sequence", "seq", "siRNA", "sense", "antisense", "guide", "passenger",
                    "序列", "修饰", "modification"]
    for table_data in tables:
        html = table_data.get("html", "")
        ts = parse_html_table(html)
        if ts is None or not ts.rows:
            continue
        header_text = " ".join(c.text for c in ts.rows[0].cells if c.text).lower()
        score = sum(1 for kw in seq_keywords if kw in header_text)
        if score >= 2:
            log.info("检测到序列表（匹配 %d 个关键词）", score)
            ts.source_html = html
            return ts
    return None


def extract_knockdown_table(tables: list[dict[str, Any]]) -> TableStructure | None:
    """从识别结果中识别并提取敲低效应表。

    通过表头关键词（如剩余、百分比、剩余、knockdown、剩余%等）匹配。

    Args:
        tables: process() 返回的表格列表。

    Returns:
        匹配到的敲低效应表，未找到时返回 None。
    """
    kd_keywords = ["remaining", "percent", "knockdown", "cell line", "concentration",
                   "剩余", "百分比", "mRNA", "表达", "活性"]
    for table_data in tables:
        html = table_data.get("html", "")
        ts = parse_html_table(html)
        if ts is None or not ts.rows:
            continue
        header_text = " ".join(c.text for c in ts.rows[0].cells if c.text).lower()
        score = sum(1 for kw in kd_keywords if kw in header_text)
        if score >= 2:
            log.info("检测到敲低效应表（匹配 %d 个关键词）", score)
            ts.source_html = html
            return ts
    return None