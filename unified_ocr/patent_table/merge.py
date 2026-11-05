"""序列表格与敲低效应表格的合并。

论文（FENNEC, Methods → Data curation）描述：
  "Following extraction, sequence tables were merged with their
   corresponding knockdown readout tables, linking each siRNA to its
   measured mRNA remaining percentage per cell line and concentration."

本模块将 PP-StructureV3 识别出的序列表与敲低效应表，
按行对应关系合并为每条 siRNA 的完整数据行。
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

from .parser import TableStructure

log = logging.getLogger(__name__)


@dataclass
class MergedEntry:
    """合并后的单条 siRNA 数据项。"""

    sequence: str = ""
    modifications: str = ""
    guide: str = ""
    passenger: str = ""
    target_gene: str = ""
    patent_id: str = ""

    knockdown_data: dict[str, dict[str, float]] = field(default_factory=dict)
    raw: dict[str, str] = field(default_factory=dict)


def merge_sequence_knockdown(
    sequence_table: TableStructure,
    knockdown_table: TableStructure,
    seq_col_map: dict[str, int] | None = None,
    kd_col_map: dict[str, int] | None = None,
    link_column: str | None = None,
) -> list[MergedEntry]:
    """合并序列表与敲低效应表。

    论文的做法是"按 siRNA 序列将序列表与敲低效应表关联"。
    在专利表格中，每行通常对应一条 siRNA，顺序一致，可直接
    按行索引合并。若两表有共同的标识列（如 SEQ ID NO 或序列号），
    则按该列做匹配。

    Args:
        sequence_table: 序列表结构。
        knockdown_table: 敲低效应表结构。
        seq_col_map: 序列表列索引映射 {列名: 列号}。
        kd_col_map: 敲低效应表列索引映射 {列名: 列号}。
        link_column: 两表共有的链接列名（如 SEQ ID NO）。

    Returns:
        合并后的 MergedEntry 列表。
    """
    seq_rows = _rows_to_dicts(sequence_table, seq_col_map) if sequence_table else []
    kd_rows = _rows_to_dicts(knockdown_table, kd_col_map) if knockdown_table else []

    if not seq_rows:
        log.warning("序列表为空，无法合并")
        return []
    if not kd_rows:
        log.warning("敲低效应表为空，无法合并")
        return []

    if link_column and link_column in seq_rows[0] and link_column in kd_rows[0]:
        log.info("按列 %s 做匹配合并", link_column)
        return _merge_by_link(seq_rows, kd_rows, link_column)
    else:
        log.info("按行索引顺序合并（无共用链接列）")
        return _merge_by_index(seq_rows, kd_rows)


def _rows_to_dicts(
    table: TableStructure,
    col_map: dict[str, int] | None,
) -> list[dict[str, str]]:
    """将 TableStructure 转为 [{列名: 文本}] 格式。"""
    if not table.rows:
        return []
    header = table.rows[0]
    headers = [c.text.strip() for c in header.cells]

    rows: list[dict[str, str]] = []
    for row in table.rows[1:]:
        row_dict: dict[str, str] = {}
        for i, cell in enumerate(row.cells):
            col_name = headers[i] if i < len(headers) else f"col_{i}"
            mapped = col_name
            if col_map and col_name in col_map:
                mapped = [k for k, v in col_map.items() if v == i][0]
            row_dict[mapped] = cell.text
        rows.append(row_dict)
    return rows


def _merge_by_link(
    seq_rows: list[dict[str, str]],
    kd_rows: list[dict[str, str]],
    link_column: str,
) -> list[MergedEntry]:
    """通过链接列合并两表。"""
    merged: list[MergedEntry] = []
    kd_by_link = {row.get(link_column, ""): row for row in kd_rows if row.get(link_column)}

    for seq_row in seq_rows:
        link_val = seq_row.get(link_column, "")
        if link_val and link_val in kd_by_link:
            kd_row = kd_by_link[link_val]
            merged.append(_build_entry(seq_row, kd_row))
        else:
            merged.append(_build_entry(seq_row, {}))

    log.info("链接合并: %d 条匹配, %d 条无对应敲低数据",
             sum(1 for m in merged if m.knockdown_data), merged.count(None))
    return merged


def _merge_by_index(
    seq_rows: list[dict[str, str]],
    kd_rows: list[dict[str, str]],
) -> list[MergedEntry]:
    """按行索引顺序合并（支持序列表行数更多）。"""
    merged: list[MergedEntry] = []
    for i, seq_row in enumerate(seq_rows):
        kd_row = kd_rows[i] if i < len(kd_rows) else {}
        merged.append(_build_entry(seq_row, kd_row))

    log.info("索引合并: %d 条序列表行, %d 条敲低表行",
             len(seq_rows), len(kd_rows))
    return merged


def _build_entry(
    seq_row: dict[str, str],
    kd_row: dict[str, str],
) -> MergedEntry:
    """从两行数据构建 MergedEntry。"""
    entry = MergedEntry(raw={**seq_row, **kd_row})

    seq_keys = ["sequence", "siRNA序列", "修饰序列", "序列", "seq"]
    guide_keys = ["guide strand", "反义链", "antisense", "guide", "guide链"]
    passenger_keys = ["passenger strand", "正义链", "sense", "passenger", "passenger链"]
    mod_keys = ["化学修饰", "modification", "修饰", "pattern"]
    gene_keys = ["target gene", "target", "gene", "mRNA", "靶标", "基因"]

    def _word_set(key: str) -> set[str]:
        return set(key.lower().replace("_", " ").replace("-", " ").split())

    wsets = {k: _word_set(k) for k in seq_row}
    ws_list = [(k, v, wsets[k]) for k, v in seq_row.items()]

    # 优先匹配完整词 "sequence"，其次短词 "seq"
    def has_word(words: set[str], target: str) -> bool:
        return target in words

    for key, val, words in ws_list:
        if has_word(words, "sequence"):
            entry.sequence = val
    if not entry.sequence:
        for key, val, words in ws_list:
            if has_word(words, "seq"):
                entry.sequence = val

    for key, val, words in ws_list:
        if has_word(words, "guide") and (has_word(words, "strand") or has_word(words, "反义链") or has_word(words, "antisense")):
            entry.guide = val
    if not entry.guide:
        for key, val, words in ws_list:
            if has_word(words, "guide"):
                entry.guide = val

    for key, val, words in ws_list:
        if has_word(words, "passenger") and (has_word(words, "strand") or has_word(words, "正义链") or has_word(words, "sense")):
            entry.passenger = val
    if not entry.passenger:
        for key, val, words in ws_list:
            if has_word(words, "passenger"):
                entry.passenger = val

    for key, val, words in ws_list:
        if any(w in words for w in ["modification", "修饰"]):
            entry.modifications = val
    for key, val, words in ws_list:
        if any(w in words for w in ["target", "gene", "mRNA", "靶标", "基因"]):
            entry.target_gene = val

    if not entry.sequence:
        entry.sequence = list(seq_row.values())[0] if seq_row else ""

    knockdown_data: dict[str, dict[str, float]] = {}
    for key, val in kd_row.items():
        kl = key.lower()
        if any(k in kl for k in ["remaining", "余", "percent", "%", "活性", "expression"]):
            try:
                numeric_val = float(val.strip().rstrip("%"))
            except (ValueError, AttributeError):
                continue
            cell_line = _extract_cell_line(key, seq_row)
            concentration = _extract_concentration(key, seq_row)
            if cell_line not in knockdown_data:
                knockdown_data[cell_line] = {}
            knockdown_data[cell_line][concentration] = numeric_val

    entry.knockdown_data = knockdown_data
    return entry


def _extract_cell_line(key: str, row: dict[str, str]) -> str:
    """从列名和行数据中提取细胞系名。"""
    cell_line_keywords = ["hela", "hek293", "a549", "huh7", "mcf7", "pc3",
                          "shsy5y", "u2os", "raw264", "b16", "caco2", "hct116"]
    kl = key.lower()
    for cl in cell_line_keywords:
        if cl in kl:
            return cl.upper()
    return "default"


def _extract_concentration(key: str, row: dict[str, str]) -> str:
    """从列名中提取浓度信息。"""
    conc_patterns = [("nm", "nM"), ("μm", "μM"), ("mm", "mM"), ("pm", "pM")]
    kl = key.lower()
    for pat, unit in conc_patterns:
        if pat in kl:
            match = re.search(r"(\d+\.?\d*)\s*" + re.escape(pat), kl)
            if match:
                return f"{match.group(1)}{unit}"
    return "default"


