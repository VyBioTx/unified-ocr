"""patent_table 模块的离线测试（不加载真实模型权重）。"""

import json

import pytest

from unified_ocr.patent_table.download_weights import list_available_models
from unified_ocr.patent_table.merge import MergedEntry, merge_sequence_knockdown
from unified_ocr.patent_table.parser import (
    TableCell,
    TableRow,
    TableStructure,
    parse_html_table,
)
from unified_ocr.patent_table.pipeline import PatentTablePipelineConfig
from unified_ocr.patent_table.qc import QCFilter, QCSpec


def test_list_available_models():
    models = list_available_models()
    components = {m["component"] for m in models}
    assert {"det", "table", "rec"} <= components
    assert "layout" in components


def test_pipeline_config_defaults():
    cfg = PatentTablePipelineConfig()
    assert cfg.det_limit_side_len == 3000
    assert cfg.det_db_thresh == 0.15
    assert cfg.det_db_box_thresh == 0.4
    assert cfg.det_db_unclip_ratio == 2.0
    assert cfg.rec_lang == "en"


def test_pipeline_config_to_kwargs():
    cfg = PatentTablePipelineConfig(det_db_unclip_ratio=1.5)
    kwargs = cfg.to_paddleocr_kwargs()
    assert kwargs["text_det_limit_side_len"] == 3000
    assert kwargs["text_det_unclip_ratio"] == 1.5
    assert kwargs["lang"] == "en"
    assert kwargs["use_table_recognition"] is True
    assert kwargs["use_doc_orientation_classify"] is False


def test_pipeline_config_device():
    # 显式 GPU
    cfg = PatentTablePipelineConfig(device="gpu:0")
    assert cfg.to_paddleocr_kwargs()["device"] == "gpu:0"
    # 旧版 use_gpu 映射
    cfg2 = PatentTablePipelineConfig(use_gpu=True)
    assert cfg2.device == "gpu:0"
    assert cfg2.to_paddleocr_kwargs()["device"] == "gpu:0"
    cfg3 = PatentTablePipelineConfig(use_gpu=False)
    assert cfg3.device == "cpu"
    # 默认 None → 不传 device（PaddleX 自动选择）
    cfg4 = PatentTablePipelineConfig()
    assert "device" not in cfg4.to_paddleocr_kwargs()


def test_page_result_defaults():
    from unified_ocr.patent_table.pipeline import PageResult
    pr = PageResult()
    assert pr.markdown == ""
    assert pr.tables == []
    assert pr.page_index == 1
    assert pr.seconds == 0.0


def test_parse_page_result_from_dict_like():
    """验证 _parse_page_result 能从 PaddleX 结果对象提取 markdown + 表格。"""
    from unified_ocr.patent_table.pipeline import PatentTablePipeline

    class FakeTable(dict):
        def __init__(self, html):
            super().__init__(pred_html=html)

    class FakeItem(dict):
        markdown = {"markdown_texts": "# 标题\n正文内容"}

    class FakeResult(list):
        pass

    item = FakeItem()
    item["table_res_list"] = [FakeTable("<table><tr><td>A</td></tr></table>")]
    item["width"] = 100
    item["height"] = 200

    pipe = PatentTablePipeline()
    page = pipe._parse_page_result(FakeResult([item]), page_index=3, seconds=1.5)
    assert page.page_index == 3
    assert page.seconds == 1.5
    assert "标题" in page.markdown
    assert len(page.tables) == 1
    assert "<table>" in page.tables[0]
    assert page.width == 100
    assert page.height == 200


SAMPLE_TABLE_HTML = """
<table>
<tr><th>SEQ ID NO</th><th>Sequence</th><th>Modification</th></tr>
<tr><td>1</td><td>mAmCmGmUmA</td><td>2'-OMe</td></tr>
<tr><td>2</td><td>fUfCfGfUfA</td><td>2'-F</td></tr>
</table>
"""


def test_parse_html_table():
    ts = parse_html_table(SAMPLE_TABLE_HTML)
    assert ts is not None
    assert ts.num_rows == 3
    assert ts.num_cols == 3
    assert ts.rows[0].cells[0].text == "SEQ ID NO"
    assert ts.rows[1].cells[1].text == "mAmCmGmUmA"
    assert ts.rows[2].cells[2].text == "2'-F"


def test_parse_html_table_to_dicts():
    ts = parse_html_table(SAMPLE_TABLE_HTML)
    dicts = ts.to_dicts()
    assert len(dicts) == 3
    assert dicts[1]["1"] == "mAmCmGmUmA"
    assert dicts[2]["2"] == "2'-F"


def test_table_structure_markdown():
    ts = TableStructure(
        rows=[
            TableRow(cells=[TableCell(text="Gene"), TableCell(text="Value")]),
            TableRow(cells=[TableCell(text="TP53"), TableCell(text="1.2")]),
        ],
        num_rows=2, num_cols=2,
    )
    md = ts.to_markdown()
    assert "| Gene | Value |" in md
    assert "| TP53 | 1.2 |" in md
    assert "| ---" in md


def test_table_structure_find_column():
    ts = parse_html_table(SAMPLE_TABLE_HTML)
    assert ts.find_column(["sequence"]) == 1
    assert ts.find_column(["modification"]) == 2
    assert ts.find_column(["nonexistent"]) is None


def test_qc_count_modifications():
    qc = QCFilter()
    assert qc.count_modifications("mAmCmGmU") == 4
    assert qc.count_modifications("fUfCfG") == 3
    assert qc.count_modifications("ACGU") == 0
    assert qc.count_modifications("invAb") == 1


def test_qc_calculate_modification_ratio():
    qc = QCFilter()
    ratio = qc.calculate_modification_ratio("mAmCmGmUmA")
    assert ratio > 0.4
    assert qc.calculate_modification_ratio("ACGU") == 0.0


def test_qc_edit_distance():
    qc = QCFilter()
    assert qc.edit_distance("ACGU", "ACGU") == 0
    assert qc.edit_distance("ACGU", "ACGG") == 1
    assert qc.edit_distance("AAAA", "UUUU") == 4


def test_qc_reverse_complement():
    qc = QCFilter()
    assert qc.reverse_complement("ACGU") == "ACGU"
    assert qc.reverse_complement("AAAA") == "UUUU"


def test_qc_check_sequence_passes():
    qc = QCFilter()
    result = qc.check_sequence("mAmCmGmUmAmCmGmUmAmCmGmUmAmCmGmU")
    assert result["passed"] is True
    assert result["length"] >= 16
    assert result["mod_count"] >= 4


def test_qc_check_sequence_too_short():
    qc = QCFilter()
    result = qc.check_sequence("ACGU")
    assert result["passed"] is False
    assert any("长度" in r for r in result["reasons"])


def test_qc_check_sequence_no_mods():
    qc = QCFilter()
    result = qc.check_sequence("ACGUACGUACGUACGU")
    assert result["passed"] is False
    assert any("修饰" in r for r in result["reasons"])


def test_qc_filter_rows():
    qc = QCFilter()
    rows = [
        {"sequence": "mAmCmGmUmAmCmGmUmAmCmGmUmAmCmGmU"},
        {"sequence": "ACGUACGU"},  # too short + no mods
        {"sequence": "ACGUACGUACGUACGU"},  # no mods
    ]
    filtered = qc.filter_rows(rows)
    assert len(filtered) == 1


def test_merge_sequence_knockdown():
    seq_table = parse_html_table("""
    <table>
    <tr><th>SEQ</th><th>Sequence</th></tr>
    <tr><td>1</td><td>mAmCmGmU</td></tr>
    <tr><td>2</td><td>fUfCfGfU</td></tr>
    </table>
    """)
    kd_table = parse_html_table("""
    <table>
    <tr><th>SEQ</th><th>HeLa_10nM</th><th>HeLa_100nM</th></tr>
    <tr><td>1</td><td>85%</td><td>45%</td></tr>
    <tr><td>2</td><td>90%</td><td>50%</td></tr>
    </table>
    """)

    merged = merge_sequence_knockdown(seq_table, kd_table)
    assert len(merged) == 2
    assert merged[0].sequence == "mAmCmGmU"


def test_merge_empty_sequence_table():
    result = merge_sequence_knockdown(None, None)
    assert result == []


def test_cli_list_models():
    from unified_ocr.patent_table.cli import main
    assert main(["list-models"]) == 0


def test_cli_download():
    from unified_ocr.patent_table.cli import main
    # 不实际下载，仅测试参数解析
    models = list_available_models()
    assert len(models) >= 3