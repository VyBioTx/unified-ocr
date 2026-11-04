"""MiniCPM-OCR 引擎与 Jev 打分的离线测试（不加载真实 MLX 权重）。"""

from __future__ import annotations

import json
import sys
import types

import pytest

from unified_ocr import registry
from unified_ocr.models import OCRResult
from unified_ocr.qc import JevScorer, apply_qc, render_candidate

MARKDOWN = "| 基因 | 表达量 |\n|------|--------|\n| TP53 | 1.2 |"


class _FakeModel:
    config = {"model_type": "minicpm_ocr"}


def _install_fake_mlx_vlm(monkeypatch, *, prob=0.93, return_object=True):
    """把 mlx_vlm.generate / minicpm_ocr_jev.score 换成可观测的假实现。"""
    calls: dict[str, object] = {}

    def fake_load(path):
        calls["loaded"] = path
        return _FakeModel(), object()

    def fake_generate(model, processor, prompt, image=None, **kwargs):
        calls["prompt"] = prompt
        calls["image"] = image
        calls["kwargs"] = kwargs
        if return_object:
            return types.SimpleNamespace(text=MARKDOWN)
        return MARKDOWN

    def fake_score(model, processor, orig, rendered, text):
        assert str(rendered).endswith(".png")
        calls["jev"] = (orig, rendered, text)
        return {"prob_0": 1 - prob, "prob_1": prob}

    mlx = types.ModuleType("mlx_vlm")
    mlx.load = fake_load
    mlx.generate = fake_generate
    prompt_utils = types.ModuleType("mlx_vlm.prompt_utils")
    prompt_utils.apply_chat_template = (
        lambda processor, config, prompt, num_images=1: f"<|user|>\n{prompt}"
    )
    models_pkg = types.ModuleType("mlx_vlm.models")
    jev = types.ModuleType("mlx_vlm.models.minicpm_ocr_jev")
    jev.score = fake_score
    mlx.prompt_utils = prompt_utils
    mlx.models = models_pkg

    for name, module in (
        ("mlx_vlm", mlx),
        ("mlx_vlm.prompt_utils", prompt_utils),
        ("mlx_vlm.models", models_pkg),
        ("mlx_vlm.models.minicpm_ocr_jev", jev),
    ):
        monkeypatch.setitem(sys.modules, name, module)
    return calls


def test_engine_registered_as_mlx():
    cls = registry.get_backend_class("minicpm-ocr")
    assert cls.spec.accelerator.startswith("MLX")
    assert cls.default_model.endswith("-bf16")


def test_recognize_applies_chat_template_and_reads_generation_text(tmp_path, monkeypatch):
    calls = _install_fake_mlx_vlm(monkeypatch)
    result = registry.create_backend("minicpm-ocr").recognize(tmp_path / "page.png")

    assert result.engine == "minicpm-ocr"
    assert result.text == MARKDOWN
    assert calls["prompt"].startswith("<|user|>")   # chat template 已套上
    assert calls["image"] == [str(tmp_path / "page.png")]
    assert calls["kwargs"]["temperature"] == 0.0


def test_recognize_accepts_plain_string_output(tmp_path, monkeypatch):
    _install_fake_mlx_vlm(monkeypatch, return_object=False)
    result = registry.create_backend("minicpm-ocr").recognize(tmp_path / "page.png")
    assert result.text == MARKDOWN


def test_render_candidate_grows_with_text():
    short = render_candidate("一行字")
    tall = render_candidate("\n".join(f"第 {i} 行内容" for i in range(40)))
    assert short.width == tall.width == 1024
    assert tall.height > short.height
    assert short.mode == "RGB"


def test_jev_score_renders_candidate_and_returns_probability(tmp_path, monkeypatch):
    calls = _install_fake_mlx_vlm(monkeypatch, prob=0.42)
    info = JevScorer().score(tmp_path / "crop.png", MARKDOWN)

    assert info["prob_consistent"] == pytest.approx(0.42)
    assert info["renderer"] == "pillow"
    assert calls["jev"][0] == str(tmp_path / "crop.png")
    assert calls["jev"][2] == MARKDOWN          # 候选文本原样传给模型


def test_jev_score_skips_empty_text():
    info = JevScorer().score("irrelevant.png", "   \n")
    assert info["prob_consistent"] is None


def test_apply_qc_writes_metadata_and_review_flag(tmp_path, monkeypatch):
    _install_fake_mlx_vlm(monkeypatch, prob=0.31)
    result = OCRResult.from_text("glm-ocr", MARKDOWN)
    info = apply_qc(result, tmp_path / "page.png", JevScorer(), threshold=0.5)

    assert info["needs_review"] is True
    assert info["scope"] == "page"
    payload = json.loads(result.to_json())
    assert payload["metadata"]["qc"]["jev"]["prob_consistent"] == pytest.approx(0.31)


def test_cli_qc_prints_scores(tmp_path, monkeypatch, capsys):
    from unified_ocr import cli

    _install_fake_mlx_vlm(monkeypatch, prob=0.8)
    image = tmp_path / "page.png"
    image.write_bytes(b"fake")
    assert cli.main(["qc", str(image), "-e", "minicpm-ocr", "-o", "text"]) == 0
    out = capsys.readouterr().out
    assert "jev=0.80000" in out
    assert "needs_review=None" in out


def test_stale_mlx_vlm_load_error_is_actionable(tmp_path, monkeypatch):
    """旧版 mlx-vlm：加载报 Model type not supported，提示要指名 pixi 命令。"""
    mlx = types.ModuleType("mlx_vlm")

    def fake_load(path):
        raise ValueError("Model type minicpm_ocr not supported.")

    mlx.load = fake_load
    monkeypatch.setitem(sys.modules, "mlx_vlm", mlx)

    with pytest.raises(RuntimeError, match="pixi update mlx-vlm -e mlx"):
        registry.create_backend("minicpm-ocr").recognize(tmp_path / "page.png")


def test_stale_mlx_vlm_score_error_is_actionable(tmp_path, monkeypatch):
    """旧版 mlx-vlm：没有 minicpm_ocr_jev 子模块，报错要能直接看懂。"""
    _install_fake_mlx_vlm(monkeypatch)
    monkeypatch.setitem(sys.modules, "mlx_vlm.models.minicpm_ocr_jev", None)

    with pytest.raises(RuntimeError, match="pixi update mlx-vlm -e mlx"):
        JevScorer().score(tmp_path / "crop.png", MARKDOWN)
