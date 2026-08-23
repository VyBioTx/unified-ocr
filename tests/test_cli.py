"""注册表 / 门面 / CLI 流程的离线测试（不加载真实模型）。"""

import json

import pytest

import unified_ocr
from unified_ocr import OCR, registry
from unified_ocr.base import EngineSpec, OCRBackend
from unified_ocr.models import OCRResult
from unified_ocr.registry import has_engine


class FakeBackend(OCRBackend):
    """不依赖任何推理库的假后端，用于验证统一流程。"""

    spec = EngineSpec(
        id="fake-engine",
        display_name="Fake",
        model_hint="none",
        accelerator="none",
    )
    default_model = "fake"

    def recognize(self, image, **options):
        return OCRResult.from_text(
            self.spec.id, "假识别结果", model=self.model, image=str(image)
        )


@pytest.fixture(autouse=True)
def register_fake():
    if not registry.has_engine("fake-engine"):
        registry.register(FakeBackend)
    yield


def test_registry_lists_three_real_engines():
    ids = {s.id for s in registry.list_engines()}
    assert {"glm-ocr", "paddleocr-vl", "hunyuanocr"} <= ids


def test_registry_lists_dots_engines():
    """dots.mocr 的 transformers(MPS) 与 MLX 引擎均已注册并可见。"""
    ids = {s.id for s in registry.list_engines()}
    assert "dots-mocr" in ids
    assert "dots-mocr-mlx" in ids
    assert has_engine("dots-mocr")
    assert has_engine("dots-mocr-mlx")


def test_ocr_facade_runs_multiple_engines(tmp_path, monkeypatch):
    img = tmp_path / "a.png"
    img.write_bytes(b"fake-image")

    # 替换 create_backend，让所有引擎都走 FakeBackend（离线验证流程）
    def fake_create(engine_id, **kwargs):
        return FakeBackend(**kwargs)

    monkeypatch.setattr(unified_ocr, "create_backend", fake_create)

    ocr = OCR()
    results = ocr.run(img, engines=["glm-ocr", "paddleocr-vl"])
    assert len(results) == 2
    assert all(r.engine == "fake-engine" for r in results)
    assert results[0].text == "假识别结果"
    assert results[0].metadata["image"] == str(img)


def test_cli_list_engines_json(capsys):
    from unified_ocr.cli import main

    assert main(["list-engines", "-o", "json"]) == 0
    out = json.loads(capsys.readouterr().out)
    ids = {e["id"] for e in out}
    assert {"glm-ocr", "paddleocr-vl", "hunyuanocr"} <= ids


def test_cli_run_offline_via_fake(tmp_path, monkeypatch, capsys):
    """monkeypatch 掉 create_backend 后，CLI run 走 FakeBackend，全流程离线可测。"""
    from unified_ocr import cli

    img = tmp_path / "b.png"
    img.write_bytes(b"fake-image")

    def fake_create(engine_id, **kwargs):
        return FakeBackend(**kwargs)

    monkeypatch.setattr(cli, "create_backend", fake_create)

    assert cli.main(["run", str(img), "-e", "glm-ocr", "-o", "json"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out[0]["engine"] == "fake-engine"
    assert out[0]["text"] == "假识别结果"


def test_cli_unknown_engine_rejected(capsys):
    from unified_ocr.cli import main

    assert main(["run", "x.png", "-e", "nope"]) == 1
    err = capsys.readouterr().err
    assert "未知引擎" in err