"""MCP server 核心逻辑的离线测试（不加载真实模型 / 不起 HTTP 服务）。

覆盖:
  - 服务端模型配置：名称解析、拒绝客户端路径、相对路径解析
  - _safe_filename 路径净化
  - 任务队列：注册 -> 运行 -> done，以及 error 路径
  - start_ocr_task / get_task_status / list_tasks 的返回结构（含 download_url）
"""

import json
from pathlib import Path

import pytest

import mcp_server.server as server


def _cfg(default: str = "glm-ocr") -> tuple[str, dict]:
    return (
        default,
        {
            "glm-ocr": server.ModelEntry("glm-ocr", "glm-ocr"),
            "paddleocr-vl": server.ModelEntry("paddleocr-vl", "paddleocr-vl"),
            "hunyuanocr": server.ModelEntry("hunyuanocr", "hunyuanocr"),
            "glm-ocr-local": server.ModelEntry("glm-ocr-local", "glm-ocr", "models/GLM-OCR"),
        },
    )


@pytest.fixture(autouse=True)
def _isolated_dirs(tmp_path, monkeypatch):
    """把数据目录指到临时目录，避免污染仓库 data/。"""
    monkeypatch.setattr(server, "UPLOAD_DIR", tmp_path / "uploads")
    monkeypatch.setattr(server, "RESULT_DIR", tmp_path / "results")
    monkeypatch.setattr(server, "PUBLIC_BASE_URL", "http://test:8802")
    # 服务端模型配置固定为测试用条目（不读仓库 models.json）
    monkeypatch.setattr(server, "_model_config_cache", _cfg())
    server.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    server.RESULT_DIR.mkdir(parents=True, exist_ok=True)
    # 每测试清空任务表与已加载后端
    monkeypatch.setattr(server, "_tasks", {})
    monkeypatch.setattr(server, "_model_backends", {})
    yield


def test_read_model_config(tmp_path, monkeypatch):
    cfg = tmp_path / "models.json"
    cfg.write_text(json.dumps({
        "default_model": "fast",
        "models": {
            "fast": {"engine": "glm-ocr", "path": "models/GLM-OCR"},
            "hunyuanocr": "hunyuanocr",
        },
    }), encoding="utf-8")
    default, entries = server._read_model_config(cfg)
    assert default == "fast"
    assert entries["fast"].engine == "glm-ocr"
    assert entries["fast"].path == "models/GLM-OCR"
    assert entries["hunyuanocr"].engine == "hunyuanocr"
    assert entries["hunyuanocr"].path == ""


def test_read_model_config_missing_falls_back(tmp_path):
    default, entries = server._read_model_config(tmp_path / "nope.json")
    assert default
    assert "glm-ocr" in entries


def test_resolve_model_spec_default():
    assert server._resolve_model_spec("")[:2] == ("glm-ocr", "glm-ocr")
    assert server._resolve_model_spec("paddleocr-vl")[:2] == ("paddleocr-vl", "paddleocr-vl")


def test_resolve_model_spec_rejects_client_paths():
    # 客户端只允许传配置内的模型名称，路径 / HF id / engine=path 一律拒绝
    for bad in (
        "glm-ocr=./models/GLM-OCR",
        "./models/GLM-OCR",
        "/abs/path/GLM-OCR",
        "mlx-community/GLM-OCR-bf16",
        "PaddlePaddle/PaddleOCR-VL",
        "../models/HunyuanOCR",
    ):
        with pytest.raises(ValueError):
            server._resolve_model_spec(bad)


def test_resolve_model_path_relative(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "REPO_DIR", tmp_path)
    (tmp_path / "models" / "GLM-OCR").mkdir(parents=True)
    assert server._resolve_model_path("models/GLM-OCR") == str(tmp_path / "models" / "GLM-OCR")
    # 不存在的相对路径（如 HF repo id）原样透传
    assert server._resolve_model_path("mlx-community/GLM-OCR-bf16") == "mlx-community/GLM-OCR-bf16"
    assert server._resolve_model_path("") == ""


def test_safe_filename():
    assert server._safe_filename("../../etc/passwd") == "passwd"
    assert server._safe_filename("a/b/c.pdf") == "c.pdf"
    assert server._safe_filename("") == "file"


class _FakeTaskRecorder:
    """记录 _run_task 的调用，用于注入 server._executor。"""

    def __init__(self, result: str = "# 标题\n\n正文内容"):
        self.result = result
        self.runs: list[tuple] = []

    def submit(self, fn, *args, **kwargs):
        self.runs.append((fn, args, kwargs))
        fn(*args, **kwargs)  # 同步执行，测试可立即断言
        return None


def test_task_flow_single_image(tmp_path, monkeypatch):
    # 模拟一个已上传的图片文件
    upload = server.UPLOAD_DIR / "abc123_test.png"
    upload.write_bytes(b"fake-image")

    monkeypatch.setattr(server, "_executor", _FakeTaskRecorder())
    monkeypatch.setattr(
        server,
        "_infer_image",
        lambda image_path, model="glm-ocr", max_tokens=8192: "# 识别结果\n\n第一段。",
    )

    # 注册 + 运行（同步 executor）
    task_id = server._register_task("abc123", "test.png", {"model": "glm-ocr"})
    server._run_task(task_id, "abc123", "test.png", {"model": "glm-ocr"})

    with server._tasks_lock:
        task = server._tasks[task_id]
    assert task["status"] == "done"
    assert task["progress"] == 100
    assert task["result_file"] and Path(task["result_file"]).exists()
    assert task["download_url"].startswith("http://test:8802/files/")

    # 结果文件内容为干净 Markdown
    content = Path(task["result_file"]).read_text(encoding="utf-8")
    assert "# 识别结果" in content


def test_task_flow_pdf_multipage(tmp_path, monkeypatch):
    upload = server.UPLOAD_DIR / "abc123_doc.pdf"
    upload.write_bytes(b"%PDF-1.4 fake")

    calls: dict = {}

    import fitz

    class FakeDoc:
        page_count = 2

        class FakePage:
            def get_pixmap(self, matrix=None):
                class Pix:
                    def save(self, path):
                        from pathlib import Path as P

                        P(path).write_bytes(b"png")
                        calls.setdefault("pages", []).append(P(path).name)

                return Pix()

        def __iter__(self):
            return iter([self.FakePage(), self.FakePage()])

        def close(self):
            pass

    monkeypatch.setattr(fitz, "open", lambda path: FakeDoc())
    monkeypatch.setattr(
        server,
        "_infer_image",
        lambda image_path, model="glm-ocr", max_tokens=8192: "## 第 X 页内容",
    )

    task_id = server._register_task("abc123", "doc.pdf", {"model": "glm-ocr"})
    server._run_task(task_id, "abc123", "doc.pdf", {"model": "glm-ocr"})

    with server._tasks_lock:
        task = server._tasks[task_id]
    assert task["status"] == "done"
    content = Path(task["result_file"]).read_text(encoding="utf-8")
    assert "<!-- page 1 / 2 -->" in content
    assert "<!-- page 2 / 2 -->" in content
    assert "第 X 页内容" in content


def test_task_error_sets_status(tmp_path, monkeypatch):
    upload = server.UPLOAD_DIR / "abc123_test.png"
    upload.write_bytes(b"fake-image")

    def boom(image_path, model="glm-ocr", max_tokens=8192):
        raise RuntimeError("inference exploded")

    monkeypatch.setattr(server, "_infer_image", boom)

    task_id = server._register_task("abc123", "test.png", {"model": "glm-ocr"})
    server._run_task(task_id, "abc123", "test.png", {"model": "glm-ocr"})

    with server._tasks_lock:
        task = server._tasks[task_id]
    assert task["status"] == "error"
    assert "inference exploded" in task["error"]


def test_start_and_status_tools(tmp_path, monkeypatch):
    upload = server.UPLOAD_DIR / "abc123_img.png"
    upload.write_bytes(b"fake-image")

    monkeypatch.setattr(server, "_submit_task", lambda fid, fn, opts: "task00000001")

    resp = server.start_ocr_task("abc123")
    assert resp["task_id"] == "task00000001"
    assert resp["model"] == "glm-ocr"

    # 客户端传路径 -> 拒绝，返回允许的模型名称
    bad = server.start_ocr_task("abc123", model="./models/GLM-OCR")
    assert "error" in bad
    assert "glm-ocr" in bad["allowed_models"]

    # 直接注入一个 done 任务验证 get_task_status 输出
    server._tasks["task00000001"] = {
        "task_id": "task00000001",
        "file_id": "abc123",
        "filename": "img.png",
        "status": "done",
        "progress": 100,
        "message": "done",
        "options": {"model": "glm-ocr"},
        "error": None,
        "download_url": "http://test:8802/files/task00000001_img.md",
        "result_file": "ignored",
    }
    st = server.get_task_status("task00000001")
    assert st["status"] == "done"
    assert st["result"]["download_url"] == "http://test:8802/files/task00000001_img.md"

    st_missing = server.get_task_status("nope")
    assert st_missing["error"]


def test_list_tasks_shape():
    server._tasks["a"] = {
        "task_id": "a", "filename": "x.png", "status": "queued", "progress": 0,
        "options": {"model": "glm-ocr"}, "download_url": None,
    }
    resp = server.list_tasks()
    assert resp["count"] == 1
    assert resp["tasks"][0]["model"] == "glm-ocr"


def test_upload_instructions_hides_paths():
    resp = server.upload_instructions()
    assert resp["default_model"] == "glm-ocr"
    names = {m["name"] for m in resp["available_models"]}
    assert {"glm-ocr", "paddleocr-vl", "hunyuanocr"} <= names
    # 不得向客户端暴露任何路径 / 权重位置信息
    for m in resp["available_models"]:
        assert "path" not in m
        assert "model_hint" not in m
