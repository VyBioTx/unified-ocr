#!/usr/bin/env python3
"""
unified-ocr — HTTP (Streamable) MCP server with file upload + async task queue.

Exposes the unified-ocr framework (GLM-OCR / PaddleOCR-VL / HunyuanOCR) as MCP
tools over the `streamable-http` transport, plus companion HTTP routes on the
same ASGI app:

  POST /upload         multipart file upload  -> {"file_id": "...", "filename": "...", "size": N}
  GET  /files/<path>   serve OCR result files (download links)

MCP tools:
  - upload_instructions()          -> where/how to POST a file (URL, field name, limits)
  - start_ocr_task(file_id, ...)   -> enqueue an OCR job, returns {task_id, status}
  - get_task_status(task_id)       -> poll progress; when done, returns result + download_url
  - list_tasks()                   -> all tasks with status
  - model_status()                 -> model loaded? device? load time

The model to use is chosen per task via `model` (engine id or model path). If
omitted it defaults to `glm-ocr`. Supported engine ids: glm-ocr (default),
paddleocr-vl, hunyuanocr.

Task queue: OCR jobs run on a bounded thread pool. `MAX_OCR_PARALLEL` env var
sets the maximum number of concurrently-running OCR tasks (default 1). OCR
output is converted into clean standard Markdown (bbox removed).

Env vars:
  MCP_HOST              bind host                 (default 0.0.0.0)
  MCP_PORT              bind port                 (default 8802)
  MAX_OCR_PARALLEL      concurrent OCR tasks      (default 1)
  MCP_UPLOAD_DIR        uploaded file dir         (default <repo>/data/uploads)
  MCP_RESULT_DIR        result file dir           (default <repo>/data/results)
  MCP_PUBLIC_BASE_URL   base URL for download links (default http://localhost:8802)

Run:
  pixi run -e mcp mcp            # starts HTTP MCP server
  python mcp_server/server.py    # same, from an env with the mcp deps installed

Connect an MCP client to: http://<host>:<port>/mcp
"""
from __future__ import annotations

import json
import logging
import os
import re
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Optional

REPO_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_DIR))

# ── Config (env-overridable) ───────────────────────────────────────────
MCP_HOST = os.environ.get("MCP_HOST", "0.0.0.0")
MCP_PORT = int(os.environ.get("MCP_PORT", "8802"))
MAX_OCR_PARALLEL = int(os.environ.get("MAX_OCR_PARALLEL", "1"))
UPLOAD_DIR = Path(
    os.environ.get("MCP_UPLOAD_DIR", str(REPO_DIR / "data" / "uploads"))
).expanduser().resolve()
RESULT_DIR = Path(
    os.environ.get("MCP_RESULT_DIR", str(REPO_DIR / "data" / "results"))
).expanduser().resolve()
PUBLIC_BASE_URL = os.environ.get(
    "MCP_PUBLIC_BASE_URL", f"http://localhost:{MCP_PORT}"
).rstrip("/")

ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".pdf"}
MAX_UPLOAD_BYTES = int(os.environ.get("MCP_MAX_UPLOAD_MB", "200")) * 1024 * 1024

DEFAULT_MODEL = "glm-ocr"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)

log = logging.getLogger("unified_ocr_mcp")
if not log.handlers:
    _h = logging.StreamHandler(sys.stderr)
    _h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    log.addHandler(_h)
log.setLevel(os.environ.get("MCP_LOG_LEVEL", "INFO").upper())


# ── Model registry (shared, lazy, serialized) ──────────────────────────
# Keyed by the model spec string the agent passed to start_ocr_task.
# We keep several backends alive (one per distinct model) so switching model
# between tasks does not reload weights; inference is still serialized through
# a single lock to keep the Metal GPU busy with only one job at a time.
_model_backends: dict[str, Any] = {}
_model_load_time: dict[str, Optional[float]] = {}
_model_lock = threading.Lock()
_infer_lock = threading.Lock()  # serialize inference across all models


def _parse_model_spec(model: str) -> tuple[str, str]:
    """Normalize a model argument into (engine_id, path_or_hf_id).

    Supported forms:
      - engine id:            "glm-ocr" | "paddleocr-vl" | "hunyuanocr"
      - "engine=path":        "glm-ocr=./models/GLM-OCR" (explicit local dir / HF id)
      - bare path / HF id:    "./models/GLM-OCR" | "mlx-community/GLM-OCR-bf16"
                              -> engine inferred from directory name prefix.
    """
    model = (model or DEFAULT_MODEL).strip()
    if not model:
        model = DEFAULT_MODEL
    if "=" in model:
        engine, _, path = model.partition("=")
        engine = engine.strip()
        path = path.strip() or None
        return engine, path or ""

    from unified_ocr import list_engines

    ids = {s.id for s in list_engines()}
    if model in ids:
        return model, ""

    lower = model.lower()
    if lower.startswith("glm"):
        return "glm-ocr", model
    if lower.startswith("paddle") or lower.startswith("paddleocr"):
        return "paddleocr-vl", model
    if "hunyuan" in lower:
        return "hunyuanocr", model
    # Unknown -> keep the default engine and treat the string as a model path/id.
    return DEFAULT_MODEL, model


def _ensure_model(model: str) -> Any:
    """Lazy-load (and cache) the backend for a model spec. Returns a backend."""
    global _model_backends, _model_load_time
    key = model.strip() or DEFAULT_MODEL
    if key in _model_backends:
        return _model_backends[key]
    with _model_lock:
        if key in _model_backends:
            return _model_backends[key]
        engine, path = _parse_model_spec(key)

        from unified_ocr import create_backend

        kwargs: dict[str, Any] = {}
        if path:
            kwargs["model"] = path
        log.info("loading model spec %r (engine=%s path=%r)", key, engine, path)
        t0 = time.time()
        backend = create_backend(engine, **kwargs)
        backend.load()
        _model_load_time[key] = time.time() - t0
        _model_backends[key] = backend
        log.info("model %r ready in %.1fs", key, _model_load_time[key])
        return backend


def _infer_image(image_path: str, model: str = DEFAULT_MODEL, max_tokens: int = 8192) -> str:
    """Run OCR on one image with the given model, returning clean Markdown text."""
    backend = _ensure_model(model)
    with _infer_lock:
        result = backend.recognize(image_path, max_tokens=max_tokens)
    return result.to_markdown()


# ── Task queue ─────────────────────────────────────────────────────────
_tasks: dict[str, dict] = {}
_tasks_lock = threading.Lock()
_executor = ThreadPoolExecutor(
    max_workers=max(1, MAX_OCR_PARALLEL),
    thread_name_prefix="ocr-task",
)


def _safe_filename(name: str) -> str:
    name = Path(name).name
    name = re.sub(r"[^A-Za-z0-9._-]", "_", name)
    return name or "file"


def _register_task(file_id: str, filename: str, options: dict) -> str:
    task_id = uuid.uuid4().hex[:12]
    with _tasks_lock:
        _tasks[task_id] = {
            "task_id": task_id,
            "file_id": file_id,
            "filename": filename,
            "status": "queued",
            "progress": 0,
            "message": "queued",
            "options": options,
            "created_at": time.time(),
            "started_at": None,
            "finished_at": None,
            "error": None,
            "result_file": None,
            "download_url": None,
        }
    return task_id


def _run_task(task_id: str, file_id: str, filename: str, options: dict) -> None:
    task: Optional[dict] = None
    try:
        with _tasks_lock:
            task = _tasks[task_id]
            task["status"] = "running"
            task["started_at"] = time.time()
            task["message"] = "loading model..."
            task["progress"] = 5

        upload_path = UPLOAD_DIR / f"{file_id}_{_safe_filename(filename)}"
        if not upload_path.exists():
            raise FileNotFoundError(f"uploaded file not found: {upload_path}")

        model = options.get("model") or DEFAULT_MODEL
        max_tokens = int(options.get("max_tokens", 8192))

        task["message"] = "converting/reading input"
        task["progress"] = 10

        # PDF -> page images; images -> single page
        suffix = upload_path.suffix.lower()
        if suffix == ".pdf":
            import fitz

            doc = fitz.open(str(upload_path))
            page_paths: list[Path] = []
            tmp_pdf_dir = RESULT_DIR / f"_pdf_{task_id}"
            tmp_pdf_dir.mkdir(parents=True, exist_ok=True)
            mat = fitz.Matrix(200 / 72, 200 / 72)
            for i, page in enumerate(doc):
                out = tmp_pdf_dir / f"page_{i + 1:04d}.png"
                page.get_pixmap(matrix=mat).save(str(out))
                page_paths.append(out)
            doc.close()
        else:
            page_paths = [upload_path]

        total = len(page_paths)
        task["message"] = f"OCR in progress (0/{total} pages)"
        md_pages: list[str] = []
        for i, p in enumerate(page_paths, 1):
            task["message"] = f"OCR in progress ({i - 1}/{total} pages)"
            md_pages.append(_infer_image(str(p), model=model, max_tokens=max_tokens))
            task["progress"] = int(10 + (i / total) * 85)

        task["message"] = "writing result"
        task["progress"] = 96

        stem = Path(filename).stem
        result_path = RESULT_DIR / f"{task_id}_{_safe_filename(stem)}.md"
        with open(result_path, "w", encoding="utf-8") as f:
            if total > 1:
                for i, md in enumerate(md_pages, 1):
                    f.write(f"\n\n---\n\n<!-- page {i} / {total} -->\n\n{md}\n".strip())
            else:
                f.write((md_pages[0] + "\n") if md_pages else "")

        task["result_file"] = str(result_path)
        task["download_url"] = f"{PUBLIC_BASE_URL}/files/{result_path.relative_to(RESULT_DIR)}"
        task["status"] = "done"
        task["progress"] = 100
        task["message"] = "done"
        task["finished_at"] = time.time()
    except Exception as e:  # noqa: BLE001 - keep the task record on failure
        if task is not None:
            task["status"] = "error"
            task["message"] = str(e)
            task["error"] = str(e)
            task["finished_at"] = time.time()
        log.exception("task %s failed", task_id)


def _submit_task(file_id: str, filename: str, options: dict) -> str:
    task_id = _register_task(file_id, filename, options)
    _executor.submit(_run_task, task_id, file_id, filename, options)
    return task_id


# ── MCP Server ─────────────────────────────────────────────────────────
try:
    from mcp.server.fastmcp import FastMCP
except ImportError:  # pragma: no cover - dependency path
    print(
        "[mcp] ERROR: mcp package not installed. Run: pixi install -e mcp",
        file=sys.stderr,
    )
    sys.exit(1)

mcp = FastMCP(
    "unified-ocr",
    instructions=(
        "Unified OCR service over HTTP. Workflow: 1) call upload_instructions() "
        "to get the /upload endpoint, 2) POST your image/PDF there with multipart field "
        "'file' to obtain a file_id, 3) call start_ocr_task(file_id, model=...) to "
        "enqueue OCR and get a task_id, 4) poll get_task_status(task_id) until status "
        "is 'done', then use result.download_url. Results are clean Markdown without "
        "bbox. The 'model' argument accepts an engine id (glm-ocr, paddleocr-vl, "
        "hunyuanocr) or 'engine=path'; if omitted it defaults to glm-ocr."
    ),
)


@mcp.tool()
def upload_instructions() -> dict:
    """Return the endpoint and format for uploading a file to this OCR server.

    The agent should POST a multipart request to the returned URL with the file in
    a field named 'file'. The server replies with {"file_id": "...", "filename": ...}.
    """
    from unified_ocr import list_engines

    return {
        "upload_url": f"{PUBLIC_BASE_URL}/upload",
        "method": "POST",
        "field": "file",
        "content_type": "multipart/form-data",
        "allowed_extensions": sorted(ALLOWED_EXTENSIONS),
        "max_size_bytes": MAX_UPLOAD_BYTES,
        "default_model": DEFAULT_MODEL,
        "available_models": [
            {"id": s.id, "model_hint": s.model_hint, "accelerator": s.accelerator}
            for s in list_engines()
        ],
        "example_curl": f'curl -F "file=@/path/to/doc.pdf" {PUBLIC_BASE_URL}/upload',
    }


@mcp.tool()
def start_ocr_task(
    file_id: str,
    model: str = DEFAULT_MODEL,
    max_tokens: int = 8192,
) -> dict:
    """Start an OCR task for an uploaded file (image or PDF).

    Args:
        file_id: id returned by POST /upload.
        model: engine id or 'engine=path'. Engine ids: glm-ocr (default),
            paddleocr-vl, hunyuanocr. A bare path/HF id is also accepted.
        max_tokens: max tokens per page (default 8192).

    Returns the task_id; poll get_task_status(task_id) for progress.
    """
    matches = [f for f in UPLOAD_DIR.glob(f"{file_id}_*") if f.is_file()]
    if not matches:
        return {"error": f"unknown file_id: {file_id}", "hint": "POST a file to /upload first"}
    upload_path = matches[0]
    filename = upload_path.name[len(file_id) + 1:]

    task_id = _submit_task(file_id, filename, {"model": model, "max_tokens": max_tokens})
    return {
        "task_id": task_id,
        "file_id": file_id,
        "filename": filename,
        "model": model,
        "status": "queued",
        "progress": 0,
        "note": "poll get_task_status(task_id) for progress and download_url when done",
    }


@mcp.tool()
def get_task_status(task_id: str) -> dict:
    """Poll an OCR task's progress and, when finished, its result download URL."""
    with _tasks_lock:
        task = _tasks.get(task_id)
        if not task:
            return {"error": f"unknown task_id: {task_id}"}
        t = dict(task)  # snapshot
    status = t["status"]
    resp: dict[str, Any] = {
        "task_id": task_id,
        "file_id": t["file_id"],
        "filename": t["filename"],
        "model": t["options"].get("model"),
        "status": status,
        "progress": t["progress"],
        "message": t["message"],
        "error": t.get("error"),
        "result": None,
    }
    if status == "done":
        resp["result"] = {
            "download_url": t["download_url"],
            "markdown_file": t["result_file"],
        }
    return resp


@mcp.tool()
def list_tasks() -> dict:
    """List all OCR tasks with their current status."""
    with _tasks_lock:
        items = []
        for tid, t in _tasks.items():
            items.append({
                "task_id": tid,
                "filename": t["filename"],
                "model": t["options"].get("model"),
                "status": t["status"],
                "progress": t["progress"],
                "download_url": t.get("download_url"),
            })
        items.sort(key=lambda x: x["task_id"])
    return {"count": len(items), "tasks": items}


@mcp.tool()
def model_status() -> dict:
    """Check which OCR models are loaded and on which device."""
    specs = []
    for key, backend in _model_backends.items():
        specs.append({
            "model": key,
            "loaded": backend.is_loaded,
            "load_time_seconds": round(_model_load_time[key], 2) if _model_load_time.get(key) else None,
        })
    return {
        "models": specs,
        "max_parallel_ocr_tasks": max(1, MAX_OCR_PARALLEL),
    }


# ── Companion HTTP routes on the same ASGI app ────────────────────────
async def _upload_handler(request: Any) -> Any:
    """POST /upload — accept a multipart 'file', store it, return a file_id."""
    try:
        form = await request.form()
        file = form.get("file")
        if file is None:
            return json_response({"error": "missing 'file' field"}, status=400)
        data = await file.read()
        if not data:
            return json_response({"error": "empty file"}, status=400)
        if len(data) > MAX_UPLOAD_BYTES:
            return json_response({"error": "file too large"}, status=413)

        original = getattr(file, "filename", None) or "file"
        suffix = Path(original).suffix.lower()
        if suffix not in ALLOWED_EXTENSIONS:
            return json_response(
                {"error": f"unsupported extension {suffix}", "allowed": sorted(ALLOWED_EXTENSIONS)},
                status=415,
            )

        file_id = uuid.uuid4().hex[:12]
        stored = UPLOAD_DIR / f"{file_id}_{_safe_filename(original)}"
        with open(stored, "wb") as f:
            f.write(data)

        return json_response({
            "file_id": file_id,
            "filename": original,
            "size": len(data),
            "content_type": getattr(file, "content_type", None),
            "next_step": "call MCP tool start_ocr_task(file_id)",
        })
    except Exception as e:  # noqa: BLE001 - HTTP route needs a clean 500
        return json_response({"error": str(e)}, status=500)


def json_response(data: dict, status: int = 200) -> Any:
    from starlette.responses import JSONResponse

    return JSONResponse(data, status_code=status)


async def _index_handler(request: Any) -> Any:
    from starlette.responses import PlainTextResponse

    return PlainTextResponse(
        "unified-ocr MCP (streamable-http)\n"
        f"MCP endpoint: {PUBLIC_BASE_URL}/mcp\n"
        f"Upload endpoint: {PUBLIC_BASE_URL}/upload  (multipart field 'file')\n"
        f"Result files: {PUBLIC_BASE_URL}/files/...\n\n"
        "MCP tools: upload_instructions, start_ocr_task, get_task_status, list_tasks, model_status"
    )


def build_app() -> Any:
    """Compose the FastMCP streamable-http app + our HTTP routes into one ASGI app."""
    from starlette.routing import Mount, Route
    from starlette.staticfiles import StaticFiles

    mcp._custom_starlette_routes.extend([
        Route("/", _index_handler),
        Route("/upload", _upload_handler, methods=["POST"]),
        Mount("/files", app=StaticFiles(directory=str(RESULT_DIR)), name="files"),
    ])
    return mcp.streamable_http_app()


if __name__ == "__main__":
    import uvicorn

    print(f"[mcp] HTTP MCP server starting on {MCP_HOST}:{MCP_PORT}")
    print(f"[mcp]   MCP endpoint:    http://{MCP_HOST}:{MCP_PORT}/mcp")
    print(f"[mcp]   Upload endpoint: http://{MCP_HOST}:{MCP_PORT}/upload")
    print(f"[mcp]   MAX_OCR_PARALLEL={max(1, MAX_OCR_PARALLEL)}")
    print(f"[mcp]   default model:   {DEFAULT_MODEL}")
    uvicorn.run(build_app(), host=MCP_HOST, port=MCP_PORT, log_level="info")
