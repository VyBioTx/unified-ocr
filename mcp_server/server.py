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

The model to use is chosen per task via `model`, but clients only pass a
**model name**. The name -> (engine, weights path/HF id) mapping lives in a
server-side config file (`mcp_server/models.json`), so filesystem paths and
directory structure are never exposed to — or accepted from — MCP clients.
If `model` is omitted it defaults to the config's `default_model`.

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
  MCP_MODEL_CONFIG      server-side model config  (default <repo>/mcp_server/models.json)

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
from dataclasses import dataclass
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

# Built-in fallback default, used only when the config omits `default_model`.
DEFAULT_MODEL = "glm-ocr"

# Server-side model config: maps a public model NAME -> {engine, path}.
# Paths live here (server-side) and are never accepted from MCP clients.
MODEL_CONFIG_PATH = Path(
    os.environ.get("MCP_MODEL_CONFIG", str(REPO_DIR / "mcp_server" / "models.json"))
).expanduser()

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)

log = logging.getLogger("unified_ocr_mcp")
if not log.handlers:
    _h = logging.StreamHandler(sys.stderr)
    _h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    log.addHandler(_h)
log.setLevel(os.environ.get("MCP_LOG_LEVEL", "INFO").upper())


# ── Server-side model config ───────────────────────────────────────────
# The client sends only a model NAME. This module resolves that name to an
# (engine, path) pair using the server-side config, so no filesystem paths
# ever cross the MCP boundary. Config schema (JSON)::
#
#   {
#     "default_model": "glm-ocr",
#     "models": {
#       "glm-ocr":      {"engine": "glm-ocr",      "path": "models/GLM-OCR"},
#       "paddleocr-vl": {"engine": "paddleocr-vl", "path": "models/PaddleOCR-VL"},
#       "hunyuanocr":   {"engine": "hunyuanocr",   "path": "models/HunyuanOCR"}
#     }
#   }
#
# `path` is optional: a relative path is resolved against the repo root when it
# exists, otherwise it is passed through (e.g. a HF repo id); an empty path
# uses the backend's own default model. The config path can be overridden with
# the MCP_MODEL_CONFIG env var.


@dataclass(frozen=True)
class ModelEntry:
    """One configured model: public name + engine + optional weights path/HF id."""

    name: str
    engine: str
    path: str = ""


_model_config_cache: Optional[tuple[str, dict[str, ModelEntry]]] = None


def _fallback_entries() -> dict[str, ModelEntry]:
    """No config file -> expose registry engine ids with their default weights."""
    from unified_ocr import list_engines

    return {s.id: ModelEntry(name=s.id, engine=s.id) for s in list_engines()}


def _read_model_config(path: Path) -> tuple[str, dict[str, ModelEntry]]:
    """Parse the JSON model config, returning (default_name, entries).

    Missing/invalid config falls back to the registered engines so the server
    still starts, but no client-supplied path is ever honored.
    """
    raw: Any = None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        log.warning("model config not found: %s; falling back to engine defaults", path)
    except (OSError, ValueError) as exc:
        log.warning("cannot parse model config %s: %s; falling back", path, exc)

    entries: dict[str, ModelEntry] = {}
    default_name = ""
    if isinstance(raw, dict):
        default_name = str(raw.get("default_model") or "").strip()
        models_raw = raw.get("models")
        if not isinstance(models_raw, dict):
            models_raw = {k: v for k, v in raw.items() if k != "default_model"}
        for name, spec in models_raw.items():
            name = str(name).strip()
            if not name:
                continue
            if isinstance(spec, str):
                entries[name] = ModelEntry(name=name, engine=spec.strip() or name)
            elif isinstance(spec, dict):
                engine = str(spec.get("engine") or name).strip()
                path_val = str(spec.get("path") or "").strip()
                entries[name] = ModelEntry(name=name, engine=engine, path=path_val)
            else:
                log.warning("ignoring invalid model config entry %r", name)

    if not entries:
        return DEFAULT_MODEL, _fallback_entries()
    if default_name not in entries:
        default_name = DEFAULT_MODEL if DEFAULT_MODEL in entries else next(iter(entries))
    return default_name, entries


def _get_model_config() -> tuple[str, dict[str, ModelEntry]]:
    """Cached (default_name, entries) view of the server-side config."""
    global _model_config_cache
    if _model_config_cache is None:
        _model_config_cache = _read_model_config(MODEL_CONFIG_PATH)
    return _model_config_cache


def reload_model_config() -> None:
    """Drop the cached config so the next lookup re-reads MODEL_CONFIG_PATH."""
    global _model_config_cache
    _model_config_cache = None


def _default_model_name() -> str:
    return _get_model_config()[0]


def _allowed_model_names() -> list[str]:
    return sorted(_get_model_config()[1])


def _resolve_model_path(path: str) -> str:
    """Resolve a configured path: repo-relative if it exists, else pass through."""
    if not path:
        return ""
    p = Path(path).expanduser()
    if p.is_absolute():
        return str(p)
    candidate = REPO_DIR / p
    if candidate.exists():
        return str(candidate)
    return path


def _resolve_model_spec(model: str) -> tuple[str, str, str]:
    """Resolve a client-supplied model NAME into (name, engine, path).

    Only names present in the server-side config are accepted. Filesystem
    paths, HF ids, and "engine=path" strings sent by clients are rejected.
    """
    default_name, entries = _get_model_config()
    name = (model or "").strip() or default_name
    if name not in entries:
        raise ValueError(
            f"unknown model {name!r}; allowed models: {', '.join(_allowed_model_names())}"
        )
    entry = entries[name]
    return entry.name, entry.engine, _resolve_model_path(entry.path)


# ── Model registry (shared, lazy, serialized) ──────────────────────────
# Keyed by the resolved model NAME. We keep several backends alive (one per
# distinct model) so switching model between tasks does not reload weights;
# inference is still serialized through a single lock to keep the Metal GPU
# busy with only one job at a time.
_model_backends: dict[str, Any] = {}
_model_load_time: dict[str, Optional[float]] = {}
_model_lock = threading.Lock()
_infer_lock = threading.Lock()  # serialize inference across all models


def _ensure_model(model: str) -> Any:
    """Lazy-load (and cache) the configured backend for a model name."""
    global _model_backends, _model_load_time
    name, engine, path = _resolve_model_spec(model)
    if name in _model_backends:
        return _model_backends[name]
    with _model_lock:
        if name in _model_backends:
            return _model_backends[name]

        from unified_ocr import create_backend

        kwargs: dict[str, Any] = {}
        if path:
            kwargs["model"] = path
        log.info("loading model %r (engine=%s)", name, engine)
        t0 = time.time()
        backend = create_backend(engine, **kwargs)
        backend.load()
        _model_load_time[name] = time.time() - t0
        _model_backends[name] = backend
        log.info("model %r ready in %.1fs", name, _model_load_time[name])
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

        model = options.get("model") or _default_model_name()
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
        "bbox. The 'model' argument MUST be one of the model names returned by "
        "upload_instructions().available_models (resolved server-side); filesystem "
        "paths are not accepted. If omitted it uses the configured default model."
    ),
)


@mcp.tool()
def upload_instructions() -> dict:
    """Return the endpoint and format for uploading a file to this OCR server.

    The agent should POST a multipart request to the returned URL with the file in
    a field named 'file'. The server replies with {"file_id": "...", "filename": ...}.
    `available_models` lists the model names accepted by start_ocr_task.
    """
    from unified_ocr import list_engines

    engine_meta = {s.id: s for s in list_engines()}
    default_name, entries = _get_model_config()
    available_models = []
    for name in sorted(entries):
        entry = entries[name]
        meta = engine_meta.get(entry.engine)
        available_models.append({
            "name": name,
            "engine": entry.engine,
            "display_name": meta.display_name if meta else entry.engine,
            "accelerator": meta.accelerator if meta else "unknown",
        })

    return {
        "upload_url": f"{PUBLIC_BASE_URL}/upload",
        "method": "POST",
        "field": "file",
        "content_type": "multipart/form-data",
        "allowed_extensions": sorted(ALLOWED_EXTENSIONS),
        "max_size_bytes": MAX_UPLOAD_BYTES,
        "default_model": default_name,
        "available_models": available_models,
        "example_curl": f'curl -F "file=@/path/to/doc.pdf" {PUBLIC_BASE_URL}/upload',
    }


@mcp.tool()
def start_ocr_task(
    file_id: str,
    model: str = "",
    max_tokens: int = 8192,
) -> dict:
    """Start an OCR task for an uploaded file (image or PDF).

    Args:
        file_id: id returned by POST /upload.
        model: a model name from upload_instructions().available_models
            (e.g. glm-ocr). Resolved server-side; filesystem paths are NOT
            accepted. Defaults to the configured default model.
        max_tokens: max tokens per page (default 8192).

    Returns the task_id; poll get_task_status(task_id) for progress.
    """
    matches = [f for f in UPLOAD_DIR.glob(f"{file_id}_*") if f.is_file()]
    if not matches:
        return {"error": f"unknown file_id: {file_id}", "hint": "POST a file to /upload first"}
    upload_path = matches[0]
    filename = upload_path.name[len(file_id) + 1:]

    try:
        name, _engine, _path = _resolve_model_spec(model)
    except ValueError as exc:
        return {"error": str(exc), "allowed_models": _allowed_model_names()}

    task_id = _submit_task(file_id, filename, {"model": name, "max_tokens": max_tokens})
    return {
        "task_id": task_id,
        "file_id": file_id,
        "filename": filename,
        "model": name,
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
    print(f"[mcp]   model config:    {MODEL_CONFIG_PATH}")
    print(f"[mcp]   default model:   {_default_model_name()}")
    print(f"[mcp]   models:          {', '.join(_allowed_model_names()) or '(none)'}")
    uvicorn.run(build_app(), host=MCP_HOST, port=MCP_PORT, log_level="info")
