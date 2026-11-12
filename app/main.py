"""Web server: JSON API + the single-page web UI."""
from __future__ import annotations

import logging
import re
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from . import auth, config
from .jobs import store

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("musicapp")


@asynccontextmanager
async def lifespan(_app):
    store()  # start the worker
    log.info("processing device: %s", config.device())
    yield


app = FastAPI(title="Music App", lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)
app.middleware("http")(auth.middleware)


@app.middleware("http")
async def revalidate_ui(request: Request, call_next):
    # without this, browsers keep an old app.js/style.css after an update (cheap 304s when unchanged)
    response = await call_next(request)
    if request.url.path == "/" or request.url.path.startswith("/static/"):
        response.headers.setdefault("Cache-Control", "no-cache")
    return response

AUDIO_EXTENSIONS = {".mp3", ".wav", ".flac", ".m4a", ".aac", ".ogg", ".opus", ".wma", ".aiff", ".aif",
                    ".mp4", ".mkv", ".webm", ".mov"}


@app.get("/api/system")
def system() -> dict:
    info = {"device": config.device(), "stems_model": config.stems_model(), "minus_model": config.minus_model()}
    if info["device"] == "cuda":
        import torch

        info["gpu"] = torch.cuda.get_device_name(0)
    return info


@app.get("/api/jobs")
def list_jobs() -> list[dict]:
    return store().list()


@app.post("/api/jobs")
async def create_job(url: str = Form(""), file: UploadFile | None = File(None)) -> dict:
    url = url.strip()
    if file is not None and file.filename:
        ext = Path(file.filename).suffix.lower()
        if ext not in AUDIO_EXTENSIONS:
            raise HTTPException(400, f"Unsupported file type {ext or '(none)'}. Use MP3, WAV, FLAC, M4A, ...")
        title = Path(file.filename).stem
        filename = "input" + ext
        job = store().create({"type": "file", "filename": filename, "title": title}, title)
        dest = store().job_dir(job["id"]) / filename
        size = 0
        with dest.open("wb") as out:
            while chunk := await file.read(1 << 20):
                size += len(chunk)
                if size > config.MAX_UPLOAD_MB * 1024 * 1024:
                    out.close()
                    store().delete(job["id"])
                    raise HTTPException(413, f"File is larger than {config.MAX_UPLOAD_MB} MB.")
                out.write(chunk)
    elif url:
        if not re.match(r"^https?://", url):
            raise HTTPException(400, "Paste a full link starting with https://")
        job = store().create({"type": "url", "url": url}, "Downloading...")
    else:
        raise HTTPException(400, "Paste a YouTube link or choose an audio file.")
    store().enqueue(job["id"])
    return job


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> dict:
    try:
        return store().get(job_id)
    except KeyError:
        raise HTTPException(404, "Song not found")


@app.delete("/api/jobs/{job_id}")
def delete_job(job_id: str) -> dict:
    try:
        store().delete(job_id)
    except KeyError:
        raise HTTPException(404, "Song not found")
    except RuntimeError as exc:
        raise HTTPException(409, str(exc))
    return {"ok": True}


@app.get("/api/jobs/{job_id}/files/{name:path}")
def job_file(job_id: str, name: str, download: int = 0):
    try:
        root = store().job_dir(job_id).resolve()
    except KeyError:
        raise HTTPException(404)
    path = (root / name).resolve()
    if root not in path.parents or not path.is_file():
        raise HTTPException(404, "File not found")
    filename = None
    if download:
        job = store().get(job_id)
        safe_title = re.sub(r'[\\/:*?"<>|]+', "", job.get("title") or "song").strip()[:80] or "song"
        filename = f"{safe_title} - {path.name}"
    media = {".musicxml": "application/vnd.recordare.musicxml+xml", ".flac": "audio/flac"}.get(path.suffix)
    return FileResponse(path, media_type=media, filename=filename)


@app.get("/login")
def login_form():
    return RedirectResponse("/") if not auth.enabled() else auth.login_page()


@app.post("/login")
def login(request: Request, password: str = Form("")):
    return auth.check_login(request, password)


@app.get("/")
def index():
    return FileResponse(config.WEB_DIR / "index.html")


app.mount("/static", StaticFiles(directory=config.WEB_DIR), name="static")


def run() -> None:
    import os
    import threading
    import webbrowser

    import uvicorn

    url = f"http://{config.HOST}:{config.PORT}"
    print(f"\n  Music App is running at {url}  (close this window to stop it)")
    print(f"  Password protection: {'ON' if auth.enabled() else 'off (set MUSICAPP_PASSWORD for a live link)'}\n")
    if os.environ.get("MUSICAPP_OPEN_BROWSER") == "1":
        threading.Timer(2.0, lambda: webbrowser.open(url)).start()
    uvicorn.run(app, host=config.HOST, port=config.PORT, log_level="warning")


if __name__ == "__main__":
    run()
