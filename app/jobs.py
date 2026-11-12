"""A tiny persistent job queue: one song is processed at a time in a worker thread."""
from __future__ import annotations

import json
import logging
import queue
import threading
import time
import traceback
import uuid
from pathlib import Path

from . import config, pipeline

log = logging.getLogger(__name__)


class JobStore:
    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._queue: queue.Queue[str] = queue.Queue()
        self._last_write: dict[str, float] = {}
        self._recover()
        self._worker = threading.Thread(target=self._work, name="job-worker", daemon=True)
        self._worker.start()

    # -- persistence ----------------------------------------------------------
    def job_dir(self, job_id: str) -> Path:
        if not job_id.isalnum():
            raise KeyError(job_id)
        return self.root / job_id

    def get(self, job_id: str) -> dict:
        path = self.job_dir(job_id) / "job.json"
        if not path.exists():
            raise KeyError(job_id)
        with self._lock:
            return json.loads(path.read_text(encoding="utf-8"))

    def _save(self, job: dict) -> None:
        path = self.job_dir(job["id"]) / "job.json"
        tmp = path.with_suffix(".tmp")
        with self._lock:
            tmp.write_text(json.dumps(job, indent=1), encoding="utf-8")
            tmp.replace(path)

    def _update(self, job_id: str, **fields) -> dict:
        job = self.get(job_id)
        job.update(fields)
        self._save(job)
        return job

    def list(self) -> list[dict]:
        jobs = []
        for d in self.root.iterdir():
            if (d / "job.json").exists():
                try:
                    job = self.get(d.name)
                except Exception:
                    continue
                job.pop("result", None)
                jobs.append(job)
        return sorted(jobs, key=lambda j: j.get("created", 0), reverse=True)

    def _recover(self) -> None:
        """Requeue jobs that were waiting, and fail ones interrupted mid-run."""
        for d in sorted(self.root.iterdir(), key=lambda p: p.stat().st_mtime):
            f = d / "job.json"
            if not f.exists():
                continue
            job = json.loads(f.read_text(encoding="utf-8"))
            if job["status"] == "running":
                job.update(status="error", error="The app was closed while this song was processing. Delete it and try again.")
                self._save(job)
            elif job["status"] == "queued":
                self._queue.put(job["id"])

    # -- API ------------------------------------------------------------------
    def create(self, source: dict, title: str) -> dict:
        job_id = uuid.uuid4().hex[:12]
        self.job_dir(job_id).mkdir(parents=True)
        job = {"id": job_id, "title": title, "source": source, "status": "queued", "step": "queued",
               "message": "Waiting to start", "progress": 0.0, "error": None, "created": time.time()}
        self._save(job)
        return job

    def enqueue(self, job_id: str) -> None:
        self._queue.put(job_id)

    def delete(self, job_id: str) -> None:
        job = self.get(job_id)
        if job["status"] == "running":
            raise RuntimeError("This song is still processing.")
        pipeline.cleanup_job(self.job_dir(job_id))

    # -- worker ---------------------------------------------------------------
    def _work(self) -> None:
        while True:
            job_id = self._queue.get()
            try:
                job = self.get(job_id)
            except KeyError:
                continue  # deleted while queued
            self._update(job_id, status="running", message="Starting", started=time.time())

            def report(step: str, message: str, progress: float, job_id=job_id) -> None:
                now = time.time()
                if now - self._last_write.get(job_id, 0) < 0.4 and progress < 1.0:
                    return
                self._last_write[job_id] = now
                self._update(job_id, step=step, message=message, progress=round(progress, 4))

            try:
                result = pipeline.run(self.job_dir(job_id), job["source"], report)
                self._update(job_id, status="done", step="done", message="Done", progress=1.0,
                             title=result["title"], result=result, finished=time.time())
            except Exception as exc:  # report every failure to the UI
                log.error("job %s failed:\n%s", job_id, traceback.format_exc())
                self._update(job_id, status="error", error=_friendly_error(exc), finished=time.time())


def _friendly_error(exc: Exception) -> str:
    text = str(exc) or exc.__class__.__name__
    if "Sign in to confirm" in text or "not a bot" in text:
        return ("YouTube asked to confirm you're not a bot. Set MUSICAPP_COOKIES_FROM_BROWSER=chrome "
                "(or edge/firefox) in settings.bat, sign in to YouTube in that browser, and try again. "
                f"Details: {text[:300]}")
    if "CUDA out of memory" in text:
        return "The GPU ran out of memory. Close other GPU-heavy apps (games, video editors) and try again."
    return text[:1000]


_store: JobStore | None = None


def store() -> JobStore:
    global _store
    if _store is None:
        _store = JobStore(config.JOBS_DIR)
    return _store
