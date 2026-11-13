"""WebUI for remove-ai-watermarks: upload video, strip visible AI marks + metadata, preview & download.

Run (cwd = webui/):
    ..\\.venv\\Scripts\\python.exe app.py
"""

from __future__ import annotations

import contextlib
import os
import re
import shutil
import socket
import subprocess
import sys
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import quote

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

REPO_ROOT = Path(__file__).resolve().parent.parent
CLI = REPO_ROOT / ".venv" / "Scripts" / "remove-ai-watermarks.exe"
WEBUI_DIR = Path(__file__).resolve().parent
DATA_DIR = WEBUI_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
OUTPUT_DIR = DATA_DIR / "outputs"
STATIC_DIR = WEBUI_DIR / "static"

VIDEO_EXTS = {".mp4", ".mov", ".m4v", ".webm", ".mkv", ".avi", ".flv"}
BACKENDS = {"cv2", "migan", "lama"}
MARKS = {"auto", "sora", "veo", "seedance", "doubao", "dola", "hailuo", "kling", "vidu"}
# 处理模式：auto=自动识别来源选策略；all=自动检测可见水印+元数据；metadata=仅元数据；erase=手动/预设区域擦除
MODES = {"auto", "all", "metadata", "erase"}
# 区域预设
ERASE_PRESETS = {"jimeng": "即梦 / Dreamina（左上药丸 + 右下 logo）"}
JOB_TTL_HOURS = 24
LOG_LINES = 200
PORT = int(os.environ.get("PORT", "8000"))

# 国内网络无法直连 huggingface.co；MI-GAN/LaMa 模型下载统一走镜像。
# 必须在导入上游包之前设置，huggingface_hub 首次调用时读取该变量。
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "30")

# 国内网络无法直连 huggingface.co；MI-GAN/LaMa 模型下载统一走镜像。
# 必须在导入上游包之前设置，huggingface_hub 在首次调用时读取该变量。
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "30")

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class Job:
    id: str
    name: str               # display filename
    mode: str               # "all" | "metadata" | "erase"
    backend: str
    mark: str
    trim_end: int = 0       # 剪掉结尾秒数，0 = 不剪
    preset: str = ""        # 擦除区域预设，如 "jimeng"
    temporal: bool = False  # 时间一致性光流（每帧约 +1.1 秒，默认关闭）
    status: str = "queued"  # queued | running | done | failed | cancelled
    message: str = ""
    created: float = field(default_factory=time.time)
    started: float | None = None
    finished: float | None = None
    input_path: Path | None = None
    output_path: Path | None = None
    output_name: str | None = None
    removed_mark: str | None = None
    log: deque[str] = field(default_factory=lambda: deque(maxlen=LOG_LINES))
    proc: subprocess.Popen | None = None

    def to_json(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "mode": self.mode,
            "backend": self.backend,
            "mark": self.mark,
            "trim_end": self.trim_end,
            "preset": self.preset,
            "temporal": self.temporal,
            "status": self.status,
            "message": self.message,
            "created": self.created,
            "elapsed": round((self.finished or time.time()) - (self.started or self.created), 1),
            "removed_mark": self.removed_mark,
            "output_name": self.output_name,
            "media_url": f"/media/{self.id}/{self.output_name}" if self.output_path else None,
            "download_url": f"/download/{self.id}" if self.output_path else None,
            "log": list(self.log)[-14:],
        }


JOBS: dict[str, Job] = {}
JOBS_LOCK = threading.Lock()
WORKER_WAKE = threading.Event()
WORKER_STOP = threading.Event()


def _video_tool(name: str) -> str:
    """解析上游包自带的 ffmpeg/ffprobe 路径（无需系统安装）。"""
    try:
        from remove_ai_watermarks.video_tools import require_video_tool
        return require_video_tool(name)
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(f"无法解析 {name} 路径（请确认已用上游 venv 运行）: {e}") from e


def trim_video_end(path: Path, seconds: int, log: deque[str]) -> str:
    """流复制剪掉视频结尾 seconds 秒，不重编码。返回给用户的提示语，失败返回空串。"""
    try:
        ffprobe = _video_tool("ffprobe")
        ffmpeg = _video_tool("ffmpeg")
        probe = subprocess.run(
            [ffprobe, "-v", "error", "-show_entries", "format=duration",
             "-of", "default=nw=1:nk=1", str(path)],
            capture_output=True, text=True, timeout=60,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
        duration = float(probe.stdout.strip())
        new_dur = duration - seconds
        if new_dur < 1.0:
            return f"视频仅 {duration:.1f} 秒，剪掉 {seconds} 秒后不足 1 秒，已跳过剪尾"
        tmp = path.with_name(path.stem + "_trim" + path.suffix)
        proc = subprocess.run(
            [ffmpeg, "-y", "-v", "error", "-i", str(path), "-map", "0",
             "-t", f"{new_dur:.6f}", "-c", "copy", str(tmp)],
            capture_output=True, text=True, timeout=300,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
        if proc.returncode == 0 and tmp.exists() and tmp.stat().st_size > 0:
            os.replace(tmp, path)
            log.append(f"  [webui] 已剪掉最后 {seconds} 秒: {duration:.2f}s -> {new_dur:.2f}s (流复制，不重编码)")
            return f"已剪掉最后 {seconds} 秒 ({duration:.1f}s → {new_dur:.1f}s)"
        log.append(f"  [webui] 剪尾失败: {(proc.stderr or '').strip()[:300]}")
        tmp.unlink(missing_ok=True)
    except Exception as e:  # noqa: BLE001
        log.append(f"  [webui] 剪尾出错: {e}")
    return ""


def next_queued_job() -> Job | None:
    with JOBS_LOCK:
        for j in JOBS.values():
            if j.status == "queued":
                return j
        return None


def worker_loop() -> None:
    while not WORKER_STOP.is_set():
        WORKER_WAKE.wait(timeout=5)
        WORKER_WAKE.clear()
        while True:
            job = next_queued_job()
            if job is None:
                break
            run_job(job)


def run_job(job: Job) -> None:
    with JOBS_LOCK:
        if job.status != "queued":
            return
        job.status = "running"
        job.started = time.time()
    out_dir = OUTPUT_DIR / job.id
    out_dir.mkdir(parents=True, exist_ok=True)
    ext = job.input_path.suffix.lower()
    out_path = out_dir / f"clean{ext}"
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    # 国内网络无法直连 huggingface.co；未显式配置时走镜像，保证 MI-GAN/LaMa 模型可下载。
    env.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

    # 手动/预设区域擦除走独立引擎（上游未把即梦等水印注册进视频检测表）
    if job.mode == "erase":
        try:
            import erase_engine

            regions = []
            if job.preset == "jimeng":
                w, h, _ = erase_engine.probe_geometry(job.input_path)
                regions = erase_engine.jimeng_regions(w, h)
            if not regions:
                raise RuntimeError("未指定可用的擦除区域")
            job.log.append(f"$ erase preset={job.preset or '-'} regions={regions} backend={job.backend}")
            erased = erase_engine.erase_video_regions(
                job.input_path, out_path, regions,
                backend=job.backend, strip_metadata=True, temporal_consistency=job.temporal,
            )
            job.log.append(f"  [webui] 区域擦除完成: {erased} 帧")
            with JOBS_LOCK:
                if job.status != "cancelled":
                    job.status = "done"
                    job.output_path = out_path
                    job.output_name = f"clean_{Path(job.name).stem}{ext}"
                    label = ERASE_PRESETS.get(job.preset, job.preset)
                    job.message = f"已擦除 {label} 区域 {erased} 帧（后端 {job.backend}）并清理 AI 元数据"
        except Exception as e:  # noqa: BLE001
            job.log.append(f"  [webui] 擦除失败: {e}")
            with JOBS_LOCK:
                if job.status != "cancelled":
                    job.status = "failed"
                    job.message = f"区域擦除失败: {e}"
        finally:
            with JOBS_LOCK:
                job.finished = time.time()
        if job.status == "done" and job.output_path and job.trim_end > 0:
            note = trim_video_end(out_path, job.trim_end, job.log)
            if note:
                with JOBS_LOCK:
                    job.message += f"；{note}"
        return

    if job.mode == "metadata":
        cmd = [str(CLI), "video", "metadata", str(job.input_path), "--remove", "-o", str(out_path)]
    else:
        cmd = [
            str(CLI), "video", "all", str(job.input_path), "-o", str(out_path),
            "--backend", job.backend, "--mark", job.mark,
        ]

    job.log.append(f"$ {' '.join(cmd[1:])}")
    try:
        proc = subprocess.Popen(
            cmd, cwd=str(REPO_ROOT), env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace",
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
        job.proc = proc
        for line in proc.stdout or []:
            line = line.rstrip()
            if line:
                job.log.append(line)
                m = re.search(r"removed (\S+) from (\d+)/(\d+) frames", line, re.IGNORECASE)
                if m:
                    job.removed_mark = m.group(1)
        code = proc.wait()
        with JOBS_LOCK:
            job.proc = None
            if job.status == "cancelled":
                pass
            elif code == 0 and out_path.exists():
                job.output_path = out_path
                job.output_name = f"clean_{Path(job.name).stem}{ext}"
        if job.status == "running" and job.output_path and job.trim_end > 0:
            trim_note = trim_video_end(out_path, job.trim_end, job.log)
        else:
            trim_note = ""
        with JOBS_LOCK:
            if job.status == "cancelled":
                pass
            elif code == 0 and out_path.exists():
                job.status = "done"
                if job.removed_mark:
                    job.message = f"已移除可见水印 ({job.removed_mark}) 并清理 AI 元数据"
                elif "No stable" in "\n".join(job.log) or "no stable" in "\n".join(job.log):
                    job.message = "未检测到稳定的可见水印，已输出去 AI 元数据的直通版本"
                else:
                    job.message = "处理完成"
                if trim_note:
                    job.message += f"；{trim_note}"
            else:
                job.status = "failed"
                job.message = "处理失败，详见日志"
    except Exception as e:  # noqa: BLE001
        with JOBS_LOCK:
            job.status = "failed"
            job.message = f"内部错误: {e}"
    finally:
        with JOBS_LOCK:
            job.finished = time.time()


def cleanup_loop() -> None:
    while not WORKER_STOP.is_set():
        time.sleep(600)
        cutoff = time.time() - JOB_TTL_HOURS * 3600
        with JOBS_LOCK:
            old = [j for j in JOBS.values() if j.finished and j.finished < cutoff]
        for j in old:
            shutil.rmtree(OUTPUT_DIR / j.id, ignore_errors=True)
            if j.input_path:
                try:
                    j.input_path.unlink()
                except OSError:
                    pass
            with JOBS_LOCK:
                JOBS.pop(j.id, None)


@contextlib.asynccontextmanager
async def lifespan(_app: FastAPI):
    # 必须在任何推理前安装：MI-GAN/LaMa 优先 CUDA，失败自动回退 CPU
    try:
        import gpu_accel

        gpu_accel.install()
    except Exception as e:  # noqa: BLE001
        print(f"[gpu] 加速补丁未安装（将使用 CPU）: {e}")
    threading.Thread(target=worker_loop, daemon=True).start()
    threading.Thread(target=cleanup_loop, daemon=True).start()
    yield


app = FastAPI(title="Remove AI Watermarks WebUI", lifespan=lifespan)


def lan_ips() -> list[str]:
    ips: set[str] = set()
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, family=socket.AF_INET):
            ip = info[4][0]
            if not ip.startswith("127."):
                ips.add(ip)
    except OSError:
        pass
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("223.5.5.5", 80))
        ips.add(s.getsockname()[0])
        s.close()
    except OSError:
        pass
    return sorted(ips)


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    return HTMLResponse((STATIC_DIR / "index.html").read_text(encoding="utf-8"))


@app.get("/api/info")
def info() -> dict:
    gpu: dict = {}
    try:
        import gpu_accel

        gpu = gpu_accel.status()
    except Exception as e:  # noqa: BLE001
        gpu = {"device": "unknown", "reason": str(e)}
    return {"ips": lan_ips(), "port": PORT, "gpu": gpu}


@app.post("/api/upload")
async def upload(
    file: UploadFile = File(...),
    mode: str = Form("all"),
    backend: str = Form("cv2"),
    mark: str = Form("auto"),
    trim_end: int = Form(0),
    preset: str = Form(""),
    temporal: int = Form(0),
) -> JSONResponse:
    if mode not in MODES:
        raise HTTPException(400, "mode 无效")
    if backend not in BACKENDS:
        raise HTTPException(400, "backend 无效")
    if mark not in MARKS:
        raise HTTPException(400, "mark 无效")
    if trim_end < 0 or trim_end > 60:
        raise HTTPException(400, "trim_end 需在 0-60 秒之间")
    if preset and preset not in ERASE_PRESETS:
        raise HTTPException(400, "preset 无效")
    if mode == "erase" and not preset:
        raise HTTPException(400, "擦除模式需要选择区域预设")
    orig = Path(file.filename or "video.mp4").name
    ext = Path(orig).suffix.lower()
    if ext not in VIDEO_EXTS:
        raise HTTPException(400, f"不支持的格式 {ext}；支持 {', '.join(sorted(VIDEO_EXTS))}")
    job_id = uuid.uuid4().hex[:12]
    in_path = UPLOAD_DIR / f"{job_id}{ext}"
    size = 0
    with in_path.open("wb") as f:
        while chunk := await file.read(1024 * 1024):
            size += len(chunk)
            f.write(chunk)
    if size == 0:
        in_path.unlink(missing_ok=True)
        raise HTTPException(400, "空文件")
    # auto 模式：先看容器来源，即梦/Dreamina 走区域擦除 + MI-GAN，其余走上游自动检测
    if mode == "auto":
        try:
            import erase_engine

            if erase_engine.detect_jimeng(in_path):
                mode, preset = "erase", "jimeng"
                if backend == "cv2":
                    backend = "migan"
            else:
                mode = "all"
        except Exception:  # noqa: BLE001
            mode = "all"
    job = Job(id=job_id, name=orig, mode=mode, backend=backend, mark=mark,
              trim_end=trim_end, preset=preset, temporal=bool(temporal), input_path=in_path)
    with JOBS_LOCK:
        JOBS[job_id] = job
    WORKER_WAKE.set()
    return {"id": job_id}


@app.post("/api/detect/{job_id}")
def detect_source(job_id: str) -> dict:
    """识别已上传视频的来源，供前端自动预选擦除预设。"""
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job or not job.input_path or not job.input_path.exists():
            raise HTTPException(404, "job not found")
        path = job.input_path
    try:
        import erase_engine

        jimeng = erase_engine.detect_jimeng(path)
    except Exception:  # noqa: BLE001
        jimeng = False
    return {"jimeng": jimeng, "suggest_mode": "erase" if jimeng else "all",
            "suggest_preset": "jimeng" if jimeng else "",
            "suggest_backend": "migan" if jimeng else "cv2"}


@app.get("/api/jobs")
def jobs() -> list[dict]:
    with JOBS_LOCK:
        return [j.to_json() for j in sorted(JOBS.values(), key=lambda x: -x.created)]


@app.get("/api/job/{job_id}")
def job_status(job_id: str) -> dict:
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job:
            raise HTTPException(404, "job not found")
        return job.to_json()


@app.post("/api/job/{job_id}/cancel")
def job_cancel(job_id: str) -> dict:
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job:
            raise HTTPException(404, "job not found")
        if job.status == "queued":
            job.status = "cancelled"
            job.finished = time.time()
        elif job.status == "running" and job.proc:
            job.proc.terminate()
            job.status = "cancelled"
            job.finished = time.time()
    return {"ok": True}


@app.delete("/api/job/{job_id}")
def job_delete(job_id: str, bg: BackgroundTasks) -> dict:
    with JOBS_LOCK:
        job = JOBS.pop(job_id, None)
    if not job:
        raise HTTPException(404, "job not found")
    if job.status == "running" and job.proc:
        job.proc.terminate()
    bg.add_task(shutil.rmtree, OUTPUT_DIR / job_id, True)
    if job.input_path:
        bg.add_task(lambda p=job.input_path: p.unlink(missing_ok=True))
    return {"ok": True}


MEDIA_TYPES = {
    ".mp4": "video/mp4", ".m4v": "video/mp4", ".mov": "video/quicktime",
    ".webm": "video/webm", ".mkv": "video/x-matroska", ".avi": "video/x-msvideo",
    ".flv": "video/x-flv",
}


def _range_response(
    request: Request, path: Path, media_type: str, filename: str | None = None,
) -> StreamingResponse:
    """Stream a file honoring HTTP Range so LAN browsers can seek/preview large videos."""
    size = path.stat().st_size
    start, end = 0, size - 1
    status_code = 200
    rng = request.headers.get("range")
    if rng:
        m = re.match(r"bytes=(\d*)-(\d*)", rng.strip())
        if m:
            if m.group(1):
                start = int(m.group(1))
            if m.group(2):
                end = min(int(m.group(2)), size - 1)
            if not m.group(1) and m.group(2):
                start = max(0, size - int(m.group(2)))
            status_code = 206
    length = end - start + 1
    headers = {
        "Content-Range": f"bytes {start}-{end}/{size}",
        "Accept-Ranges": "bytes",
        "Cache-Control": "no-store",
    }
    if filename:
        headers["Content-Disposition"] = f"attachment; filename*=UTF-8''{quote(filename)}"
    fh = path.open("rb")
    fh.seek(start)

    def iter_file():
        remaining = length
        try:
            while remaining > 0:
                chunk = fh.read(min(1024 * 1024, remaining))
                if not chunk:
                    break
                remaining -= len(chunk)
                yield chunk
        finally:
            fh.close()

    return StreamingResponse(iter_file(), status_code=status_code, media_type=media_type, headers=headers)


@app.get("/media/{job_id}/{name}")
def media(job_id: str, name: str, request: Request) -> StreamingResponse:
    with JOBS_LOCK:
        job = JOBS.get(job_id)
    if not job or not job.output_path or not job.output_path.exists() or name != job.output_name:
        raise HTTPException(404, "not found")
    return _range_response(
        request, job.output_path,
        MEDIA_TYPES.get(job.output_path.suffix.lower(), "video/mp4"),
    )


@app.get("/download/{job_id}")
def download(job_id: str, request: Request) -> StreamingResponse:
    with JOBS_LOCK:
        job = JOBS.get(job_id)
    if not job or not job.output_path or not job.output_path.exists():
        raise HTTPException(404, "not found")
    safe = re.sub(r"[^\w.\-\u4e00-\u9fff]+", "_", job.output_name or "clean.mp4")
    return _range_response(
        request, job.output_path,
        MEDIA_TYPES.get(job.output_path.suffix.lower(), "video/mp4"), filename=safe,
    )


app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


def main() -> None:
    import uvicorn

    if not CLI.exists():
        print("未找到 remove-ai-watermarks CLI，请先部署上游代码（把本 webui 放进上游克隆目录内运行，或执行 deploy.ps1）")
        print(f"  期望路径: {CLI}")
        sys.exit(1)
    print("Remove AI Watermarks WebUI  (Ctrl+C 退出)")
    print(f"  本机访问:   http://127.0.0.1:{PORT}/")
    for ip in lan_ips():
        print(f"  局域网访问: http://{ip}:{PORT}/")
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="warning")


if __name__ == "__main__":
    main()