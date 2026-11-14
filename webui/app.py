"""WebUI for remove-ai-watermarks: upload video, strip visible AI marks + metadata, preview & download.

Run (cwd = webui/):
    ..\\.venv\\Scripts\\python.exe app.py
"""

from __future__ import annotations

import contextlib
import hashlib
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
from email.utils import formatdate
from pathlib import Path
from urllib.parse import quote

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, Response, StreamingResponse
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
    output_nomusic_path: Path | None = None   # 无背景音乐版本
    output_nomusic_name: str | None = None
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
            "output_nomusic_name": self.output_nomusic_name,
            "media_url_nomusic": (f"/media/{self.id}/{self.output_nomusic_name}"
                                   if self.output_nomusic_path else None),
            "download_url_nomusic": (f"/download/{self.id}/nomusic"
                                     if self.output_nomusic_path else None),
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
            import jimeng_detect

            regions: list = []
            if job.preset == "jimeng":
                w, h, _ = erase_engine.probe_geometry(job.input_path)
                # 即梦的 logo 会在四个角之间**分时轮换**（实测每 4 秒换一个角），
                # 而且 @昵称 长度因人而异。所以不能写死一个区域，必须逐帧定位。
                job.log.append("$ 正在自动定位即梦水印（四角 + 昵称长度自适应）...")
                timeline = jimeng_detect.detect_timeline(job.input_path)
                if timeline:
                    for s, e, b in timeline:
                        job.log.append(
                            f"  [detect] 帧 {s}-{e - 1} 区域 {b} ({b[2]}x{b[3]})"
                        )
                    # 逐帧区域表必须覆盖全部可解码帧：检测时段只到最后一块
                    # 有命中的地方，结尾未覆盖的帧（黑场/解码帧数差）也要有
                    # 行可索引，否则擦除循环会越界。
                    frame_count = erase_engine.probe_frame_count(job.input_path)
                    regions = jimeng_detect.regions_by_frame(
                        timeline, w, h, frame_count=frame_count
                    )
                else:
                    # 检测失败时退化为固定四角全覆盖，宁可多擦也不漏水印
                    job.log.append("  [detect] 未定位到 logo，回退为固定四角全覆盖")
                    regions = erase_engine.jimeng_regions(w, h)
            if not regions:
                raise RuntimeError("未指定可用的擦除区域")
            job.log.append(
                f"$ erase preset={job.preset or '-'} backend={job.backend} "
                f"regions={'逐帧定位' if regions and isinstance(regions[0], list) else '静态'}"
            )
            try:
                erased = erase_engine.erase_video_regions(
                    job.input_path, out_path, regions,
                    backend=job.backend, strip_metadata=True, temporal_consistency=job.temporal,
                )
            except Exception as gpu_err:  # noqa: BLE001
                # GPU 在真实推理中途失败（如 Volta 缺 kernel、显存抖动）时，
                # 禁用 CUDA 后用 CPU 完整重跑一次，保证用户总能拿到结果。
                job.log.append(f"  [gpu] 加速推理失败，回退 CPU 重试: {gpu_err}")
                os.environ["WEBUI_ONNX_DEVICE"] = "cpu"
                try:
                    import gpu_accel

                    gpu_accel.reset()
                except Exception:  # noqa: BLE001
                    pass
                shutil.rmtree(OUTPUT_DIR / job.id, ignore_errors=True)
                out_dir.mkdir(parents=True, exist_ok=True)
                erased = erase_engine.erase_video_regions(
                    job.input_path, out_path, regions,
                    backend=job.backend, strip_metadata=True, temporal_consistency=job.temporal,
                )
                job.log.append("  [gpu] 已回退 CPU 完成")
            job.log.append(f"  [webui] 区域擦除完成: {erased} 帧")
            with JOBS_LOCK:
                if job.status != "cancelled":
                    job.status = "done"
                    # 统一命名 zc-日期-序号：先改名再做剪尾（剪尾作用于新文件）
                    daily = daily_output_name(ext)
                    final_path = out_dir / daily
                    out_path.rename(final_path)
                    job.output_path = final_path
                    job.output_name = daily
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
            note = trim_video_end(job.output_path, job.trim_end, job.log)
            if note:
                with JOBS_LOCK:
                    job.message += f"；{note}"
        if job.status == "done" and job.output_path:
            nomusic = make_no_music(job.output_path, job.log)
            if nomusic:
                job.output_nomusic_path = nomusic
                job.output_nomusic_name = nomusic.name
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
                # 统一命名 zc-日期-序号（先改名再做剪尾）
                daily = daily_output_name(ext)
                final_path = out_dir / daily
                out_path.rename(final_path)
                job.output_path = final_path
                job.output_name = daily
        if job.status == "running" and job.output_path and job.trim_end > 0:
            trim_note = trim_video_end(job.output_path, job.trim_end, job.log)
        else:
            trim_note = ""
        if job.status == "running" and job.output_path:
            nomusic = make_no_music(job.output_path, job.log)
            if nomusic:
                job.output_nomusic_path = nomusic
                job.output_nomusic_name = nomusic.name
        with JOBS_LOCK:
            if job.status == "cancelled":
                pass
            elif code == 0 and job.output_path and job.output_path.exists():
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


def daily_output_name(ext: str) -> str:
    """去水印输出统一命名：zc-2026-11-14-001.mp4。

    ``zc`` 固定前缀 + 当天日期 + 当天第几个视频（001 起）。序号取输出
    目录里当天已有文件的最大值 +1：清理任务删除的是整个任务目录，
    因此编号不会被已清理的文件重新占用；任务串行执行，无并发竞争。
    """
    date = time.strftime("%Y-%m-%d")
    max_seq = 0
    for p in OUTPUT_DIR.glob(f"*/zc-{date}-*"):
        try:
            seq = int(p.stem.rsplit("-", 1)[-1])
        except ValueError:
            continue
        max_seq = max(max_seq, seq)
    return f"zc-{date}-{max_seq + 1:03d}{ext}"


def make_no_music(path: Path, log: deque[str] | None = None) -> Path | None:
    """生成无背景音乐版本：视频轨流复制（-c:v copy，秒级）、去掉音频轨。

    文件名加 ``-无背景音乐`` 后缀，后缀本身就是标记，避免与原声版混淆。
    生成失败只记日志、不影响主流程（主文件照常交付）。
    """
    try:
        ffmpeg = _video_tool("ffmpeg")
        out = path.with_name(path.stem + "-无背景音乐" + path.suffix)
        if out.exists():
            return out
        proc = subprocess.run(
            [ffmpeg, "-y", "-v", "error", "-i", str(path), "-an",
             "-c:v", "copy", str(out)],
            capture_output=True, text=True, timeout=600,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
        if proc.returncode == 0 and out.exists() and out.stat().st_size > 0:
            if log is not None:
                log.append(f"  [webui] 已生成无背景音乐版本: {out.name}")
            return out
        if log is not None:
            log.append(f"  [webui] 无背景音乐版本生成失败: {(proc.stderr or '').strip()[:200]}")
    except Exception as e:  # noqa: BLE001
        if log is not None:
            log.append(f"  [webui] 无背景音乐版本生成失败: {e}")
    return None


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


# ---------------------------------------------------------------------------
# 访问口令门禁：经 frp/Caddy 穿透的远程来访者需输入口令，局域网与本机直连
# 不需要。判定「远程」的依据（满足任一即视为远程）：
#   1. 请求带反向代理注入的转发头（Caddy 对每个反代请求都会加
#      X-Forwarded-For；走 Cloudflare 时还有 CF-Connecting-IP）；
#   2. Host 头不是本机地址——兜住「绕过 Caddy 直连 frp 公网端口」的
#      访客：这种请求没有转发头，但 Host 是公网域名或服务器地址。
# 口令来源：环境变量 WEBUI_PASSCODE，或本文件同目录的 gate_passcode.txt
# （部署机本地文件，不入库）。两处都没有 = 不启用门禁（行为与旧版一致）。
_GATE_COOKIE = "shuiyin_gate"
_GATE_SALT = "shuiyin-gate-v1"


def _gate_passcode() -> str:
    code = os.environ.get("WEBUI_PASSCODE", "").strip()
    if code:
        return code
    try:
        return (Path(__file__).resolve().parent / "gate_passcode.txt").read_text(
            encoding="utf-8").strip()
    except OSError:
        return ""


def _gate_token() -> str:
    return hashlib.sha256((_GATE_SALT + _gate_passcode()).encode()).hexdigest()


def _is_local_request(request: Request) -> bool:
    if (request.headers.get("x-forwarded-for")
            or request.headers.get("x-real-ip")
            or request.headers.get("cf-connecting-ip")):
        return False
    host = (request.headers.get("host") or "").split(":")[0].strip("[]").lower()
    if host in ("", "127.0.0.1", "localhost", "::1"):
        return True
    return host in {ip.lower() for ip in lan_ips()}


_GATE_PAGE = """<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>访问口令 · AI 视频水印去除</title>
<style>
  body {{ margin:0; min-height:100vh; display:flex; align-items:center;
         justify-content:center; background:#0f1420; color:#e8ecf4;
         font-family:"Microsoft YaHei",system-ui,sans-serif; }}
  .card {{ background:#171e2e; border:1px solid #2a3450; border-radius:14px;
          padding:36px 40px; width:min(360px,90vw); text-align:center; }}
  h1 {{ font-size:18px; margin:0 0 8px; }}
  p {{ font-size:13px; color:#9aa7bd; margin:0 0 20px; }}
  input {{ width:100%; box-sizing:border-box; padding:11px 13px; font-size:16px;
          border-radius:8px; border:1px solid #2a3450; background:#0f1420;
          color:#e8ecf4; text-align:center; letter-spacing:2px; }}
  button {{ width:100%; margin-top:14px; padding:11px; font-size:15px;
           border:0; border-radius:8px; background:#3b82f6; color:#fff;
           cursor:pointer; }}
  button:hover {{ background:#2f6fe0; }}
  .err {{ color:#f87171; font-size:13px; margin-top:12px; min-height:18px; }}
</style></head><body><div class="card">
<h1>🔒 需要访问口令</h1>
<p>远程访问受口令保护，请输入口令继续</p>
<form method="post" action="/api/gate">
  <input name="passcode" type="password" inputmode="numeric" autocomplete="off"
         placeholder="访问口令" autofocus>
  <button type="submit">进入</button>
  <div class="err">{error}</div>
</form>
</div></body></html>"""


@app.middleware("http")
async def gate_middleware(request: Request, call_next):
    if _gate_passcode():
        path = request.url.path
        if path not in ("/@gate", "/api/gate") and not _is_local_request(request):
            if request.cookies.get(_GATE_COOKIE) != _gate_token():
                return RedirectResponse("/@gate", status_code=302)
    return await call_next(request)


@app.get("/@gate", response_class=HTMLResponse)
def gate_page(error: str = "") -> HTMLResponse:
    return HTMLResponse(_GATE_PAGE.format(error="口令不正确，请重试" if error else ""))


@app.post("/api/gate")
def gate_submit(passcode: str = Form("")) -> RedirectResponse:
    if passcode.strip() and passcode.strip() == _gate_passcode():
        resp = RedirectResponse("/", status_code=303)
        resp.set_cookie(_GATE_COOKIE, _gate_token(), httponly=True,
                        samesite="lax", max_age=30 * 24 * 3600)
        return resp
    return RedirectResponse("/@gate?e=1", status_code=303)


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    return HTMLResponse((STATIC_DIR / "index.html").read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# 创作者中心（/ai/index.html）：面向展示的独立视图。
# 只输出展示必需的字段（文件名/时间/播放地址/虚拟积分），**不包含任何
# 去水印相关元数据**（模式细节/预设/后端/处理消息一律不出口）。
# 该路径与全站一样受口令门禁保护。

def _ai_credits(job_id: str) -> int:
    """虚拟积分展示值：由任务 id 稳定映射到 66~200（同视频恒定）。"""
    try:
        h = int(job_id[:12], 16)
    except ValueError:
        h = abs(hash(job_id))
    return 66 + h % 135


@app.get("/api/ai/videos")
def ai_videos() -> list[dict]:
    with JOBS_LOCK:
        done = [j for j in JOBS.values()
                if j.status == "done" and j.output_path and j.output_path.exists()]
        out = []
        for j in sorted(done, key=lambda x: x.created, reverse=True):
            item = {
                "id": j.id,
                "name": j.output_name or j.name,
                "media_url": f"/media/{j.id}/{j.output_name}",
                "download_url": f"/download/{j.id}",
                "created": j.created,
                "credits": _ai_credits(j.id),
            }
            if j.output_nomusic_path and j.output_nomusic_path.exists():
                item["name_nomusic"] = j.output_nomusic_name
                item["media_url_nomusic"] = f"/media/{j.id}/{j.output_nomusic_name}"
                item["download_url_nomusic"] = f"/download/{j.id}/nomusic"
            out.append(item)
    return out


@app.get("/ai/index.html", response_class=HTMLResponse)
@app.get("/ai/", response_class=HTMLResponse)
@app.get("/ai", response_class=HTMLResponse)
def ai_index() -> HTMLResponse:
    return HTMLResponse((STATIC_DIR / "ai" / "index.html").read_text(encoding="utf-8"))


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
    """Stream a file honoring HTTP Range so LAN browsers can seek/preview large videos.

    Content-Length is mandatory here: Chrome/Safari use it (together with the
    media duration) to lay out the seek bar and to decide how much to buffer
    ahead. Omitting it makes some browsers stop after the first few seconds.
    """
    size = path.stat().st_size
    start, end = 0, size - 1
    status_code = 200
    rng = request.headers.get("range")
    if rng:
        m = re.match(r"bytes=(\d*)-(\d*)", rng.strip())
        if m and (m.group(1) or m.group(2)):
            if not m.group(1) and m.group(2):
                # suffix range: last N bytes
                start = max(0, size - int(m.group(2)))
            else:
                start = int(m.group(1))
                if m.group(2):
                    end = min(int(m.group(2)), size - 1)
            if start >= size or start > end:
                return Response(
                    status_code=416,
                    headers={"Content-Range": f"bytes */{size}", "Accept-Ranges": "bytes"},
                )
            status_code = 206
    length = end - start + 1
    headers = {
        "Content-Length": str(length),
        "Content-Range": f"bytes {start}-{end}/{size}",
        "Accept-Ranges": "bytes",
        "Cache-Control": "no-store",
        "Last-Modified": formatdate(path.stat().st_mtime, usegmt=True),
    }
    if status_code == 200:
        # 非分段请求不需要 Content-Range，带上部分客户端会误解为分段响应
        headers.pop("Content-Range", None)
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
        # 无背景音乐版本走同一接口：名字与已知变体严格相等才放行（防目录穿越）
        if (job and job.output_nomusic_path and job.output_nomusic_path.exists()
                and name == job.output_nomusic_name):
            return _range_response(
                request, job.output_nomusic_path,
                MEDIA_TYPES.get(job.output_nomusic_path.suffix.lower(), "video/mp4"),
            )
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


@app.get("/download/{job_id}/nomusic")
def download_nomusic(job_id: str, request: Request) -> StreamingResponse:
    """无背景音乐版本下载（文件名带 -无背景音乐 后缀，天然区分）。"""
    with JOBS_LOCK:
        job = JOBS.get(job_id)
    if (not job or not job.output_nomusic_path
            or not job.output_nomusic_path.exists()):
        raise HTTPException(404, "not found")
    safe = re.sub(r"[^\w.\-\u4e00-\u9fff]+", "_",
                  job.output_nomusic_name or "nomusic.mp4")
    return _range_response(
        request, job.output_nomusic_path,
        MEDIA_TYPES.get(job.output_nomusic_path.suffix.lower(), "video/mp4"),
        filename=safe,
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
