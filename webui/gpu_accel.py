"""GPU 加速补丁：让上游 MI-GAN / LaMa 的 ONNX 推理优先用 CUDA，失败自动回退 CPU。

为什么需要补丁而不是改上游：上游 `region_eraser._get_session()` 把 provider 写死为
``["CPUExecutionProvider"]``，并且上游代码属于 uv 管理的第三方依赖，直接修改会在
``uv sync`` 后被覆盖。这里在进程内替换 ``region_eraser._get_session``，保持上游
其余行为（模型下载、缓存、推理包装）完全不变。

GPU 不可用的情况全部静默降级到 CPU，绝不让整条处理链失败：
  * onnxruntime 未安装 GPU 版（无 CUDAExecutionProvider）
  * NVIDIA 驱动 / CUDA 运行时 / cuDNN 缺失
  * 显存不足（初始化或首次推理抛异常）
  * 用户显式要求 CPU（环境变量 WEBUI_ONNX_DEVICE=cpu）

另外补一个 Windows 专有的坑：onnxruntime-gpu 通过 pip 安装的 CUDA/cuDNN
DLL 位于 ``site-packages/nvidia/*/bin``，Windows 不搜索这些目录，必须先调用
``os.add_dll_directory`` 并把路径注入 PATH，否则 CUDA provider 加载会报
"Error 126 ... which is missing"。
"""

from __future__ import annotations

import logging
import os
import sys
import threading
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

_LOCK = threading.Lock()
_STATE: dict[str, Any] = {"patched": False, "device": "unknown", "reason": ""}

# 显存低于该值时拒绝使用 GPU（单位 MB）。MI-GAN 约 28MB 模型，LaMa 约 200MB，
# 但推理峰值远高于权重；留足余量避免 OOM 把整批任务拖垮。
_MIN_FREE_VRAM_MB = 1024


def _prepare_cuda_dll_search_path() -> list[str]:
    """Windows 下让 onnxruntime 能找到 pip 安装的 CUDA/cuDNN DLL。"""
    if sys.platform != "win32":
        return []
    site_packages = Path(sys.prefix) / "Lib" / "site-packages" / "nvidia"
    if not site_packages.is_dir():
        return []
    added: list[str] = []
    for bin_dir in site_packages.glob("*/bin"):
        if not bin_dir.is_dir():
            continue
        try:
            os.add_dll_directory(str(bin_dir))
        except (OSError, AttributeError):
            continue
        added.append(str(bin_dir))
    if added:
        os.environ["PATH"] = os.pathsep.join([*added, os.environ.get("PATH", "")])
    return added


def _gpu_free_memory_mb() -> int | None:
    """Best-effort free VRAM probe via nvidia-smi; None when unavailable."""
    try:
        import subprocess

        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        if result.returncode != 0 or not result.stdout.strip():
            return None
        return int(result.stdout.strip().splitlines()[0].strip())
    except Exception:  # noqa: BLE001
        return None


def _cuda_usable() -> tuple[bool, str]:
    """Decide whether CUDA inference should be attempted."""
    forced = os.environ.get("WEBUI_ONNX_DEVICE", "").strip().lower()
    if forced == "cpu":
        return False, "环境变量 WEBUI_ONNX_DEVICE=cpu"
    try:
        import onnxruntime as ort
    except Exception as e:  # noqa: BLE001
        return False, f"onnxruntime 不可用: {e}"
    available = ort.get_available_providers()
    if "CUDAExecutionProvider" not in available:
        return False, f"未安装 GPU 版 onnxruntime（providers={available}）"
    if forced == "cuda":
        return True, "环境变量强制 CUDA"
    free = _gpu_free_memory_mb()
    if free is not None and free < _MIN_FREE_VRAM_MB:
        return False, f"显存不足（空闲 {free}MB < {_MIN_FREE_VRAM_MB}MB）"
    return True, f"CUDA 可用（空闲显存 {free}MB）" if free is not None else "CUDA 可用"


def _make_session_factory(original: Any) -> Any:
    """Wrap upstream ``_get_session`` with GPU-first, CPU-fallback behavior."""
    import onnxruntime as ort

    def factory(name: str, repo_id: str, filename: str, label: str) -> Any:
        # 命中缓存直接返回（上游语义）
        cached = _STATE.get(f"session:{name}")
        if cached is not None:
            return cached

        usable, reason = _cuda_usable()
        model_path = None
        attempts: list[tuple[str, str]] = []
        if usable:
            attempts.append(("cuda", "CUDAExecutionProvider"))
        attempts.append(("cpu", "CPUExecutionProvider"))

        for device, provider in attempts:
            try:
                cached_any = _STATE.get(f"session:{name}:{device}")
                if cached_any is not None:
                    _STATE[f"session:{name}"] = cached_any
                    _STATE["device"] = device
                    return cached_any
                if model_path is None:
                    from huggingface_hub import hf_hub_download

                    model_path = hf_hub_download(repo_id=repo_id, filename=filename)
                options = ort.SessionOptions()
                if device == "cuda":
                    # 显存不足时自动把拿不到显存的节点退回 CPU，而不是整块失败
                    session = ort.InferenceSession(
                        model_path,
                        sess_options=options,
                        providers=[provider, "CPUExecutionProvider"],
                    )
                else:
                    session = ort.InferenceSession(
                        model_path, sess_options=options, providers=[provider],
                    )
                actual = session.get_providers()
                if device == "cuda" and "CUDAExecutionProvider" not in actual:
                    raise RuntimeError(f"CUDA provider 未生效（actual={actual}）")
                _STATE[f"session:{name}:{device}"] = session
                _STATE[f"session:{name}"] = session
                _STATE["device"] = device
                _STATE["reason"] = reason if device == "cuda" else f"{reason}；已回退 CPU"
                log.info("[gpu] %s 使用 %s 推理（%s）", label, actual[0], _STATE["reason"])
                return session
            except Exception as e:  # noqa: BLE001
                log.warning("[gpu] %s 在 %s 上初始化失败，尝试回退: %s", label, device, e)
                continue
        # 全部失败时交回上游原始实现，保证异常信息与上游一致
        return original(name, repo_id, filename, label)

    return factory


def install() -> dict[str, Any]:
    """Idempotently patch the upstream ONNX session factory."""
    with _LOCK:
        if _STATE["patched"]:
            return dict(_STATE)
        dll_dirs = _prepare_cuda_dll_search_path()
        try:
            from remove_ai_watermarks import region_eraser

            original = region_eraser._get_session  # noqa: SLF001
            region_eraser._get_session = _make_session_factory(original)  # noqa: SLF001
            _STATE["patched"] = True
            usable, reason = _cuda_usable()
            _STATE["device"] = "cuda" if usable else "cpu"
            _STATE["reason"] = reason
            _STATE["dll_dirs"] = len(dll_dirs)
            log.info("[gpu] 补丁已安装: device=%s (%s), dll_dirs=%s", _STATE["device"], reason, len(dll_dirs))
        except Exception as e:  # noqa: BLE001
            _STATE["reason"] = f"补丁安装失败: {e}"
            log.warning("[gpu] 补丁安装失败: %s", e)
        return dict(_STATE)


def reset() -> None:
    """Drop cached sessions so the next call re-resolves the device (used for CPU fallback)."""
    with _LOCK:
        for key in [k for k in _STATE if str(k).startswith("session:")]:
            _STATE.pop(key, None)
        _STATE["device"] = "cpu"
        _STATE["reason"] = "运行中检测到 GPU 推理失败，已回退 CPU"


def status() -> dict[str, Any]:
    """Report the resolved inference device for the UI."""
    info = dict(_STATE)
    if not info.get("patched"):
        info = install()
    free = _gpu_free_memory_mb()
    return {
        "patched": info.get("patched", False),
        "device": info.get("device", "unknown"),
        "reason": info.get("reason", ""),
        "free_vram_mb": free,
    }