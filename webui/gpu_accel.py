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


def _cuda_selftest() -> tuple[bool, str]:
    """真正跑一次最小 GPU 推理，验证 CUDA 不是「名义可用」。

    为什么必须有这一步：``get_available_providers()`` 里出现
    ``CUDAExecutionProvider`` 只代表这个 wheel 编译时带了 CUDA 支持，
    **不代表你的显卡能执行**。典型反例：V100(sm_70) 配 onnxruntime-gpu
    1.20+，provider 加载成功、会话创建成功，但一执行算子就抛
    ``cudaErrorNoKernelImageForDevice``（官方已移除 sm_70 kernel）。

    实现上用 MI-GAN 的真实 ONNX 模型跑一张极小图（32×32）。不额外引入
    ``onnx`` 依赖，且这个模型本来就要下载、必然可用。开销约几十毫秒。

    注意：内部会先确保 CUDA DLL 搜索路径已注入。独立进程（如
    ``check-gpu.bat``）直接调用本函数时，若没先走 ``install()``，
    Windows 会找不到 ``cublasLt64_*.dll``，从而误判为不可用。
    """
    # 独立进程调用时 DLL 路径可能还没注入，这里补上（幂等）
    _prepare_cuda_dll_search_path()

    try:
        import numpy as np
        import onnxruntime as ort
    except Exception as e:  # noqa: BLE001
        return False, f"onnxruntime/numpy 不可用: {e}"

    try:
        from huggingface_hub import hf_hub_download

        model_path = hf_hub_download(repo_id="andraniksargsyan/migan", filename="migan.onnx")
    except Exception as e:  # noqa: BLE001
        # 模型拿不到时无法自检；此时不应阻止使用 GPU（模型下载是另一回事）
        return True, f"CUDA provider 存在（未自检，模型不可达: {str(e)[:60]}）"

    try:
        session = ort.InferenceSession(
            model_path, providers=["CUDAExecutionProvider"],
        )
        actual = session.get_providers()
        if "CUDAExecutionProvider" not in actual:
            return False, f"CUDA provider 未生效（actual={actual}）"
        # MI-GAN 输入：image (1,3,H,W) uint8 / mask (1,1,H,W) uint8（0=hole）
        size = 32
        session.run(None, {
            "image": np.zeros((1, 3, size, size), np.uint8),
            "mask": np.zeros((1, 1, size, size), np.uint8),
        })
        return True, "CUDA 自检通过"
    except Exception as e:  # noqa: BLE001
        msg = str(e)
        if "NoKernelImageForDevice" in msg or "no kernel image" in msg:
            return False, (
                "当前显卡架构不被此版本 onnxruntime-gpu 支持"
                "（V100/Volta 需 onnxruntime-gpu==1.19.2）"
            )
        if "CUDNN_STATUS_EXECUTION_FAILED" in msg:
            return False, (
                "cuDNN 不支持当前显卡架构"
                "（V100/Volta 需 nvidia-cudnn-cu12==9.10.2.21）"
            )
        return False, f"CUDA 自检失败: {msg[:160]}"


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

    # 关键：只有 provider 存在还不够，必须真正跑一次算子。
    # 否则会在处理几十帧后才因 NoKernelImageForDevice 崩溃，再整段回退重跑，
    # 既浪费几分钟，也让 /api/info 长时间谎报「GPU 加速」。
    ok, why = _cuda_selftest()
    if not ok:
        return False, why
    return True, (f"CUDA 可用（空闲显存 {free}MB，自检通过）"
                  if free is not None else "CUDA 可用（自检通过）")


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
