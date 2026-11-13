"""手动区域擦除引擎：复用上游 remove-ai-watermarks 的逐帧填充与原子编码机制。

背景：上游刻意不把所有厂商的视频水印注册进检测表（即梦 CN 水印在
docs/module-internals.md 中被记录为 "PARKED"——模板匹配在视频上分不开正负样本，
左上 "AI生成" 药丸的误报模式是静态纹理角，时间递归仲裁器无法拦截）。本模块
不触碰上游检测注册表，而是把用户指定的区域交给上游同款逐帧填充后端
（cv2 / MI-GAN / LaMa）与同一条原子编码管线（含色彩档案、VFR 时间戳、
HDR 拒绝、音频流复制、元数据剥离），质量与上游已注册水印的移除路径一致。
"""

from __future__ import annotations

from itertools import pairwise
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from dataclasses import replace
from fractions import Fraction

from remove_ai_watermarks.video_decode import open_video_capture
from remove_ai_watermarks.video_encoding import (
    abort_raw_video_encoder,
    finish_raw_video_encoder,
    mux_encoded_video,
    probe_video_encode_profile,
    probe_video_timestamps,
    raw_video_command,
    staged_video_output,
    start_raw_video_encoder,
)
from remove_ai_watermarks.video_temporal import stabilize_filled_frame
from remove_ai_watermarks.video_visible import (
    _HDR_TRANSFERS,
    _TimestampedNutWriter,
    _mask_for_region,
    _timestamp_time_base,
)
from remove_ai_watermarks.watermark_registry import fill, resolve_backend

Region = tuple[int, int, int, int]

# 即梦 / Dreamina（字节跳动）视频水印预设区域，按 1280x720 实测标定，以帧比例表达。
# 保留作为「检测失败时的保守兜底」：当自动定位拿不到结果时，用这几个角
# 的固定区域全部擦一遍，避免漏掉水印（宁可多擦，也不能留水印）。
JIMENG_REGIONS_RELATIVE: tuple[tuple[float, float, float, float], ...] = (
    (0.0047, 0.0111, 0.0625, 0.0556),   # 左上 AI生成 药丸（整片常驻）
    (0.8242, 0.7333, 0.1328, 0.1500),   # 右下 即梦 logo + 作者名
    (0.0200, 0.7300, 0.1900, 0.1600),   # 左下（分时轮换位）
    (0.8200, 0.0500, 0.1800, 0.1500),   # 右上（分时轮换位）
)


def jimeng_regions(width: int, height: int) -> list[Region]:
    """Scale the measured Jimeng/Dreamina preset regions to a frame size."""
    return [
        (round(width * x), round(height * y), round(width * w), round(height * h))
        for x, y, w, h in JIMENG_REGIONS_RELATIVE
    ]


def detect_jimeng(source: Path) -> bool:
    """Whether container provenance says this is a Jimeng/Dreamina export.

    Mirrors the upstream video provenance notion: any marker naming ByteDance's
    Dreamina/Jimeng surface. Used only to pick default regions for the UI, never
    to authorize a removal on its own.
    """
    try:
        from remove_ai_watermarks.video import inspect_video_metadata

        report = inspect_video_metadata(source)
        markers = report.markers or {}
    except Exception:  # noqa: BLE001
        return False
    identity = " ".join(str(v) for k, v in markers.items() if k != "aigc_producer").lower()
    producer = str(markers.get("aigc_producer", "")).lower()
    return any(token in identity for token in ("dreamina", "jimeng")) or "dreamina" in producer



def probe_geometry(source: Path) -> tuple[int, int, float]:
    """Return (width, height, fps) without scanning watermark detectors."""
    capture = open_video_capture(source)
    if not capture.isOpened():
        raise RuntimeError(f"OpenCV could not decode video: {source}")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    capture.release()
    if width <= 0 or height <= 0 or fps <= 0:
        raise RuntimeError(f"Video has invalid stream geometry: {source}")
    return width, height, fps


def probe_frame_count(source: Path) -> int:
    """Number of decodable frames, counted the same way erase does.

    ``CAP_PROP_FRAME_COUNT`` 按容器元数据估算，与真实解码结果可能差几帧
    （实测 -c copy 剪出的 4s 片段元数据 96 帧、实际解码 98 帧）。
    逐帧读一遍才能拿到与擦除循环一致的数量。
    """
    capture = open_video_capture(source)
    if not capture.isOpened():
        raise RuntimeError(f"OpenCV could not decode video: {source}")
    count = 0
    try:
        while True:
            ok, _frame = capture.read()
            if not ok:
                break
            count += 1
    finally:
        capture.release()
    return count


def resolve_regions(
    source: Path,
    regions: list[Region],
    *,
    keep_margin: int = 1,
) -> list[Region]:
    """Clamp regions into the frame; drop anything degenerate after clamping."""
    width, height, _ = probe_geometry(source)
    resolved: list[Region] = []
    for x, y, w, h in regions:
        x = max(0, min(int(x), width - keep_margin))
        y = max(0, min(int(y), height - keep_margin))
        w = max(0, min(int(w), width - x - keep_margin))
        h = max(0, min(int(h), height - y - keep_margin))
        if w >= 8 and h >= 8:
            resolved.append((x, y, w, h))
    return resolved


def _variable_timestamps(timestamps: tuple[float, ...], time_base: Fraction) -> bool:
    if len(timestamps) < 3:
        return False
    ticks = tuple(round(t / float(time_base)) for t in timestamps)
    intervals = tuple(cur - prev for prev, cur in pairwise(ticks))
    return all(i > 0 for i in intervals) and len(set(intervals)) > 1


def erase_video_regions(
    source: Path,
    output: Path,
    regions: list[Region] | list[list[Region]],
    *,
    backend: str = "cv2",
    strip_metadata: bool = True,
    temporal_consistency: bool = True,
    padding_fraction: float = 0.06,
) -> int:
    """Fill the given regions on every frame and atomically encode the result.

    Mirrors upstream ``encode_clean_video`` but takes static user regions
    (multiple per frame) instead of per-mark detector output. Returns the
    number of erased frames.

    ``regions`` accepts two shapes:

    * ``list[Region]`` — one fixed set of boxes applied to every frame
      (manual boxes, single-position presets).
    * ``list[list[Region]]`` — a **per-frame** region list, used by the
      Jimeng preset whose logo hops between corners over time.
    """
    per_frame: bool = isinstance(regions, list) and len(regions) > 0 and \
        isinstance(regions[0], list)
    if not regions:
        return 0
    resolved_backend = resolve_backend(backend)  # type: ignore[arg-type]

    # Pass 1: geometry + timestamps (same strategy as upstream _scan_video_detectors).
    width, height, fps = probe_geometry(source)
    capture = open_video_capture(source)
    if not capture.isOpened():
        raise RuntimeError(f"OpenCV could not decode video: {source}")
    timestamps: list[float] = []
    try:
        while True:
            ok, _frame = capture.read()
            if not ok:
                break
            timestamps.append(float(capture.get(cv2.CAP_PROP_POS_MSEC)) / 1000)
    finally:
        capture.release()
    frame_count = len(timestamps)
    if frame_count == 0:
        raise RuntimeError(f"Video contains no decodable frames: {source}")
    shared = probe_video_timestamps(source)
    timestamp_tuple = tuple(shared) if len(shared) == frame_count else tuple(timestamps)

    with staged_video_output(output) as (encoded_video, temporary_output):
        profile = probe_video_encode_profile(source)
        if (profile.component_depth or 0) > 8 or profile.color_transfer in _HDR_TRANSFERS:
            source_format = profile.source_pixel_format or "unknown pixel format"
            raise RuntimeError(
                "Visible video removal currently supports SDR 8-bit input only; "
                f"refusing to silently reduce {source_format} / "
                f"{profile.color_transfer or 'unknown transfer'} to 8-bit SDR"
            )
        time_base = _timestamp_time_base(profile.time_base)
        preserve_start_offset = profile.start_pts not in (None, 0)
        timestamped_input = preserve_start_offset or _variable_timestamps(timestamp_tuple, time_base)
        if timestamped_input and profile.time_base is None:
            profile = replace(profile, time_base=f"{time_base.numerator}/{time_base.denominator}")

        process = start_raw_video_encoder(
            raw_video_command(
                encoded_video,
                width=width,
                height=height,
                fps=fps,
                crf=14,
                profile=profile,
                timestamped_input=timestamped_input,
                copy_input_timestamps=preserve_start_offset,
            )
        )
        frame_pipe = process.stdin
        writer: _TimestampedNutWriter | None = None
        previous_source: np.ndarray | None = None
        previous_cleaned: np.ndarray | None = None
        previous_mask: np.ndarray | None = None
        erased = 0
        capture = open_video_capture(source)
        try:
            if timestamped_input:
                writer = _TimestampedNutWriter(
                    frame_pipe,
                    width=width,
                    height=height,
                    time_base=time_base,
                    start_pts=profile.start_pts or 0,
                )
            for frame_index in range(frame_count):
                ok, frame = capture.read()
                if not ok:
                    raise RuntimeError(f"Video ended while reading frame {frame_index}: {source}")
                source_frame = frame
                if per_frame:
                    # 解码帧数可能多于检测时段覆盖的帧数（如结尾黑场、两次解码
                    # 帧数不一致）。越界的帧没有已定位的水印，直接原样写出。
                    frame_regions = regions[frame_index] \
                        if frame_index < len(regions) else []
                else:
                    frame_regions = regions
                combined_mask: np.ndarray | None = None
                for region in frame_regions:  # type: ignore[union-attr]
                    mask = _mask_for_region(
                        frame,
                        region,
                        padding_fraction=padding_fraction,
                        mask_style="box",
                    )
                    combined_mask = mask if combined_mask is None else np.maximum(combined_mask, mask)
                # 所有区域一次性填充：一次前向代替每区域一次，MI-GAN/LaMa 快近一倍
                if combined_mask is not None:
                    frame = fill(frame, combined_mask, backend=resolved_backend)  # type: ignore[arg-type]
                    if (
                        temporal_consistency
                        and previous_source is not None
                        and previous_cleaned is not None
                        and previous_mask is not None
                    ):
                        frame = stabilize_filled_frame(
                            previous_source,
                            previous_cleaned,
                            previous_mask,
                            source_frame,
                            frame,
                            combined_mask,
                            copy=False,
                        )
                    erased += 1
                if writer is None:
                    frame_pipe.write(frame.tobytes())  # type: ignore[union-attr]
                else:
                    writer.write(frame, timestamp_tuple[frame_index])  # type: ignore[arg-type]
                previous_source = source_frame
                previous_cleaned = frame
                previous_mask = combined_mask
            if writer is not None:
                writer.close()
                writer = None
            finish_raw_video_encoder(process, encoded_video, operation="webui region erase")
            mux_encoded_video(
                encoded_video,
                source,
                temporary_output,
                strip_metadata=strip_metadata,
                copy_input_timestamps=preserve_start_offset,
            )
        except Exception:
            if writer is not None:
                writer.close()
            abort_raw_video_encoder(process)
            raise
        finally:
            capture.release()

    return erased
