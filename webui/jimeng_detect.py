"""即梦 / Dreamina 视频水印的自动定位（时间感知）。

为什么需要这个模块
------------------
即梦视频里有两类叠加物：

1. **"AI生成" 药丸**：固定在左上角，整片存在（中国 AIGC 标识要求）。
2. **"即梦" logo + @作者昵称**：位置是**分时轮换**的——实测一支 16 秒视频里
   0-4s 在右下、4-8s 在左下、8-12s 在右上。只写死一个角必然漏擦。

而且 **@昵称长度因人而异**（@娥酱 / @张三 / @李思思思思……），所以擦除框的
宽度**不能写死**，必须由检测到的文字实际范围决定。

检测原理
--------
"即梦"两个字是固定的白色字形，横向相邻、高度相近；其下方一行是 @昵称，
字形更小。这种"规整的两行文字排版"在自然画面里极罕见，因此：

  1. 用**时间中位数**把运动背景平均掉（水印是静态叠加，会留在中位数图里）；
  2. 在白/高亮掩码上找**高度相近、横向相邻的两个字形**（= 即梦 wordmark）；
  3. 再向下找**昵称行**，把它的实际右边界纳入 bbox（← 昵称长度自适应）；
  4. 左边界用**时间持久性**收敛到渐变箭头的真实起点（箭头是彩色，不在白色掩码里）；
  5. 逐块（默认 1 秒）判定水印在哪个角，再做时间平滑、合并成时段。

产出 ``[(起始帧, 结束帧, 区域), ...]``，交给擦除引擎按帧应用。
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

from remove_ai_watermarks.video_decode import open_video_capture

# 四角搜索窗（相对帧宽高的比例）。窗口要足够宽以容纳任意长度的昵称
# （@娥酱 很短，@李思思思思 可能比 logo 本身还宽）；同时不能大到把
# 画面主体吃进来。55% 宽 × 34% 高在实测里既有余量又不误报。
WINDOW_W_FRAC = 0.55
WINDOW_H_FRAC = 0.34

CORNERS: dict[str, tuple[float, float]] = {
    "TL": (0.0, 0.0),
    "TR": (1.0 - WINDOW_W_FRAC, 0.0),
    "BL": (0.0, 1.0 - WINDOW_H_FRAC),
    "BR": (1.0 - WINDOW_W_FRAC, 1.0 - WINDOW_H_FRAC),
}

# "AI生成" 药丸在左上角的固定位置（相对坐标，1280x720 实测标定）。
# 它是整片常驻的合规标识，不需要随时间变化。
PILL_REL: tuple[float, float, float, float] = (0.0047, 0.0111, 0.0625, 0.0556)

# 白/高亮判据（HSV）。水印是半透明白字，阈值放宽一点以容纳不同背景亮度。
_WHITE_V = 178
_WHITE_S = 90


@dataclass(frozen=True)
class LogoHit:
    """One detected logo placement inside a corner window (window-relative)."""

    corner: str
    bbox: tuple[int, int, int, int]  # x, y, w, h
    wordmark: tuple[int, int, int, int]
    nickname: tuple[int, int, int, int] | None
    confidence: float


def _white_mask(img: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    return (((hsv[:, :, 2] > _WHITE_V) & (hsv[:, :, 1] < _WHITE_S))
            .astype(np.uint8) * 255)


def _glyphs(img: np.ndarray) -> list[dict]:
    """White/highlight connected components that look like glyphs."""
    mask = cv2.morphologyEx(_white_mask(img), cv2.MORPH_CLOSE,
                            np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out: list[dict] = []
    for c in contours:
        x, y, bw, bh = cv2.boundingRect(c)
        area = float(cv2.contourArea(c))
        if area < 110 or bh < 11 or bw < 5:
            continue
        if bh > 170 or bw > 340:
            continue
        if area / max(1, bw * bh) < 0.13:      # 填充率过低 = 细长噪点
            continue
        out.append({"x": x, "y": y, "w": bw, "h": bh, "area": area})
    return out


def _find_wordmark(glyphs: list[dict]) -> list[dict] | None:
    """The "即梦" wordmark: all same-height glyphs on one text line.

    注意：不能只取「前两块」。中文笔画常被白色掩码切成多个连通域——
    实测左下角「即梦」被切成三块：``即`` 拆成两块（117 和 138），
    ``梦`` 单独一块（160，宽 35）。只取前两块会把 ``梦`` 整个漏掉，
    擦除框右边界就停在 156，而 ``梦`` 一直延伸到 196。

    因此这里收集**同一行、高度一致、横向连续**的全部字形，整体作为词标。
    """
    cands = [g for g in glyphs if g["h"] >= 18]
    if len(cands) < 2:
        return None
    cands.sort(key=lambda g: g["x"])
    best: tuple[float, list[dict]] | None = None
    for i in range(len(cands)):
        group = [cands[i]]
        for j in range(i + 1, len(cands)):
            ref, b = group[-1], cands[j]
            if abs((b["y"] + b["h"] / 2) - (ref["y"] + ref["h"] / 2)) > max(ref["h"], b["h"]) * 0.30:
                continue
            gap = b["x"] - (ref["x"] + ref["w"])
            if gap > max(ref["h"], b["h"]) * 1.2 or gap < -8:
                continue
            group.append(b)
        if len(group) < 2:
            continue
        gx1 = min(g["x"] for g in group)
        gx2 = max(g["x"] + g["w"] for g in group)
        if gx2 - gx1 < 26:
            continue
        hs = [g["h"] for g in group]
        # "即梦" 二字高度几乎一致（同一字号）；背景物体的高度比要乱得多。
        # 实测三个角的真实 wordmark 高度比为 1.00~1.08。
        if max(hs) / max(1, min(hs)) > 1.15:
            continue
        # 保留组内全部字形（可能 2 块，也可能因笔画断裂有 3~4 块）。
        # 只过滤掉高度明显偏离主体的（如混进来的标点/噪点）。
        h_med = float(np.median(hs))
        kept = [g for g in group if abs(g["h"] - h_med) / h_med <= 0.20]
        if len(kept) < 2:
            continue
        # 保留的字形必须彼此横向连续（防止把同行远处的东西并入）
        kept.sort(key=lambda g: g["x"])
        h_main = max(g["h"] for g in kept)
        reach = h_main * 3.0          # "即梦" 两字跨度约 1.5~3 倍字高
        anchor = kept[0]["x"]
        kept = [g for g in kept if g["x"] - anchor <= reach]
        if len(kept) < 2:
            continue
        # "即梦" 是画面上最大的文字：用字形高度做主排序依据，
        # 昵称行的小字（高度约一半）永远排不到前面。
        score = h_main * 1000.0 + sum(g["area"] for g in kept) / 100.0
        if best is None or score > best[0]:
            best = (score, kept)
    return best[1] if best else None


def _find_nickname(glyphs: list[dict], wordmark: list[dict]) -> list[dict]:
    """The '@nickname' line directly below the wordmark.

    Width comes from the actual glyphs, so any nickname length works
    (@娥酱, @张三, @李思思思思, ...).
    """
    wx1 = min(g["x"] for g in wordmark)
    wx2 = max(g["x"] + g["w"] for g in wordmark)
    wy1 = min(g["y"] for g in wordmark)
    wy2 = max(g["y"] + g["h"] for g in wordmark)
    wh = max(g["h"] for g in wordmark)
    picked: list[dict] = []
    for g in glyphs:
        if any(g is w for w in wordmark):
            continue
        # 必须落在 wordmark 行下方、且不太远
        if g["y"] < wy1 + wh * 0.55 or g["y"] > wy2 + wh * 1.5:
            continue
        if g["h"] < 10 or g["h"] > wh * 1.15:
            continue
        # 水平上要与 wordmark 有重叠或紧邻（昵称常与 logo 左对齐或向右溢出）
        if g["x"] + g["w"] < wx1 - wh * 1.5 or g["x"] > wx2 + wh * 9.0:
            continue
        picked.append(g)
    if not picked:
        return []
    # 只保留水平上连成一片的字形，避免把远处无关亮点算进来
    picked.sort(key=lambda g: g["x"])
    chain = [picked[0]]
    for g in picked[1:]:
        ref = chain[-1]
        gap = g["x"] - (ref["x"] + ref["w"])
        if gap > max(ref["h"], g["h"]) * 2.2:
            break
        chain.append(g)
    return chain


def _union(objs: list[dict]) -> tuple[int, int, int, int]:
    """Window-relative union bbox of glyph dicts."""
    x1 = min(o["x"] for o in objs)
    y1 = min(o["y"] for o in objs)
    x2 = max(o["x"] + o["w"] for o in objs)
    y2 = max(o["y"] + o["h"] for o in objs)
    return (x1, y1, x2 - x1, y2 - y1)


def detect_in_window(
    median_bgr: np.ndarray,
    corner: str,
) -> LogoHit | None:
    """Locate the logo (wordmark + nickname) inside one corner window.

    All returned coordinates are **window-relative**; the caller adds the window
    origin. Keeping this function pure (median image in, box out) makes it
    trivially testable on crops.
    """
    glyphs = _glyphs(median_bgr)
    wordmark = _find_wordmark(glyphs)
    if not wordmark:
        return None
    nickname = _find_nickname(glyphs, wordmark)
    wm = _union(wordmark)
    full = _union(wordmark + nickname) if nickname else wm
    wh = wm[3]
    # 退化检测：真正的水印框必须同时包含"即梦"二字行与下方昵称行
    # （高度约为字高的 2~3 倍）。只框住一行小字的，是把昵称行误当成了
    # wordmark —— 这种结果宁可丢弃，交给时间平滑用邻段补齐。
    if nickname:
        span_h = (full[1] + full[3]) - wm[1]
        if span_h < wh * 1.7:
            return None
    elif full[3] < wh * 1.0:
        return None

    # 左侧补上彩色渐变箭头。"即梦" logo 是同一份设计资产，箭头位置相对
    # 首字左边缘是固定的，只有缩放差异；三点实测比值 0.96 / 1.83 / 1.59，
    # 取 1.9 作为安全上限（宁可多擦几像素，也不能漏掉箭头）。
    x1 = max(0, wm[0] - int(round(wh * 1.9)))
    y1 = max(0, wm[1] - int(round(wh * 0.10)))
    x2 = full[0] + full[2]
    y2 = full[1] + full[3] + int(round(wh * 0.10))

    # 右/下各留一点安全余量：字形最右一列的抗锯齿像素可能低于白色阈值，
    # 导致检测框紧贴笔画边缘（实测左下角右侧只剩 2px，擦完仍有残影）。
    # 半个字高以内都是纯背景，多擦这一点不会有观感损失。
    pad = max(3, int(round(wh * 0.18)))
    x2 += pad
    y2 += pad

    bbox = (x1, y1, x2 - x1, y2 - y1)
    # 置信度：wordmark 字形数 + 昵称参与度 + 面积
    conf = float(len(wordmark) * 100 + (50.0 if nickname else 0.0)
                 + sum(g["area"] for g in wordmark + nickname) / 200.0)
    return LogoHit(
        corner=corner,
        bbox=bbox,
        wordmark=wm,
        nickname=_union(nickname) if nickname else None,
        confidence=conf,
    )


def _read_frames(source: Path) -> tuple[list[np.ndarray], float]:
    cap = open_video_capture(source)
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV could not decode video: {source}")
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 25.0
    frames: list[np.ndarray] = []
    try:
        while True:
            ok, f = cap.read()
            if not ok:
                break
            frames.append(f)
    finally:
        cap.release()
    return frames, fps


def _window(width: int, height: int, corner: str) -> tuple[int, int, int, int]:
    fx, fy = CORNERS[corner]
    wx, wy = int(round(width * fx)), int(round(height * fy))
    ww = min(int(round(width * WINDOW_W_FRAC)), width - wx)
    wh = min(int(round(height * WINDOW_H_FRAC)), height - wy)
    return wx, wy, ww, wh


def detect_timeline(
    source: Path,
    *,
    chunk_seconds: float = 1.0,
    samples_per_chunk: int = 10,
    min_segment_seconds: float = 0.8,
) -> list[tuple[int, int, tuple[int, int, int, int]]]:
    """Return ``[(start_frame, end_frame_exclusive, region), ...]`` for the logo.

    Regions are frame-absolute ``(x, y, w, h)``. Empty list means "not detected".
    """
    frames, fps = _read_frames(source)
    n = len(frames)
    if not n:
        return []
    h, w = frames[0].shape[:2]
    chunk = max(4, int(round(fps * chunk_seconds)))

    # --- 1. 逐块检测：每块用时间中位数压掉运动背景 ---
    chunks: list[tuple[int, int, LogoHit | None]] = []
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        if end - start < max(4, chunk // 3):
            break
        idx = np.linspace(start, end - 1, min(samples_per_chunk, end - start)).astype(int)
        best: LogoHit | None = None
        for corner in CORNERS:
            wx, wy, ww, wh = _window(w, h, corner)
            if ww <= 8 or wh <= 8:
                continue
            stack = np.stack([frames[i][wy:wy + wh, wx:wx + ww] for i in idx])
            med = np.median(stack, axis=0).astype(np.uint8)
            hit = detect_in_window(med, corner)
            if hit is None:
                continue
            if best is None or hit.confidence > best.confidence:
                best = hit
        chunks.append((start, end, best))

    # --- 2. 时间平滑：前后都未命中的孤立块视为误报 ---
    labels = [c[2].corner if c[2] else None for c in chunks]
    for i in range(1, len(chunks) - 1):
        if labels[i] is not None and labels[i - 1] is None and labels[i + 1] is None:
            chunks[i] = (chunks[i][0], chunks[i][1], None)
    labels = [c[2].corner if c[2] else None for c in chunks]

    # --- 3. 合并连续同角块；每段取最大范围（避免截断长昵称） ---
    # 检测抖动会让同一角的水印在某 1 秒块上短暂漏检（如 6-7s）。
    # 若空档很短、且空档之后第一个命中仍是同一个角，说明水印并没有换角，
    # 只是这一刻没检出 —— 直接把它并进当前段，而不是切成多段。
    segments: list[dict] = []
    pending_gap: dict | None = None
    for (start, end, hit) in chunks:
        label = hit.corner if hit else None
        if label is None:
            if pending_gap is None:
                pending_gap = {"start": start, "end": end}
            else:
                pending_gap["end"] = end
            continue
        wx, wy, _, _ = _window(w, h, label)
        box = (hit.bbox[0] + wx, hit.bbox[1] + wy, hit.bbox[2], hit.bbox[3])

        if pending_gap is not None:
            prev = segments[-1] if segments else None
            short = (pending_gap["end"] - pending_gap["start"]) <= max(
                4, int(round(fps * min_segment_seconds * 1.5)))
            if prev is not None and prev["corner"] == label and short:
                # 同一角、空档很短 -> 视为连续
                prev["end"] = end
                prev["boxes"].append(box)
                pending_gap = None
                continue
            segments.append({"corner": None, "start": pending_gap["start"],
                             "end": pending_gap["end"], "boxes": []})
            pending_gap = None

        if segments and segments[-1]["corner"] == label:
            segments[-1]["end"] = end
            segments[-1]["boxes"].append(box)
        else:
            segments.append({"corner": label, "start": start, "end": end, "boxes": [box]})

    if pending_gap is not None:
        segments.append({"corner": None, "start": pending_gap["start"],
                         "end": pending_gap["end"], "boxes": []})

    def merge_max(boxes: list[tuple[int, int, int, int]]) -> tuple[int, int, int, int]:
        x1 = min(b[0] for b in boxes)
        y1 = min(b[1] for b in boxes)
        x2 = max(b[0] + b[2] for b in boxes)
        y2 = max(b[1] + b[3] for b in boxes)
        return (x1, y1, x2 - x1, y2 - y1)

    # --- 4. 空档处理：短的由相邻段补齐；长的（如淡出黑场）擦所有已知角 ---
    corner_boxes = {s["corner"]: merge_max(s["boxes"]) for s in segments if s["corner"]}
    out: list[tuple[int, int, tuple[int, int, int, int]]] = []
    for si, seg in enumerate(segments):
        if seg["corner"] is not None:
            out.append((seg["start"], seg["end"], merge_max(seg["boxes"])))
            continue
        gap = seg["end"] - seg["start"]
        if gap <= max(4, int(round(fps * min_segment_seconds * 1.5))):
            prev_seg = next((s for s in reversed(segments[:si]) if s["corner"]), None)
            nxt = next((s for s in segments[si + 1:] if s["corner"]), None)
            pick = prev_seg or nxt
            if pick:
                out.append((seg["start"], seg["end"], merge_max(pick["boxes"])))
            continue
        # 长空档：把所有检测到的角都擦一遍，确保转场/淡出时不留残影
        for corner in corner_boxes:
            out.append((seg["start"], seg["end"], corner_boxes[corner]))

    out.sort(key=lambda x: x[0])
    return out


def pill_region(width: int, height: int) -> tuple[int, int, int, int]:
    """Absolute box of the always-on top-left 'AI生成' pill."""
    x, y, w, h = PILL_REL
    return (round(width * x), round(height * y), round(width * w), round(height * h))


def regions_by_frame(
    timeline: list[tuple[int, int, tuple[int, int, int, int]]],
    width: int,
    height: int,
    *,
    include_pill: bool = True,
    frame_count: int | None = None,
) -> list[list[tuple[int, int, int, int]]]:
    """Expand the timeline into a per-frame region list."""
    total = frame_count or (max(e for _, e, _ in timeline) if timeline else 0)
    per_frame: list[list[tuple[int, int, int, int]]] = [[] for _ in range(total)]
    for start, end, box in timeline:
        for i in range(max(0, start), min(total, end)):
            per_frame[i].append(box)
    if include_pill:
        pill = pill_region(width, height)
        for row in per_frame:
            row.append(pill)
    return per_frame


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    src = Path(sys.argv[1] if len(sys.argv) > 1
               else r"C:\Users\Administrator\Desktop\去水印素材\jimeng.mp4")
    hits = detect_timeline(src)
    frames, fps = _read_frames(src)
    h, w = frames[0].shape[:2]
    print(f"{w}x{h} {len(frames)}f {fps}fps")
    print(f"\n=== 检测到 {len(hits)} 个时段 ===")
    for s, e, b in hits:
        print(f"  {s/fps:5.2f}-{e/fps:5.2f}s  frames {s}-{e-1}  bbox={b}  ({b[2]}x{b[3]})")
    print(f"\n药丸: {pill_region(w, h)}")