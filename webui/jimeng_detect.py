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
    "TR": (1.0 - WINDOW_W_FRAC, 0.0),
    "BL": (0.0, 1.0 - WINDOW_H_FRAC),
    "BR": (1.0 - WINDOW_W_FRAC, 1.0 - WINDOW_H_FRAC),
}

# 竖屏的第二个轮换位在**左中**而非左下：真实竖屏样本（720×1280）实测
# 水印位于 y 0.478~0.540H（垂直居中偏上），BL 窗口（0.66H~H）完全罩不住。
# 因此竖屏用 {TR, ML, BR}，横屏沿用 1280×720 标定的三角。
# 两套都**不含 TL**：左上角永远只有「AI生成」药丸（无条件擦除，不需要
# 检测），而扫描 TL 只会引入药丸/画面内容的假命中（实测）。
ML_Y_FRAC = 0.42


def corner_windows(width: int, height: int) -> dict[str, tuple[float, float]]:
    """Aspect-aware corner windows (relative origins)."""
    if height > width:
        return {
            "TR": (1.0 - WINDOW_W_FRAC, 0.0),
            "ML": (0.0, ML_Y_FRAC),
            "BR": (1.0 - WINDOW_W_FRAC, 1.0 - WINDOW_H_FRAC),
        }
    return dict(CORNERS)

# "AI生成" 药丸在左上角的固定位置（相对坐标，1280x720 实测标定）。
# 它是整片常驻的合规标识，不需要随时间变化。
PILL_REL: tuple[float, float, float, float] = (0.0047, 0.0111, 0.0625, 0.0556)

# 白/高亮判据（HSV）。水印是半透明白字，阈值放宽一点以容纳不同背景亮度。
_WHITE_V = 178
_WHITE_S = 90

# 单个角窗口内白色像素占比超过该值时判定「背景过亮，白色判据失效」：
# 浅色/近白背景会把水印笔画和背景粘成一片，此时检出的"字形"不可信，
# 实测会给出随机位置的假命中。返回 None 交给上层走四角兜底。
# 真实水印块的白占比在 1%~14%（实测），大面积浅色背景在 40%+，阈值取中间。
_MAX_WHITE_RATIO = 0.30

# 1280×720 实测标定的绝对像素阈值。帧宽 <1280 时按比例放宽（竖屏水印若
# 随宽度缩放，字形会比横屏标定小），帧宽 ≥1280 时保持原值——不因放大而
# 收紧，避免误伤"保持绝对像素尺寸的叠加物"。
_CALIB_WIDTH = 1280


def _adaptive(px: int, frame_width: int, *, lo: int) -> int:
    """Scale an absolute pixel threshold with frame width (floor at ``lo``)."""
    if frame_width >= _CALIB_WIDTH:
        return px
    return max(lo, int(round(px * frame_width / _CALIB_WIDTH)))


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


def _glyphs(img: np.ndarray, frame_width: int | None = None) -> list[dict]:
    """White/highlight connected components that look like glyphs.

    返回的字形带 ``strong`` 标记：强档（面积/填充率达到 1280×720 标定值）
    才有资格当 wordmark 候选；弱档仅供昵称行使用——昵称里的简单字
    （「十」「三」「之」「语」）笔画细、白色核心面积小，按强档阈值会被
    整字丢弃，导致昵称链中断（实测 @+12 字昵称被拦腰截断，框宽只有
    真实宽度的 46%）。

    ``frame_width`` 用于缩放绝对像素阈值：默认从窗口宽度反推
    （窗口恒为帧宽的 ``WINDOW_W_FRAC``）。
    """
    fw = frame_width or max(1, int(round(img.shape[1] / WINDOW_W_FRAC)))
    strong_area = _adaptive(110, fw, lo=30)
    strong_fill = 0.13
    # 弱档下探到 25：昵称行尾的「」括号笔画极瘦（实测白色核心仅 ~35px²），
    # 40 的门槛会把收尾括号整字丢弃、擦除框右缘差出一个字宽。
    base_area = _adaptive(25, fw, lo=8)
    base_fill = 0.08
    bh_min = _adaptive(11, fw, lo=5)
    bw_min = _adaptive(5, fw, lo=2)
    mask = cv2.morphologyEx(_white_mask(img), cv2.MORPH_CLOSE,
                            np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out: list[dict] = []
    for c in contours:
        x, y, bw, bh = cv2.boundingRect(c)
        area = float(cv2.contourArea(c))
        if area < base_area or bh < bh_min or bw < bw_min:
            continue
        if bh > 170 or bw > 340:
            continue
        fill = area / max(1, bw * bh)
        if fill < base_fill:                   # 填充率过低 = 细长噪点
            continue
        out.append({"x": x, "y": y, "w": bw, "h": bh, "area": area,
                    "strong": area >= strong_area and fill >= strong_fill})
    return out


def _find_wordmark(glyphs: list[dict], frame_width: int | None = None) -> list[dict] | None:
    """The "即梦" wordmark: a strict same-height seed pair + tolerant neighbours.

    结构判定分两层：

    1. **种子对**：同一行、横向相邻、高度几乎一致（比值 ≤1.15）的两个
       强字形——"即""梦"是同一字号，这是横屏/竖屏/强噪声下都稳定的
       核心签名（实测比 1.00~1.08）。
    2. **有界邻居扩展**：与种子对同行、横向邻接、高度在
       [0.55, 1.25]×种子高 之间的字形并入词标。下限容忍笔画洗淡
       （强噪声背景如秋林径向模糊会把「梦」的上半细笔画洗淡，实测
       中位数里只剩 ~0.7×「即」高），上限挡住比字还高的背景物体
       （实测秋林样本里 37px 的亮斑紧贴 28px 的字）。

    注意：不能只取「前两块」。中文笔画常被白色掩码切成多个连通域
    （实测「即」拆成两块、「梦」一块，或多达 3~4 块），必须按几何
    连续性整体收集。
    """
    fw = frame_width or _CALIB_WIDTH
    h_min = _adaptive(18, fw, lo=11)
    span_min = _adaptive(26, fw, lo=15)
    cands = [g for g in glyphs if g["h"] >= h_min and g.get("strong", True)]
    if len(cands) < 2:
        return None
    cands.sort(key=lambda g: g["x"])
    n = len(cands)
    best: tuple[float, list[dict]] | None = None

    def _same_row(a: dict, b: dict) -> bool:
        return abs((b["y"] + b["h"] / 2) - (a["y"] + a["h"] / 2)) <= max(a["h"], b["h"]) * 0.30

    for i in range(n - 1):
        for j in range(i + 1, n):
            a, b = cands[i], cands[j]
            gap = b["x"] - (a["x"] + a["w"])
            if gap > max(a["h"], b["h"]) * 1.2:
                break                      # 按 x 排序，再往后只会更远
            if not _same_row(a, b) or gap < -8:
                continue
            if max(a["h"], b["h"]) / max(1, min(a["h"], b["h"])) > 1.15:
                continue                   # 种子对必须严格同高
            pair = [a, b]
            pair_h = max(a["h"], b["h"])
            lo, hi = 0.55 * pair_h, 1.25 * pair_h

            # 向右扩展邻居（同行、邻接、高度有界）
            k = j + 1
            while k < n:
                g = cands[k]
                ref = pair[-1]
                g2 = g["x"] - (ref["x"] + ref["w"])
                if g2 > max(ref["h"], g["h"]) * 1.2:
                    break
                if (_same_row(ref, g) and g2 >= -8
                        and lo <= g["h"] <= hi):
                    pair.append(g)
                k += 1
            # 向左扩展（只允许正常字距，不允许跨洞——左伸昵称与 logo
            # 之间字距连续，远处背景亮点隔着大空隙）
            k = i - 1
            while k >= 0:
                g = cands[k]
                ref = pair[0]
                g2 = ref["x"] - (g["x"] + g["w"])
                if g2 > max(ref["h"], g["h"]) * 1.2:
                    break
                if (_same_row(ref, g) and g2 >= -8
                        and lo <= g["h"] <= hi):
                    pair.insert(0, g)
                k -= 1

            gx1 = min(g["x"] for g in pair)
            gx2 = max(g["x"] + g["w"] for g in pair)
            if gx2 - gx1 < span_min:
                continue
            # 保留的字形必须彼此横向连续（防止把同行远处的东西并入）
            reach = pair_h * 3.0
            anchor = pair[0]["x"]
            kept = [g for g in pair if g["x"] - anchor <= reach]
            if len(kept) < 2:
                continue
            score = pair_h * 1000.0 + sum(g["area"] for g in kept) / 100.0
            if best is None or score > best[0]:
                best = (score, kept)
    return best[1] if best else None


def _find_nickname(glyphs: list[dict], wordmark: list[dict],
                   frame_width: int | None = None) -> list[dict]:
    """The '@nickname' line directly below the wordmark.

    Width comes from the actual glyphs, so any nickname length works
    (@娥酱, @张三, @李思思思思, ...).
    """
    fw = frame_width or _CALIB_WIDTH
    nick_h_min = _adaptive(10, fw, lo=6)
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
        if g["h"] < nick_h_min or g["h"] > wh * 1.15:
            continue
        # 水平上要与 wordmark 有重叠或紧邻。左门坎放宽到 4×字高：真实竖屏
        # 样本里昵称比 wordmark 左伸 2.6×字高（右下角时昵称行更宽、左对齐
        # 位置更靠左），1.5×字高会把行首的 @ 和第一个字拒掉、造成残留。
        if g["x"] + g["w"] < wx1 - wh * 4.0 or g["x"] > wx2 + wh * 9.0:
            continue
        picked.append(g)
    if not picked:
        return []
    # 链式收集：从**最靠近 wordmark 左缘**的字形出发（昵称与 logo 在那里
    # 相接），向右、向两个方向延伸。
    #   向右：正常字距 ≤2.2×字高；跨洞续链 ≤4.5×字高且必须同行——昵称里的
    #     简单字/细瘦括号（十/三/之/」）白色核心小、可能整字不达字形门槛，
    #     在链上留下 1~3 个字宽的空洞；擦除框是并集矩形，只要链延伸过去，
    #     缺失字符自然被盖住。真实视频里 @娥酱 右侧 121px 的碎石反光
    #     （不同行）会被同行判据挡掉。
    #   向左：只允许正常字距（≤2.2×字高），不允许跨洞——真实竖屏样本里
    #     昵称左伸部分与 wordmark 之间字距连续；而横屏样本 @娥酱 左侧
    #     92px 的背景亮点隔着大空隙，若允许跨洞会被误链、框左界多擦
    #     40px（实测回归）。
    picked.sort(key=lambda g: g["x"])
    start_i = min(range(len(picked)),
                  key=lambda i: abs(picked[i]["x"] - wx1))
    chain = [picked[start_i]]
    for g in picked[start_i + 1:]:
        ref = chain[-1]
        gap = g["x"] - (ref["x"] + ref["w"])
        if gap > max(ref["h"], g["h"]) * 4.5:
            break
        if gap > max(ref["h"], g["h"]) * 2.2:
            ref_c = ref["y"] + ref["h"] / 2
            g_c = g["y"] + g["h"] / 2
            if abs(g_c - ref_c) > max(ref["h"], g["h"]) * 0.35:
                break
        chain.append(g)
    left_part: list[dict] = []
    for g in reversed(picked[:start_i]):
        ref = left_part[0] if left_part else chain[0]
        gap = ref["x"] - (g["x"] + g["w"])
        if gap > max(ref["h"], g["h"]) * 2.2:
            break
        ref_c = ref["y"] + ref["h"] / 2
        g_c = g["y"] + g["h"] / 2
        if abs(g_c - ref_c) > max(ref["h"], g["h"]) * 0.35:
            break
        left_part.append(g)
    chain = list(reversed(left_part)) + chain
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
    *,
    frame_width: int | None = None,
) -> LogoHit | None:
    """Locate the logo (wordmark + nickname) inside one corner window.

    All returned coordinates are **window-relative**; the caller adds the window
    origin. Keeping this function pure (median image in, box out) makes it
    trivially testable on crops.

    ``frame_width`` 是整帧宽度（非窗口宽度），用于缩放绝对像素阈值；
    缺省时从窗口宽度按 ``WINDOW_W_FRAC`` 反推。
    """
    fw = frame_width or max(1, int(round(median_bgr.shape[1] / WINDOW_W_FRAC)))
    # 亮背景守卫：窗口几乎全白时白色判据已失效，检出的任何"字形"都不可信
    white_ratio = float((_white_mask(median_bgr) > 0).mean())
    if white_ratio > _MAX_WHITE_RATIO:
        return None
    glyphs = _glyphs(median_bgr, fw)
    wordmark = _find_wordmark(glyphs, fw)
    if not wordmark:
        return None
    nickname = _find_nickname(glyphs, wordmark, fw)
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
    # 但昵称行可能比 wordmark 更靠左（真实竖屏样本左伸 2.6×字高），
    # 此时框左界必须外扩到昵称左缘，否则行首字符残留。
    pad = max(4, int(round(wh * 0.25)))
    x1 = max(0, wm[0] - int(round(wh * 1.9)))
    if nickname and full[0] < x1:
        x1 = max(0, full[0] - pad)
    # 顶部余量 0.35×字高：星标箭头比文字行更高（真实竖屏样本实测顶尖
    # 超出 wordmark 顶约 0.3×字高），0.10 会留下彩色箭头尖残留。
    y1 = max(0, wm[1] - int(round(wh * 0.35)))
    x2 = full[0] + full[2]
    y2 = full[1] + full[3] + int(round(wh * 0.10))

    # 右/下各留一点安全余量：字形最右一列的抗锯齿像素可能低于白色阈值，
    # 导致检测框紧贴笔画边缘（实测左下角右侧只剩 2px，擦完仍有残影）。
    # 半个字高以内都是纯背景，多擦这一点不会有观感损失。
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
    fx, fy = corner_windows(width, height)[corner]
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
        for corner in corner_windows(w, h):
            wx, wy, ww, wh = _window(w, h, corner)
            if ww <= 8 or wh <= 8:
                continue
            stack = np.stack([frames[i][wy:wy + wh, wx:wx + ww] for i in idx])
            med = np.median(stack, axis=0).astype(np.uint8)
            hit = detect_in_window(med, corner, frame_width=w)
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
    # 每个角的输出框 = 该角**全部检出**的并集：水印按角固定位置，但逐块
    # 中位数质量有起伏（强噪声背景会把部分字形洗淡），单块的框可能只含
    # 局部（实测 BL 段一块只检出昵称细条，主体暴露）；并集才完整。
    # 注意：同一个角可能被中段空档拆成多个 segment（如强噪声视频里
    # 某几块误检出假角、把 BL 拆成两段）。字典推导式的同名键会互相
    # 覆盖——只剩最后一段的框，前一段的框被丢弃（真实样本实测导致
    # 主体暴露 34 px/帧残留）。必须按角**聚合全部框**再取并集。
    per_corner: dict[str, list] = {}
    for sgm in segments:
        if sgm["corner"]:
            per_corner.setdefault(sgm["corner"], []).extend(sgm["boxes"])
    corner_boxes = {c: merge_max(bs) for c, bs in per_corner.items()}
    # 星标/渐变箭头是彩色装饰，白色掩码抓不到（或白色核心太小不稳定），
    # 检测框只能锚定在文字字形上——按框高 30% 向四周扩一圈，把装饰
    # 元素盖住（真实样本实测：TR 星标左缘超出文字框 13px、顶部 2px）。
    expanded: dict[str, tuple[int, int, int, int]] = {}
    for c, (bx, by, bw, bh) in corner_boxes.items():
        m = max(6, int(round(bh * 0.3)))
        nx1, ny1 = max(0, bx - m), max(0, by - m)
        nx2, ny2 = min(w, bx + bw + m), min(h, by + bh + m)
        expanded[c] = (nx1, ny1, nx2 - nx1, ny2 - ny1)
    corner_boxes = expanded
    out: list[tuple[int, int, tuple[int, int, int, int]]] = []
    for si, seg in enumerate(segments):
        if seg["corner"] is not None:
            out.append((seg["start"], seg["end"], corner_boxes[seg["corner"]]))
            continue
        gap = seg["end"] - seg["start"]
        if gap <= max(4, int(round(fps * min_segment_seconds * 1.5))):
            prev_seg = next((s for s in reversed(segments[:si]) if s["corner"]), None)
            nxt = next((s for s in segments[si + 1:] if s["corner"]), None)
            pick = prev_seg or nxt
            if pick:
                out.append((seg["start"], seg["end"], merge_max(pick["boxes"])))
            continue
        # 长空档分两种：
        #   片中空档（之后还有检出）：水印大概率仍在轮换，只是没检出，
        #     全程擦所有已知角（宁可多擦）。
        #   片尾空档（直到视频结束都没有检出）：水印已随淡出消失，只擦
        #     最后检出段之后 ~1s 的过渡窗。不再全程兜底——黑场上的品牌
        #     片尾动画（如「即梦AI」居中出场）是视频内容，兜底框会盖住
        #     它的边缘，全程擦除会毁掉它（真实竖屏样本实测）。
        is_tail_gap = not any(s["corner"] for s in segments[si + 1:])
        if is_tail_gap:
            # 1.5s：覆盖淡出过程（真实样本水印在最后检出后 ~1.2s 才完全隐去）
            gap_end = min(seg["end"], seg["start"] + int(round(fps * 1.5)))
        else:
            gap_end = seg["end"]
        for gs in range(seg["start"], gap_end, chunk):
            ge = min(gs + chunk, gap_end)
            for corner in corner_boxes:
                out.append((gs, ge, corner_boxes[corner]))

    out.sort(key=lambda x: x[0])
    return out


def detect_jimeng_visual(path: Path, samples: int = 48) -> bool:
    """轻量视觉探测：均匀抽帧（默认 ≤48 帧、不整段载入内存），
    每 4 帧一组做时间中位数后跑角窗检测；任一组检出即梦 wordmark
    即返回 True。仅用作 auto 模式的路由信号（转存视频的容器元数据
    常被剥离），正式擦除仍走完整的 detect_timeline。
    """
    cap = open_video_capture(path)
    if not cap.isOpened():
        return False
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    w = h = 0
    group: list[np.ndarray] = []

    def flush() -> bool:
        if len(group) < 2 or w == 0:
            return False
        med = np.median(np.stack(group), axis=0).astype(np.uint8)
        for corner in corner_windows(w, h):
            wx, wy, ww, wh = _window(w, h, corner)
            if ww <= 8 or wh <= 8:
                continue
            if detect_in_window(med[wy:wy + wh, wx:wx + ww], corner,
                                frame_width=w) is not None:
                return True
        return False

    try:
        if total > 0:
            idxs = np.linspace(0, total - 1, min(samples, total)).astype(int)
            for i in idxs:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
                ok, f = cap.read()
                if not ok:
                    continue
                if w == 0:
                    h, w = f.shape[:2]
                group.append(f)
                if len(group) >= 4 and flush():
                    return True
                if len(group) >= 4:
                    group.clear()
            if flush():
                return True
        else:
            for _ in range(samples):
                ok, f = cap.read()
                if not ok:
                    break
                if w == 0:
                    h, w = f.shape[:2]
                group.append(f)
                if len(group) >= 4 and flush():
                    return True
                if len(group) >= 4:
                    group.clear()
            if flush():
                return True
    finally:
        cap.release()
    return False


def pill_region(width: int, height: int) -> tuple[int, int, int, int]:
    """Absolute box of the always-on top-left 'AI生成' pill.

    1280×720 横屏实测：药丸位于 (6, 8)，尺寸 80×40。非 16:9 分辨率上
    「按帧高缩放」会把药丸高度拉到不符合实际的值（720×1280 竖屏下
    0.0556×1280 = 71px，而药丸设计高度只有 40px），于是叠加三种缩放假设
    取并集，保证竖屏上药丸一定被盖住：

    1. 原始标定（x/w 随帧宽、y/h 随帧高）——16:9 下与按宽缩放完全一致；
    2. 整体按帧宽等比缩放（叠加层随播放器宽度缩放的假设）；
    3. 位置随帧宽、尺寸保持 720p 设计值（叠加层不随分辨率缩放的假设）；
    4. 非 16:9 额外并入**真实竖屏样本实测框**：720×1280 实测药丸位于
       (18, 14, 73, 32)，边距与横屏标定明显不同（x 边距 18 vs 6，
       高度 32 vs 40），前三种假设在竖屏上都盖不住它的右端。

    16:9 分辨率上 1~3 重合或互为子集，行为与旧版一致；竖屏上四个框
    取并集。多擦区域都在左上角内，无观感损失。
    """
    rx, ry, rw, rh = PILL_REL
    k = width / 1280.0
    boxes = [
        (round(width * rx), round(height * ry), round(width * rw), round(height * rh)),
        (round(6 * k), round(8 * k), round(80 * k), round(40 * k)),
        (6, 8, 80, 40),
    ]
    if abs(height * 16 - width * 9) > max(2, round(width * 0.002)):
        boxes.append((round(width * 0.025), round(height * 0.011),
                      round(width * 0.102), round(height * 0.025)))
    x1 = min(b[0] for b in boxes)
    y1 = min(b[1] for b in boxes)
    x2 = max(b[0] + b[2] for b in boxes)
    y2 = max(b[1] + b[3] for b in boxes)
    return (x1, y1, x2 - x1, y2 - y1)


def regions_by_frame(
    timeline: list[tuple[int, int, tuple[int, int, int, int]]],
    width: int,
    height: int,
    *,
    include_pill: bool = True,
    frame_count: int | None = None,
    extend_tail: bool = True,
) -> list[list[tuple[int, int, int, int]]]:
    """Expand the timeline into a per-frame region list.

    ``detect_timeline`` 会跳过末尾不足一个分块的帧（默认分块 1 秒），
    所以它覆盖的帧数可能略少于视频总帧数。传 ``frame_count`` 时按真实帧数
    生成长度对齐的列表；``extend_tail`` 让尾部继承最后一个已检测区域，
    避免最后不足 1 秒的画面漏擦。
    """
    detected_end = max((e for _, e, _ in timeline), default=0)
    total = frame_count if frame_count is not None else detected_end
    per_frame: list[list[tuple[int, int, int, int]]] = [[] for _ in range(total)]
    for start, end, box in timeline:
        for i in range(max(0, start), min(total, end)):
            per_frame[i].append(box)
    # 尾部补齐：检测没覆盖到的帧沿用最后一个区域
    if extend_tail and timeline and detected_end < total:
        last_box = timeline[-1][2]
        for i in range(detected_end, total):
            per_frame[i].append(last_box)
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