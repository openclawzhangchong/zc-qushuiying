# -*- coding: utf-8 -*-
"""逐 1s 块检测体检：背景亮度/白占比/字形粘连/命中漏检误检。

用法：python tests/analyze_chunks.py <视频路径>

（原为 P0-2 亮背景分析所写，适用于任何即梦素材的快速体检。）

对每 1s 块输出：
- 期望角的窗口内背景平均亮度 (V 均值) 与高亮占比 (V>178 & S<90)
- 检测结果（bbox / 置信度 / None）
- 最大字形连通域的宽高（看是否把水印和浅色背景粘成一片）
- 其它角的误检
"""
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, r"D:\remove-ai-watermarks\webui")
sys.path.insert(0, r"D:\remove-ai-watermarks\src")

import cv2
import numpy as np

import jimeng_detect as jd

SRC = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(r"C:\Users\Administrator\Desktop\去水印素材\jimeng.mp4")
# 实测轮换：0-4s BR、4-8s BL、8-12s TR、12s+ 黑场（每 4 秒 = 4 个 1s 块）
def expect(ci: int):
    if ci < 4:
        return "BR"
    if ci < 8:
        return "BL"
    if ci < 12:
        return "TR"
    return None

frames, fps = jd._read_frames(SRC)
n = len(frames)
h, w = frames[0].shape[:2]
chunk = int(round(fps * 1.0))

print(f"{w}x{h} {n}f {fps}fps")
hdr = (f"{'块':>3} {'秒':>6} {'期望角':>5} | {'角':>3} {'V均值':>6} {'白占比%':>7} "
       f"{'最大字形wxh':>11} {'bbox':>22} {'conf':>6}  判定")
print(hdr)
print("=" * len(hdr.expandtabs()))

for ci in range(0, int(np.ceil(n / chunk))):
    start = ci * chunk
    end = min(start + chunk, n)
    if end - start < 4:
        break
    idx = np.linspace(start, end - 1, 10).astype(int)
    exp = expect(ci)

    for corner in jd.CORNERS:
        wx, wy, ww, wh = jd._window(w, h, corner)
        stack = np.stack([frames[i][wy:wy + wh, wx:wx + ww] for i in idx])
        med = np.median(stack, axis=0).astype(np.uint8)
        hsv = cv2.cvtColor(med, cv2.COLOR_BGR2HSV)
        v_mean = float(hsv[:, :, 2].mean())
        white = ((hsv[:, :, 2] > jd._WHITE_V) & (hsv[:, :, 1] < jd._WHITE_S)).astype(np.uint8)
        wpct = float(white.mean()) * 100
        glyphs = jd._glyphs(med)
        gw = max((g["w"] for g in glyphs), default=0)
        gh = max((g["h"] for g in glyphs), default=0)
        hit = jd.detect_in_window(med, corner)
        is_target = corner == exp
        is_fp = (hit is not None) and not is_target
        if not (is_target or is_fp):
            continue
        bbox = str(hit.bbox) if hit else "None"
        conf = f"{hit.confidence:.0f}" if hit else "-"
        if is_target and hit:
            verdict = "命中"
        elif is_target:
            verdict = "漏检!"
        else:
            verdict = "误检!"
        secs = f"{start/fps:.1f}-{end/fps:.1f}"
        print(f"{ci:>3} {secs:>6} {str(exp):>5} | {corner:>3} {v_mean:>6.1f} {wpct:>7.2f} "
              f"{gw:>4}x{gh:<6} {bbox:>22} {conf:>6}  {verdict}")
    print("-" * 110)
