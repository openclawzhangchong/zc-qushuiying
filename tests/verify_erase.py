# -*- coding: utf-8 -*-
"""端到端擦除效果量化验收。

用法::

    python tests/verify_erase.py <原始视频> <处理后视频> [gt.json]

不传 gt.json 时用 jimeng_detect 在原始视频上自动定位水印区域
（与擦除管线同源）+ pill_region 计算药丸。输出处理后视频比原始短
（剪尾）时只对重叠部分采样。

指标与验收线（详见 docs/jimeng-watermark-notes.md 第 8/15 节）：
- 水印区域笔画能量（拉普拉斯绝对值均值）降幅 ≥ 90%
- 静态水印像素残留 ≈ 0（静态 = 原始 ≥95% 帧皆为白）
- 对照角能量变化 ±3% 内（cv2 后端在暗场内容上放宽到 ±10%，主要是编码噪声）
- 药丸：区域内亮度 std 大幅下降（半透明叠加不能用白色判据）
- 全片 PSNR ≥ 29 dB，改动像素占比 1~2%
"""
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "webui"))
try:
    sys.path.insert(0, str(Path(r"D:\remove-ai-watermarks\src")))
except Exception:  # noqa: BLE001
    pass

import cv2  # noqa: E402
import numpy as np  # noqa: E402

import jimeng_detect as jd  # noqa: E402


def read_all(path: Path):
    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()
    return frames


def energy(frame, box):
    x, y, w, h = box
    g = cv2.cvtColor(frame[y:y + h, x:x + w], cv2.COLOR_BGR2GRAY)
    return float(np.abs(cv2.Laplacian(g, cv2.CV_32F)).mean())


def white_mask(frame, box):
    x, y, w, h = box
    hsv = cv2.cvtColor(frame[y:y + h, x:x + w], cv2.COLOR_BGR2HSV)
    return (hsv[:, :, 2] > 178) & (hsv[:, :, 1] < 90)


def main() -> None:
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    print("读取视频（大文件稍慢）...")
    orig, proc = read_all(src), read_all(dst)
    n = min(len(orig), len(proc))
    if len(orig) != len(proc):
        print(f"注意: 帧数不同 orig={len(orig)} proc={len(proc)}（可能已剪尾），按重叠部分比较")
    h, w = orig[0].shape[:2]
    print(f"{w}x{h} orig {len(orig)}f / proc {len(proc)}f")

    # --- 水印区域：检测器定位 ---
    timeline = jd.detect_timeline(src)
    print(f"\n检出时段 {len(timeline)} 个：")
    for s, e, b in timeline:
        print(f"  {s / 24:5.2f}-{e / 24:5.2f}s bbox={b}")
    pill = jd.pill_region(w, h)

    print("\n=== 1) 水印区域（能量降幅 / 静态白像素残留） ===")
    for s, e, box in timeline:
        s2, e2 = max(0, s), min(e, n)
        if e2 - s2 < 4:
            continue
        x, y, bw, bh = box
        # 静态白掩码：原始 ≥95% 帧皆为白（排除移动的内容亮斑）
        cnt = np.zeros((bh, bw), np.float32)
        for i in range(s2, e2):
            cnt += white_mask(orig[i], box).astype(np.float32)
        static = cnt >= (e2 - s2) * 0.95
        resid = 0
        for i in range(s2, e2):
            resid += int((white_mask(proc[i], box) & static).sum())
        idx = list(range(s2, e2, max(1, (e2 - s2) // 12)))
        eo = float(np.mean([energy(orig[i], box) for i in idx]))
        ep = float(np.mean([energy(proc[i], box) for i in idx]))
        print(f"  {s2/24:5.2f}-{e2/24:5.2f}s {str(box):>22} 能量 {eo:7.2f}→{ep:6.2f} "
              f"(降 {(1 - ep / max(eo, 1e-6)) * 100:5.1f}%)  静态残留 {resid / (e2 - s2):.2f} px/帧")

    # --- 对照组：各检出时段内，未含水印的角窗口 ---
    print("\n=== 2) 对照组（无水印的角，能量变化应小） ===")
    corners = jd.corner_windows(w, h)
    for s, e, box in timeline:
        active = None
        for corner in corners:
            c = jd._window(w, h, corner)
            if (box[0] >= c[0] and box[1] >= c[1]
                    and box[0] + box[2] <= c[0] + c[2]
                    and box[1] + box[3] <= c[1] + c[3]):
                active = corner
                break
        if active is None:
            continue
        for corner in corners:
            if corner == active:
                continue
            wx, wy, ww, wh = jd._window(w, h, corner)
            cbox = (wx + 30, wy + 30, ww - 60, wh - 60)
            if cbox[2] <= 0 or cbox[3] <= 0:
                continue
            s2, e2 = max(0, s), min(e, n)
            if e2 - s2 < 4:
                continue
            idx = list(range(s2, e2, max(1, (e2 - s2) // 12)))
            eo = float(np.mean([energy(orig[i], cbox) for i in idx]))
            ep = float(np.mean([energy(proc[i], cbox) for i in idx]))
            chg = (ep / max(eo, 1e-6) - 1) * 100
            print(f"  {s2/24:5.2f}-{e2/24:5.2f}s 对照[{corner}] {eo:7.2f}→{ep:7.2f} ({chg:+5.1f}%)")
        break  # 每支视频列一个时段的对照即可，避免刷屏

    # --- 药丸 ---
    x, y, pw, ph = pill
    idx = list(range(0, min(288, n), max(1, min(288, n) // 12)))
    so = float(np.mean([orig[i][y:y + ph, x:x + pw].std() for i in idx]))
    sp = float(np.mean([proc[i][y:y + ph, x:x + pw].std() for i in idx]))
    vo = float(np.mean([(cv2.cvtColor(orig[i][y:y + ph, x:x + pw], cv2.COLOR_BGR2HSV)[:, :, 2] > 90).sum() for i in idx]))
    vp = float(np.mean([(cv2.cvtColor(proc[i][y:y + ph, x:x + pw], cv2.COLOR_BGR2HSV)[:, :, 2] > 90).sum() for i in idx]))
    print(f"\n=== 3) 药丸 {pill} ===")
    print(f"  std {so:6.2f} → {sp:5.2f} | V>90 像素 {vo:7.1f} → {vp:6.1f}")

    # --- 全片 ---
    psnrs, chgs = [], []
    for i in range(0, n, 3):
        a = orig[i].astype(np.float32)
        b = proc[i].astype(np.float32)
        mse = float(((a - b) ** 2).mean())
        if mse > 0:
            psnrs.append(10 * np.log10(255 ** 2 / mse))
        chgs.append(float((np.abs(a - b).max(axis=2) > 8).mean()))
    print(f"\n=== 4) 全片 ===")
    print(f"  PSNR {np.mean(psnrs):.2f} dB (采样 {len(psnrs)} 帧) | 改动像素 {np.mean(chgs) * 100:.2f}% (|diff|>8)")


if __name__ == "__main__":
    main()
