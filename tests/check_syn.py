# -*- coding: utf-8 -*-
"""合成视频检测结果 vs ground truth 对比（P0-1 / P1-5 / P1-6）。

判定：
- 每个真实水印时段：检测结果是否命中同一角、检测框是否**完整包含**真实
  水印像素区（容差 3px）
- 药丸：pill_region(W,H) 是否完整覆盖真实药丸位置
- 昵称长度序列：框宽是否随字数增长
"""
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, r"D:\remove-ai-watermarks\webui")
sys.path.insert(0, r"D:\remove-ai-watermarks\src")

import jimeng_detect as jd

SYN = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(r"D:\testwork\syn")
truth = json.loads((SYN / "truth.json").read_text(encoding="utf-8"))

TOL = 3

def contains(big, small, tol=TOL):
    bx, by, bw, bh = big
    sx, sy, sw, sh = small
    return (bx <= sx + tol and by <= sy + tol and
            bx + bw >= sx + sw - tol and by + bh >= sy + sh - tol)

def inter(a, b):
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[0] + a[2], b[0] + b[2]); y2 = min(a[1] + a[3], b[1] + b[3])
    return max(0, x2 - x1) * max(0, y2 - y1)

for name, t in truth.items():
    w, h = t["size"]
    src = SYN / f"{name}.mp4"
    hits = jd.detect_timeline(src)
    print(f"===== {name}  {w}x{h} scale={t['scale_mode']} nick={t['nickname']} =====")

    # --- logo 分段对比 ---
    for seg in t["segments"]:
        f0, f1 = seg["f0"], seg["f1"]
        patch = seg["patch"]
        fps = t["fps"]
        # 找时间重叠的检测段
        cands = [(s, e, b) for (s, e, b) in hits
                 if s < f1 and e > f0]
        if not cands:
            print(f"  [{seg['corner']} {f0/fps:.1f}-{f1/fps:.1f}s] 漏检! patch={patch}")
            continue
        s, e, b = cands[0]
        ok = contains(b, patch)
        # 覆盖率（检测框与真实 patch 的交 / patch 面积）
        cov = inter(b, patch) / max(1, patch[2] * patch[3])
        print(f"  [{seg['corner']} {f0/fps:.1f}-{f1/fps:.1f}s] 检出 {s/fps:.1f}-{e/fps:.1f}s "
              f"bbox={b} 包含={'OK' if ok else 'NO'} patch覆盖率={cov*100:.1f}% "
              f"patch={patch}")

    # --- 药丸 ---
    pb = jd.pill_region(w, h)
    tp = t["pill"]
    ok = contains(pb, tp, tol=1)
    print(f"  药丸 truth={tp} pill_region={pb} 覆盖={'OK' if ok else 'NO'}")

    # --- 昵称框宽（nick 系列） ---
    if name.startswith("nick") and hits:
        print(f"  框宽: {hits[0][2][2]}px")
    print()
