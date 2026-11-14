# -*- coding: utf-8 -*-
"""合成即梦风格测试视频生成器（P0-1 / P1-5 / P1-6）。

模拟真实即梦版式（依据 docs/jimeng-watermark-notes.md 的实测几何）：
  [彩色箭头] 即 梦        ← 白色 wordmark，两字同高
             @昵称         ← 下一行，字号约一半
  左上角整片常驻半透明「AI生成」药丸
  logo 每 4 秒换角：0-4s 右下 → 4-8s 左下 → 8-12s 右上，12s 后无叠加

背景为程序生成的运动场景（渐变 + 移动色块 + 噪声），时间中位数可将其压掉。

两种缩放假设（竖屏上 PILL_REL 的行为依赖于此）：
  scale_mode="absolute"  叠加物保持 720p 横屏时的绝对像素尺寸
  scale_mode="width"     叠加物随帧宽等比缩放（k = W/1280）

输出写入 D:/testwork/syn/，ground truth 写入同目录 truth.json。
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.stdout.reconfigure(encoding="utf-8")

OUT_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(r"D:\testwork\syn")
OUT_DIR.mkdir(parents=True, exist_ok=True)
FONT_PATH = r"C:\Windows\Fonts\simhei.ttf"

# 真实 720p 横屏实测（像素）：
WORD_H = 28          # wordmark 字形高
NICK_H = 16          # 昵称字形高
ROW_GAP = 10         # 两行间距
ARROW_W = 26         # 箭头宽
ARROW_H = 16         # 箭头高
ARROW_GAP = 5        # 箭头与 wordmark 间距
PILL_W, PILL_H = 80, 40
PILL_X, PILL_Y = 6, 8
# 各角边距（水印块外沿到帧边的距离，720p 实测近似）
MARGINS = {
    "BR": {"right": 76, "bottom": 90},
    "BL": {"left": 55, "bottom": 90},
    "TR": {"right": 62, "top": 61},
}


def render_watermark(nickname: str, k: float) -> np.ndarray:
    """Render arrow + wordmark + nickname onto an RGBA canvas, scaled by k."""
    wh = max(6, round(WORD_H * k))
    nh = max(6, round(NICK_H * k))
    f_word = ImageFont.truetype(FONT_PATH, round(wh * 1.25))
    f_nick = ImageFont.truetype(FONT_PATH, round(nh * 1.25))
    aw = max(4, round(ARROW_W * k))
    ah = max(4, round(ARROW_H * k))
    agap = max(2, round(ARROW_GAP * k))
    gap = max(3, round(ROW_GAP * k))

    # 先量尺寸
    tmp = Image.new("RGBA", (8, 8))
    td = ImageDraw.Draw(tmp)
    wb = td.textbbox((0, 0), "即梦", font=f_word)
    w_w = wb[2] - wb[0]
    nb = td.textbbox((0, 0), nickname, font=f_nick)
    n_w = nb[2] - nb[0]
    n_h = nb[3] - nb[1]

    cw = aw + agap + w_w + max(n_w - w_w, 0) + round(8 * k)
    ch = wh + gap + n_h + round(8 * k)
    img = Image.new("RGBA", (cw, ch), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)

    # wordmark（白色）
    wx = aw + agap - wb[0]
    wy = -wb[1]
    d.text((wx, wy), "即梦", font=f_word, fill=(255, 255, 255, 235))
    # 昵称（白色，略小）
    nx = aw + agap - nb[0]
    ny = wh + gap - nb[1]
    d.text((nx, ny), nickname, font=f_nick, fill=(255, 255, 255, 235))
    # 渐变箭头（橙→粉，三角旗）
    arrow = Image.new("RGBA", (aw, ah), (0, 0, 0, 0))
    ad = ImageDraw.Draw(arrow)
    for i in range(aw):
        t = i / max(1, aw - 1)
        r = int(255 - 60 * t)
        g = int(120 + 40 * t)
        b = int(40 + 160 * t)
        half = int(ah / 2 * (1 - 0.75 * i / max(1, aw - 1))) + 1
        ad.line([(i, ah // 2 - half), (i, ah // 2 + half)], fill=(r, g, b, 255))
    img.alpha_composite(arrow, (0, (wh - ah) // 2))

    arr = np.array(img)  # RGBA
    return arr


def render_pill(k: float) -> np.ndarray:
    """Semi-transparent rounded 'AI生成' pill (absolute design, scaled by k)."""
    pw, ph = max(20, round(PILL_W * k)), max(12, round(PILL_H * k))
    img = Image.new("RGBA", (pw, ph), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle([0, 0, pw - 1, ph - 1], radius=ph // 2,
                        fill=(255, 255, 255, 140))
    fs = round(ph * 0.45)
    font = ImageFont.truetype(FONT_PATH, fs)
    tb = d.textbbox((0, 0), "AI生成", font=font)
    d.text(((pw - (tb[2] - tb[0])) // 2 - tb[0], (ph - (tb[3] - tb[1])) // 2 - tb[1]),
           "AI生成", font=font, fill=(60, 60, 60, 255))
    return np.array(img)


def composite(frame: np.ndarray, rgba: np.ndarray, x: int, y: int) -> None:
    """Alpha-composite an RGBA patch onto a BGR frame at (x, y)."""
    h, w = rgba.shape[:2]
    fh, fw = frame.shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(fw, x + w), min(fh, y + h)
    if x1 <= x0 or y1 <= y0:
        return
    patch = rgba[y0 - y:y1 - y, x0 - x:x1 - x]
    a = (patch[:, :, 3:4].astype(np.float32)) / 255.0
    rgb = patch[:, :, :3].astype(np.float32)
    dst = frame[y0:y1, x0:x1].astype(np.float32)
    frame[y0:y1, x0:x1] = (rgb * a + dst * (1 - a)).astype(np.uint8)


def make_background(w: int, h: int, frames: int, bright: bool = False,
                    seed: int = 7, calm: bool = False) -> list[np.ndarray]:
    """Moving gradient + drifting blobs + noise; median collapses it.

    ``calm``：色块小、快、且避开近白色（用于昵称长度等几何测试，
    排除「大块近白色块压住水印行」这类比真实素材更对抗的干扰）。
    """
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    base = np.dstack([
        90 + 50 * (xx / max(1, w - 1)),
        70 + 60 * (yy / max(1, h - 1)),
        110 + 30 * ((xx + yy) / max(1, w + h - 2)),
    ]).astype(np.float32)
    if bright:
        base = base * 1.55 + 60   # 大面积浅色/近白背景
    base = np.clip(base, 0, 235)
    blobs = []
    for i in range(7):
        if calm:
            r_lo, r_hi, v_lo, v_hi, spd_lo, spd_hi = 12, 35, 40, 170, 6, 12
        else:
            r_lo, r_hi, v_lo, v_hi, spd_lo, spd_hi = 40, 130, 0, 255, 2, 6
        c = np.array([rng.integers(v_lo, v_hi), rng.integers(v_lo, v_hi),
                      rng.integers(v_lo, v_hi)], dtype=np.float32)
        blobs.append({
            "r": rng.integers(r_lo, r_hi), "c": c,
            "x": rng.integers(0, w), "y": rng.integers(0, h),
            "vx": rng.integers(spd_lo, spd_hi) * rng.choice([-1, 1]),
            "vy": rng.integers(spd_lo, spd_hi) * rng.choice([-1, 1]),
        })
    out = []
    for f in range(frames):
        img = base.copy()
        for b in blobs:
            b["x"] = (b["x"] + b["vx"]) % w
            b["y"] = (b["y"] + b["vy"]) % h
            cv2.circle(img, (int(b["x"]), int(b["y"])), int(b["r"]),
                       tuple(int(v) for v in b["c"]), -1, lineType=cv2.LINE_AA)
        img += rng.normal(0, 5, img.shape).astype(np.float32)
        out.append(np.clip(img, 0, 255).astype(np.uint8))
    return out


def paste_pos(corner: str, patch_wh: tuple[int, int], w: int, h: int,
              margins: dict) -> tuple[int, int]:
    pw, ph = patch_wh
    m = margins[corner]
    x = w - m["right"] - pw if "right" in m else m["left"]
    y = h - m["bottom"] - ph if "bottom" in m else m["top"]
    return x, y


def gen_video(path: Path, size: tuple[int, int], *, nickname: str,
              scale_mode: str, bright: bool = False, seconds: int = 14,
              fps: int = 24, corners: list[str] | None = None,
              calm: bool = False) -> dict:
    w, h = size
    k = 1.0 if scale_mode == "absolute" else w / 1280.0
    km = k  # 边距与叠加物同缩放
    wm = render_watermark(nickname, k)
    pill = render_pill(min(k, 1.0) if scale_mode == "absolute" else k)
    margins = {
        "BR": {"right": round(76 * km), "bottom": round(90 * km)},
        "BL": {"left": round(55 * km), "bottom": round(90 * km)},
        "TR": {"right": round(62 * km), "top": round(61 * km)},
    }
    pill_x, pill_y = round(PILL_X * km), round(PILL_Y * km)

    rotation = corners or ["BR", "BL", "TR"]
    total = seconds * fps
    frames_bg = make_background(w, h, total, bright=bright, calm=calm)

    truth = {"size": [w, h], "fps": fps, "scale_mode": scale_mode,
             "nickname": nickname, "segments": [], "pill": None}
    pw, ph = wm.shape[1], wm.shape[0]
    qw, qh = pill.shape[1], pill.shape[0]

    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (w, h))
    for f in range(total):
        frame = frames_bg[f]
        sec = f / fps
        ci = min(int(sec // 4), len(rotation) - 1)
        corner = rotation[ci] if sec < 4 * len(rotation) else None
        if corner is not None:
            x, y = paste_pos(corner, (pw, ph), w, h, margins)
            composite(frame, wm, x, y)
        composite(frame, pill, pill_x, pill_y)
        writer.write(frame)
        if corner is not None:
            if not truth["segments"] or truth["segments"][-1]["corner"] != corner:
                truth["segments"].append({"corner": corner,
                                          "f0": f, "f1": total,
                                          "patch": [x, y, pw, ph]})
    writer.release()
    tp = [pill_x, pill_y, qw, qh]
    truth["pill"] = tp
    return truth


def main() -> None:
    jobs = [
        # P0-1 竖屏两种缩放假设
        ("v_A_absolute", (720, 1280), dict(nickname="@娥酱", scale_mode="absolute")),
        ("v_B_width", (720, 1280), dict(nickname="@娥酱", scale_mode="width")),
        # P1-5 1080p 横屏（按宽等比）
        ("h_1080", (1920, 1080), dict(nickname="@娥酱", scale_mode="width")),
        # P0-2 亮背景压力（1280x720，大面积浅色）
        ("h_bright", (1280, 720), dict(nickname="@娥酱", scale_mode="absolute",
                                       bright=True, seconds=12)),
        # P1-6 昵称长度（单角 BR，5 秒即可；calm 背景 = 几何测试不受对抗色块干扰）
        ("nick2", (1280, 720), dict(nickname="@张三", scale_mode="absolute",
                                    seconds=5, corners=["BR"], calm=True)),
        ("nick5", (1280, 720), dict(nickname="@李思思思思", scale_mode="absolute",
                                    seconds=5, corners=["BR"], calm=True)),
        ("nick8", (1280, 720), dict(nickname="@欧阳思思思思思思思", scale_mode="absolute",
                                    seconds=5, corners=["BR"], calm=True)),
        ("nick12", (1280, 720), dict(nickname="@夏目漱石子之梦十三夜物语",
                                     scale_mode="absolute", seconds=5, corners=["BR"],
                                     calm=True)),
    ]
    truths = {}
    for name, size, kw in jobs:
        p = OUT_DIR / f"{name}.mp4"
        print(f"生成 {name} {size} ...", flush=True)
        truths[name] = gen_video(p, size, **kw)
    (OUT_DIR / "truth.json").write_text(
        json.dumps(truths, ensure_ascii=False, indent=2), encoding="utf-8")
    print("done ->", OUT_DIR)


if __name__ == "__main__":
    main()
