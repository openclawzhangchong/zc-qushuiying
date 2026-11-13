# -*- coding: utf-8 -*-
"""jimeng_detect 单元测试（零依赖，pytest 兼容，也可直接 python 运行）。

覆盖交接文档 P2 清单要求的场景：
- 昵称 2/5/8/12 字：擦除框宽度随昵称增长且完整包含
- 四角各位置：BR/BL/TR/TL 窗口都能定位
- 无 logo 时不误报：纯背景返回 None
- 笔画断裂成 3~4 块：wordmark 仍完整收集
- 亮背景守卫：近白背景不产生假命中
- 药丸三假设并集：16:9 行为不变，竖屏覆盖两种缩放假设
- 非 16:9 兜底区域：角部锚定框盖住两种缩放假设下的 logo
- 昵称链跨洞续链：同行空洞跨过、异行亮点挡住

运行方式（任选其一）::

    python tests/test_jimeng_detect.py
    pytest tests/test_jimeng_detect.py

依赖：numpy / opencv-python；PIL + 任一中文字体仅渲染字形用（缺失则跳过
渲染类用例）。上游 remove_ai_watermarks 缺失时以桩模块替代（检测逻辑不
依赖它）；erase_engine 依赖完整上游，缺失时相关用例自动跳过。
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
_REPO = Path(__file__).resolve().parent.parent

# jimeng_detect 顶部 import 了上游 video_decode；单测环境没有上游时打桩
try:
    import remove_ai_watermarks.video_decode  # noqa: F401
except ImportError:
    _pkg = types.ModuleType("remove_ai_watermarks")
    _vd = types.ModuleType("remove_ai_watermarks.video_decode")
    _vd.open_video_capture = lambda *a, **k: None
    _pkg.video_decode = _vd
    sys.modules["remove_ai_watermarks"] = _pkg
    sys.modules["remove_ai_watermarks.video_decode"] = _vd

sys.path.insert(0, str(_REPO / "webui"))

import numpy as np  # noqa: E402
import cv2  # noqa: E402

import jimeng_detect as jd  # noqa: E402

try:
    import erase_engine  # noqa: E402

    HAVE_ERASE_ENGINE = True
except Exception:  # noqa: BLE001  上游未部署时跳过 erase_engine 相关用例
    HAVE_ERASE_ENGINE = False

# ---------------------------------------------------------------- 渲染工具

_FONT_PATH: str | None = None
_FONT_CHECKED = False


def _font_path() -> str | None:
    global _FONT_PATH, _FONT_CHECKED
    if not _FONT_CHECKED:
        _FONT_CHECKED = True
        for cand in (r"C:\Windows\Fonts\simhei.ttf", r"C:\Windows\Fonts\msyh.ttc",
                     "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
                     "/System/Library/Fonts/PingFang.ttc"):
            if Path(cand).exists():
                _FONT_PATH = cand
                break
    return _FONT_PATH


def _draw_watermark(canvas: np.ndarray, x: int, y: int, nickname: str,
                    word_h: int = 28, nick_h: int = 16) -> tuple[int, int, int, int]:
    """在 BGR 画布上画 [箭头]即梦 / @昵称，返回字形像素的联合包围盒（不含箭头）。"""
    from PIL import Image, ImageDraw, ImageFont

    fh, fw = canvas.shape[:2]
    img = Image.new("RGB", (fw, fh))
    img.paste(Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)))
    d = ImageDraw.Draw(img)
    f_word = ImageFont.truetype(_font_path(), round(word_h * 1.25))
    f_nick = ImageFont.truetype(_font_path(), round(nick_h * 1.25))
    wb = d.textbbox((0, 0), "即梦", font=f_word)
    nb = d.textbbox((0, 0), nickname, font=f_nick)
    wx, wy = x + 30 - wb[0], y - wb[1]
    d.text((wx, wy), "即梦", font=f_word, fill=(250, 250, 250))
    nx, ny = x + 30 - nb[0], y + word_h + 10 - nb[1]
    d.text((nx, ny), nickname, font=f_nick, fill=(250, 250, 250))
    out = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
    # 彩色渐变箭头（不在白色掩码里，模拟真实设计）
    for i in range(24):
        t = i / 23.0
        color = (int(40 + 180 * t), 120, int(255 - 100 * t))
        cv2.line(out, (x + i, y + word_h // 2 - 7), (x + i, y + word_h // 2 + 7),
                 color, 2)
    canvas[:] = out
    gx1 = min(wx, nx)
    gy1 = min(wy, ny)
    gx2 = max(wx + (wb[2] - wb[0]), nx + (nb[2] - nb[0]))
    gy2 = max(wy + (wb[3] - wb[1]), ny + (nb[3] - nb[1]))
    return (gx1, gy1, gx2 - gx1, gy2 - gy1)


def _gradient_canvas(size: tuple[int, int], seed: int = 11) -> np.ndarray:
    w, h = size
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    base = np.dstack([80 + 60 * (xx / w), 70 + 70 * (yy / h),
                      100 + 40 * ((xx + yy) / (w + h))])
    return np.clip(base + rng.normal(0, 5, base.shape), 0, 255).astype(np.uint8)


def _corner_canvas(corner: str, size: tuple[int, int] = (1280, 720),
                   nickname: str = "@娥酱", k: float = 1.0
                   ) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """水印贴在指定角（尺寸与边距按 k 缩放），返回 (画布, 水印字形 bbox)。"""
    w, h = size
    canvas = _gradient_canvas(size)
    word_h = max(8, round(28 * k))
    nick_h = max(6, round(16 * k))
    if corner == "BR":
        ox, oy = round(w - 76 * k - 200 * k), round(h - 90 * k - 70 * k)
    elif corner == "BL":
        ox, oy = round(55 * k), round(h - 90 * k - 70 * k)
    elif corner == "TR":
        ox, oy = round(w - 62 * k - 200 * k), round(61 * k)
    else:  # TL：避开药丸常驻位，专测 TL 窗口
        ox, oy = 200, 120
    glyph_bbox = _draw_watermark(canvas, ox, oy, nickname, word_h, nick_h)
    return canvas, glyph_bbox


def _contains(big, small, tol=3):
    return (big[0] <= small[0] + tol and big[1] <= small[1] + tol
            and big[0] + big[2] >= small[0] + small[2] - tol
            and big[1] + big[3] >= small[1] + small[3] - tol)


def _detect_frame(canvas: np.ndarray, corner: str):
    """对整帧的指定角窗口跑 detect_in_window，返回窗口绝对坐标的 bbox。"""
    h, w = canvas.shape[:2]
    wx, wy, ww, wh = jd._window(w, h, corner)
    hit = jd.detect_in_window(canvas[wy:wy + wh, wx:wx + ww], corner, frame_width=w)
    if hit is None:
        return None
    return (hit.bbox[0] + wx, hit.bbox[1] + wy, hit.bbox[2], hit.bbox[3])


# ---------------------------------------------------------------- 用例

def test_adaptive_threshold_math():
    """帧宽 <1280 时阈值按比例放宽且有下限；>=1280 时保持原值。"""
    assert jd._adaptive(18, 1280, lo=11) == 18
    assert jd._adaptive(18, 1920, lo=11) == 18      # 不因放大而收紧
    assert jd._adaptive(18, 720, lo=11) == 11       # 18*720/1280=10.1 -> 下限 11
    assert jd._adaptive(110, 720, lo=30) == 62
    assert jd._adaptive(110, 640, lo=30) == 55


def test_no_logo_no_false_positive():
    """纯背景（渐变+噪声）四角都应返回 None。"""
    canvas = _gradient_canvas((1280, 720), seed=3)
    for corner in jd.CORNERS:
        assert _detect_frame(canvas, corner) is None, corner


def test_bright_background_guard():
    """大面积近白背景：白占比超阈值时应返回 None 而不是给出假命中。"""
    canvas = np.full((720, 1280, 3), 235, np.uint8)
    canvas += np.random.default_rng(5).integers(-4, 5, canvas.shape).astype(np.uint8)
    for corner in jd.CORNERS:
        assert _detect_frame(canvas, corner) is None, corner


def test_wordmark_broken_strokes_collected():
    """小字号「即梦」在白色掩码里笔画断裂成多块时，wordmark 仍完整收集。"""
    canvas, gbox = _corner_canvas("BR")
    hit = _detect_frame(canvas, "BR")
    assert hit is not None
    assert _contains(hit, gbox), f"bbox={hit} 应包含字形 {gbox}"


def test_four_corners_detected():
    """BR / BL / TR / TL 四个角都能定位且框住水印字形。"""
    for corner in ("BR", "BL", "TR", "TL"):
        canvas, gbox = _corner_canvas(corner)
        hit = _detect_frame(canvas, corner)
        assert hit is not None, f"{corner} 未检出"
        assert _contains(hit, gbox), f"{corner} bbox={hit} 应包含 {gbox}"


def test_nickname_length_adaptive():
    """昵称 2/5/8/12 字：框完整包含昵称行，宽度随字数增长。"""
    widths = []
    for nick in ("@张三", "@李思思思思", "@欧阳思思思思思思思", "@夏目漱石子之梦十三夜物语"):
        canvas, gbox = _corner_canvas("BR", nickname=nick)
        hit = _detect_frame(canvas, "BR")
        assert hit is not None, f"{nick} 未检出"
        assert _contains(hit, gbox), f"{nick} bbox={hit} 应包含 {gbox}"
        widths.append(hit[2])
    assert widths == sorted(widths), f"框宽应随昵称增长: {widths}"
    assert widths[3] - widths[0] >= 8 * 12, f"12 字昵称框宽增长不足: {widths}"


def test_nickname_chain_resync_same_row():
    """昵称链跨洞续链：同行空洞（简单字被丢）后仍继续收集。"""
    glyphs = [
        {"x": 0, "y": 100, "w": 18, "h": 18, "area": 200, "strong": True},    # @
        {"x": 22, "y": 100, "w": 16, "h": 18, "area": 180, "strong": True},   # 字1
        # ---- 空洞 62px（3 个简单字被丢）----
        {"x": 100, "y": 101, "w": 34, "h": 18, "area": 380, "strong": True},  # 字5+字6
        {"x": 140, "y": 100, "w": 16, "h": 18, "area": 170, "strong": True},  # 字7
    ]
    wordmark = [{"x": 0, "y": 40, "w": 60, "h": 30, "area": 900, "strong": True}]
    chain = jd._find_nickname(glyphs, wordmark, 1280)
    assert chain, "链不应为空"
    assert jd._union(chain)[2] >= 156, "链应延伸到最后一个字形"


def test_nickname_chain_rejects_other_row():
    """链不吸纳异行亮点：真实视频里 @娥酱 右侧 121px 的碎石反光（不同行）。"""
    glyphs = [
        {"x": 62, "y": 197, "w": 16, "h": 17, "area": 165, "strong": True},   # @
        {"x": 81, "y": 195, "w": 20, "h": 23, "area": 210, "strong": True},   # 娥
        {"x": 104, "y": 195, "w": 20, "h": 30, "area": 292, "strong": True},  # 酱
        # 碎石反光：昵称行上方 1.3 个字高，间距 121px
        {"x": 245, "y": 162, "w": 9, "h": 13, "area": 49, "strong": False},
        {"x": 276, "y": 167, "w": 22, "h": 11, "area": 59, "strong": False},
    ]
    wordmark = [{"x": 117, "y": 142, "w": 78, "h": 31, "area": 900, "strong": True}]
    chain = jd._find_nickname(glyphs, wordmark, 1280)
    assert jd._union(chain) == (62, 195, 62, 30), f"碎石不应入链: {jd._union(chain)}"


def test_wordmark_ignores_weak_glyphs():
    """弱档字形（面积/填充率不达标）不能当 wordmark 候选。"""
    glyphs = [
        {"x": 0, "y": 100, "w": 30, "h": 30, "area": 90, "strong": False},
        {"x": 40, "y": 100, "w": 30, "h": 30, "area": 95, "strong": False},
    ]
    assert jd._find_wordmark(glyphs, 1280) is None


def test_pill_region_16_9_unchanged():
    """16:9 上药丸并集必须完整包含旧版框，且面积增长有界。

    1280×720 上与旧版逐位一致；更大的 16:9 只因「绝对尺寸」假设框
    （6,8,80,40）在左上方向多出 ≤3px；更小的 16:9 多擦的也是药丸角，
    面积增长不超过 2 倍（宁可多擦，不能漏水印）。
    """
    for w, h in ((1280, 720), (1920, 1080), (2560, 1440), (960, 540)):
        rx, ry, rw, rh = jd.PILL_REL
        old = (round(w * rx), round(h * ry), round(w * rw), round(h * rh))
        got = jd.pill_region(w, h)
        assert _contains(got, old, tol=0), (w, h, got, old)
        if (w, h) == (1280, 720):
            assert got == old, (w, h, got, old)
        assert got[2] * got[3] <= old[2] * old[3] * 2, (w, h, got, old)


def test_pill_region_vertical_covers_hypotheses():
    """竖屏上药丸框必须同时盖住「绝对尺寸」与「按宽缩放」两种假设。"""
    w, h = 720, 1280
    pb = jd.pill_region(w, h)
    # 假设 A：保持 720p 设计的绝对尺寸 (6,8,80,40)
    assert _contains(pb, (6, 8, 80, 40), tol=0), f"{pb} 应覆盖绝对尺寸药丸"
    # 假设 B：随宽度等比缩放 (3.4, 4.5, 45, 22.5)
    assert _contains(pb, (3, 4, 45, 23), tol=1), f"{pb} 应覆盖按宽缩放药丸"
    # 原始 PILL_REL 缩放（被帧高拉大的框）也仍在并集内
    rx, ry, rw, rh = jd.PILL_REL
    old = (round(w * rx), round(h * ry), round(w * rw), round(h * rh))
    assert _contains(pb, old, tol=0)


def test_jimeng_regions_16_9_unchanged():
    """16:9 兜底区域仍按相对坐标缩放（行为不变）。"""
    if not HAVE_ERASE_ENGINE:
        print("-> 跳过（上游未部署）")
        return
    got = erase_engine.jimeng_regions(1280, 720)
    expect = [
        (round(1280 * x), round(720 * y), round(1280 * w), round(720 * hh))
        for x, y, w, hh in erase_engine.JIMENG_REGIONS_RELATIVE
    ]
    assert got == expect


def test_jimeng_regions_vertical_covers_logo():
    """非 16:9 兜底区域：角部锚定框必须盖住两种缩放假设下的 logo 位置。"""
    if not HAVE_ERASE_ENGINE:
        print("-> 跳过（上游未部署）")
        return
    w, h = 720, 1280
    pill, br, bl, tr = erase_engine.jimeng_regions(w, h)
    # 假设 A（绝对尺寸）：用 1280x720 实测设计框（1064,551,143,80 / 55,614,148,95 /
    # 1081,61,137,82）按角锚定到竖屏
    a_br = (w - 76 - 143, h - 169, 143, 80)
    # 假设 B（按宽缩放，k=0.5625）：块 80x45，距右边 43 / 距底 51
    k = w / 1280.0
    b_br = (round(w - 76 * k - 143 * k), round(h - 90 * k - 80 * k),
            round(143 * k), round(80 * k))
    assert _contains(br, a_br, tol=0), f"BR 兜底 {br} 应盖住 {a_br}"
    assert _contains(br, b_br, tol=2), f"BR 兜底 {br} 应盖住 {b_br}"
    a_bl = (55, h - 106, 148, 95)
    b_bl = (round(55 * k), round(h - 90 * k - 80 * k), round(148 * k), round(95 * k))
    assert _contains(bl, a_bl, tol=0), f"BL 兜底 {bl} 应盖住 {a_bl}"
    assert _contains(bl, b_bl, tol=2), f"BL 兜底 {bl} 应盖住 {b_bl}"
    a_tr = (w - 62 - 137, 61, 137, 82)
    b_tr = (round(w - 62 * k - 137 * k), round(61 * k), round(137 * k), round(82 * k))
    assert _contains(tr, a_tr, tol=0), f"TR 兜底 {tr} 应盖住 {a_tr}"
    assert _contains(tr, b_tr, tol=2), f"TR 兜底 {tr} 应盖住 {b_tr}"
    assert _contains(pill, (6, 8, 80, 40), tol=0)


def test_vertical_scaled_watermark_detected():
    """720 宽竖屏 + 按宽缩放的水印（字形高约 16px）：自适应阈值应能检出。

    这是 v_B_width 合成素材的静态等价用例：若叠加层随帧宽缩放，
    1280 标定的 h>=18 阈值会漏掉 16px 的 wordmark。
    """
    canvas, gbox = _corner_canvas("BR", size=(720, 1280), k=720 / 1280.0)
    hit = _detect_frame(canvas, "BR")
    assert hit is not None, "竖屏按宽缩放的水印应能检出"
    assert _contains(hit, gbox, tol=4), f"bbox={hit} 应包含 {gbox}"


# ---------------------------------------------------------------- 运行器

def main() -> None:
    fns = [(n, f) for n, f in sorted(globals().items())
           if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in fns:
        try:
            fn()
            print(f"PASS  {name}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL  {name}: {e}")
        except RuntimeError as e:
            print(f"SKIP  {name}: {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
