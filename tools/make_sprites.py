"""从 NICI 官方商品图生成游戏立绘（透明底）。

为什么要这条路线：要 100% 还原官方造型，最准的素材就是官方商品图本身，
而不是让生成模型去模仿。难点在于**白底 + 白色毛绒**几乎不可分，
所以不能靠颜色阈值（实测前景只剩 0.3%，玩偶被整块吃掉），
必须靠 GrabCut 做颜色建模 + 迭代。

三个关键处理：
1. GrabCut 抠图（容差法在白底白玩偶上必败）
2. 投影清除：官方图底部有柔和投影，抠下来会变成白色糊块。
   用**局部方差**区分 —— 毛绒有细密纹理（方差高），投影是平滑的（方差低）。
   再限定在下方区域，避免误伤同样又白又平滑的眼片。
3. 几何归一：裁到内容边界 → 统一画布比例 → 底部对齐 → 定高缩放。
   与 tools/optimize_assets.py 输出规格一致，游戏侧无需改布局。

用法：
    py -3 tools/make_sprites.py            # 处理候选并出联览图
    py -3 tools/make_sprites.py --final    # 按 FINAL 选定输出 assets/opt/
"""
from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

import cv2
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "_research" / "products"
WORK = ROOT / "_research" / "sprites"
FINAL_OUT = ROOT / "assets" / "opt"

CANVAS_ASPECT = 0.92   # 与 optimize_assets.py 保持一致
OUT_HEIGHT = 640


# ---------------------------------------------------------------- 读写（绕开中文路径）

def imread_u(path: Path):
    try:
        buf = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return None
    return cv2.imdecode(buf, cv2.IMREAD_COLOR) if buf.size else None


def imwrite_u(path: Path, img) -> bool:
    ext = Path(path).suffix or ".png"
    ok, buf = cv2.imencode(ext, img)
    if ok:
        with open(path, "wb") as f:
            f.write(buf.tobytes())
    return ok


# ---------------------------------------------------------------- 抠图

def grabcut_foreground(bgr, margin: int = 6, iters: int = 5) -> np.ndarray | None:
    """GrabCut 前景掩码。四边一圈标为确定背景，其余先当"可能前景"。"""
    h, w = bgr.shape[:2]
    mask = np.full((h, w), cv2.GC_PR_FGD, np.uint8)
    mask[:margin, :] = cv2.GC_BGD
    mask[-margin:, :] = cv2.GC_BGD
    mask[:, :margin] = cv2.GC_BGD
    mask[:, -margin:] = cv2.GC_BGD

    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(bgr, mask, None, bgd, fgd, iters, cv2.GC_INIT_WITH_MASK)
    except cv2.error:
        return None
    return np.isin(mask, [cv2.GC_FGD, cv2.GC_PR_FGD]).astype(np.uint8)


def break_thin_links(fg: np.ndarray, k: int = 3) -> np.ndarray:
    """形态学开运算，切断钥匙扣链条这类细连接。

    只腐蚀不掉玩偶本体 —— 链条约 2~4px，玩偶最细处（角、耳）也远粗于此。
    开运算后再闭运算补回开运算削掉的边缘。
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    opened = cv2.morphologyEx(fg, cv2.MORPH_OPEN, kernel, iterations=1)
    return cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=2)


def keep_largest(fg: np.ndarray, min_frac: float = 0.02):
    """只保留最大的连通块。返回 (掩码, 大块数量, 最大块占比)。"""
    n, lab, stats, _ = cv2.connectedComponentsWithStats(fg, 8)
    if n <= 1:
        return fg, 0, 0.0
    h, w = fg.shape
    areas = sorted(((stats[i, cv2.CC_STAT_AREA], i) for i in range(1, n)), reverse=True)
    big = [(a, i) for a, i in areas if a > h * w * min_frac]
    if not big:
        return fg, 0, 0.0
    keep = big[0][1]
    return (lab == keep).astype(np.uint8), len(big), big[0][0] / (h * w)


def fill_holes(fg: np.ndarray) -> np.ndarray:
    """填充前景内部孔洞（白肚皮、眼片容易被误判成背景）。"""
    h, w = fg.shape
    ff = fg.copy()
    m = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(ff, m, (0, 0), 1)
    return np.clip(fg + (ff == 0).astype(np.uint8), 0, 1)


def drop_shadow(fg: np.ndarray, bgr: np.ndarray,
                std_thr: float = 25.0, gray_thr: int = 235,
                sat_thr: int = 30, bottom_frac: float = 0.45) -> np.ndarray:
    """去掉底部的柔和投影。

    实测结论（tools/tune_shadow.py）：本体与投影的**局部方差几乎相同**
    （中位都是 1.6），方差分不开两者 —— 真正起作用的是亮度阈值。
    所以 std_thr 给得较宽（不让它误伤绒布纹理），靠 gray_thr 做主要判据；
    sat_thr 用来保住粉色鬃毛/嘴部这类有饱和度的部位。

    只在下半个画面动手：眼片同样是"又白又平滑"，靠区域限制保住它们。
    """
    h, w = fg.shape
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    sat = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[:, :, 1]

    mean = cv2.boxFilter(gray, -1, (9, 9))
    sq = cv2.boxFilter(gray * gray, -1, (9, 9))
    std = np.sqrt(np.maximum(sq - mean * mean, 0))

    ys, xs = np.where(fg > 0)
    if len(ys) == 0:
        return fg
    y0, y1 = ys.min(), ys.max()
    cut = y0 + int((y1 - y0) * (1 - bottom_frac))

    region = np.zeros((h, w), bool)
    region[cut:, :] = True

    shadow = region & (fg > 0) & (std < std_thr) & (gray > gray_thr) & (sat < sat_thr)
    out = fg.copy()
    out[shadow] = 0
    # 注意：这里**不能**再调 fill_holes —— 如果投影块被前景包住，
    # 补洞会把它当成"内部空洞"重新填回来，正好抵消掉删除效果。
    return out


def matte(path: Path):
    """完整抠图。返回 (RGBA, 诊断信息)。"""
    bgr = imread_u(path)
    if bgr is None:
        return None, {"error": "读取失败"}

    fg = grabcut_foreground(bgr)
    if fg is None or fg.sum() == 0:
        return None, {"error": "GrabCut 失败"}

    fg = break_thin_links(fg)
    fg, big_count, big_frac = keep_largest(fg)
    fg = fill_holes(fg)
    fg = drop_shadow(fg, bgr)

    alpha = (fg * 255).astype(np.uint8)
    alpha = cv2.GaussianBlur(alpha, (5, 5), 0)

    ys, xs = np.where(alpha > 24)
    if len(ys) == 0:
        return None, {"error": "抠完是空的"}
    box = (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1)

    rgba = cv2.cvtColor(bgr, cv2.COLOR_BGR2BGRA)
    rgba[:, :, 3] = alpha
    info = {
        "fg_ratio": float((alpha > 127).mean()),
        "big_blocks": big_count,
        "big_frac": big_frac,
        "edge_px": int(np.count_nonzero((alpha > 20) & (alpha < 235))),
        "bbox": box,
    }
    return rgba, info


# ---------------------------------------------------------------- 几何归一

def normalize(rgba, aspect: float = CANVAS_ASPECT, height: int = OUT_HEIGHT):
    """裁到内容边界 → 放到统一比例的透明画布（水平居中、底部对齐）→ 定高缩放。"""
    a = rgba[:, :, 3]
    ys, xs = np.where(a > 24)
    if len(ys) == 0:
        return rgba
    pad = 6
    x0 = max(0, xs.min() - pad)
    y0 = max(0, ys.min() - pad)
    x1 = min(rgba.shape[1], xs.max() + 1 + pad)
    y1 = min(rgba.shape[0], ys.max() + 1 + pad)
    crop = rgba[y0:y1, x0:x1]

    ch, cw = crop.shape[:2]
    tw = max(cw, int(round(ch * aspect)))
    canvas = np.zeros((ch, tw, 4), np.uint8)
    ox = (tw - cw) // 2
    canvas[:, ox:ox + cw] = crop

    scale = height / ch
    return cv2.resize(canvas, (int(round(tw * scale)), height), interpolation=cv2.INTER_AREA)


# ---------------------------------------------------------------- 联览图

def checkerboard(h: int, w: int, sq: int = 20) -> np.ndarray:
    tile = np.zeros((h, w, 3), np.uint8)
    yy, xx = np.mgrid[0:h, 0:w]
    tile[(yy // sq + xx // sq) % 2 == 0] = 205
    tile[(yy // sq + xx // sq) % 2 == 1] = 245
    return tile


def contact_sheet(items, dest: Path, cell: int = 260, cols: int = 6):
    """items: [(label, RGBA)]，输出棋盘底联览图。"""
    if not items:
        return None
    rows = (len(items) + cols - 1) // cols
    label_h = 18
    W = cols * cell
    H = rows * (cell + label_h)
    sheet = np.full((H, W, 3), 250, np.uint8)
    for i, (label, rgba) in enumerate(items):
        r, c = divmod(i, cols)
        y, x = r * (cell + label_h), c * cell
        im = cv2.resize(rgba, (cell, cell), interpolation=cv2.INTER_AREA)
        a = im[:, :, 3:4].astype(np.float32) / 255.0
        # 等比缩放后可能变形，这里只做展示用
        comp = (im[:, :, :3].astype(np.float32) * a +
                checkerboard(cell, cell).astype(np.float32) * (1 - a)).astype(np.uint8)
        sheet[y:y + cell, x:x + cell] = comp
        cv2.putText(sheet, label, (x + 4, y + cell + 13),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (60, 60, 90), 1, cv2.LINE_AA)
    imwrite_u(dest, sheet)
    return dest


# ---------------------------------------------------------------- 候选

# 联览图里看起来是独立玩偶（非周边、非抱枕）的编号
CANDIDATES = [0, 7, 10, 18, 20, 23, 24, 27, 30, 31, 32, 35]

# ---------------------------------------------------------------- 最终选定的素材

# 游戏角色 id -> (商品图编号, 裁掉顶部比例)
#
# 编号来自 tools/batch_matte.py 打出的带编号联览图（_research/matte_all.jpg）。
# 选的都是**独立玩偶、无钥匙扣挂环、构图完整**的那几张 ——
# 钥匙扣款式顶部有金属环，游戏里独角兽站在草地上，顶个环很怪，
# 所以要么选无环的，要么把环连同一段挂链裁掉（crop_top）。
FINAL = {
    # 米白身体 + 粉鬃毛，大体型四分之三视角，姿态最好的一张
    "unicorn_bubble": (20, 0.0),
    # 米白身体 + 青蓝鬃毛，是官方线里唯一的冷色款；顶部有金属环，裁掉
    "unicorn_cloudy": (0, 0.30),
    # 粉色亮片
    "unicorn_peachy": (29, 0.0),
    # 深灰亮片 + 粉鬃毛
    "unicorn_dreamy": (30, 0.0),
}


def produce_final() -> None:
    files = files_sorted()
    FINAL_OUT.mkdir(parents=True, exist_ok=True)
    print(f"{'角色 id':20s} {'源图':26s} {'裁顶':>5s} {'前景%':>7s} {'输出':>11s}")
    print("-" * 76)
    for stem, (idx, crop_top) in FINAL.items():
        if idx >= len(files):
            print(f"{stem:20s} 编号 {idx} 越界")
            continue
        src = files[idx]
        bgr = imread_u(src)
        if bgr is None:
            print(f"{stem:20s} {src.name:26s} 读不到")
            continue
        if crop_top > 0:
            cut = int(bgr.shape[0] * crop_top)
            bgr = bgr[cut:, :]
            tmp = WORK / f"_crop_{stem}.png"
            WORK.mkdir(parents=True, exist_ok=True)
            imwrite_u(tmp, bgr)
            rgba, info = matte(tmp)
            tmp.unlink(missing_ok=True)
        else:
            rgba, info = matte(src)
        if rgba is None:
            print(f"{stem:20s} {src.name:26s} 失败: {info.get('error')}")
            continue
        out = normalize(rgba)
        dest = FINAL_OUT / f"{stem}.webp"
        ok, buf = cv2.imencode(".webp", out, [cv2.IMWRITE_WEBP_QUALITY, 88])
        if ok:
            with open(dest, "wb") as f:
                f.write(buf.tobytes())
        print(f"{stem:20s} {src.name:26s} {crop_top:5.2f} {info['fg_ratio']*100:6.1f}% "
              f"{out.shape[1]:5d}x{out.shape[0]:<5d} ({dest.stat().st_size // 1024} KB)")


def files_sorted():
    return sorted(SRC.glob("p*"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--final", action="store_true", help="按 FINAL 输出到 assets/opt/")
    args = ap.parse_args()

    files = files_sorted()
    if not files:
        raise SystemExit(f"没找到商品图：{SRC}")

    if args.final:
        produce_final()
        return

    # 试跑：处理候选并出联览图
    WORK.mkdir(parents=True, exist_ok=True)
    items, rows = [], []
    print(f"{'#':>3s} {'源图':28s} {'前景%':>7s} {'大块':>5s} {'占最大块':>9s}  说明")
    print("-" * 82)
    for idx in CANDIDATES:
        if idx >= len(files):
            continue
        src = files[idx]
        rgba, info = matte(src)
        if rgba is None:
            print(f"{idx:3d} {src.name:28s}   失败: {info.get('error')}")
            continue
        norm = normalize(rgba)
        imwrite_u(WORK / f"{idx:02d}_{src.stem}.png", norm)
        note = ""
        if info["big_blocks"] > 1:
            note = f"⚠ {info['big_blocks']} 个独立块（可能是双玩偶）"
        rows.append((idx, src.name, info, note))
        items.append((f"#{idx}", norm))
        print(f"{idx:3d} {src.name:28s} {info['fg_ratio']*100:6.1f}% "
              f"{info['big_blocks']:5d} {info['big_frac']*100:8.1f}%  {note}")

    dest = contact_sheet(items, ROOT / "_research" / "sprite_preview.jpg")
    if dest:
        print(f"\n联览图: {dest}")


if __name__ == "__main__":
    main()
