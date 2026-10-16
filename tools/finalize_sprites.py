"""把选定的候选图处理成最终游戏立绘。

候选是 1024x1024 的生成图，四周留白很多，而且各只大小不一。
直接塞进游戏会忽大忽小，所以统一走三步：
  1. 按 alpha 内容边界裁掉空白
  2. 放到统一比例的透明画布上（水平居中、底部对齐）—— 四只脚踩同一水平线
  3. 定高缩放 + 转 WebP

这三步和早先 tools/make_sprites.py 的 normalize() 完全一致，
保证输出规格相同（高 640、比例 0.92），游戏侧不用改任何布局。

用法：
    py -3 tools/finalize_sprites.py            # 按 PICKS 出图
    py -3 tools/finalize_sprites.py --preview  # 只出对照预览
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_sprites as M  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
CAND = ROOT / "assets" / "cand"
OUT = ROOT / "assets" / "opt"

# 角色 id -> 选定的候选文件名
# 选型标准：四只的视角要统一（正面偏四分之三、两只眼睛都露出来、能看到全身），
# 单个看好看但角度突兀的会破坏"四只像一套"的感觉。
PICKS: dict[str, str] = {
    "unicorn_bubble": "bubble_v3",   # 正面偏 3/4，两眼清晰，微笑弧线嘴
    "unicorn_cloudy": "cloudy_v2",   # 同角度，彩虹刺绣可见
    "unicorn_peachy": "peachy_v2",   # 同角度，白鬃毛蓬松，六角宝石在侧面
    "unicorn_dreamy": "dreamy_v3",   # 同角度，粉鬃毛，菱形宝石
}


def load(p: Path):
    im = cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_UNCHANGED)
    if im is None:
        return None
    if im.shape[2] == 3:                      # 没有 alpha 就补成全不透明
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGBA)
    return im


def tighten_alpha(rgba, erode_px: int = 2, feather: int = 3):
    """收紧 alpha 边缘，抹掉带色的半透明光晕。

    生成图边缘常残留一圈彩色的半透明像素 —— 那是模型把背景的渐变光晕
    压进了 alpha 里。放在绿色草地上会显出一圈脏边。
    做法：腐蚀 alpha 把最外层那圈切掉，再轻微羽化，避免留下硬锯齿。
    只动 alpha，不碰 RGB，所以主体内部的颜色不受影响。
    """
    a = rgba[:, :, 3]
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode_px * 2 + 1,) * 2)
    a2 = cv2.erode(a, k, iterations=1)
    if feather > 0:
        a2 = cv2.GaussianBlur(a2, (feather * 2 + 1,) * 2, 0)
    out = rgba.copy()
    out[:, :, 3] = np.minimum(a, a2)     # 只收不放，避免羽化把边缘又扩出去
    return out


def finalize(rgba):
    """收边 -> 裁白边 -> 统一比例画布 -> 定高缩放。与 make_sprites.normalize 同规格。"""
    rgba = tighten_alpha(rgba)

    a = rgba[:, :, 3]
    ys, xs = np.where(a > 24)
    if len(ys) == 0:
        return rgba
    pad = 4
    x0, y0 = max(0, xs.min() - pad), max(0, ys.min() - pad)
    x1, y1 = min(rgba.shape[1], xs.max() + 1 + pad), min(rgba.shape[0], ys.max() + 1 + pad)
    crop = rgba[y0:y1, x0:x1]

    ch, cw = crop.shape[:2]
    tw = max(cw, int(round(ch * M.CANVAS_ASPECT)))
    canvas = np.zeros((ch, tw, 4), np.uint8)
    ox = (tw - cw) // 2
    canvas[:, ox:ox + cw] = crop

    scale = M.OUT_HEIGHT / ch
    return cv2.resize(canvas, (int(round(tw * scale)), M.OUT_HEIGHT),
                      interpolation=cv2.INTER_AREA)


def find_candidate(stem: str, cand_name: str) -> Path | None:
    """按候选名（如 bubble_v3）找到 assets/cand/<name>/<name>.png。

    注意角色 id 是 unicorn_bubble，而候选目录名是 bubble_v3（少一层前缀），
    所以不能拿 id 直接拼 glob —— 直接按候选名定位。
    """
    if not cand_name:
        return None
    for d in CAND.glob(f"{cand_name}*"):
        f = d / f"{d.name}.png"
        if d.is_dir() and f.exists():
            return f
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preview", action="store_true")
    args = ap.parse_args()

    if not PICKS:
        # 还没人工/评审定选，列出全部候选供参考
        print("PICKS 为空 —— 尚未确定选型。当前可用候选：")
        for d in sorted(CAND.iterdir()):
            if d.is_dir() and (d / f"{d.name}.png").exists():
                print(f"  {d.name}")
        raise SystemExit(0)

    OUT.mkdir(parents=True, exist_ok=True)
    tiles = []
    print(f"{'角色':10s} {'候选':18s} {'输出':>12s} {'KB':>6s}")
    print("-" * 52)
    for stem, cand_name in PICKS.items():
        src = find_candidate(stem, cand_name)
        if src is None:
            print(f"{stem:10s} 找不到候选 {cand_name}")
            continue
        im = load(src)
        if im is None:
            print(f"{stem:10s} {src.name:18s} 读取失败")
            continue
        out = finalize(im)
        dest = OUT / f"{stem}.webp"
        ok, buf = cv2.imencode(".webp", out, [cv2.IMWRITE_WEBP_QUALITY, 88])
        if ok:
            dest.write_bytes(buf.tobytes())
        print(f"{stem:10s} {src.name:18s} {out.shape[1]:5d}x{out.shape[0]:<5d} "
              f"{dest.stat().st_size // 1024:5d}")

        if args.preview:
            cell = 320
            h, w = out.shape[:2]
            s = cell / max(h, w)
            sm = cv2.resize(out, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
            cv = np.zeros((cell, cell, 4), np.uint8)
            oy, ox = (cell - sm.shape[0]) // 2, (cell - sm.shape[1]) // 2
            cv[oy:oy + sm.shape[0], ox:ox + sm.shape[1]] = sm
            al = cv[:, :, 3:4].astype(np.float32) / 255.0
            comp = (cv[:, :, :3].astype(np.float32) * al +
                    M.checkerboard(cell, cell).astype(np.float32) * (1 - al)).astype(np.uint8)
            cv2.putText(comp, stem.replace("unicorn_", ""), (8, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (30, 30, 80), 2, cv2.LINE_AA)
            tiles.append(comp)

    if tiles:
        sheet = np.hstack(tiles)
        dest = ROOT / "_research" / "final_sprites.jpg"
        M.imwrite_u(dest, sheet)
        print(f"\n预览: {dest}  ({sheet.shape[1]}x{sheet.shape[0]})")


if __name__ == "__main__":
    main()
