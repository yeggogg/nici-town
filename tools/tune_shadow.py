"""定位并调参：白色玩偶脚下方那团残留的成因。

现象：GrabCut 抠完后，白玩偶脚底下方留有一团白色糊块（官方图的接触投影）。
先测出该区域真实的灰度/局部方差分布，再扫一组阈值组合出联览图挑最优。
"""
import io
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_sprites as M  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")

# 有残留的那只（米白大玩偶）
PICK = "p20_20220923110846.jpg"


def local_std(gray: np.ndarray, k: int = 9) -> np.ndarray:
    g = gray.astype(np.float32)
    m = cv2.boxFilter(g, -1, (k, k))
    s = cv2.boxFilter(g * g, -1, (k, k))
    return np.sqrt(np.maximum(s - m * m, 0))


def main() -> None:
    bgr = M.imread_u(M.SRC / PICK)
    if bgr is None:
        raise SystemExit("读不到源图")

    fg = M.grabcut_foreground(bgr)
    fg = M.break_thin_links(fg)
    fg, _, _ = M.keep_largest(fg)
    fg = M.fill_holes(fg)

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    sat = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[:, :, 1]
    std = local_std(gray)

    ys, xs = np.where(fg > 0)
    y0, y1 = ys.min(), ys.max()
    body_cut = y0 + int((y1 - y0) * 0.62)   # 本体大致下缘

    # ① 本体区域 vs 下方残留区，各取前景像素的统计
    body = fg > 0
    body[:body_cut, :] = False              # 只留下半段
    foot_zone = (fg > 0) & (np.arange(fg.shape[0])[:, None] > y1 - (y1 - y0) * 0.22)

    def stat(mask, name):
        if mask.sum() == 0:
            print(f"  {name}: 空")
            return
        g = gray[mask]
        s = std[mask]
        sa = sat[mask]
        print(f"  {name} (n={mask.sum()}):")
        print(f"    灰度   p10={np.percentile(g,10):.0f} p50={np.percentile(g,50):.0f} p90={np.percentile(g,90):.0f}")
        print(f"    局部std p10={np.percentile(s,10):.1f} p50={np.percentile(s,50):.1f} p90={np.percentile(s,90):.1f}")
        print(f"    饱和   p50={np.percentile(sa,50):.0f} p90={np.percentile(sa,90):.0f}")

    print("像素统计（判断投影与毛绒是否能分开）")
    print(f"[bbox y {y0}..{y1}]")
    stat(body, "下半段前景")
    stat(foot_zone, "最下 22%（残留重灾区）")

    # ② 扫阈值组合
    combos = [
        (999, 255, "不处理基线"),
        (4, 200, "std<4 gray>200"),
        (8, 200, "std<8 gray>200"),
        (15, 200, "std<15 gray>200"),
        (8, 235, "std<8 gray>235"),
        (15, 235, "std<15 gray>235"),
        (25, 235, "std<25 gray>235"),
        (40, 240, "std<40 gray>240"),
    ]
    tiles = []
    print("\n阈值扫描")
    print(f"{'组合':26s} {'前景%':>7s} {'去掉的px':>9s}")
    print("-" * 48)
    for std_thr, gray_thr, note in combos:
        base = fg.copy()
        out = M.drop_shadow(base, bgr, std_thr=std_thr, gray_thr=gray_thr)
        removed = int((base > 0).sum() - (out > 0).sum())
        fg_ratio = (out > 0).mean()
        print(f"{note:26s} {fg_ratio*100:6.1f}% {removed:9d}")

        alpha = (out * 255).astype(np.uint8)
        alpha = cv2.GaussianBlur(alpha, (5, 5), 0)
        rgba = cv2.cvtColor(bgr, cv2.COLOR_BGR2BGRA)
        rgba[:, :, 3] = alpha
        tiles.append((note, M.normalize(rgba)))

    dest = M.contact_sheet(tiles, M.ROOT / "_research" / "shadow_tune.jpg", cell=300, cols=4)
    print(f"\n对照图: {dest}")


if __name__ == "__main__":
    main()
