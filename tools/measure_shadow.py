"""测出残留投影在 alpha 里的确切位置与形状，据此决定裁剪量。

先前的两次方法尝试都被数据推翻（记在 tools/ 里）：
  - 局部方差分离：本体与投影的方差中位都是 1.6，不可分
  - 梯度/锐边分离：亮片与锐边同为高梯度，对亮片款会把本体打碎

所以不再猜，直接量：把 alpha 按行统计宽度与灰度，找出
"本体真正的下缘"和"投影延伸到哪里"，用数据定裁剪。
"""
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_sprites as M  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")


def analyze(pick: str) -> None:
    bgr = M.imread_u(M.SRC / pick)
    if bgr is None:
        print(f"{pick} 读不到")
        return

    fg = M.grabcut_foreground(bgr)
    fg = M.break_thin_links(fg)
    fg, _, _ = M.keep_largest(fg)
    fg = M.fill_holes(fg)
    fg = M.drop_shadow(fg, bgr)

    ys, xs = np.where(fg > 0)
    y0, y1 = ys.min(), ys.max()
    x0, x1 = xs.min(), xs.max()
    h = y1 - y0 + 1
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    print(f"\n=== {pick} ===")
    print(f"bbox: x {x0}..{x1} ({x1-x0+1}px)   y {y0}..{y1} ({h}px)")
    print(f"{'行(距底%)':>10s} {'宽度px':>8s} {'占总宽%':>9s} {'灰度中位':>9s} {'灰度p90':>8s}")
    print("-" * 54)

    # 从底部往上逐行看，找出"影子"特征：宽而亮
    for frac in np.arange(0.0, 1.01, 0.05):
        yy = int(y1 - frac * h)
        if yy < y0:
            break
        row = fg[yy] > 0
        w = int(row.sum())
        if w == 0:
            print(f"{frac*100:9.0f}% {0:8d} {0:8.0f}% {'—':>9s} {'—':>8s}")
            continue
        g = gray[yy][row]
        print(f"{frac*100:9.0f}% {w:8d} {w/(x1-x0+1)*100:8.1f}% "
              f"{np.median(g):9.0f} {np.percentile(g,90):8.0f}")

    # 水平方向：影子通常偏一侧
    print("\n按列看（距左%）被占用到的最低行：")
    for frac in np.arange(0.0, 1.01, 0.1):
        xx = int(x0 + frac * (x1 - x0))
        col = np.where(fg[:, xx] > 0)[0]
        if len(col) == 0:
            print(f"  {frac*100:5.0f}% 空")
            continue
        bottom = col.max()
        depth = (bottom - y0) / h * 100
        print(f"  {frac*100:5.0f}% 最低行距顶 {depth:5.1f}%（越过本体下缘 {depth-100:5.1f}%）")


def main() -> None:
    for p in ["p20_20220923110846.jpg", "p00_20250912154041.jpg"]:
        analyze(p)


if __name__ == "__main__":
    main()
