"""用「边缘锐利度」分离毛绒本体与柔和投影。

为什么换这个方法：先前假设"毛绒有纹理所以局部方差高、投影平滑所以方差低"，
但实测两者方差的**中位数都是 1.6**，根本分不开（tools/tune_shadow.py 有记录）。

改用真正存在的差异：影棚商品图里
  - 本体的轮廓是**锐边**（与背景的对比强，梯度高）
  - 投影是**柔边**（向背景渐隐，梯度低）

做法：把图像按梯度切成"可通行"区域，再做连通域分析。
本体内部的平坦区与本体核心连通；投影被本体轮廓那道锐边挡住，
与核心不连通 —— 于是能整块剔除，而不是只削掉边缘一圈。

输出一组阈值的对照图，供肉眼选定。
"""
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_sprites as M  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")


def gradient_mag(gray: np.ndarray) -> np.ndarray:
    """Sobel 梯度幅值。在 float 上算，避免 uint8 溢出截断。"""
    g = gray.astype(np.float32)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    return cv2.magnitude(gx, gy)


def clean_by_edges(fg: np.ndarray, gray: np.ndarray,
                   grad_thr: float = 25.0, core_iter: int = 6) -> np.ndarray:
    """只保留"经由柔和区域与本体核心连通"的前景。"""
    mag = gradient_mag(gray)
    passable = (mag <= grad_thr).astype(np.uint8)

    # 本体核心：把前景使劲腐蚀，剩下的必然是本体
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    core = cv2.erode(fg, k, iterations=core_iter)
    if core.sum() == 0:            # 腐蚀过头了，退而求其次
        core = cv2.erode(fg, k, iterations=max(1, core_iter // 3))
    if core.sum() == 0:
        return fg

    n, lab = cv2.connectedComponents(passable, 8)
    keep = set(np.unique(lab[core > 0]).tolist()) - {0}
    if not keep:
        return fg
    return (np.isin(lab, list(keep)) & (fg > 0)).astype(np.uint8)


def main() -> None:
    picks = [
        ("p20_20220923110846.jpg", "bubble 米白大玩偶"),
        ("p31_20220125130513.jpg", "dreamy 深灰亮片"),
    ]
    thrs = [8, 15, 25, 40]

    for name, note in picks:
        bgr = M.imread_u(M.SRC / name)
        if bgr is None:
            print(f"{name} 读不到")
            continue
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

        fg = M.grabcut_foreground(bgr)
        fg = M.break_thin_links(fg)
        fg, _, _ = M.keep_largest(fg)
        fg = M.fill_holes(fg)
        fg0 = M.drop_shadow(fg, bgr)      # 先做现有的亮度阈值清理

        print(f"\n=== {note} ===")
        print(f"{'梯度阈值':>8s} {'前景%':>8s} {'较上一步减少':>12s}")
        tiles = []
        base_ratio = (fg0 > 0).mean()
        for thr in thrs:
            out = clean_by_edges(fg0, gray, grad_thr=thr)
            r = (out > 0).mean()
            print(f"{thr:8d} {r*100:7.2f}% {int((fg0>0).sum()-(out>0).sum()):12d}")

            alpha = (out * 255).astype(np.uint8)
            alpha = cv2.GaussianBlur(alpha, (5, 5), 0)
            rgba = cv2.cvtColor(bgr, cv2.COLOR_BGR2BGRA)
            rgba[:, :, 3] = alpha
            tiles.append((f"grad<={thr}", M.normalize(rgba)))

        # 加一张"未做梯度清理"的做对照
        alpha = (fg0 * 255).astype(np.uint8)
        alpha = cv2.GaussianBlur(alpha, (5, 5), 0)
        rgba = cv2.cvtColor(bgr, cv2.COLOR_BGR2BGRA)
        rgba[:, :, 3] = alpha
        tiles.insert(0, ("基线(无梯度清理)", M.normalize(rgba)))

        dest = M.ROOT / "_research" / f"edge_{name[:3]}.jpg"
        M.contact_sheet(tiles, dest, cell=300, cols=5)
        print(f"对照图: {dest}")


if __name__ == "__main__":
    main()
