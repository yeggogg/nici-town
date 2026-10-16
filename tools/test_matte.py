"""测试能否把官方商品图的白底抠干净。

为什么单独测这个：白色毛绒玩偶配白底是最难的抠图场景 —— 前景和背景都是白色，
单靠颜色阈值必然失败，必须靠**连通性**（背景是连成一片的，玩偶不是）。

策略：从四角 flood fill 找出与边缘连通的近白区域作为背景，
再取最大连通前景、轻微羽化边缘。附带软阴影的取舍。

用法：
    py -3 tools/test_matte.py
"""
import io
import sys
from pathlib import Path

import cv2
import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "_research" / "products"
OUT = ROOT / "_research" / "matte_test"


def imread_u(path: Path):
    """读图。cv2.imread 在 Windows 上遇到非 ASCII 路径会失败，
    所以自己读字节再 imdecode。"""
    try:
        buf = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return None
    if buf.size == 0:
        return None
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def imread_u_alpha(path: Path):
    try:
        buf = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return None
    if buf.size == 0:
        return None
    return cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)


def imwrite_u(path: Path, img) -> bool:
    """写图。同理，cv2.imwrite 也会被非 ASCII 路径卡住。"""
    ext = Path(path).suffix or ".png"
    ok, buf = cv2.imencode(ext, img)
    if not ok:
        return False
        # 通过文件对象写入，绕开路径编码问题
    with open(path, "wb") as f:
        f.write(buf.tobytes())
    return True


def matte(path: Path, tol: int = 14, feather: int = 2):
    """返回 (RGBA, 背景占比, 前景占比, 边缘像素数)。"""
    bgr = imread_u(path)
    if bgr is None:
        return None
    h, w = bgr.shape[:2]

    # 从四角 flood fill：只把"与边缘连通"的近白像素判为背景。
    # 玩偶即使也是白的，只要没和边缘连上就保得住。
    mask = np.zeros((h + 2, w + 2), np.uint8)
    flags = 4 | cv2.FLOODFILL_MASK_ONLY | (255 << 8)
    seeds = [(0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1),
             (w // 2, 0), (w // 2, h - 1), (0, h // 2), (w - 1, h // 2)]
    probe = bgr.copy()
    for seed in seeds:
        if mask[seed[1] + 1, seed[0] + 1]:
            continue  # 已被前面的种子覆盖
        cv2.floodFill(probe, mask, seed, 0,
                      (tol, tol, tol), (tol, tol, tol), flags)
    bg = mask[1:-1, 1:-1] > 0

    alpha = np.where(bg, 0, 255).astype(np.uint8)
    # 填掉前景内部的孔洞（flood fill 有时会漏掉玩偶内部的小白块）
    fg_u8 = (alpha > 0).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(fg_u8, 8)
    if n > 2:
        # 保留最大的若干连通块（玩偶本体 + 可能被分离开的部件），去掉噪点
        areas = [(stats[i, cv2.CC_STAT_AREA], i) for i in range(1, n)]
        areas.sort(reverse=True)
        keep = {i for a, i in areas if a > 400}
        alpha = np.where(np.isin(lab, list(keep)), 255, 0).astype(np.uint8)

    if feather:
        alpha = cv2.GaussianBlur(alpha, (feather * 2 + 1,) * 2, 0)

    rgba = cv2.cvtColor(bgr, cv2.COLOR_BGR2BGRA)
    rgba[:, :, 3] = alpha

    edge = int(np.count_nonzero((alpha > 20) & (alpha < 235)))
    return rgba, bg.mean(), (~bg).mean(), edge


def matte_grabcut(path: Path, margin: int = 6, iters: int = 5):
    """用 GrabCut 抠图。

    比容差法适合这个场景：容差法只看"像不像白色"，白玩偶会被一起吃掉；
    GrabCut 会同时建立前景/背景的颜色模型并迭代，能把同为白色但位置在中间的
    玩偶留住。初始化时把四周一圈标成确定背景，中心矩形内标成可能前景。
    """
    bgr = imread_u(path)
    if bgr is None:
        return None
    h, w = bgr.shape[:2]

    mask = np.zeros((h, w), np.uint8)
    # 四周一圈：确定背景（影棚背景一定贴边）
    mask[:margin, :] = cv2.GC_BGD
    mask[-margin:, :] = cv2.GC_BGD
    mask[:, :margin] = cv2.GC_BGD
    mask[:, -margin:] = cv2.GC_BGD
    # 中间：可能前景
    mask[margin:-margin, margin:-margin] = cv2.GC_PR_FGD

    bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(bgr, mask, None, bgd, fgd, iters, cv2.GC_INIT_WITH_MASK)
    except cv2.error:
        return None

    fg = np.isin(mask, [cv2.GC_FGD, cv2.GC_PR_FGD]).astype(np.uint8)
    # 只保留最大连通块，去掉零星误判
    n, lab, stats, _ = cv2.connectedComponentsWithStats(fg, 8)
    if n > 2:
        areas = sorted(((stats[i, cv2.CC_STAT_AREA], i) for i in range(1, n)), reverse=True)
        keep = areas[0][1]
        fg = (lab == keep).astype(np.uint8) if areas[0][0] > h * w * 0.005 else fg

    # 填内部孔洞（毛绒本体的白肚皮可能被判成背景）
    ff = fg.copy()
    m2 = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(ff, m2, (0, 0), 1)
    holes = (ff == 0).astype(np.uint8)
    fg = np.clip(fg + holes, 0, 1)

    alpha = (fg * 255).astype(np.uint8)
    alpha = cv2.GaussianBlur(alpha, (5, 5), 0)
    alpha = np.where(alpha > 250, 255, alpha).astype(np.uint8)

    rgba = cv2.cvtColor(bgr, cv2.COLOR_BGR2BGRA)
    rgba[:, :, 3] = alpha
    edge = int(np.count_nonzero((alpha > 20) & (alpha < 235)))
    fg_ratio = float((alpha > 127).mean())
    return rgba, 1.0 - fg_ratio, fg_ratio, edge


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    picks = [
        ("p20_20220923110846.jpg", "米白大玩偶"),
        ("p23_20220923141153.jpg", "米白钥匙扣"),
        ("p29_20220923111627.jpg", "粉色玩偶"),
        ("p34_20220125114439.jpg", "粉色钥匙扣"),
    ]

    print(f"{'文件':34s} {'方法':>10s} {'前景%':>8s} {'边缘px':>8s}  说明")
    print("-" * 88)
    results = []
    for name, note in picks:
        p = SRC / name
        if not p.exists():
            print(f"{name:34s} 缺失")
            continue
        for label, fn in (("floodfill", matte), ("grabcut", matte_grabcut)):
            r = fn(p)
            if r is None:
                print(f"{name:34s} {label:>10s}   读取/处理失败")
                continue
            rgba, bgp, fgp, edge = r
            # 前景占比是判断成败的关键：0.3% 说明玩偶被吃掉了
            verdict = "合格" if 0.08 < fgp < 0.85 else ("前景被吃掉" if fgp <= 0.08 else "背景没抠掉")
            results.append((name, label, rgba, fgp, edge, note, verdict))
            print(f"{name:34s} {label:>10s} {fgp*100:7.1f}% {edge:8d}  {note} / {verdict}")

    # 预览：棋盘底，只看 grabcut 那批
    tiles = []
    for name, label, rgba, fgp, edge, note, verdict in results:
        if label != "grabcut":
            continue
        im = cv2.resize(rgba, (400, 400), interpolation=cv2.INTER_AREA)
        h, w = im.shape[:2]
        sq = 20
        checker = np.zeros((h, w, 3), np.uint8)
        for yy in range(0, h, sq):
            for xx in range(0, w, sq):
                checker[yy:yy + sq, xx:xx + sq] = 205 if ((xx // sq) + (yy // sq)) % 2 == 0 else 245
        a = im[:, :, 3:4].astype(np.float32) / 255.0
        comp = (im[:, :, :3].astype(np.float32) * a + checker.astype(np.float32) * (1 - a)).astype(np.uint8)
        tiles.append(comp)

    if tiles:
        sheet = np.hstack(tiles)
        imwrite_u(ROOT / "_research" / "matte_preview.jpg", sheet)
        print(f"\n预览图: _research/matte_preview.jpg  ({sheet.shape[1]}x{sheet.shape[0]})")


if __name__ == "__main__":
    main()
