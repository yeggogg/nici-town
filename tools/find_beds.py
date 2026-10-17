"""从田地场景图里自动找出四条田垄的四边形坐标。

为什么要自动找：田垄是**透视**的（上窄下宽、微微倾斜），
用 2×2 均分网格去对齐永远对不上 —— 这就是"框盖在画上"违和感的来源。
眼睛读百分比也不够准，所以直接按土色分割，取每个土块的四个角。

做法：
  1. 按颜色找出深褐色土壤区域（田垄的土面）
  2. 连通域分成 4 块
  3. 对每块取"最小外接四边形"（cv2.minAreaRect -> boxPoints）
  4. 输出成 CSS clip-path 能直接用的百分比坐标

输出：_research/beds.json，并在图上画出四边形便于核对。

用法：
    py -3 tools/find_beds.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "assets" / "opt" / "scene_field.webp"


def main() -> None:
    im = cv2.imdecode(np.fromfile(str(SRC), np.uint8), cv2.IMREAD_UNCHANGED)
    if im is None:
        raise SystemExit(f"读不到 {SRC}")
    if im.shape[2] == 4:
        im = cv2.cvtColor(im, cv2.COLOR_BGRA2BGR)
    h, w = im.shape[:2]

    hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    H, S, V = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

    # 深褐土壤：色相在橙黄区（BGR->HSV 里约 5~30），饱和中等，亮度偏暗
    soil = ((H >= 3) & (H <= 32) & (S >= 45) & (V >= 30) & (V <= 165)).astype(np.uint8)

    # 只在画面下半部找（上半是天空/远山，不该有土）
    soil[:int(h * 0.45), :] = 0

    # 只做**小**的开运算去噪。
    # 早先用 25×25 的闭运算想把土块内部的小洞（石子高光）填掉，
    # 结果连通了田垄之间的石子路，四块地被并成了一整片。
    # 田垄之间本来就该是分开的，保留缝隙反而更好找。
    soil = cv2.morphologyEx(soil, cv2.MORPH_OPEN,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)), iterations=1)

    n, lab, stats, cent = cv2.connectedComponentsWithStats(soil, 8)
    blobs = []
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < w * h * 0.005:      # 太小的噪点丢掉
            continue
        blobs.append((area, i, cent[i]))
    blobs.sort(reverse=True)

    print(f"找到 {len(blobs)} 个候选土块（面积降序）")
    if len(blobs) < 4:
        raise SystemExit("不足 4 块，阈值需要调整")

    # 取最大的 4 块，按 x 从左到右排序
    picked = sorted(blobs[:4], key=lambda b: b[2][0])

    vis = im.copy()
    beds = []
    for rank, (area, idx, c) in enumerate(picked):
        m = (lab == idx).astype(np.uint8)
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        cnt = max(cnts, key=cv2.contourArea)

        # 用"上沿/下沿各取左右端点"求梯形四角。
        #
        # 为什么不用 minAreaRect：它给的是**最小外接矩形**，
        # 对倾斜的田垄会转一个角度，再按 sum/diff 排角就全乱了
        # （实测 #1 的左上角被算到 x=-1.3%，跑出画面）。
        # 田垄本身是顶边/底边近似水平的梯形，直接取上下沿更准。
        ys, xs = np.where(m > 0)
        y_lo, y_hi = int(np.percentile(ys, 4)), int(np.percentile(ys, 96))
        band = 6
        def xr(y):
            sel = xs[(ys >= y - band) & (ys <= y + band)]
            return (int(sel.min()), int(sel.max())) if len(sel) else (None, None)
        t_l, t_r = xr(y_lo)
        b_l, b_r = xr(y_hi)
        if None in (t_l, t_r, b_l, b_r):
            continue
        quad = np.array([[t_l, y_lo], [t_r, y_lo], [b_r, y_hi], [b_l, y_hi]], dtype=np.float32)

        pts = quad.astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(vis, [pts], True, (0, 0, 255), 3)
        cxy = quad.mean(axis=0)
        cv2.putText(vis, str(rank + 1), (int(cxy[0]) - 10, int(cxy[1]) + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 255), 3, cv2.LINE_AA)

        pct = [[round(float(x) / w * 100, 2), round(float(y) / h * 100, 2)] for x, y in quad]
        beds.append({
            "rank": rank + 1,
            "area": area,
            "cx": round(float(cxy[0]) / w * 100, 2),
            "cy": round(float(cxy[1]) / h * 100, 2),
            "quad": pct,
        })
        print(f"  #{rank+1} 面积={area:>7d}  中心=({cxy[0]/w*100:5.1f}%,{cxy[1]/h*100:5.1f}%)")
        for name, (x, y) in zip(["左上", "右上", "右下", "左下"], pct):
            print(f"      {name}: {x:5.1f}% , {y:5.1f}%")

    dest = ROOT / "_research" / "beds.json"
    dest.write_text(json.dumps(beds, ensure_ascii=False, indent=2), encoding="utf-8")

    ok, buf = cv2.imencode(".jpg", vis, [cv2.IMWRITE_JPEG_QUALITY, 92])
    if ok:
        (ROOT / "_research" / "beds_annot.jpg").write_bytes(buf.tobytes())
    print(f"\n结果: {dest}")
    print("  （同时输出了 _research/beds_annot.jpg 便于核对）")


if __name__ == "__main__":
    main()
