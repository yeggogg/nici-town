"""在田地场景图上叠加百分比网格，用来人工读出四条田垄的位置。

为什么要读：要让作物"长在垄上"，点击区域必须精确落在画好的田垄里。
之前用的是 2×2 均分网格，跟背景里的透视根本不吻合 ——
所以才会有"一个框盖在画上"的违和感。

输出：_research/field_grid.jpg
"""
from __future__ import annotations

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
    print(f"场景图: {w}x{h}  宽高比 {w/h:.4f}")

    vis = im.copy()
    for pct in range(0, 101, 5):
        x = int(w * pct / 100)
        y = int(h * pct / 100)
        major = (pct % 10 == 0)
        col = (0, 0, 255) if major else (0, 200, 255)
        th = 2 if major else 1
        if x < w:
            cv2.line(vis, (x, 0), (x, h), col, th)
            if major:
                cv2.putText(vis, str(pct), (x + 2, 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
        if y < h:
            cv2.line(vis, (0, y), (w, y), col, th)
            if major:
                cv2.putText(vis, str(pct), (3, y - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

    dest = ROOT / "_research" / "field_grid.jpg"
    dest.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(".jpg", vis, [cv2.IMWRITE_JPEG_QUALITY, 92])
    if ok:
        dest.write_bytes(buf.tobytes())
    print(f"网格图: {dest}")


if __name__ == "__main__":
    main()
