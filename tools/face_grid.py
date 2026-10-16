"""在立绘上叠加百分比网格，用于人工读取面部特征的确切位置。

前面试过自动找鼻孔/嘴巴，都不可靠：
  - 找暗色块：眼睛、瞳孔、蹄子阴影都是暗的，分不清哪个是嘴
  - 找粉色鼻孔：报出来的间距一会儿 0.9% 一会儿 15.5%，明显在乱认

所以改成画网格人工读。读出来的值直接写进代码 —— 只有 4 只，
一次性量准比调自动检测稳。

输出：_research/face_grid.jpg
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
OPT = ROOT / "assets" / "opt"


def load(p: Path):
    return cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_UNCHANGED)


def main() -> None:
    tiles = []
    for p in sorted(OPT.glob("unicorn_*.webp")):
        im = load(p)
        if im is None:
            continue
        h, w = im.shape[:2]
        bgr = im[:, :, :3].copy()

        # 只在头部区域画网格（x 5%~70%, y 5%~65%），太密反而看不清
        for pct in range(5, 75, 5):
            x = int(w * pct / 100)
            y = int(h * pct / 100)
            major = (pct % 10 == 0)
            col = (0, 0, 255) if major else (0, 160, 255)
            th = 2 if major else 1
            if x < w:
                cv2.line(bgr, (x, 0), (x, int(h * 0.7)), col, th)
                if major:
                    cv2.putText(bgr, str(pct), (x + 2, 16),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 255), 1, cv2.LINE_AA)
            if y < h:
                cv2.line(bgr, (0, y), (int(w * 0.75), y), col, th)
                if major:
                    cv2.putText(bgr, str(pct), (2, y - 3),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 255), 1, cv2.LINE_AA)

        # 放大头部区域
        crop = bgr[0:int(h * 0.7), 0:int(w * 0.8)]
        crop = cv2.resize(crop, (crop.shape[1] * 2, crop.shape[0] * 2),
                          interpolation=cv2.INTER_CUBIC)
        cv2.rectangle(crop, (0, 0), (crop.shape[1] - 1, crop.shape[0] - 1), (0, 128, 0), 3)
        cv2.putText(crop, p.stem.replace("unicorn_", ""), (10, crop.shape[0] - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 100, 0), 3, cv2.LINE_AA)
        tiles.append(crop)

    if not tiles:
        raise SystemExit("没有立绘")

    # 两行两列
    mh = max(t.shape[0] for t in tiles)
    mw = max(t.shape[1] for t in tiles)
    padded = []
    for t in tiles:
        c = np.full((mh, mw, 3), 255, np.uint8)
        c[:t.shape[0], :t.shape[1]] = t
        padded.append(c)
    while len(padded) % 2:
        padded.append(np.full((mh, mw, 3), 255, np.uint8))
    rows = [np.hstack(padded[i:i + 2]) for i in range(0, len(padded), 2)]
    sheet = np.vstack(rows)

    dest = ROOT / "_research" / "face_grid.jpg"
    dest.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(".jpg", sheet, [cv2.IMWRITE_JPEG_QUALITY, 92])
    if ok:
        dest.write_bytes(buf.tobytes())
    print(f"网格图: {dest}  {sheet.shape[1]}x{sheet.shape[0]}   ok={ok}")


if __name__ == "__main__":
    main()
