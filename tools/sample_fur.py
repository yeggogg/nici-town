"""按人工读出的坐标，采样每只立绘嘴巴周围的毛色。

为什么要单独采色：要做"嘴型随心情变"，得先用一块毛色补丁盖住原嘴，
再画新的嘴。补丁颜色必须和周围毛色**一致**，否则会看出一块疤。
立绘是照片级布偶，毛色不是纯色（有明暗过渡），所以取中位数最稳。

坐标来自 tools/face_grid.py 生成的网格图人工读取。

用法：
    py -3 tools/sample_fur.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
OPT = ROOT / "assets" / "opt"

# 人工从网格图读出的嘴巴位置（百分比）。略微偏右偏下，落在吻部正中。
MOUTH = {
    "bubble": (24.0, 47.0),
    "cloudy": (24.0, 51.0),
    "dreamy": (28.0, 56.0),
    "peachy": (31.5, 54.0),
}


def load(p: Path):
    return cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_UNCHANGED)


def main() -> None:
    out = {}
    for key, (mx, my) in MOUTH.items():
        p = OPT / f"unicorn_{key}.webp"
        im = load(p)
        if im is None:
            print(f"{key}: 读不到 {p.name}")
            continue
        h, w = im.shape[:2]
        bgr = im[:, :, :3]

        # 在嘴周围取一块，排除嘴的深色绣线（只统计亮像素）
        cx, cy = int(w * mx / 100), int(h * my / 100)
        rx, ry = int(w * 0.10), int(h * 0.10)
        x0, x1 = max(0, cx - rx), min(w, cx + rx)
        y0, y1 = max(0, cy - ry), min(h, cy + ry)
        patch = bgr[y0:y1, x0:x1].reshape(-1, 3).astype(np.float32)

        lum = patch.mean(axis=1)
        fur = patch[lum > np.percentile(lum, 60)]   # 取较亮的一批，避开绣线
        med = np.median(fur, axis=0)
        # 再算一下明暗范围，方便做渐变的补丁（纯色补丁会有硬边）
        lo = np.percentile(fur, 15, axis=0)
        hi = np.percentile(fur, 85, axis=0)

        hexs = lambda c: "#%02x%02x%02x" % (int(c[2]), int(c[1]), int(c[0]))
        out[key] = {
            "x": mx, "y": my,
            "fur": hexs(med), "furLo": hexs(lo), "furHi": hexs(hi),
            "origin": [round(cx / w * 100, 2), round(cy / h * 100, 2)],
        }
        print(f"  {key:8s} 嘴@({mx}%,{my}%)  毛色={hexs(med)}  "
              f"暗={hexs(lo)} 亮={hexs(hi)}  (样本 {len(fur)} px)")

    dest = ROOT / "_research" / "face.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n{json.dumps(out, ensure_ascii=False, indent=2)}")
    print(f"\n已存 {dest}")


if __name__ == "__main__":
    main()
