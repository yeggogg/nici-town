"""测量每只立绘的「鼻子中点」与「嘴巴」位置，供叠加心情嘴型用。

为什么不能靠暗色块直接找嘴：立绘上眼睛、瞳孔、蹄尖、蹄边的阴影都是暗的，
容易认错。改用语义更明确的规则 ——

    先找**两个粉红色的鼻孔点**（高饱和粉、尺寸小、左右成对、在头部），
    嘴巴必然在这两点的正下方。这是布偶固定的面部结构。

同时采样鼻孔周围的毛色，作为"擦掉原嘴"的补丁颜色。

输出可直接粘进 index.html 的 JSON。

用法：
    py -3 tools/measure_face.py
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


def load(p: Path):
    return cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_UNCHANGED)


def find_nostrils(im):
    """找两个粉色鼻孔点。返回 [(cx,cy)…]，最多两个。"""
    h, w = im.shape[:2]
    bgr = im[:, :, :3]
    a = im[:, :, 3]
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    H, S, V = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

    # 粉色：色相在洋红~红之间，饱和度高，亮度中等偏高
    pink = ((S > 70) & (V > 110) &
            (((H >= 150) & (H <= 179)) | (H <= 8)) &
            (a > 200)).astype(np.uint8)

    n, lab, stats, cent = cv2.connectedComponentsWithStats(pink, 8)
    cands = []
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        x, y = int(stats[i, 0]), int(stats[i, 1])
        bw, bh = int(stats[i, 2]), int(stats[i, 3])
        cx, cy = cent[i]
        # 鼻孔是小圆点：面积适中、接近方形、在上半部
        if not (12 <= area <= 900):
            continue
        if bh == 0 or not (0.5 <= bw / bh <= 2.2):
            continue
        if cy > h * 0.62:
            continue
        cands.append((area, cx, cy, bw, bh))

    if len(cands) < 2:
        return []
    # 取面积最接近的一对，且要求水平相邻
    cands.sort(key=lambda c: c[1])
    best = None
    for i in range(len(cands)):
        for j in range(i + 1, len(cands)):
            A, ax, ay, _, _ = cands[i]
            B, bx, by, _, _ = cands[j]
            dy = abs(ay - by)
            if dy > h * 0.05:
                continue
            gap = bx - ax
            if gap <= 0 or gap > w * 0.20:
                continue
            # 两个点的大小应该相近
            ratio = min(A, B) / max(A, B)
            if ratio < 0.35:
                continue
            score = ratio - abs(dy) / (h * 0.05) - gap / (w * 0.20)
            if best is None or score > best[0]:
                best = (score, ax, ay, bx, by)
    if best is None:
        return []
    _, ax, ay, bx, by = best
    return [(ax, ay), (bx, by)]


def main() -> None:
    out = {}
    for p in sorted(OPT.glob("unicorn_*.webp")):
        im = load(p)
        if im is None:
            continue
        h, w = im.shape[:2]
        bgr = im[:, :, :3]
        nos = find_nostrils(im)
        key = p.stem.replace("unicorn_", "")
        if not nos:
            print(f"  {key:10s} 没找到鼻孔点")
            continue

        mx = (nos[0][0] + nos[1][0]) / 2
        my = (nos[0][1] + nos[1][1]) / 2
        gap = abs(nos[1][0] - nos[0][0])

        # 嘴在鼻孔下方约 0.45~0.9 个间距处
        mouth = (mx, my + gap * 0.68)

        # 采样鼻孔周围的毛色（排除粉色本身），作为补丁色
        r = int(gap * 1.1)
        x0, x1 = max(0, int(mx - r)), min(w, int(mx + r))
        y0, y1 = max(0, int(my - r * 0.3)), min(h, int(my + r * 2.4))
        patch = bgr[y0:y1, x0:x1].reshape(-1, 3)
        lum = patch.mean(axis=1)
        fur = patch[lum > 150]
        med = np.median(fur, axis=0) if len(fur) else np.array([240, 240, 240])

        out[key] = {
            "mouthX": round(mx / w * 100, 2),
            "mouthY": round(mouth[1] / h * 100, 2),
            "noseGap": round(gap / w * 100, 2),
            "fur": "#%02x%02x%02x" % (int(med[2]), int(med[1]), int(med[0])),
        }
        print(f"  {key:10s} 鼻孔=({mx/w*100:5.1f}%,{my/h*100:5.1f}%) "
              f"间距={gap/w*100:4.1f}%  ->  嘴=({out[key]['mouthX']}%,{out[key]['mouthY']}%) "
              f"毛色={out[key]['fur']}")

    dest = ROOT / "_research" / "face.json"
    dest.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n结果: {dest}")
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
