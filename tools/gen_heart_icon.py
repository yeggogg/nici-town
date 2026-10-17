# -*- coding: utf-8 -*-
"""Convert the generated musk PNG into a game-ready transparent webp icon."""
import glob
import os
import sys

import cv2
import numpy as np

RAW_DIR = r"D:/Projects/Nici小镇/_research/ui_raw/heart"
OUT = r"D:/Projects/Nici小镇/assets/ui/heart.webp"
PAD = 6
MAX_EDGE = 256


def imread_u(path):
    with open(path, "rb") as f:
        data = np.frombuffer(f.read(), np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_UNCHANGED)


def imwrite_u(path, img, ext=".webp", quality=88):
    params = []
    if ext == ".webp":
        params = [cv2.IMWRITE_WEBP_QUALITY, quality]
    ok, buf = cv2.imencode(ext, img, params)
    if not ok:
        raise RuntimeError("imencode failed")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(buf.tobytes())


def main():
    files = sorted(glob.glob(os.path.join(RAW_DIR, "musk_*.png")), key=os.path.getmtime)
    if not files:
        sys.exit("no musk_*.png found")
    src = files[-1]
    print("src:", src, os.path.getsize(src), "bytes")

    img = imread_u(src)
    if img is None:
        sys.exit("decode failed")
    print("shape:", img.shape, img.dtype)

    if img.ndim == 3 and img.shape[2] == 4:
        alpha = img[:, :, 3]
    else:
        sys.exit("no alpha channel in source")

    ys, xs = np.where(alpha > 8)
    if len(xs) == 0:
        sys.exit("alpha empty")
    x0, y0, x1, y1 = xs.min(), ys.min(), xs.max() + 1, ys.max() + 1
    print("alpha bbox:", (x0, y0, x1, y1), "size:", (x1 - x0, y1 - y0))

    h, w = alpha.shape
    x0 = max(0, x0 - PAD)
    y0 = max(0, y0 - PAD)
    x1 = min(w, x1 + PAD)
    y1 = min(h, y1 + PAD)
    crop = img[y0:y1, x0:x1].copy()

    ch, cw = crop.shape[:2]
    scale = MAX_EDGE / float(max(ch, cw))
    if scale < 1.0:
        resample = cv2.INTER_AREA
    else:
        resample = cv2.INTER_CUBIC
        scale = min(scale, 1.0)
    nw, nh = max(1, round(cw * scale)), max(1, round(ch * scale))
    out = cv2.resize(crop, (nw, nh), interpolation=resample)
    print("output size:", out.shape)

    imwrite_u(OUT, out, ".webp", 88)
    print("wrote:", OUT, os.path.getsize(OUT), "bytes")

    # verify round-trip
    chk = imread_u(OUT)
    print("verify:", chk.shape, chk.dtype,
          "alpha min/max:", int(chk[:, :, 3].min()), int(chk[:, :, 3].max()))


if __name__ == "__main__":
    main()
