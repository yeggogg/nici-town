"""Convert a freshly generated PNG into a game UI icon webp.

Usage: py -3 tools/make_ui_icon.py <src.png> <dst.webp> [max_side] [pad]

Crops to the alpha bounding box with padding, scales so the longest side is
`max_side`, and writes a webp with alpha. Uses cv2.imencode + write_bytes
because cv2.imwrite silently fails on Windows paths containing non-ASCII.
"""
import sys

import cv2
import numpy as np


def main() -> int:
    src = sys.argv[1]
    dst = sys.argv[2]
    max_side = int(sys.argv[3]) if len(sys.argv) > 3 else 256
    pad = int(sys.argv[4]) if len(sys.argv) > 4 else 6

    # cv2.imread also chokes on non-ASCII paths on Windows, so read the bytes
    # ourselves and decode from memory.
    with open(src, "rb") as fh:
        img = cv2.imdecode(np.frombuffer(fh.read(), dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if img is None:
        print(f"ERROR: could not read {src}", file=sys.stderr)
        return 1
    if img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)

    alpha = img[:, :, 3]
    print(f"source: {img.shape[1]}x{img.shape[0]}, alpha min/max = {alpha.min()}/{alpha.max()}")
    print(f"transparent pixels: {int((alpha < 8).sum())} / {alpha.size}")

    ys, xs = np.where(alpha > 8)
    if len(xs) == 0:
        print("ERROR: image has no alpha content", file=sys.stderr)
        return 1
    x0, x1 = xs.min(), xs.max() + 1
    y0, y1 = ys.min(), ys.max() + 1
    print(f"content bbox: x[{x0},{x1}) y[{y0},{y1})  {(x1 - x0)}x{(y1 - y0)}")

    img = img[y0:y1, x0:x1]

    # Generated PNGs often keep a dark gradient in the fully transparent area.
    # Scaling would then average that darkness into the visible edge and leave a
    # grey fringe, so bleed the nearest opaque colour outward before resizing.
    alpha = img[:, :, 3]
    known = (alpha > 8).astype(np.uint8)
    if known.any() and not known.all():
        rgb = img[:, :, :3].copy()
        known_mask = known > 0
        kernel = np.ones((3, 3), np.uint8)
        for _ in range(24):  # enough to cover the padding ring
            if known_mask.all():
                break
            blurred = cv2.dilate(rgb, kernel)
            spread = (cv2.dilate(known, kernel) > 0) & ~known_mask
            rgb[spread] = blurred[spread]
            known = ((known > 0) | spread).astype(np.uint8)
            known_mask = known > 0
        img[:, :, :3] = rgb

    # Pad with transparency while keeping the artwork's own aspect ratio, so the
    # icon sits in the same visual family as the rest of assets/ui.
    h, w = img.shape[:2]
    canvas = cv2.copyMakeBorder(
        img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(0, 0, 0, 0)
    )

    scale = max_side / float(max(h + pad * 2, w + pad * 2))
    if scale != 1.0:
        interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
        nh = max(1, int(round((h + pad * 2) * scale)))
        nw = max(1, int(round((w + pad * 2) * scale)))
        canvas = cv2.resize(canvas, (nw, nh), interpolation=interp)

    ok, buf = cv2.imencode(".webp", canvas, [cv2.IMWRITE_WEBP_QUALITY, 88])
    if not ok:
        print("ERROR: webp encode failed", file=sys.stderr)
        return 1
    with open(dst, "wb") as fh:
        fh.write(buf.tobytes())

    with open(dst, "rb") as fh:
        check = cv2.imdecode(np.frombuffer(fh.read(), dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if check.shape[2] == 3:
        check = cv2.cvtColor(check, cv2.COLOR_BGR2BGRA)
    print(f"wrote {dst}: {check.shape[1]}x{check.shape[0]}, channels={check.shape[2]}, "
          f"alpha min={check[:, :, 3].min()}, {len(buf) / 1024:.1f} KB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
