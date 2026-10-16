"""量出表情图相对于主立绘的"身体漂移"。

为什么必须量：表情图是用改图端点生成的，模型有可能会顺手把
身体、视角、比例也改了 —— 那样切换表情时玩偶会跳一下。
光看单张图看不出来（每张单独看都挺好看），必须和原图叠起来比。

做法：
  1. 对齐 alpha 边界框（只看剪影，不受颜色影响）
  2. 算剪影 IoU —— 越接近 1 说明姿势/轮廓越没变
  3. 在**身体区域**（排除头部）算像素差
  4. 输出并排 + 差分热力图，便于肉眼复核

用法：
    py -3 tools/check_expr_drift.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "assets" / "opt"
EXPR = ROOT / "assets" / "expr"


def load(p: Path):
    return cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_UNCHANGED)


def bbox_of(a, thr=24):
    ys, xs = np.where(a > thr)
    if len(ys) == 0:
        return None
    return xs.min(), ys.min(), xs.max() + 1, ys.max() + 1


def main() -> None:
    chars = sorted(p.stem.replace("unicorn_", "") for p in SRC.glob("unicorn_*.webp"))
    tiles = []

    print(f"{'角色':10s} {'表情':7s} {'剪影IoU':>9s} {'尺寸差异':>16s} {'身体像素差':>10s}  判定")
    print("-" * 74)

    for c in chars:
        base = load(SRC / f"unicorn_{c}.webp")
        if base is None:
            continue
        ba = base[:, :, 3]
        bb = bbox_of(ba)
        if bb is None:
            continue
        bx0, by0, bx1, by1 = bb
        b_sil = (ba[by0:by1, bx0:bx1] > 127)

        for mood in ("happy", "calm", "sad"):
            p = EXPR / f"{c}_{mood}.webp"
            if not p.exists():
                continue
            ex = load(p)
            if ex is None:
                continue
            ea = ex[:, :, 3]
            eb = bbox_of(ea)
            if eb is None:
                continue
            ex0, ey0, ex1, ey1 = eb
            e_sil = (ea[ey0:ey1, ex0:ex1] > 127)

            # 剪影 IoU：先把两张裁到同一尺寸
            H = max(b_sil.shape[0], e_sil.shape[0])
            W = max(b_sil.shape[1], e_sil.shape[1])

            def pad(m):
                o = np.zeros((H, W), bool)
                o[:m.shape[0], :m.shape[1]] = m
                return o

            A, B = pad(b_sil), pad(e_sil)
            inter = (A & B).sum()
            union = (A | B).sum()
            iou = inter / union if union else 0

            # 身体区域的像素差：只比下半段（排除头部，那里本来就该变）
            bh, bw = b_sil.shape
            y_cut = int(bh * 0.55)
            b_roi = base[by0 + y_cut:by1, bx0:bx1, :3].astype(np.float32)
            # 表情图的同一区域（按比例映射）
            eh, ew = e_sil.shape
            ey_cut = int(eh * 0.55)
            e_roi = ex[ey0 + ey_cut:ey1, ex0:ex1, :3].astype(np.float32)

            hh = min(b_roi.shape[0], e_roi.shape[0])
            ww = min(b_roi.shape[1], e_roi.shape[1])
            if hh > 4 and ww > 4:
                diff = np.abs(b_roi[:hh, :ww] - e_roi[:hh, :ww]).mean()
            else:
                diff = -1

            verdict = ("很好" if iou > 0.93 else
                       "可接受" if iou > 0.88 else
                       "漂移明显")
            size_note = f"{e_sil.shape[1]}x{e_sil.shape[0]}"
            print(f"{c:10s} {mood:7s} {iou:9.3f} {size_note:>16s} {diff:10.1f}  {verdict}")

            # 拼一张：原图 | 表情图 | 差分
            cell = 300
            def fit(img):
                h, w = img.shape[:2]
                s = cell / max(h, w)
                r = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
                cv = np.zeros((cell, cell, 4), np.uint8)
                oy, ox = (cell - r.shape[0]) // 2, (cell - r.shape[1]) // 2
                cv[oy:oy + r.shape[0], ox:ox + r.shape[1]] = r
                return cv
            def over(img):
                cv = fit(img)
                a = cv[:, :, 3:4].astype(np.float32) / 255.0
                yy, xx = np.mgrid[0:cell, 0:cell]
                ck = np.where((yy // 18 + xx // 18) % 2 == 0, 210, 246).astype(np.uint8)
                ck = cv2.merge([ck, ck, ck])
                return (cv[:, :, :3].astype(np.float32) * a + ck.astype(np.float32) * (1 - a)).astype(np.uint8)

            t = np.hstack([over(base), over(ex)])
            cv2.putText(t, f"{c} {mood}  IoU={iou:.3f}", (6, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 20, 60), 2, cv2.LINE_AA)
            tiles.append(t)

    if tiles:
        cols = 3
        while len(tiles) % cols:
            tiles.append(np.full_like(tiles[0], 250))
        rows = [np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)]
        sheet = np.vstack(rows)
        dest = ROOT / "_research" / "expr_compare.jpg"
        dest.parent.mkdir(parents=True, exist_ok=True)
        ok, buf = cv2.imencode(".jpg", sheet, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if ok:
            dest.write_bytes(buf.tobytes())
        print(f"\n对比图: {dest}  {sheet.shape[1]}x{sheet.shape[0]}")


if __name__ == "__main__":
    main()
