"""批量抠图全部官方商品图，产出带编号的联览图，用于挑选立绘素材。

不猜哪张能用 —— 让数据说话：每张都跑一遍抠图，把前景占比、
连通块数、成图尺寸标出来，一眼就能看出哪些是真正的独立玩偶
（而不是抱枕、毛毯、马克杯、挂在杆上的商品图）。
"""
import io
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_sprites as M  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")


def main() -> None:
    files = sorted(M.SRC.glob("p*"))
    print(f"共 {len(files)} 张")

    ok_items, rows = [], []
    for i, p in enumerate(files):
        rgba, info = M.matte(p)
        if rgba is None:
            rows.append((i, p.name, None, info.get("error", "失败")))
            continue
        norm = M.normalize(rgba)
        fg = info["fg_ratio"]
        blocks = info["big_blocks"]
        note = ""
        if fg < 0.06:
            note = "前景过小(非玩偶?)"
        elif fg > 0.70:
            note = "前景过大(背景没抠净?)"
        elif blocks > 1:
            note = f"{blocks}个独立块"
        rows.append((i, p.name, info, note))
        ok_items.append((i, p.name, norm))

    print(f"\n{'#':>3s}  {'文件名':30s} {'前景%':>7s} {'块':>3s} {'尺寸':>10s}  说明")
    print("-" * 76)
    for i, name, info, note in rows:
        if info is None:
            print(f"{i:3d}  {name:30s} {'—':>7s} {'—':>3s} {'—':>10s}  {note}")
        else:
            x0, y0, x1, y1 = info["bbox"]
            size = f"{x1 - x0}x{y1 - y0}"
            print(f"{i:3d}  {name:30s} {info['fg_ratio']*100:6.1f}% "
                  f"{info['big_blocks']:3d} {size:>10s}  {note}")

    # 联览图：编号 + 文件名 + 前景占比，便于精确挑选
    CELL, COLS, LBL = 250, 6, 34
    rows_n = (len(ok_items) + COLS - 1) // COLS
    W, H = COLS * CELL, rows_n * (CELL + LBL)
    sheet = np.full((H, W, 3), 250, np.uint8)
    for k, (i, name, norm) in enumerate(ok_items):
        r, c = divmod(k, COLS)
        y, x = r * (CELL + LBL), c * CELL
        im = cv2.resize(norm, (CELL, CELL), interpolation=cv2.INTER_AREA)
        a = im[:, :, 3:4].astype(np.float32) / 255.0
        comp = (im[:, :, :3].astype(np.float32) * a +
                M.checkerboard(CELL, CELL).astype(np.float32) * (1 - a)).astype(np.uint8)
        sheet[y:y + CELL, x:x + CELL] = comp
        cv2.putText(sheet, f"#{i}  {name[1:9]}", (x + 4, y + CELL + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (40, 40, 90), 1, cv2.LINE_AA)
    dest = M.ROOT / "_research" / "matte_all.jpg"
    M.imwrite_u(dest, sheet)
    print(f"\n联览图: {dest}  ({W}x{H}，共 {len(ok_items)} 张)")


if __name__ == "__main__":
    main()
