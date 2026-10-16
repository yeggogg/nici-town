"""定位立绘上嘴巴的位置，并把检测框画在图上便于核对。

为什么要"画出来"：立绘是照片级布偶，嘴上是一条细细的深色绣线，
光看打印出来的坐标很难判断哪个框才是嘴。画上编号再肉眼看，才不会认错。

关键：只靠"暗"会把粉色鬃毛也算进来（粉色算出来的亮度刚好压在阈值下），
所以必须同时要求**低饱和**——绣线是黑灰色，鬃毛是高饱和的。
"""
import sys
from pathlib import Path
import cv2
import numpy as np

sys.stdout.reconfigure(encoding='utf-8')
ROOT = Path(__file__).resolve().parent.parent
OPT = ROOT/'assets'/'opt'

def load(p):
    return cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_UNCHANGED)

tiles = []
for p in sorted(OPT.glob('unicorn_*.webp')):
    im = load(p)
    if im is None:
        continue
    h, w = im.shape[:2]
    bgr = im[:, :, :3]
    a = im[:, :, 3]
    lum = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.int16)
    sat = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[:, :, 1]
    # 绣线 = 暗 + 低饱和
    mask = ((lum < 140) & (sat < 80) & (a > 200)).astype(np.uint8)

    n, lab, stats, cent = cv2.connectedComponentsWithStats(mask, 8)
    vis = bgr.copy()
    rows = []
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < 15:
            continue
        x, y = int(stats[i, 0]), int(stats[i, 1])
        bw, bh = int(stats[i, 2]), int(stats[i, 3])
        cx, cy = cent[i]
        rows.append((cy, cx, x, y, bw, bh, area))
    rows.sort()

    print(f'\n=== {p.name} {w}x{h} 深暗低饱和块 {len(rows)} 个 ===')
    for k, (cy, cx, x, y, bw, bh, area) in enumerate(rows):
        cv2.rectangle(vis, (x, y), (x+bw, y+bh), (0, 0, 255), 2)
        cv2.putText(vis, str(k), (x-2, y-4), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        print(f'  #{k:2d} 中心 pct(x={cx/w*100:5.1f}% y={cy/h*100:5.1f}%) '
              f'box=({x:4d},{y:4d} {bw:3d}x{bh:3d}) 长宽比={bw/max(bh,1):4.1f} area={area:5d}')

    vis = cv2.resize(vis, (int(w*1.35), int(h*1.35)), interpolation=cv2.INTER_CUBIC)
    cv2.putText(vis, p.stem.replace('unicorn_',''), (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 128, 0), 2)
    tiles.append(vis)

if tiles:
    while len(tiles) % 2:
        tiles.append(np.full_like(tiles[0], 250))
    rows_img = [np.hstack(tiles[i:i+2]) for i in range(0, len(tiles), 2)]
    sheet = np.vstack(rows_img)
    # 注意：cv2.imwrite 在 Windows 上遇到中文路径会静默失败，
    # 必须先编码再自己写文件。
    dest = ROOT/'_research'/'mouth_annot.jpg'
    dest.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode('.jpg', sheet, [cv2.IMWRITE_JPEG_QUALITY, 92])
    if ok:
        dest.write_bytes(buf.tobytes())
    print(f'\n标注图: {dest}  {sheet.shape[1]}x{sheet.shape[0]}  ok={ok}')
