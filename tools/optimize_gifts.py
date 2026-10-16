"""把生成的礼品插画压成网页用的 WebP。

原图 1024x1024 PNG、单个 1~1.6 MB，13 个合计 17 MB —— 当网页资源太重。
礼品在游戏里最大只显示到 76px（开箱动画），按 2x 视网膜算 256px 足够；
但考虑到用户可能放大看，留到 320px。
"""
import sys
from pathlib import Path
import cv2, numpy as np

sys.stdout.reconfigure(encoding='utf-8')
ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT/'assets'/'gifts'
DST = ROOT/'assets'/'gifts_opt'
SIZE = 320

DST.mkdir(parents=True, exist_ok=True)
before = after = 0
print(f'{"文件":14s} {"输出":>10s} {"压缩前":>9s} {"压缩后":>8s} {"比例":>7s}')
print('-'*56)
for p in sorted(SRC.glob('*.png')):
    im = cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_UNCHANGED)
    if im is None:
        print(f'{p.stem:14s} 读取失败'); continue
    h, w = im.shape[:2]
    s = SIZE / max(h, w)
    out = cv2.resize(im, (int(round(w*s)), int(round(h*s))), interpolation=cv2.INTER_AREA)
    dest = DST / f'{p.stem}.webp'
    ok, buf = cv2.imencode('.webp', out, [cv2.IMWRITE_WEBP_QUALITY, 88])
    if ok:
        dest.write_bytes(buf.tobytes())
    b, a = p.stat().st_size, dest.stat().st_size
    before += b; after += a
    print(f'{p.stem:14s} {out.shape[1]:4d}x{out.shape[0]:<5d} {b/1024:8.0f}K {a/1024:7.0f}K {a/b*100:6.1f}%')
print(f'\n合计 {before/1024/1024:.1f} MB -> {after/1024/1024:.2f} MB (省 {(1-after/before)*100:.0f}%)')
