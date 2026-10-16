"""预览 assets/opt 下四只最终立绘（棋盘底），用于肉眼验收。"""
import sys
from pathlib import Path
import cv2, numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_sprites as M
sys.stdout.reconfigure(encoding='utf-8')

names = ['unicorn_bubble','unicorn_cloudy','unicorn_peachy','unicorn_dreamy']
CELL = 340
tiles = []
for n in names:
    p = M.FINAL_OUT / f'{n}.webp'
    if not p.exists():
        print(f'缺失: {p}'); continue
    buf = np.fromfile(str(p), dtype=np.uint8)
    im = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
    h, w = im.shape[:2]
    s = CELL / max(h, w)
    im = cv2.resize(im, (int(w*s), int(h*s)), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((CELL, CELL, 4), np.uint8)
    oy, ox = (CELL-im.shape[0])//2, (CELL-im.shape[1])//2
    canvas[oy:oy+im.shape[0], ox:ox+im.shape[1]] = im
    a = canvas[:,:,3:4].astype(np.float32)/255.0
    comp = (canvas[:,:,:3].astype(np.float32)*a +
            M.checkerboard(CELL, CELL).astype(np.float32)*(1-a)).astype(np.uint8)
    cv2.putText(comp, n.replace('unicorn_',''), (8, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (40,40,90), 2, cv2.LINE_AA)
    tiles.append(comp)
    print(f'{n:20s} {im.shape[1]}x{im.shape[0]}')

sheet = np.hstack(tiles)
M.imwrite_u(M.ROOT/'_research'/'final_preview.jpg', sheet)
print(f'\n预览: _research/final_preview.jpg ({sheet.shape[1]}x{sheet.shape[0]})')
