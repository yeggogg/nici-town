"""生成带 p 编号的联览图，用于准确挑选素材。"""
import sys
from pathlib import Path
import cv2, numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_sprites as M
sys.stdout.reconfigure(encoding='utf-8')

files = sorted(M.SRC.glob('p*'))
CELL, COLS, LBL = 200, 8, 20
rows = (len(files) + COLS - 1) // COLS
W, H = COLS*CELL, rows*(CELL+LBL)
sheet = np.full((H, W, 3), 252, np.uint8)
for i, p in enumerate(files):
    r, c = divmod(i, COLS)
    y, x = r*(CELL+LBL), c*CELL
    bgr = M.imread_u(p)
    if bgr is None: continue
    h, w = bgr.shape[:2]
    s = CELL / max(h, w)
    im = cv2.resize(bgr, (int(w*s), int(h*s)), interpolation=cv2.INTER_AREA)
    oy, ox = (CELL-im.shape[0])//2, (CELL-im.shape[1])//2
    sheet[y+oy:y+oy+im.shape[0], x+ox:x+ox+im.shape[1]] = im
    cv2.putText(sheet, f'{i:02d} {p.name[1:9]}', (x+3, y+CELL+14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.34, (40,40,90), 1, cv2.LINE_AA)
M.imwrite_u(M.ROOT/'_research'/'sheet_labeled.jpg', sheet)
print(f'联览图: _research/sheet_labeled.jpg  {W}x{H}  共 {len(files)} 张')
