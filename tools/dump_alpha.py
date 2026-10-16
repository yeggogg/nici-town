"""把立绘的 alpha 通道直接可视化，定位残留问题。"""
import sys
from pathlib import Path
import cv2, numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_sprites as M
sys.stdout.reconfigure(encoding='utf-8')

p = M.FINAL_OUT / 'unicorn_bubble.webp'
buf = np.fromfile(str(p), dtype=np.uint8)
im = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
h, w = im.shape[:2]
a = im[:, :, 3]
print(f'尺寸 {w}x{h}')
print(f'alpha: min={a.min()} max={a.max()} 半透明像素={( (a>20)&(a<235) ).sum()} ({( (a>20)&(a<235) ).mean()*100:.2f}%)')

# alpha 可视化：黑底白前景
vis = np.zeros((h, w, 3), np.uint8)
vis[:, :, 0] = vis[:, :, 1] = vis[:, :, 2] = a
# 底部 45% 放大 2 倍
crop = a[int(h*0.55):, :]
crop = cv2.resize(crop, (crop.shape[1]*2, crop.shape[0]*2), interpolation=cv2.INTER_NEAREST)
vis2 = np.zeros((crop.shape[0], crop.shape[1], 3), np.uint8)
vis2[:, :, 0] = vis2[:, :, 1] = vis2[:, :, 2] = crop

M.imwrite_u(M.ROOT/'_research'/'alpha_full.png', vis)
M.imwrite_u(M.ROOT/'_research'/'alpha_bottom.png', vis2)
print('已存: _research/alpha_full.png  _research/alpha_bottom.png')
