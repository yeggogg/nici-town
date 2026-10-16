"""把商品图拼成联览图，便于快速挑选。"""
import io, sys, pathlib
from PIL import Image, ImageDraw

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
root = pathlib.Path(__file__).resolve().parent.parent
src = root / '_research' / 'products'
files = sorted(src.glob('p*.jpg')) + sorted(src.glob('p*.png'))

COLS, CELL, PAD, LBL = 8, 190, 6, 16
rows = (len(files) + COLS - 1) // COLS
W = COLS * (CELL + PAD) + PAD
H = rows * (CELL + PAD + LBL) + PAD
sheet = Image.new('RGB', (W, H), (245, 245, 248))
d = ImageDraw.Draw(sheet)

for i, p in enumerate(files):
    r, c = divmod(i, COLS)
    x = PAD + c * (CELL + PAD)
    y = PAD + r * (CELL + PAD + LBL)
    try:
        im = Image.open(p).convert('RGB')
        im.thumbnail((CELL, CELL), Image.LANCZOS)
        sheet.paste(im, (x + (CELL - im.width) // 2, y + (CELL - im.height) // 2))
        d.text((x + 2, y + CELL + 2), f'{i:02d}', fill=(60, 60, 80))
    except Exception as e:
        d.text((x + 2, y + 40), f'{i:02d} ERR', fill=(200, 0, 0))

dest = root / '_research' / 'sheet.jpg'
sheet.save(dest, quality=88)
print(f'联览图: {dest}  {sheet.size[0]}x{sheet.size[1]}  共 {len(files)} 张')
