"""下载 NICI 官网独角兽分类页的全部商品图，并拼一张联览图用于挑选。仅供参考研究。"""
import re, io, sys, pathlib, urllib.request, concurrent.futures as cf

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
root = pathlib.Path(__file__).resolve().parent.parent
out = root / '_research' / 'products'
out.mkdir(parents=True, exist_ok=True)

h = (root / '_research' / 'unicorn.html').read_text(encoding='utf-8', errors='replace')
urls, seen = [], set()
for u in re.findall(r'(?:src|data-src|data-original)=["\']([^"\']+\.(?:jpg|jpeg|png))["\']', h, re.I):
    if 'UpFile' not in u:
        continue
    full = u if u.startswith('http') else 'https://www.nici.com.cn/' + u.lstrip('/')
    if full not in seen:
        seen.add(full)
        urls.append(full)

UA = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/124.0.0.0 Safari/537.36'}

def grab(i_u):
    i, u = i_u
    dest = out / f'p{i:02d}_{u.split("/")[-1]}'
    if dest.exists() and dest.stat().st_size > 1000:
        return dest
    try:
        req = urllib.request.Request(u, headers=UA)
        dest.write_bytes(urllib.request.urlopen(req, timeout=30).read())
        return dest
    except Exception:
        return None

with cf.ThreadPoolExecutor(max_workers=8) as ex:
    got = [p for p in ex.map(grab, enumerate(urls)) if p]

from PIL import Image
print(f'下载 {len(got)} 张')
sizes = []
for p in got:
    try:
        im = Image.open(p)
        sizes.append((p, im.size))
    except Exception:
        pass
print(f'可读 {len(sizes)} 张；尺寸分布：')
from collections import Counter
print(' ', Counter(s for _, s in sizes).most_common(8))
