"""从 NICI 官网独角兽分类页提取产品图，仅用于研究造型比例（不随游戏分发）。"""
import re, io, sys, pathlib, urllib.request

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
root = pathlib.Path(__file__).resolve().parent.parent
html = (root / '_research' / 'unicorn.html').read_text(encoding='utf-8', errors='replace')
print('[debug] html bytes:', len(html))

imgs = re.findall(r'(?:src|data-src|data-original)=["\']([^"\']+\.(?:jpg|jpeg|png))["\']', html, re.I)
names = re.findall(r'<[^>]*class=["\'][^"\']*(?:name|title)[^"\']*["\'][^>]*>(.*?)<', html, re.S)
cands, seen = [], set()
for u in imgs:
    if 'UpFile' not in u and 'A_UpLoad' not in u:
        continue
    full = u if u.startswith('http') else 'https://www.nici.com.cn/' + u.lstrip('/')
    if full in seen:
        continue
    seen.add(full)
    cands.append(full)

print('候选图:', len(cands))
for u in cands[:12]:
    print('  ', u)

ua = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/124.0.0.0 Safari/537.36'}
saved = []
for i, u in enumerate(cands[:5]):
    try:
        req = urllib.request.Request(u, headers=ua)
        data = urllib.request.urlopen(req, timeout=25).read()
        p = root / '_research' / 'ref' / f'unicorn_ref_{i}.jpg'
        p.write_bytes(data)
        saved.append((p.name, len(data)))
    except Exception as e:
        print(f'  [失败] {u[-40:]}: {type(e).__name__}')

print()
print('已保存:')
for n, sz in saved:
    print(f'  _research/ref/{n}  ({sz//1024} KB)')
