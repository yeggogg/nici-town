"""解析下载的 NICI 官网首页，找出独角兽相关分类与图片。只读。"""
import re, io, sys, pathlib

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
root = pathlib.Path(__file__).resolve().parent.parent
h = (root / '_research' / 'home.html').read_text(encoding='utf-8', errors='replace')

print('=== “独角兽”出现位置 ===')
for m in re.finditer('独角兽', h):
    s, e = max(0, m.start()-260), min(len(h), m.end()+260)
    print(re.sub(r'\s+', ' ', h[s:e])[:500])
    print('---')

print('\n=== 产品分类链接 ===')
seen = set()
for m in re.finditer(r'<a[^>]+href=["\']([^"\']*product\.aspx[^"\']*)["\'][^>]*>(.*?)</a>', h, re.S|re.I):
    href, txt = m.group(1), re.sub(r'<[^>]+>', '', m.group(2)).strip()
    txt = re.sub(r'\s+', ' ', txt)
    if href not in seen and txt:
        seen.add(href)
        print(f'  {txt[:26]:28s} {href}')

print('\n=== 首页图片 ===')
imgs = re.findall(r'(?:src|data-src|data-original)=["\']([^"\']+\.(?:jpg|jpeg|png|webp|gif))', h, re.I)
for u in sorted(set(imgs))[:30]:
    print('  ', u[:120])
