"""从官网首页找 logo 与品牌素材的实际文件路径。只读。"""
import re, io, sys, pathlib

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
root = pathlib.Path(__file__).resolve().parent.parent
h = (root / '_research' / 'home.html').read_text(encoding='utf-8', errors='replace')

print('=== 含 logo / 标识 的标签 ===')
for m in re.finditer(r'<[^>]*(?:logo|Logo|LOGO)[^>]*>', h):
    print('  ', re.sub(r'\s+', ' ', m.group(0))[:200])

print('\n=== 头部区域图片（前 20 个 img）===')
for i, m in enumerate(re.finditer(r'<img[^>]+>', h)):
    if i >= 20:
        break
    tag = re.sub(r'\s+', ' ', m.group(0))
    if any(k in tag.lower() for k in ['logo', 'head', 'banner', 'top']):
        print('  ', tag[:200])

print('\n=== CSS 里引用的背景图 ===')
css = re.findall(r'url\((["\']?)([^)"\']+)\1\)', h)
for _, u in sorted(set(css))[:20]:
    print('  ', u[:120])

print('\n=== favicon / icon ===')
for m in re.finditer(r'<link[^>]+(?:icon|apple-touch)[^>]*>', h, re.I):
    print('  ', re.sub(r'\s+', ' ', m.group(0))[:180])
