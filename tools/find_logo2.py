import re, io, sys, pathlib
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
root = pathlib.Path(__file__).resolve().parent.parent
h = (root / '_research' / 'home.html').read_text(encoding='utf-8', errors='replace')

# logo 那个 <a> 的完整内容
m = re.search(r'<a[^>]*class="logo"[^>]*>(.*?)</a>', h, re.S)
print('=== <a class="logo"> 内容 ===')
print(re.sub(r'\s+', ' ', m.group(1))[:500] if m else 'none')
print()

# 页面引用的 css 文件
print('=== 引用的 CSS ===')
for m in re.finditer(r'<link[^>]+href=["\']([^"\']+\.css[^"\']*)', h, re.I):
    print('  ', m.group(1))
