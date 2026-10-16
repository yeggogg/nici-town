"""把 musk-image 生成的原始 PNG 处理成游戏用的 WebP 素材。

三步：
1. 按 alpha 内容边界裁掉空白，让每只独角兽在画面里占据相同比例；
2. 统一画布宽高比、底部对齐 —— 四只脚踩在同一水平线上，游戏里才不会忽大忽小；
3. 缩到 2x 视网膜足够的分辨率并转 WebP。

用法：
    py -3 tools/optimize_assets.py
"""
from pathlib import Path
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "assets"
DST = SRC / "opt"

UNICORNS = ["unicorn_bubble", "unicorn_cloudy", "unicorn_peachy", "unicorn_dreamy"]

# 统一画布宽高比（宽/高）。四只实测 0.85~0.92，取 0.92 让最宽的那只刚好贴边。
CANVAS_ASPECT = 0.92
UNICORN_H = 640          # 输出高度
ALPHA_THRESHOLD = 24     # 低于此值视为完全透明，忽略边缘噪点
PAD = 6                  # 裁剪后留一点余量，避免描边被切


def trim_to_content(im: Image.Image) -> Image.Image:
    """按 alpha 通道裁掉四周空白。"""
    alpha = im.getchannel("A").point(lambda v: 255 if v > ALPHA_THRESHOLD else 0)
    box = alpha.getbbox()
    if not box:
        return im
    x0, y0, x1, y1 = box
    x0 = max(0, x0 - PAD)
    y0 = max(0, y0 - PAD)
    x1 = min(im.width, x1 + PAD)
    y1 = min(im.height, y1 + PAD)
    return im.crop((x0, y0, x1, y1))


def normalize_canvas(im: Image.Image, aspect: float) -> Image.Image:
    """把裁好的图放到统一比例的透明画布上：水平居中、底部对齐。"""
    w, h = im.size
    target_w = max(w, int(round(h * aspect)))
    canvas = Image.new("RGBA", (target_w, h), (0, 0, 0, 0))
    canvas.paste(im, ((target_w - w) // 2, 0), im)
    return canvas


def main() -> None:
    DST.mkdir(parents=True, exist_ok=True)
    total_before = total_after = 0
    report = []

    for stem in UNICORNS:
        src = SRC / f"{stem}.png"
        if not src.exists():
            print(f"[skip] {src.name} 不存在")
            continue

        im = Image.open(src).convert("RGBA")
        im = trim_to_content(im)
        im = normalize_canvas(im, CANVAS_ASPECT)

        scale = UNICORN_H / im.height
        im = im.resize((int(round(im.width * scale)), UNICORN_H), Image.LANCZOS)

        out = DST / f"{stem}.webp"
        im.save(out, "WEBP", quality=88, method=6)

        before, after = src.stat().st_size, out.stat().st_size
        total_before += before
        total_after += after
        report.append(f"{stem:18s} {im.size[0]:>4}x{im.size[1]:<4} "
                      f"{before/1024:7.0f} KB -> {after/1024:5.0f} KB")

    # 场景背景：不裁剪、不透明，直接缩放
    scene = SRC / "scene_meadow.png"
    if scene.exists():
        im = Image.open(scene).convert("RGB").resize((1280, 853), Image.LANCZOS)
        out = DST / "scene_meadow.webp"
        im.save(out, "WEBP", quality=85, method=6)
        before, after = scene.stat().st_size, out.stat().st_size
        total_before += before
        total_after += after
        report.append(f"{'scene_meadow':18s} {im.size[0]:>4}x{im.size[1]:<4} "
                      f"{before/1024:7.0f} KB -> {after/1024:5.0f} KB")

    print("\n".join(report))
    if total_before:
        print(f"\n总计 {total_before/1024/1024:.2f} MB -> {total_after/1024/1024:.2f} MB "
              f"(省下 {(1-total_after/total_before)*100:.0f}%)")


if __name__ == "__main__":
    main()
