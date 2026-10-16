"""批量生成 13 件魔法礼品的插画（NICI 风格，透明底）。

礼品原本用的是 emoji —— 和照片级毛绒立绘放在一起很违和，
换成统一风格的插画能明显提升整体观感。

注意：当前 musk_image.py 的 CLI 只认 --prompt/--size/--n/--background/
--output-format/--out-dir/--attempts，**没有 --name**，输出自动命名为
musk_<时间戳>.png，所以生成后由本脚本按礼品名重命名。

用法：
    py -3 tools/gen_gifts.py
    py -3 tools/gen_gifts.py --sheet
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
SKILL = Path("C:/Users/yeggo/.agents/skills/musk-image")
SCRIPT = SKILL / "scripts" / "musk_image.py"
OUT = ROOT / "assets" / "gifts"
WORK = ROOT / "_research" / "gift_raw"
SHEET = ROOT / "_research" / "gift_sheet.jpg"

# 统一风格尾串：保证 13 张彼此一致，也与游戏的马卡龙调性一致
STYLE = (
    "风格：NICI 毛绒玩具的可爱风格，圆润饱满的造型，柔和哑光的短毛绒质感，"
    "粉白奶油配色加玫红点缀，造型干净简洁，不要复杂背景元素。"
    "柔和均匀的影棚灯光，轻微自然阴影。"
    "纯透明背景，画面里只有这一件物品，主体居中，边缘干净利落。"
)

GIFTS = [
    ("clover",   "一片小巧的四叶幸运草，四片圆润饱满的心形叶子，嫩绿色哑光毛绒质感，"
                 "叶片正中绣着一颗小小的粉色爱心。"),
    ("cloud",    "一小团云朵形状的棉花糖，蓬松圆润，奶白色到淡粉色的柔和渐变，看起来软软的。"),
    ("ribbon",   "一卷彩虹色缎带绕成的小蝴蝶结，从粉红、橙、黄、绿到蓝的柔和渐变，缎面微微反光。"),
    ("lollipop", "一支小巧的棒棒糖，糖体是粉白相间的螺旋纹，半透明带一点闪光，下面一根白色小棍。"),
    ("carrot",   "一根小巧的金色胡萝卜，圆润饱满，闪亮的金色，顶部一簇嫩绿色的短叶。"),
    ("candle",   "一支小巧的蜡烛，奶白色蜡身，顶端一小簇淡蓝色火苗，蜡身有细细的云朵压纹，底部一圈粉色。"),
    ("crystal",  "一个可爱的小水晶球，透明圆球里有一朵小小的粉色云和一颗黄色星星，下面连着粉色圆形底座。"),
    ("gem",      "一颗多面切割的宝石，淡粉色半透明，棱面闪闪发光，造型圆润可爱。"),
    ("rainbow",  "一道小巧的彩虹拱门，五条圆润的色带从粉红、橙、黄到蓝，两端各有一小朵白云。"),
    ("carousel", "一个迷你旋转木马模型，粉色和奶白色的木马，顶上彩色条纹的小帐篷顶，造型圆润可爱。"),
    ("crown",    "一顶小巧的王冠，金色亮片质感，尖角圆润，正中间镶嵌一颗粉色爱心宝石。"),
    ("star",     "一颗五角星，柔和的淡黄色星光，轮廓圆润饱满，周围环绕几点细小的闪光。"),
    ("golden",   "一只小小的金色独角兽玩偶，圆润饱满的毛绒造型，通体闪亮的金色，"
                 "头顶一根小小独角，配粉色鬃毛和尾巴。"),
]


def one(item):
    name, desc = item
    d = WORK / name
    d.mkdir(parents=True, exist_ok=True)
    for old in d.glob("musk_*.png"):
        old.unlink(missing_ok=True)

    cmd = [sys.executable, str(SCRIPT), "generate",
           "--prompt", f"{desc}\n\n{STYLE}",
           "--size", "1024x1024", "--background", "transparent",
           "--out-dir", str(d)]
    t0 = time.time()
    try:
        p = subprocess.run(cmd, cwd=str(SKILL), capture_output=True, text=True,
                           encoding="utf-8", errors="replace", timeout=900)
    except subprocess.TimeoutExpired:
        return name, None, "超时"

    got = [q for q in d.glob("musk_*.png") if q.stat().st_mtime >= t0 - 2]
    if not got:
        tail = ((p.stdout or "") + (p.stderr or "")).strip().splitlines()
        return name, None, (tail[-1] if tail else f"rc={p.returncode}")

    OUT.mkdir(parents=True, exist_ok=True)
    dest = OUT / f"{name}.png"
    max(got, key=lambda q: q.stat().st_mtime).replace(dest)
    return name, dest, f"{time.time()-t0:.0f}s"


def build_sheet():
    tiles = []
    for name, _, _ in [(g[0], 0, 0) for g in GIFTS]:
        p = OUT / f"{name}.png"
        if not p.exists():
            continue
        im = cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_UNCHANGED)
        if im is None:
            continue
        cell = 200
        h, w = im.shape[:2]
        s = cell / max(h, w)
        im = cv2.resize(im, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
        canvas = np.zeros((cell, cell, 4), np.uint8)
        oy, ox = (cell - im.shape[0]) // 2, (cell - im.shape[1]) // 2
        canvas[oy:oy + im.shape[0], ox:ox + im.shape[1]] = im
        a = canvas[:, :, 3:4].astype(np.float32) / 255.0
        yy, xx = np.mgrid[0:cell, 0:cell]
        ck = np.where((yy // 18 + xx // 18) % 2 == 0, 208, 246).astype(np.uint8)
        ck = cv2.merge([ck, ck, ck])
        comp = (canvas[:, :, :3].astype(np.float32) * a +
                ck.astype(np.float32) * (1 - a)).astype(np.uint8)
        tiles.append(comp)

    if not tiles:
        print("还没有礼品图")
        return
    cols = 7
    rows = (len(tiles) + cols - 1) // cols
    while len(tiles) < rows * cols:
        tiles.append(np.full_like(tiles[0], 250))
    sheet = np.vstack([np.hstack(tiles[r * cols:(r + 1) * cols]) for r in range(rows)])
    ok, buf = cv2.imencode(".jpg", sheet, [cv2.IMWRITE_JPEG_QUALITY, 88])
    if ok:
        SHEET.write_bytes(buf.tobytes())
    print(f"联览图: {SHEET}  ({sheet.shape[1]}x{sheet.shape[0]})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sheet", action="store_true")
    args = ap.parse_args()
    if args.sheet:
        build_sheet()
        return

    print(f"生成 {len(GIFTS)} 件礼品插画（4 路并行）")
    ok = 0
    with ThreadPoolExecutor(max_workers=4) as ex:
        for name, dest, note in ex.map(one, GIFTS):
            flag = "OK " if dest else "FAIL"
            if dest:
                ok += 1
            print(f"  [{flag}] {name:10s} {note[:70]}")
    print(f"\n完成 {ok}/{len(GIFTS)}")
    build_sheet()


if __name__ == "__main__":
    main()
