"""生成田地场景图 + 4 种作物插画。

田地要从"弹窗列表"改成"主画面换场景"，所以需要：
  1. 一张田地场景背景（和现有的 scene_meadow 同一画风、同一尺寸比例）
  2. 4 种作物长成时的插画（透明底，叠在田地格子上）

画风必须和已有的 scene_meadow 一致 —— 它是日系动画的清新调子、
阳光柔和、远端有山和云。田地是同一片小镇的另一角，所以：
  - 同样的光照方向和色温
  - 同样的俯视角度（略微俯视，能看到地面）
  - 不要出现独角兽（角色是单独叠上去的）

用法：
    py -3 tools/gen_field.py            # 全部生成
    py -3 tools/gen_field.py --only crops
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
OUT_SCENE = ROOT / "assets" / "opt"
OUT_CROP = ROOT / "assets" / "crops"
WORK = ROOT / "_research" / "field_raw"

# 场景：16:10 左右，和 stage 的比例接近，cover 裁切时不会损失太多
SCENE_SIZE = "1536x1024"
CROP_SIZE = "1024x1024"

# 场景提示词。要点：
#   1. 画风对齐 scene_meadow（日系清新、柔和阳光）
#   2. **不要独角兽** —— 角色是运行时叠上去的
#   3. 中间要留出空位放田垄，别让主体全挤在正中
SCENE = (
    "日系动画风格的清新田园场景，用作手机游戏背景。\n"
    "画面内容：小镇边上的一小片菜园/田地。近景是四块整齐的长方形田垄，"
    "土壤是松软的深褐色，垄边用浅色木条围出边框，田垄之间的空隙铺着细碎的浅色砂石小路。\n"
    "田垄后方有一圈低矮的白色木栅栏，栅栏外是铺展开的绿色草坡，"
    "再远处是连绵的青山和几朵白云。\n"
    "左下角放一只小小的木桶和一把靠在栅栏上的锄头，右侧有一丛白色小野花。\n"
    "光照：温暖的午后阳光从左上角斜射下来，柔和均匀，有淡淡的影子。\n"
    "色调：明亮、清新、饱和但不刺眼，草绿、天蓝、奶油白、浅木色为主。\n"
    "构图：**略微俯视**，能清楚看到四块田垄的地面。田垄位于画面下半部，"
    "上方留出天空和远山。整体构图干净，不要有主体挡在正中央。\n"
    "绝对不要出现任何动物、人物、独角兽 —— 这是一张纯场景背景图。\n"
    "不要文字、不要 UI 元素、不要边框。"
)

# 四种作物。要能一眼分辨，且和各自的籽对应。
CROPS = [
    ("clover", "一丛三叶草，圆润饱满的三片心形叶子，嫩绿色哑光，"
               "每株都很短小可爱，簇拥在一起，其中一两株有四片叶子。"),
    ("timothy", "一丛高一些的提摩西草，细长的浅绿色叶片从根部向上散开，"
                "顶端有几个毛茸茸的淡黄色小穗，造型干净利落。"),
    ("alfalfa", "一丛苜蓿，叶片是椭圆形的小三出复叶，深一些的翠绿色，"
                "中间点缀几朵极小的紫色花苞，长得浓密茂盛。"),
    ("dandelion", "一丛蒲公英，锯齿状的叶片贴地铺开，中间抽出两三支花茎，"
                  "顶端是明亮的黄色小绒花，其中一支已经变成白色绒球。"),
]

CROP_STYLE = (
    "风格：NICI 毛绒玩具的可爱风格，圆润饱满的造型，柔和哑光的短毛绒质感，"
    "绿色系为主，粉白点缀，造型干净简洁。"
    "柔和均匀的影棚灯光，轻微自然阴影。"
    "纯透明背景，画面里只有这一丛植物，主体居中，边缘干净利落。"
)


def load(p: Path):
    return cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_UNCHANGED)


def save_webp(p: Path, img, quality=86):
    ok, buf = cv2.imencode(".webp", img, [cv2.IMWRITE_WEBP_QUALITY, quality])
    if ok:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(buf.tobytes())
    return ok


def run(prompt: str, size: str, background: str, outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)
    for old in outdir.glob("musk_*"):
        old.unlink(missing_ok=True)
    cmd = [sys.executable, str(SCRIPT), "generate",
           "--prompt", prompt, "--size", size,
           "--background", background, "--out-dir", str(outdir)]
    t0 = time.time()
    try:
        p = subprocess.run(cmd, cwd=str(SKILL), capture_output=True, text=True,
                           encoding="utf-8", errors="replace", timeout=900)
    except subprocess.TimeoutExpired:
        return None, "超时"
    got = [q for q in outdir.glob("musk_*") if q.stat().st_mtime >= t0 - 2]
    if not got:
        tail = ((p.stdout or "") + (p.stderr or "")).strip().splitlines()
        return None, (tail[-1][:120] if tail else f"rc={p.returncode}")
    return max(got, key=lambda q: q.stat().st_mtime), f"{time.time()-t0:.0f}s"


def do_scene():
    im_path, note = run(SCENE, SCENE_SIZE, "opaque", WORK / "scene")
    if not im_path:
        return "scene", None, note
    im = load(im_path)
    if im is None:
        return "scene", None, "读不出图"
    dest = OUT_SCENE / "scene_field.webp"
    save_webp(dest, im, quality=84)
    return "scene", dest, note


def do_crop(item):
    name, desc = item
    p, note = run(f"{desc}\n\n{CROP_STYLE}", CROP_SIZE, "transparent", WORK / name)
    if not p:
        return name, None, note
    im = load(p)
    if im is None:
        return name, None, "读不出图"
    if im.shape[2] == 3:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2BGRA)
    # 裁到内容边界，去掉多余留白（作物在格子里要尽量占满）
    a = im[:, :, 3]
    ys, xs = np.where(a > 24)
    if len(ys):
        pad = 6
        y0, y1 = max(0, ys.min()-pad), min(im.shape[0], ys.max()+1+pad)
        x0, x1 = max(0, xs.min()-pad), min(im.shape[1], xs.max()+1+pad)
        im = im[y0:y1, x0:x1]
    dest = OUT_CROP / f"{name}.webp"
    save_webp(dest, im, quality=88)
    return name, dest, note


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["scene", "crops"], help="只生成场景或只生成作物")
    args = ap.parse_args()

    if not SCRIPT.exists():
        raise SystemExit(f"找不到生图脚本: {SCRIPT}")

    jobs = []
    if args.only in (None, "scene"):
        jobs.append(("scene", do_scene, ()))
    if args.only in (None, "crops"):
        jobs += [(n, do_crop, ((n, d),)) for n, d in CROPS]

    print(f"生成 {len(jobs)} 张图（3 路并行）\n")
    print(f"{'名称':12s} {'耗时':>7s}  说明")
    print("-" * 60)
    ok = 0
    with ThreadPoolExecutor(max_workers=3) as ex:
        futs = [ex.submit(fn, *a) for _, fn, a in jobs]
        for f in futs:
            name, dest, note = f.result()
            if dest:
                ok += 1
            print(f"{name:12s} {note:>7s}  [{'OK' if dest else 'FAIL'}] {'' if dest else note}")

    print(f"\n完成 {ok}/{len(jobs)}")


if __name__ == "__main__":
    main()
