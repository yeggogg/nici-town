"""生成竖幅田地场景（手机用）。

为什么需要单独一张：
  现有的 scene_field.webp 是 1536x1024 的横幅（宽高比 1.5）。
  手机竖屏的舞台是 347x422（宽高比 0.82），用 cover 缩放时
  会被撑到 630px 宽、左右各裁掉约 127px ——
  最左和最右两块田垄整个跑到屏幕外。
  无论怎么调 CSS 都救不了，因为"横幅图放进竖屏"必然裁切。

所以给手机单独出一张竖幅图：4 块田垄摆成 2x2，下半部为主。
运行时按舞台宽高比自动选横幅还是竖幅（见 layoutFieldScene）。

用法：
    py -3 tools/gen_field_portrait.py
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
SKILL = Path("C:/Users/yeggo/.agents/skills/musk-image")
SCRIPT = SKILL / "scripts" / "musk_image.py"
OUT = ROOT / "assets" / "opt"
WORK = ROOT / "_research" / "field_portrait"

# 竖幅：1024x1280 -> 宽高比 0.8，和手机舞台的 0.82 几乎一致
SIZE = "1024x1280"

PROMPT = (
    "日系动画风格的清新田园场景，**竖版构图**，专门用作手机游戏背景。\n"
    "画面内容：小镇边上的一小片菜园。画面下半部分整齐排列着**四块**长方形田垄，"
    "摆成 2x2（上排两块、下排两块），每块之间留出浅色砂石小路的空隙。\n"
    "**四块田垄全部是刚刚翻好的空地** —— 只有松软的深褐色土壤，"
    "表面有细碎的土粒和浅浅的耙痕，**地里绝对不要有任何植物、菜苗、蔬菜、杂草**，"
    "是空的、等着播种的土。\n"
    "田垄边缘用浅色木条围出边框。\n"
    "田垄后方（画面中部）有一圈低矮的白色木栅栏，栅栏外是铺展开的绿色草坡。\n"
    "画面上部三分之一是天空：明亮的天蓝，几朵蓬松的白云，远处有连绵青山。\n"
    "光照：温暖的午后阳光从左上角斜射下来，柔和均匀，有淡淡的影子。\n"
    "色调：明亮、清新、饱和但不刺眼，草绿、天蓝、奶油白、浅木色为主。\n"
    "构图：**略微俯视**，能清楚看到四块空田垄的地面。四块田垄完整地出现在画面内，"
    "**左右两侧不要裁切**，每块都完整可见。田垄集中在画面下半部，上方留出天空。\n"
    "绝对不要出现任何动物、人物、独角兽 —— 这是一张纯场景背景图。\n"
    "不要文字、不要 UI 元素、不要边框。"
)


def main() -> None:
    WORK.mkdir(parents=True, exist_ok=True)
    for old in WORK.glob("musk_*"):
        old.unlink(missing_ok=True)

    cmd = [sys.executable, str(SCRIPT), "generate",
           "--prompt", PROMPT, "--size", SIZE,
           "--background", "opaque", "--out-dir", str(WORK)]
    t0 = time.time()
    print("生成竖幅田地场景…")
    p = subprocess.run(cmd, cwd=str(SKILL), capture_output=True, text=True,
                       encoding="utf-8", errors="replace", timeout=900)
    got = [q for q in WORK.glob("musk_*") if q.stat().st_mtime >= t0 - 2]
    if not got:
        tail = ((p.stdout or "") + (p.stderr or "")).strip().splitlines()
        raise SystemExit("生成失败: " + (tail[-1] if tail else f"rc={p.returncode}"))

    src = max(got, key=lambda q: q.stat().st_mtime)
    im = cv2.imdecode(np.fromfile(str(src), np.uint8), cv2.IMREAD_UNCHANGED)
    if im is None:
        raise SystemExit(f"读不到 {src}")
    if im.shape[2] == 4:
        im = cv2.cvtColor(im, cv2.COLOR_BGRA2BGR)

    dest = OUT / "scene_field_portrait.webp"
    ok, buf = cv2.imencode(".webp", im, [cv2.IMWRITE_WEBP_QUALITY, 84])
    if not ok:
        raise SystemExit("webp 编码失败")
    dest.write_bytes(buf.tobytes())
    print(f"输出: {dest}  {im.shape[1]}x{im.shape[0]}  {len(buf)//1024} KB  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
