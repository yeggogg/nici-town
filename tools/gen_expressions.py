"""用 musk-image 的改图端点，把四只立绘各生成三种表情。

为什么不用 CSS 叠加嘴型：
先试过"用毛色补丁盖住原嘴、再画新弧线"。补丁色是从立绘上实测采样的，
但在照片级布偶上，毛色有明暗过渡和绒毛纹理，
一块纯色圆斑盖上去就是一块**补丁**（粉毛上尤其明显，像贴了创可贴）。
所以改成直接生成 —— 让模型把嘴和眼神一起改掉，比贴图自然。

关键约束：**除了表情，其他都不能变**。
否则切换表情时身体会跳一下（视角、姿态、比例全变了，看着像换了只玩偶）。
所以：
  1. 提示词反复强调"只改嘴巴和眼神，其余完全不变"
  2. 以**当前立绘**作为参考图送进改图端点
  3. 生成后用同一套 normalize 归一化（底部对齐、定高），至少保证脚和大小一致
  4. 生成后做**差分检查**，量出身体到底漂了多少（见 tools/check_expr_drift.py）

用法：
    py -3 tools/gen_expressions.py            # 生成全部
    py -3 tools/gen_expressions.py --only bubble
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
SRC = ROOT / "assets" / "opt"
OUT = ROOT / "assets" / "expr"
WORK = ROOT / "_research" / "expr_raw"

CANVAS_ASPECT = 0.92
OUT_HEIGHT = 640

# 三种表情。阈值要和游戏里的 MOODS 分级对得上（见 index.html）。
MOODS = {
    "happy": (
        "把嘴巴改成可爱的笑容 —— 嘴角明显向上扬，露出开心的神态。"
        "眼睛可以稍微眯一点、更弯更亮，显得心情很好。",
        70,   # happy >= 70 用这张
    ),
    "calm": (
        "把嘴巴改成平静的淡淡微笑 —— 一条很浅的、几乎平的小弧线。"
        "眼神保持平和放松。",
        40,
    ),
    "sad": (
        "把嘴巴改成难过委屈的表情 —— 嘴角向下撇成一条小弧线，看起来没精打采、有点委屈。"
        "眼睛可以稍微下垂、显得可怜巴巴的。"
        # 强调"委屈"而不是"生气"：早先的描述让模型画成了倒八字怒眉，
        # 那只适合发脾气，不适合"没被照顾好所以蔫了"。
        "注意是**委屈没精神**，不是生气 —— 绝对不要画眉毛、不要怒目、不要凶。",
        0,
    ),
}

# 提示词的共同部分：身份 + 不许改的东西
KEEP = (
    "这是一只 NICI 独角兽毛绒玩偶的商品图，请你**只修改它的表情（嘴巴和眼神）**。\n"
    "必须严格保持不变的部分：\n"
    "- 身体姿势、四肢位置、头的朝向和角度\n"
    "- 身体大小、在画面中的位置和占比\n"
    "- 毛绒面料的质感（短密圈绒）、颜色、明暗和光影\n"
    "- 鬃毛、尾巴、独角的形状和位置\n"
    "- 身上的刺绣标记（星星/彩虹/宝石等）\n"
    "- 纯透明背景，画面里只有这只玩偶\n"
    "绝对不要：改变体型比例、改变视角、换姿势、加配饰、改背景、改毛色。"
)


def load(p: Path):
    return cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_UNCHANGED)


def save(p: Path, img, quality: int = 90):
    ok, buf = cv2.imencode(".webp", img, [cv2.IMWRITE_WEBP_QUALITY, quality])
    if ok:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(buf.tobytes())
    return ok


def normalize(rgba, aspect=CANVAS_ASPECT, height=OUT_HEIGHT):
    """裁到内容边界 -> 统一比例画布（水平居中、底部对齐）-> 定高缩放。

    与 tools/finalize_sprites.py 同一套规格 —— 保证表情图和主立绘
    尺寸一致、脚踩同一水平线，切换时才不会跳。
    """
    a = rgba[:, :, 3]
    ys, xs = np.where(a > 24)
    if len(ys) == 0:
        return rgba
    pad = 4
    x0, y0 = max(0, xs.min() - pad), max(0, ys.min() - pad)
    x1 = min(rgba.shape[1], xs.max() + 1 + pad)
    y1 = min(rgba.shape[0], ys.max() + 1 + pad)
    crop = rgba[y0:y1, x0:x1]

    ch, cw = crop.shape[:2]
    tw = max(cw, int(round(ch * aspect)))
    canvas = np.zeros((ch, tw, 4), np.uint8)
    ox = (tw - cw) // 2
    canvas[:, ox:ox + cw] = crop

    sc = height / ch
    return cv2.resize(canvas, (int(round(tw * sc)), height), interpolation=cv2.INTER_AREA)


def one(job):
    char, mood, prompt, ref = job
    d = WORK / f"{char}_{mood}"
    d.mkdir(parents=True, exist_ok=True)
    for old in d.glob("musk_*"):
        old.unlink(missing_ok=True)

    cmd = [sys.executable, str(SCRIPT), "edit",
           "--prompt", f"{prompt}\n\n{KEEP}",
           "--image", str(ref),
           "--size", "1024x1024", "--background", "transparent",
           "--out-dir", str(d)]
    t0 = time.time()
    try:
        p = subprocess.run(cmd, cwd=str(SKILL), capture_output=True, text=True,
                           encoding="utf-8", errors="replace", timeout=900)
    except subprocess.TimeoutExpired:
        return char, mood, None, "超时"

    got = [q for q in d.glob("musk_*") if q.stat().st_mtime >= t0 - 2]
    if not got:
        tail = ((p.stdout or "") + (p.stderr or "")).strip().splitlines()
        return char, mood, None, (tail[-1] if tail else f"rc={p.returncode}")

    im = load(max(got, key=lambda q: q.stat().st_mtime))
    if im is None:
        return char, mood, None, "读不出生成图"
    if im.shape[2] == 3:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2BGRA)

    out = normalize(im)
    dest = OUT / f"{char}_{mood}.webp"
    save(dest, out)
    return char, mood, dest, f"{time.time()-t0:.0f}s"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", help="只生成某些角色，逗号分隔，例如 bubble,cloudy")
    ap.add_argument("--moods", help="只生成某些表情，逗号分隔，例如 sad")
    args = ap.parse_args()

    if not SCRIPT.exists():
        raise SystemExit(f"找不到生图脚本: {SCRIPT}")

    chars = [p.stem.replace("unicorn_", "") for p in sorted(SRC.glob("unicorn_*.webp"))]
    if args.only:
        want = {c.strip() for c in args.only.split(",") if c.strip()}
        chars = [c for c in chars if c in want]

    moods = list(MOODS)
    if args.moods:
        want = {m.strip() for m in args.moods.split(",") if m.strip()}
        moods = [m for m in moods if m in want]

    jobs = []
    for c in chars:
        ref = SRC / f"unicorn_{c}.webp"
        for mood in moods:
            prompt, _thr = MOODS[mood]
            jobs.append((c, mood, prompt, ref))

    print(f"生成 {len(jobs)} 张表情图（{len(chars)} 只 × {len(moods)} 种，4 路并行）\n")
    print(f"{'角色':10s} {'表情':7s} {'耗时':>7s}  说明")
    print("-" * 56)
    ok = 0
    with ThreadPoolExecutor(max_workers=4) as ex:
        for char, mood, dest, note in ex.map(one, jobs):
            flag = "OK " if dest else "FAIL"
            if dest:
                ok += 1
            print(f"{char:10s} {mood:7s} {note:>7s}  [{flag}]")
    print(f"\n完成 {ok}/{len(jobs)}  ->  {OUT}")


if __name__ == "__main__":
    main()
