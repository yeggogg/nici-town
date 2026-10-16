"""用 musk-image 生成四只 NICI 独角兽立绘（透明底）。

为什么改回生成：先前试过"抠官方商品图当立绘"，虽然几何与配色 100% 准确，
但白底白玩偶的接触投影抠不干净（脚下一团白糊，见 _research/matte_all.jpg），
观感不可接受。改为生成 —— 不追求像素级复刻，追求"一眼就是 NICI"。

提高相似度的两个关键：
1. **带参考图**走 edit 端点。纯文字很难稳定复现 NICI 的呆萌神态，
   把官方商品图喂进去能显著提高相似度。
2. 提示词把 NICI 的辨识点写死（下面 BASE 里逐条列了）。
   这些是从官方实物图上真正观察到的，不是套话 —— 早先画成"长毛+动漫大眼"
   就是没抓准这几条。

用法：
    py -3 tools/gen_unicorns.py            # 生成全部候选
    py -3 tools/gen_unicorns.py --sheet    # 只重出联览图
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
SKILL = Path("C:/Users/yeggo/.agents/skills/musk-image")
SCRIPT = SKILL / "scripts" / "musk_image.py"
OUT = ROOT / "assets" / "cand"          # 候选先放这里，挑完再进 assets/opt
SHEET = ROOT / "_research" / "cand_sheet.jpg"

# 造型要点。每一条都对应一次实测踩坑：
#
# 「正面偏四分之三侧」—— 早先只写"四分之三侧"，模型漂到接近正侧面、
#   只露一只眼睛，脸的可爱度全丢了。必须强调"正面为主"。
#
# 「弧线微笑嘴」—— 官方实物确实是粉红椭圆嘴，但照实写了之后，
#   在游戏 191px 的显示尺寸下就是个奇怪的小粉块。改成弧线微笑。
BASE = """一只 NICI 品牌的独角兽毛绒玩偶，商品图。

视角（这条最重要）：**正面偏四分之三侧**，脸和身体朝向观众，头微微转向一侧约 20 度。
必须能完整看到整张脸、两只眼睛、整个口鼻和身体正面的轮廓。
绝对不要正侧面、不要只露一只眼睛、不要背对观众。

必须严格遵循的 NICI 造型特征：
1. 面料是超短密实的圈绒（bouclé），由一颗颗小卷圈组成，哑光、有均匀细密的颗粒感。
   绝对不是长毛绒、不是蓬松炸毛、不是光滑布料、不是亮面。
2. 眼睛是贴在脸部的一对大的扁平白色椭圆眼片，外沿有细细的深色绣线包边。
   眼片中央只有一个很小的黑色圆点瞳孔，眼神憨厚呆萌。
   绝对不要大眼睛、不要睫毛、不要高光点、不要动漫式眼睛。
3. 口鼻圆鼓鼓地向前凸出。两个粉红色的小圆点做鼻孔。
   **嘴巴是一条短短的上扬弧线，像温柔的微笑** —— 不要在嘴的位置放粉色椭圆或色块。
4. 鬃毛是一簇簇高饱和的短毛，从头顶一路排到颈背，体积中等、不夸张、不低伏贴地。
5. 尾巴是一小团圆球状的绒球。
6. 独角是细小的圆锥形，表面有浅浅的螺旋纹。
7. 身体矮胖圆桶形，四条短粗的圆柱腿，腿末端圆钝、没有蹄子、没有分趾。

姿态：四足站立，头微微歪着，精神饱满、生动可爱。
灯光：柔和均匀的影棚灯光，轻微自然阴影。
背景：纯透明背景，画面里只有这只玩偶，边缘干净利落。"""

# 角色 id -> (显示名, 配色描述, 参考图文件名)
CHARS = [
    ("bubble", "泡泡",
     "米白色身体，亮粉红色的鬃毛和尾巴，身体侧面绣着两枚小小的粉色星星。",
     "p20_20220923110846.jpg"),
    ("cloudy", "蓝宝",
     "米白色身体，高饱和的青蓝色鬃毛和尾巴，身体侧面绣着一道小小的彩虹。",
     "p00_20250912154041.jpg"),
    ("peachy", "桃桃",
     "通体粉色、带细密反光亮片的身体，雪白色的蓬松鬃毛和尾巴，身体侧面绣着一枚粉色六角形宝石。",
     "p30_20220125130406.jpg"),
    ("dreamy", "梦梦",
     "通体深灰蓝色、带细密反光亮片的身体，淡粉色的蓬松鬃毛和尾巴，身体侧面绣着一枚粉色菱形宝石。",
     "p31_20220125130513.jpg"),
]

VARIANTS = 4      # 每只出几个候选
WORKERS = 4       # 并行路数


def newest_png(d: Path, since: float) -> Path | None:
    """取目录里刚生成的那张（脚本自动命名为 musk_<时间戳>.png）。"""
    cands = [p for p in d.glob("musk_*.png") if p.stat().st_mtime >= since - 2]
    return max(cands, key=lambda p: p.stat().st_mtime) if cands else None


def one(job) -> tuple:
    """生成一张候选。job = (char_id, variant_no, prompt, ref_or_None)"""
    char_id, vno, prompt, ref = job
    outdir = OUT / f"{char_id}_v{vno}"
    outdir.mkdir(parents=True, exist_ok=True)
    for old in outdir.glob("musk_*.png"):
        old.unlink(missing_ok=True)

    cmd = [sys.executable, str(SCRIPT)]
    if ref is not None:
        cmd += ["edit", "--prompt", prompt, "--image", str(ref)]
    else:
        cmd += ["generate", "--prompt", prompt]
    cmd += ["--size", "1024x1024", "--background", "transparent",
            "--out-dir", str(outdir)]

    import time
    t0 = time.time()
    try:
        p = subprocess.run(cmd, cwd=str(SKILL), capture_output=True, text=True,
                           encoding="utf-8", errors="replace", timeout=900)
    except subprocess.TimeoutExpired:
        return char_id, vno, None, "超时"

    got = newest_png(outdir, t0)
    if got is None:
        tail = ((p.stdout or "") + (p.stderr or "")).strip().splitlines()
        return char_id, vno, None, (tail[-1] if tail else f"rc={p.returncode}")
    dest = outdir / f"{char_id}_v{vno}.png"
    got.replace(dest)
    return char_id, vno, dest, f"{time.time()-t0:.0f}s"


def build_sheet() -> None:
    """候选联览图：棋盘底，标出角色与候选号。"""
    tiles = []
    for char_id, name, _, _ in CHARS:
        for vno in range(1, VARIANTS + 1):
            p = OUT / f"{char_id}_v{vno}" / f"{char_id}_v{vno}.png"
            if not p.exists():
                continue
            im = cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_UNCHANGED)
            if im is None:
                continue
            cell = 300
            h, w = im.shape[:2]
            s = cell / max(h, w)
            im = cv2.resize(im, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
            canvas = np.zeros((cell, cell, 4), np.uint8)
            oy, ox = (cell - im.shape[0]) // 2, (cell - im.shape[1]) // 2
            canvas[oy:oy + im.shape[0], ox:ox + im.shape[1]] = im
            a = canvas[:, :, 3:4].astype(np.float32) / 255.0
            yy, xx = np.mgrid[0:cell, 0:cell]
            checker = np.where((yy // 20 + xx // 20) % 2 == 0, 205, 245).astype(np.uint8)
            checker = cv2.merge([checker, checker, checker])
            comp = (canvas[:, :, :3].astype(np.float32) * a +
                    checker.astype(np.float32) * (1 - a)).astype(np.uint8)
            cv2.putText(comp, f"{name} v{vno}", (8, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.62, (30, 30, 80), 2, cv2.LINE_AA)
            tiles.append(comp)

    if not tiles:
        print("还没有候选图")
        return
    cols = VARIANTS
    rows = (len(tiles) + cols - 1) // cols
    while len(tiles) < rows * cols:
        tiles.append(np.full_like(tiles[0], 250))
    sheet = np.vstack([np.hstack(tiles[r * cols:(r + 1) * cols]) for r in range(rows)])
    SHEET.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(".jpg", sheet, [cv2.IMWRITE_JPEG_QUALITY, 88])
    if ok:
        SHEET.write_bytes(buf.tobytes())
    print(f"联览图: {SHEET}  ({sheet.shape[1]}x{sheet.shape[0]})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sheet", action="store_true", help="只重出联览图")
    args = ap.parse_args()

    if args.sheet:
        build_sheet()
        return

    if not SCRIPT.exists():
        raise SystemExit(f"找不到生图脚本: {SCRIPT}")

    jobs = []
    for char_id, name, color, _refname in CHARS:
        for vno in range(1, VARIANTS + 1):
            # 全部走纯文生图，**不喂参考图**。
            # 上一轮 v1/v2 带了官方商品图做参考，结果视角被参考图带歪了 ——
            # 官方那几张本身就是侧视钥匙扣，模型照抄了侧脸角度，只露一只眼睛。
            # 参考图对"面料质感"有帮助，但对"视角"是负作用，得不偿失。
            jobs.append((char_id, vno, f"{BASE}\n\n配色：{color}", None))

    print(f"生成 {len(jobs)} 张候选（{VARIANTS} 变体 × {len(CHARS)} 只，{WORKERS} 路并行）\n")
    print(f"{'角色':8s} {'候选':>4s} {'耗时':>7s}  说明")
    print("-" * 60)
    done = 0
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        for char_id, vno, dest, note in ex.map(one, jobs):
            name = next(n for c, n, _, _ in CHARS if c == char_id)
            flag = "OK " if dest else "FAIL"
            if dest:
                done += 1
            print(f"{name:8s} v{vno:<3d} {note:>7s}  [{flag}]")

    print(f"\n完成 {done}/{len(jobs)}")
    build_sheet()


if __name__ == "__main__":
    main()
