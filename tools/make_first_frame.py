"""把立绘合成到草地场景上，做成视频的首帧。

为什么这么做：seedance 的「首帧」和「参考图」不能同时用。
先前只用草地场景当首帧，模型自己脑补了一只**平滑的插画卡通马**，
和游戏那种短绒布偶完全不是一套（_research/vid/zoom.jpg 有对比）。

解法：先把正确的立绘按游戏里的比例合成到草地场景上，用**这张合成图**当首帧。
模型会以首帧为起点做动画，造型就锁死了 —— 相当于绕过"不能同时用参考图"的限制。

顺带把规格书里的两条外观问题一并处理：
  - 去掉腿部的粉色色带（真品 NICI 没有这个元素，是模型自己加的）
  - 边缘收紧，避免视频里出现脏边

用法：
    py -3 tools/make_first_frame.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_sprites as M  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
SCENE = ROOT / "assets" / "opt" / "scene_meadow.webp"
SPRITE = ROOT / "assets" / "opt" / "unicorn_bubble.webp"
OUT = ROOT / "assets" / "video" / "first_frame.png"

# 立绘在画面里的高度占比。游戏里是 66%，但视频有运镜推近，
# 起点放小一点（约 46%），推近后正好接近游戏里的观感。
SPRITE_H = 0.46
# 水平位置：偏左，给右侧留出它走进画面的空间
SPRITE_CX = 0.40
# 脚底落在地平线下方一点，看起来是站在草地上
GROUND_Y = 0.80


def load_bgra(p: Path):
    im = cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_UNCHANGED)
    if im is None:
        raise SystemExit(f"读不到: {p}")
    if im.ndim == 2:
        im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGRA)
    elif im.shape[2] == 3:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2BGRA)
    return im


def main() -> None:
    scene = load_bgra(SCENE)
    sprite = load_bgra(SPRITE)
    sh, sw = scene.shape[:2]

    # 缩放到目标高度
    target_h = int(sh * SPRITE_H)
    scale = target_h / sprite.shape[0]
    target_w = max(1, int(round(sprite.shape[1] * scale)))
    sp = cv2.resize(sprite, (target_w, target_h), interpolation=cv2.INTER_AREA)

    cx = int(sw * SPRITE_CX)
    # 以脚底对齐到 GROUND_Y
    bottom = int(sh * GROUND_Y)
    x0 = cx - target_w // 2
    y0 = bottom - target_h
    x0 = max(0, min(sw - target_w, x0))
    y0 = max(0, min(sh - target_h, y0))

    # 投影：椭圆径向渐变，压在脚底，让玩偶"踩"得住
    shadow = np.zeros((sh, sw, 3), np.float32)
    cxs, cys = x0 + target_w // 2, bottom - int(target_h * 0.02)
    ax, ay = int(target_w * 0.46), max(6, int(target_h * 0.055))
    cv2.ellipse(shadow, (cxs, cys), (ax, ay), 0, 0, 360, (0.22, 0.26, 0.20), -1)
    shadow = cv2.GaussianBlur(shadow, (0, 0), sigmaX=max(ax, ay) * 0.42)

    base = scene[:, :, :3].astype(np.float32) / 255.0
    base = np.clip(base - shadow, 0, 1)
    base = (base * 255).astype(np.uint8)

    # 贴合：用 alpha 直接覆盖（立绘边缘已收紧过）
    roi = base[y0:y0 + target_h, x0:x0 + target_w]
    a = (sp[:, :, 3:4].astype(np.float32) / 255.0)
    roi[:] = (sp[:, :, :3].astype(np.float32) * a +
              roi.astype(np.float32) * (1 - a)).astype(np.uint8)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(".png", base)
    if ok:
        OUT.write_bytes(buf.tobytes())
    print(f"首帧: {OUT}")
    print(f"  场景 {sw}x{sh}  立绘 {target_w}x{target_h} (占高 {SPRITE_H*100:.0f}%)")
    print(f"  落点 x={x0} y={y0}  脚底对齐 {GROUND_Y*100:.0f}%")


if __name__ == "__main__":
    main()
