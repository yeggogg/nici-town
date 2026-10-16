"""用 reach-seedance 生成开场动画。

为什么视频不能做常驻立绘、却能做开场：
  视频输出是 RGB，**没有 alpha 通道**。我们的立绘要贴在草地上，必须透明，
  白毛绒玩偶事后抠图会逐帧抖动。但开场动画是整屏播放、自带背景的，
  不需要透明 —— 正好是视频的强项。

以游戏里那张草地场景图作为**首帧**，保证转场时背景完全衔接得上。

用法：
    py -3 tools/gen_intro_video.py --dry-run     # 只验证参数，不花钱
    py -3 tools/gen_intro_video.py               # 真跑
"""
import argparse
import subprocess
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
SKILL = Path("C:/Users/yeggo/.agents/skills/reach-seedance")
SCRIPT = SKILL / "scripts" / "reach_video.py"
FIRST_FRAME = ROOT / "assets" / "scene_meadow.png"
OUT = ROOT / "assets" / "video"

# 提示词要点（按 skill 的建议）：
#   主体外观要具体、动作过程要写清、要有镜头语言、
#   还要把"玩偶质感"写死，否则模型会给一只写实卡通马而不是毛绒玩偶。
PROMPT = (
    "一只可爱的米白色毛绒独角兽玩偶，短密的圈绒面料，亮粉红色的鬃毛和尾巴，"
    "一对扁平的大白眼球配小黑点瞳孔，粉红色的小鼻孔，"
    "从画面左侧轻快地小跑进入草地中央，然后放慢脚步停下来。"
    "停下后它歪了歪头，两只耳朵轻轻抖动，尾巴微微摇摆，像是在好奇地打量镜头。"
    "镜头缓慢向前推近，最后停在它半身的位置。"
    "阳光明媚，微风拂过草地，白色和黄色的小野花轻轻摇摆，远处云影缓缓移动。"
    "温暖柔和、安静治愈的氛围，日系动画的清新调子。"
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--resolution", default="720p")
    ap.add_argument("--duration", default="5")
    args = ap.parse_args()

    if not SCRIPT.exists():
        raise SystemExit(f"找不到 seedance 脚本: {SCRIPT}")
    if not FIRST_FRAME.exists():
        raise SystemExit(f"找不到首帧图: {FIRST_FRAME}")

    OUT.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, str(SCRIPT), "generate",
        "--prompt", PROMPT,
        "--model", "seedance-2-fast",
        "--resolution", args.resolution,
        "--duration", args.duration,
        "--aspect-ratio", "adaptive",
        "--first-frame", str(FIRST_FRAME),
        "--out-dir", str(OUT),
    ]
    if args.dry_run:
        cmd.append("--dry-run")

    print(f"首帧: {FIRST_FRAME.name}")
    print(f"参数: {args.resolution} / {args.duration}s / seedance-2-fast / adaptive\n")
    p = subprocess.run(cmd, cwd=str(SKILL), text=True, encoding="utf-8",
                       errors="replace", timeout=1200)
    print(f"\n退出码: {p.returncode}")
    if not args.dry_run and OUT.exists():
        for f in sorted(OUT.iterdir()):
            print(f"  {f.name}  {f.stat().st_size // 1024} KB")


if __name__ == "__main__":
    main()
