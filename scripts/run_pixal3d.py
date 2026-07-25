"""
运行 Pixal3D 3D 网格生成推理脚本。

用法:
    pixi run python scripts/run_pixal3d.py \
        --image results/real_imgs/resized_img/0d285f5be7fa09c3dbbf1c9334047888.png \
        --output results/test_pixal3d/output.glb
"""

import sys, os, subprocess
import argparse

def main():
    parser = argparse.ArgumentParser(description="Pixal3D 3D网格生成一键推理脚本")
    parser.add_argument("--image", required=True, help="输入图片路径 (e.g., results/real_imgs/resized_img/xxx.png)")
    parser.add_argument("--output", default="results/test_pixal3d/output.glb", help="输出 GLB 文件路径")
    parser.add_argument("--seed", type=int, default=0, help="随机种子")
    parser.add_argument("--fov", type=float, default=-1.0, help="手动指定相机 FOV (弧度，例如 0.2)。默认 -1 表示使用 MoGe-2 自动估计")
    parser.add_argument("--low_vram", action="store_true", help="启用低显存模式 (显存需求降至 10-12GB)")
    parser.add_argument("--resolution", type=int, default=-1, help="分辨率 (1024 或 1536)")
    args = parser.parse_args()

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    pixal3d_dir = os.path.join(project_root, "ext", "Pixal3D")
    
    # 转换为绝对路径
    image_path = os.path.abspath(args.image)
    output_path = os.path.abspath(args.output)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    print("=" * 60)
    print(f"[Pixal3D 推理] 输入图片: {image_path}")
    print(f"[Pixal3D 推理] 输出路径: {output_path}")
    print("=" * 60)
    
    cmd = [
        "conda", "run", "-n", "trellis2",
        "python", "inference.py",
        "--image", image_path,
        "--output", output_path,
        "--seed", str(args.seed),
    ]
    if args.fov > 0:
        cmd.extend(["--fov", str(args.fov)])
    if args.low_vram:
        cmd.append("--low_vram")
    if args.resolution > 0:
        cmd.extend(["--resolution", str(args.resolution)])
    
    try:
        res = subprocess.run(cmd, cwd=pixal3d_dir, check=True)
        print("\n[成功] Pixal3D 推理完成！GLB 已保存至:", output_path)
    except subprocess.CalledProcessError as e:
        print(f"\n[错误] Pixal3D 推理失败，退出码: {e.returncode}")
        sys.exit(e.returncode)

if __name__ == "__main__":
    main()
