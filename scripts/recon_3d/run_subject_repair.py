"""按 Image ID 执行头皮适配、原图守卫下重连接、缺口生长和渲染。"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir', type=Path, required=True)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--head', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--baseline-count', type=int, required=True)
    p.add_argument('--adapt-head', action='store_true', help='实验选项，需另行通过多视图形态验收')
    p.add_argument('--views', nargs='+', choices=['front', 'left', 'right', 'back'], default=['front'])
    a = p.parse_args()
    if 'front' not in a.views or a.baseline_count <= 0:
        p.error('必须包含原图 front，baseline-count 必须为正数')
    for key in ['data_dir', 'input', 'head', 'output_dir']:
        setattr(a, key, getattr(a, key).resolve())
    if not a.output_dir.is_relative_to(a.data_dir):
        p.error('输出必须位于该 Image ID 数据目录内')
    a.output_dir.mkdir(parents=True, exist_ok=False)
    geometry, regrow, final = [a.output_dir/name for name in ['geometry', 'regrow', 'final']]
    env = dict(os.environ, ALSOFT_DRIVERS='null')
    stages = []
    def run(name, command):
        command = list(map(str, command))
        print(f'[{name}]', flush=True)
        subprocess.run(command, cwd=ROOT, env=env, check=True)
        stages.append({'stage': name, 'command': command, 'status': 'completed'})
        (a.output_dir/'stages.json').write_text(json.dumps(stages, indent=2))
    run('prepare', [sys.executable, 'scripts/recon_3d/prepare_subject_scalp.py',
        '--data-dir', a.data_dir, '--head', a.head, '--output-dir', geometry,
        *([] if a.adapt_head else ['--preserve-head']), '--views', *a.views])
    run('template', ['pixi', 'run', 'blender', '-b', 'assets/render_template.blend', '--python-exit-code', '1',
        '-P', 'scripts/render/build_subject_template.py', '--', '--head', geometry/'subject_head_candidate.npz',
        '--output', geometry/'subject_template.blend'])
    run('export_head', ['pixi', 'run', 'blender', '-b', geometry/'subject_template.blend', '--python-exit-code', '1',
        '-P', 'scripts/render/export_template_head.py', '--', '--output-dir', geometry/'template_head'])
    run('regrow', [sys.executable, 'scripts/recon_3d/regrow_subject_strands.py', '--data-dir', a.data_dir,
        '--input', a.input, '--geometry-dir', geometry, '--output-dir', regrow,
        '--baseline-count', a.baseline_count, '--preserve-guide-body', '--views', *a.views])
    run('fill', [sys.executable, 'scripts/recon_3d/fill_subject_surface_gaps.py', '--data-dir', a.data_dir,
        '--geometry-dir', geometry, '--input-dir', regrow, '--output-dir', final,
        '--baseline-count', a.baseline_count, '--views', *a.views])
    run('render', ['pixi', 'run', 'blender', '-b', '--python-exit-code', '1',
        '-P', 'scripts/render/render_all_prefixes.py', '--', '--template', geometry/'subject_template.blend',
        '--input', final/'all_root_prefixes.npz', '--output-dir', a.output_dir/'render', '--views', *a.views])
    print(f'结果：{a.output_dir}', flush=True)


if __name__ == '__main__':
    main()
