"""测试 scripts/run_pipeline.sh 的阶段调度与参数解析。"""
import subprocess
from pathlib import Path


def run_pipeline_dry(args: list[str]) -> subprocess.CompletedProcess[str]:
    # 使用 bash 检查参数解析与阶段构建
    cmd = ["bash", "-c", """
set -e
source <(sed '/^set -e/q' scripts/run_pipeline.sh | sed '$d') "$@"
# 截取到 DATA_DIR 定义前，仅测试参数解析与 STAGES 构建
echo "IMG_ID=$IMG_ID"
echo "RAW_IMG=$RAW_IMG"
echo "PDE_MODE=$PDE_MODE"
echo "WITH_PARTITION=$WITH_PARTITION"
echo "WITH_GROW_SURFACE=$WITH_GROW_SURFACE"
echo "STAGES=${STAGES[*]}"
""", "_"] + args
    return subprocess.run(cmd, capture_output=True, text=True)


def test_default_stages_without_raw_img():
    res = run_pipeline_dry(["--img_id", "test_id"])
    assert res.returncode == 0
    output = res.stdout
    assert "IMG_ID=test_id" in output
    assert "STAGES=pixal3d extract align_calib render flux maps pde fit_head preview" in output


def test_default_stages_with_raw_img():
    res = run_pipeline_dry(["--img_id", "test_id", "--raw_img", "results/real_imgs/test.png"])
    assert res.returncode == 0
    output = res.stdout
    assert "STAGES=prepare pixal3d extract align_calib render flux maps pde fit_head preview" in output


def test_with_partition_flag():
    res = run_pipeline_dry(["--img_id", "test_id", "--with-partition"])
    assert res.returncode == 0
    output = res.stdout
    assert "WITH_PARTITION=true" in output
    assert "STAGES=pixal3d extract align_calib render flux maps partition pde fit_head preview" in output


def test_with_grow_surface_flag():
    res = run_pipeline_dry(["--img_id", "test_id", "--with-grow-surface"])
    assert res.returncode == 0
    output = res.stdout
    assert "WITH_GROW_SURFACE=true" in output
    assert "STAGES=pixal3d extract align_calib render flux maps pde fit_head grow_surface preview" in output


def test_with_both_partition_and_grow_surface():
    res = run_pipeline_dry(["--img_id", "test_id", "--with-partition", "--with-grow-surface"])
    assert res.returncode == 0
    output = res.stdout
    assert "STAGES=pixal3d extract align_calib render flux maps partition pde fit_head grow_surface preview" in output


def test_explicit_stages_override():
    res = run_pipeline_dry(["--img_id", "test_id", "--stages", "partition", "fit_head", "grow_surface"])
    assert res.returncode == 0
    output = res.stdout
    assert "STAGES=partition fit_head grow_surface" in output


def test_custom_growth_and_partition_options():
    res = run_pipeline_dry([
        "--img_id", "test_id",
        "--partition-seed-geometry", "mesh_raycast",
        "--surface-growth-view", "back",
        "--surface-growth-budget", "500",
        "--surface-offset-m", "0.001"
    ])
    assert res.returncode == 0
    assert "IMG_ID=test_id" in res.stdout


def test_partition_stage_missing_prerequisites():
    fake_id = "_test_dummy_missing_partition"
    cmd = ["bash", "scripts/run_pipeline.sh", "--img_id", fake_id, "--stages", "partition"]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode != 0
    assert "错误: 找不到 3D 毛发网格" in res.stdout or "错误: partition 阶段需要" in res.stdout
    # 清理测试目录
    import shutil
    shutil.rmtree(f"results/multiview_data/{fake_id}", ignore_errors=True)


def test_fit_head_stage_missing_prerequisites():
    fake_id = "_test_dummy_missing_fit_head"
    cmd = ["bash", "scripts/run_pipeline.sh", "--img_id", fake_id, "--stages", "fit_head"]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode != 0
    assert "未找到发丝输入文件" in res.stdout
    import shutil
    shutil.rmtree(f"results/multiview_data/{fake_id}", ignore_errors=True)


def test_grow_surface_stage_missing_prerequisites():
    fake_id = "_test_dummy_missing_grow_surface"
    cmd = ["bash", "scripts/run_pipeline.sh", "--img_id", fake_id, "--stages", "grow_surface"]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode != 0
    assert "grow_surface 需要输入发丝" in res.stdout
    import shutil
    shutil.rmtree(f"results/multiview_data/{fake_id}", ignore_errors=True)

