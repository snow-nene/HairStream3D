# 历史实验脚本

此目录保存不属于默认重建入口的早期头模、Alpha Wrap 和完整发丝提取实验。
它们用于复现实验报告，不作为当前 PDE 或模板安全渲染管线的依赖。

当前正式入口保留在上一级目录：

- `run_pde_multiview.py`
- `build_volume_partition_bundle.py`
- `run_volume_partition_smoke.py`
- `fit_strands_to_template_head.py`

历史脚本的输入通常包含固定样例路径或实验参数。新增实验应使用显式命令行参数，
不要把历史脚本重新接回 `scripts/run_pipeline.sh`。
