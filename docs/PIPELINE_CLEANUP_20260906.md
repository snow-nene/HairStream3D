# 管线与 OpenSpec 整理记录

日期：2026-09-06。

本次整理把当前工作分成三类：正式入口、仍在进行的 OpenSpec 变更和历史实验。默认流水线不再静默回退到旧 PDE。

## OpenSpec 状态

已经完成的三个变更已移到 `openspec/changes/archive/2026-09-06-*`：

- `add-local-parting-groom-layer`
- `detect-strand-depth-partitions`
- `fix-parting-topology-with-scalp-cut-pde`

仍在进行的变更保留在 `openspec/changes/`：

- `enforce-rk4-partition-invariant`：64³ 分区 smoke 仍未完成；
- `integrate-volume-aware-partitioned-pde`：128³/10k 生产验收和真实全域消融仍未完成。

空的 `couple-scalp-prefix-to-volume-pde` 变更没有被伪造为完成状态，后续应删除或重新创建 proposal。

## 默认入口

`scripts/run_pipeline.sh` 的 PDE 阶段现在默认为 `governed`，要求显式存在并通过加载校验的体积分区 bundle，并自动使用 `screened_poisson`。旧链路只能显式调用：

```bash
bash scripts/run_pipeline.sh --img_id <image_id> --stages pde --legacy-pde
```

治理路径缺少 bundle 时会停止，不会生成未受体积约束的旧结果。治理 PDE 也不会直接把未绑定模板的发丝交给预览渲染。

## 测试与脚本

`.gitignore` 现在只忽略 `tests/outputs/` 和缓存，新增测试可以进入版本库。`pixi run test` 只运行测试；样例重建单独使用 `pixi run pipeline-sample`。

明确属于早期头模、Alpha Wrap 或完整发丝提取实验的脚本已移到 `scripts/recon_3d/historical/`。正式入口仍在 `scripts/recon_3d/` 顶层。

本次验证：`openspec validate --all` 通过，`pixi run pytest -q tests` 通过 189 项。

## 结果清理

已删除未被文档或脚本引用的重复目录 `pde_parting_hybrid_cap_v30_rerun_20260828/`。其余 `results/` 目录保留，因为它们仍可能是历史报告的证据；后续删除应以报告引用和输入 fingerprint 为依据，不能按目录名称批量删除。
