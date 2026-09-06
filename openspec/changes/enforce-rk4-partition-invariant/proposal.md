## Why

分区 screened-Poisson 已经阻断跨标签扩散，但当前样例导出的 9,291 根发丝中仍有 727 根进入非根部分区，跨区率为 7.82%，且一半在第 3 个采样点前发生。需要把分区约束延伸到 RK4 查询、候选更新和聚束阶段，才能保证 PDE 隔离最终落实到发丝轨迹。

## What Changes

- 为每根发根从三维分区体中读取并固定正整数标签。
- 为 RK4 的场查询和候选更新增加标签感知门禁，禁止轨迹进入非根部分区。
- 对跨区候选执行最小回退或冻结，并记录发生位置和次数。
- 限制 guide strand 分配和 clumping 只在同一分区内进行。
- 输出跨区候选率、修正率、最终违规率及分区发丝数量，未提供标签时保持原行为。

## Capabilities

### New Capabilities

- `rk4-partition-invariant`: 保证分区 PDE 生成的发丝在 RK4 积分和聚束过程中保持其根部分区，并提供可审计指标。

### Modified Capabilities

无。

## Impact

主要影响 `scripts/recon_3d/run_pde_multiview.py` 中的 RK4 合成、guide strand 分配和诊断输出，并增加 `tests/` 下的分区轨迹回归测试。分区标签仍为显式可选输入；未启用 `--front_partition_labels` 时不改变现有重建结果。
