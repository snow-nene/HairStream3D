## Why

当前默认重建在单个连续三维 PDE 域内平滑方向场，`strand_map` 的散度只用于 RK4 聚散调节，不能阻止发缝或深度层次两侧相互平滑，导致完整密度下头发糊成一体。需要先验证能否仅依靠 `strand_map`、`depth_map` 和 `seg` 稳定检测真实方向/层次突变，并形成可供后续分区 PDE 使用的边界。

## What Changes

- 新增隔离的正面突变检测实验，从无向双角方向场计算局部方向不连续度。
- 融合平滑深度梯度与深度局部跳变，降低仅凭二维方向产生的伪边缘。
- 在头发 `seg` 内生成边缘置信图、候选阻隔线与连通分区标签。
- 将检测边界和分区叠加到 `raw_img.png`，并用已有发缝 mask 进行事后命中率评估。
- 人工确认二维边界后，允许显式传入二维分区标签，并在三维 screened-Poisson 中阻断跨标签扩散。

## Capabilities

### New Capabilities

- `strand-depth-partition-audit`: 从方向图和深度图检测头发内部突变边界，生成候选分区、诊断图和机器可读报告。

### Modified Capabilities

无。

## Impact

二维审计仍保持隔离；新增的 `--front_partition_labels` 只在显式启用 `screened_poisson` 时改变 PDE 邻接关系。未传标签时保持现有行为。RK4 根点到分区的归属逻辑本阶段暂不改变。
