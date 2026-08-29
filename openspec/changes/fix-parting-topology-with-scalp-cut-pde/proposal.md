## Why

当前发缝约束只以有限权重 PDE 法向惩罚、二维 RK4 投影和根部后处理存在，离散邻接仍跨越发缝，因此无法同时保证零桥接、根段贴附和端点自然闭合。需要把发缝侧别提升为求解域的拓扑状态，并先用非全量冒烟证明硬不变量，再接入真实重建链路。

## What Changes

- 新增通用头皮 cut graph，将弯曲发缝表示为带端点 cap 的左右双图册
- 在切开头皮上求解根部切向 PDE，并把结果提升为现有三维 screened-Poisson 的根部薄层边界
- 为发根和发束携带不可变侧别标签，使根部积分与 guide 查找不能跨侧
- 用统一的根部薄层坐标替代“只移动第 0 点”的独立最近点吸附
- 新增 S0–S3 合成拓扑冒烟和 S4 真实冠部小样本冒烟，不运行 `10,000` 根或 `128³` 全量重建
- 新路径默认关闭，现有重建基线和 CLI 行为保持不变

## Capabilities

### New Capabilities

- `parting-topology-pde`: 定义发缝 cut graph、根部薄层 PDE、侧别保持、端点 cap 和无符号线场行为
- `parting-topology-smoke`: 定义合成与真实冠部非全量冒烟、数值指标和硬停止条件

### Modified Capabilities

当前没有既有 OpenSpec capability 需要修改。

## Impact

主要影响 `lib/recon_strategy/` 下的 PDE/曲面场工具、`scripts/recon_3d/run_pde_multiview.py` 的可选接线以及 `tests/` 下的拓扑冒烟。实验输出仅允许写入 `tests/outputs/parting_topology_smoke/` 和指定 Image ID 的 `results/multiview_data/.../pde_governance/parting_topology_smoke/`。首轮不新增模型依赖，不改变默认参数，不执行全量重建。
