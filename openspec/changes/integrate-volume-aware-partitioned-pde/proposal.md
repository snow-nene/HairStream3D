## Why

现有体积证据审计、strand/depth 分区 PDE 和 RK4 分区守卫尚未形成统一闭环：体积证据未进入正式求解域，正面标签沿视线贯穿全体积，最终导出仍存在跨区发丝。需要以同一头发体积域和三维分区驱动方向场求解与发丝积分，保留真实方向/深度层次突变。

## What Changes

- 将头发、空气、实体、未知及冲突证据整理为有坐标元数据的体积输入，构建统一有效域和物理尺度 SDF；未知不直接判为空气。
- 复用无向双角 strand_map 与 mask-aware depth_map 突变检测，增加边界类型、置信度和可见性约束，以 seg 限定头发区域。
- 用网格首个可见交点（显式 mesh_raycast 模式）或已校准深度表面附近的二维分区种子向体积内部传播标签，替代正面标签无限视线拉伸；支持单 front 和任意可用视角集合。
- 在统一域内复用标签门控 screened-Poisson，明确分区锚定、符号一致化与数值验收。
- 统一根点、方向插值、RK4 步进、碰撞/聚束和最终导出的域与分区约束，检查整条线段，防止跨区后返回。
- 建立 64³ 对照和 128³/10k 验证，联合检查跨区、穿头、越域、长度和覆盖率，避免靠截短或删发丝通过验收。

参考 `docs/HairLRM.md` 的几何与观测协同思路，增加以下免新增训练能力：

- 固定 FLUX 重绘与已有 HairStep 推理链路，原始 front 独立保留；不依赖 DOAE 训练、权重或 latent inversion。
- 统一网格/相机配准门禁、头发与实体语义分离、共同可见区域筛选及同源生成证据权重预算。
- 增加双可见交点差分提升与切平面提升一致性检查，保护遮挡层和突变分区。
- 组合全局均匀、发流显著区域与无向方向均衡采样，提供具有类型和距离权重的表面方向约束。
- 增加分区内符号锚与多股流向歧义报告，采用可靠 guide 优先和覆盖驱动补生长。
- 建立免编码器的重建反馈、回退和等预算消融，分别报告全投影与遮挡后的可见指标。
- 提供物理尺度一致的稀疏计算策略和可选多视角人工曲线接口，保留全自动默认流程。

## Capabilities

### New Capabilities

- `volume-aware-partitioned-integration`: 体积证据到三维分区、分区方向 PDE、受约束发丝积分和导出审计的端到端契约。

### Modified Capabilities

无。现有相关能力仍位于未归档 change 中；本 change 复用其实现并补齐统一体积闭环，不重复声明已有基础检测能力。

## Impact

预期涉及 `lib/multiview_pde.py`、`lib/recon_strategy/weighted_poisson.py`、`scripts/recon_3d/run_pde_multiview.py`，复用 `scripts/vis/audit_strand_depth_partitions.py` 和体积证据审计产物。新增独立脚本按功能放入 `scripts/recon_3d/` 或 `scripts/vis/`，测试放入 `tests/`；使用 pixi。

本次扩展更新现有 OpenSpec 工件，新增要求尚待实施与验收，按 CLI 解析路径存放；实施说明和实验报告放入 `docs/`，生成数据放入 `results/multiview_data/<image_id>/pde_governance/` 的独立实验子目录。实施前必须对实际修改符号执行 GitNexus upstream impact 并报告风险，提交前执行 detect_changes。

前置关联：`detect-strand-depth-partitions`、`enforce-rk4-partition-invariant`。后者的最终跨区验收尚未完成，本 change 将复核并补齐，不能视为已经零违规。保持新链路显式启用，不覆盖现有基线产物。
