# 体积分区 PDE 样例验收报告

样例：`0d285f5be7fa09c3dbbf1c9334047888`。本报告只汇总已经落盘的独立 run，不把诊断产物当作生产 bundle。

## 已完成的运行

| run | 状态 | 证据 |
| --- | --- | --- |
| `mesh_front_64` | 拒绝完整 bundle | 35 个无种子体素、22 个小组件 |
| `mesh_four_view_64` | 拒绝完整 bundle | 四视图仍有同一组无种子组件 |
| `mesh_front_pde_diagnostic_64_canonical_bounds` | 诊断通过 | 34,845 体素，逐区残差约 `1e-8`，957,330 条生长线段零违规 |
| `mesh_front_root_connections_64` | 增量诊断通过 | 168 条短连接审计通过，973,962 条生长线段零违规 |
| `lifting_ablation_front_64` | 消融完成 | 切平面、交点差分和一致性筛选均记录接受率与角度统计 |
| `mesh_front_root_connections_64_snapshotted` | 诊断通过 | 9,838 根参与，168 条短连接审计通过，973,962 条生长线段零违规，并保存 correction/integration/export 三阶段快照 |

## 门禁结论

完整候选域包含 34,880 个体素和 23 个连通组件；35 个体素所在的 22 个组件没有网格可见交点种子。系统保留这些组件并拒绝正式 bundle，未通过扩大表面带、删除组件或随机补标签来掩盖问题。原始根集合有 330 个域外根点，短连接只允许唯一分区且通过实体/冲突/整段审计的 168 根。

因此当前不能宣称 64³ 三组生产验收通过，也不能启动 128³/10k 门禁。诊断场只用于检查 PDE 残差、根状态、覆盖和泄漏，不切换默认配置。

四视图补证据复核：`mesh_four_view_64` 的 front/left/right/back 表面 seed 分别为 7,676/8,660/10,639/12,008 个体素；四视图表面 seed union 仍不能作为体积 coverage，传播后 authoritative coverage 仍为 34,845/34,880，剩余 35 个体素分属 22 个组件。系统因此拒绝把表面 seed union 直接写成体积分区标签。

重新运行的根连接诊断现在保存 `stage_snapshots/stage_manifest.json`，其中包含每个阶段的数组形状、SHA256、根标签、连接状态和审计元数据。该快照能归因新的违规来源；它不能追溯旧 PLY 中已经丢失的历史中间状态。

## 后续解锁条件

需要为无种子组件提供可审计的可见交点或人工/多视图证据，并重新运行三组固定输入对照；只有完整域、根覆盖、整线段实体/分区守卫和可见评价全部通过后，才允许运行 128³/10k。所有失败报告保留在对应 `pde_governance/volume_partition_integration/<run_id>/` 目录。
# 降级域验收（2026-09-06）

对于没有可见交点、注册视图或人工曲线证据的 22 个组件，采用
`unknown/excluded` 策略。35 个体素不赋予 partition 标签，也不参与 PDE
传播、积分和正式发丝导出。可信域包含 34,845 个体素，coverage 为 100%；
原始 34,880 体素全域验收仍标记为未通过。机器可读证据位于
`volume_partition_integration/mesh_front_64/downgraded_domain_acceptance/`。

该策略允许在可信域上继续进行 64³ PDE 质量检查，但不能把结果描述为原始
全域覆盖，也不能据此跳过 128³/10k 的同策略资源和泄漏检查。
