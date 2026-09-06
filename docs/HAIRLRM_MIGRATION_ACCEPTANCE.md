# HairLRM 思路迁移与当前验收记录

本文记录 `integrate-volume-aware-partitioned-pde` 对 `docs/HairLRM.md` 的采纳范围。当前链路使用 FLUX 重绘和现有 HairStep 推理生成 `strand_map`、`depth_map` 与 `seg`，不引入 HairLRM 所需的大规模发型训练、DOAE/VAE/Transformer 权重、latent inversion 或新增编码器训练。

## 已采纳

| 主题 | 当前实现 | 证据 |
| --- | --- | --- |
| 几何/语义解耦 | geometry、direction、semantic、partition 独立接口；头部实体抑制方向 | `lib/multiview_interfaces.py` |
| 可见交点提升 | mesh raycast 种子、可见交点差分、轴向一致性门禁 | `lib/multiview_direction_lifting.py`、`scripts/vis/audit_direction_lifting_ablation.py` |
| 观测来源管理 | 原始 front 保护、FLUX 来源 fingerprint、同源预算 | `lib/multiview_observation.py` |
| 突变分区 | strand 双角梯度、depth 跳变/平面残差、拓扑边界和加权传播 | `scripts/vis/audit_strand_depth_partitions.py`、`lib/recon_strategy/volume_partition.py` |
| 表面能量 | 类型化法向惩罚、距离支持带和退化法线屏蔽 | `lib/surface_energy.py`、`lib/recon_strategy/weighted_poisson.py` |
| 多流方向 | 轴向符号锚、未定向和非同轴歧义保留 | `lib/direction_orientation.py` |
| guide/根覆盖 | 根状态保留、同区 guide 门禁、可见缺口补根和随机基线 | `lib/guide_quality.py`、`lib/supplemental_growth.py` |
| 反馈与评价 | 有界 solve→integrate→render→score、局部重求解区域、可见投影指标 | `lib/feedback_controller.py`、`lib/local_pde_resolve.py`、`scripts/vis/audit_visibility_metrics.py` |
| 稀疏与消融 | 物理尺度分块缓存、固定输入/种子/预算的 scenario manifest | `scripts/recon_3d/build_spatial_cache_manifest.py`、`lib/scenario_matrix.py` |

## 明确排除

不采纳 HairLRM 的训练型模块，不下载或依赖其训练资产；不把生成视图当作独立真实观测；不以轨迹覆盖改写硬实体、分区或原始 front 锚；不以端点合法替代整线段体积审计。

## 当前证据边界

单元和合成测试已覆盖接口约束、分区隔离、方向提升、根连接、反馈预算和评价指标。样例数据仍存在 35 个无种子体素、330 个域外根点以及神经深度校准未通过的问题；因此 64³ 三组完整对照、128³/10k 门禁、真实渲染反馈和生产默认切换仍未宣称通过。所有失败状态、拒绝原因和诊断产物必须保留在 `results/multiview_data/<image_id>/pde_governance/` 下。

## 验收顺序

先验证固定输入和来源清单，再验证配准、分区、逐组件 PDE 残差、根/guide 状态和整线段审计；随后运行固定预算消融和可见投影报告，最后才允许比较 64³/128³ 资源与质量指标。任何一项门禁失败都保留报告并阻止生产导出。
