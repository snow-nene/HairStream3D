# 体积约束分区 PDE 实施记录

_2026-09-06；OpenSpec：integrate-volume-aware-partitioned-pde；当前为增量实施，尚未完成样例验收。_

## 当前结果

已新增统一体积输入契约、旧证据显式适配、深度表面带播种、域内测地标签传播、跨视图局部编号对应和方向相容性检查。重建入口的 `--volume_partition_bundle` 显式启用新路径；不开启时保留已有路径。严格路径增加同区观测查找、逐组件 PDE 残差、RK4 低支持/曲率缩步、整线段合法性检查、最终候选及序列化 PLY 审计。

用户已确认使用网格可见交点定位三维种子，strand_map/depth_map 继续负责突变分区。已生成主体组件的 64³ PDE／积分诊断，最终 957,330 条线段审计全部通过。完整候选域仍有无种子组件和域外根点，因此这不是全域生产验收。

## 样例证据

样例目录为 `results/multiview_data/0d285f5be7fa09c3dbbf1c9334047888/pde_governance/volume_partition_integration/`。

| 检查 | 实测结果 | 含义 |
| --- | --- | --- |
| 历史最终 PLY 逐点审计 | 8 根、34 个点违规 | 复现此前记录 |
| 历史最终 PLY 整线段审计 | 29 根、62 条线段违规 | 端点合法不能证明中间不跨区；包含保守接触判定 |
| 候选体积 | 34,880 体素、23 个连通组件 | 来源于旧证据和封闭包络 |
| 无根且无原始方向锚组件 | 22 个 | 未静默裁剪，也未宣称可解 |
| 原始根点 | 9,670 个域内、330 个域外 | 未执行投影或合法短连接 |
| 全局 neural depth 校准 | 留出 P90 46.2 mm | 未通过 15 mm 门禁 |
| 左右分区 neural depth 校准 | P90 41.7 / 60.4 mm | 简单分区仿射仍不合格 |
| 方向/深度边界拓扑审计 | q90/q93、5 像素支持带均得到左右两大区 | 复用已有拓扑筛选；未手工画线 |

原始日志、输入与代码 SHA256 位于 `baseline_20260905/manifest.json`；根点域关系位于 `baseline_20260905/root_domain_report.json`；最终线段定位位于 `baseline_20260905/export_segment_audit.json`。二维分区、标定样本和拒绝原因位于 `front_bundle_64/`。历史命令没有完整保存，当前 manifest 明确记录这一限制，不能把历史结果称为已严格复现的配置对照。

## 接口与运行

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 pixi run python scripts/recon_3d/build_volume_partition_bundle.py \
  --data-dir results/multiview_data/0d285f5be7fa09c3dbbf1c9334047888 \
  --output-dir results/multiview_data/0d285f5be7fa09c3dbbf1c9334047888/pde_governance/volume_partition_integration/front_bundle_64 \
  --mesh results/multiview_data/0d285f5be7fa09c3dbbf1c9334047888/pixal3d/hair_mesh_aligned_best.obj
```

此命令当前预期在深度校准门禁失败并保留诊断。`candidate_volume.npz` 是审计中间数据，其标签尚未完成，不是可消费的 `volume_partition_bundle.npz`。有效包要求域内标签为正、域外为零、域与实体不重叠，并具有 XYZ、米、world-to-grid 等明确元数据。

## 验证与影响范围

已通过新增 17 项体积契约/积分测试（包含自适应缩步、最低步长终止、同组件观测引用）、原有 5 项 RK4 分区测试、3 项分区 Poisson 测试、5 项 strand/depth 测试及 7 项基础 Poisson 测试。测试输出集中在 `tests/outputs/volume_partition_integration/`，汇总日志为 `volume_all.log`。这些测试不代表当前图像的端到端验收已经通过。

GitNexus 实际分支索引为 `feat/multi-view-fusion`，仓库名为 `HairStream3D`。本轮实际修改符号的 upstream impact 为 LOW，直接调用者主要是多视图 filter、重建 main 和审计 main。全工作区 detect_changes 为 CRITICAL（23 个符号、18 条流程）；其中包含开始实施前已有的未提交修改，也未完整计入全部新文件，因此不能将该汇总解释为本轮独立差异的精确范围。本轮不提交、不覆盖基线、不切换默认。

## 尚待完成

三维播种来源已按用户确认固定为显式 `--seed-geometry mesh_raycast`。神经深度几何校准门禁在此模式标记为不适用（passed=null），没有将其记录为通过。完整域仍需处理无种子组件及 330 个域外根点。

随后处理组件证据、根部合法连接、分区内符号定向、硬边界从二维到三维的接线、统一根集合的 64³ 三组对照。只有联合几何/长度/覆盖门禁通过，才运行 128³/10k；当前未进行这些全量验收。

## 网格交点实验（2026-09-06）

正面模式 `mesh_front_64/` 得到 50,895 个可见表面样本、7,676 个体素种子；方向由 strand_map 图像切向与网格法线约束提升到三维，不消费神经深度梯度作为几何。二维突变分区在主体域内传播为两区，分别为 20,515 / 14,330 体素。

完整候选域保留 34,880 体素。正面及四视图模式均因 22 个小组件共 35 体素没有种子而拒绝生成正式 bundle。四视图的 left/right/back 暂分配新的全局编号 3/4/5，不能把新增编号解释为已验证的跨视图语义对应。

独立脚本 `scripts/recon_3d/run_volume_partition_smoke.py` 显式只求解有种子的组件，并输出原候选域、隔离掩码、全部根点及选择索引。此脚本不删除违规发丝以通过检查，也不输出生产 bundle。最终诊断目录为 `mesh_front_pde_diagnostic_64_canonical_bounds/`。

| 指标 | 结果 |
| --- | --- |
| 求解体素 | 34,845；隔离 35 |
| PDE 逐区相对残差 | 9.54e-9 / 9.72e-9 |
| PCG | 218 次迭代，CPU 约 3 秒（仅求解） |
| 根点 | 原始 10,000，参与 9,670，域外 330 |
| 最终整线段检查 | 957,330 条，0 违规 |
| 长度 P10 / 中位数 / P90 | 6.27 / 20.23 / 28.56 cm |
| 长度不足 1 cm | 131 根 |

初次诊断 `mesh_front_pde_diagnostic_64/` 发现 1 根发丝的 71 条线段在分区面附近触发审计。定位为积分 float32 边界与审计原始边界不一致，现通过 RK4 的 `volume_bounds` 参数将原始边界传给根点及最终候选整线段守卫；正式入口也接入这一参数。原失败产物保留，修复后重新计算 PDE 与积分，未修改审计阈值。

可视化 `diagnostic_strands.png` 展示两区流向，部分发尾横向弯折及短发丝仍需质量评估。当前没有根部连接、生产后处理、统一基线覆盖率对照和 128³ 验收。局部几何检查通过不能替代这些质量门禁。

复现：

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLCONFIGDIR=/tmp/hairstream-volume-mpl pixi run python scripts/recon_3d/build_volume_partition_bundle.py \
  --data-dir results/multiview_data/0d285f5be7fa09c3dbbf1c9334047888 \
  --output-dir results/multiview_data/0d285f5be7fa09c3dbbf1c9334047888/pde_governance/volume_partition_integration/mesh_front_64 \
  --mesh results/multiview_data/0d285f5be7fa09c3dbbf1c9334047888/pixal3d/hair_mesh_aligned_best.obj \
  --seed-geometry mesh_raycast
# 上一步预期保留种子诊断并拒绝完整 bundle。
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLCONFIGDIR=/tmp/hairstream-volume-mpl pixi run python scripts/recon_3d/run_volume_partition_smoke.py \
  --data-dir results/multiview_data/0d285f5be7fa09c3dbbf1c9334047888 \
  --seed-dir results/multiview_data/0d285f5be7fa09c3dbbf1c9334047888/pde_governance/volume_partition_integration/mesh_front_64 \
  --output-dir results/multiview_data/0d285f5be7fa09c3dbbf1c9334047888/pde_governance/volume_partition_integration/mesh_front_pde_diagnostic_64_canonical_bounds
```

## 短根连接及覆盖率诊断（2026-09-06 续）

新增 `lib/recon_strategy/volume_root_connection.py`，用一个物理体素对角线（本例 10.9137 mm）作为初始化短连接上限。仅允许沿无冲突未知/头发证据接入唯一可判断的分区，对实体、可信空气、其他分区及整段越界进行保守检查。根身份不变，PDE 域及方向场保持固定；连接段是明确记录的初始化例外，不能将其表述为完全位于原生长域内。

实验 `mesh_front_root_connections_64/` 保存 `root_connection_plan.npz`、`connected_strands.npz`、`report.json`，全部 10,000 个原始根有状态。原域内 9,670 根，新增短连接 168 根，共参与 9,838 根；153 根超过距离限制，9 根分区归属模糊。未修改阈值来接入其余根点。35 个无种子体素的 22 个组件继续保留隔离记录、空间中心及证据类别。

168 条连接段审计通过；后续 973,962 条生长线段零违规。发丝长度中位数 20.066 cm，P10/P90 为 5.676/28.511 cm，150 根不足 1 cm。连接产物将第 0 条线段明确作为初始化段，后续段使用严格生长域；这不是生产导出及根连接全流程验收。

固定相同相机和 1 像素线宽，对上一轮网格诊断做增量对照：

| 指标 | 原主体诊断 | 加短连接 |
| --- | --- | --- |
| 发丝数 | 9,670 | 9,838 |
| 有效长度至少 1 cm | 9,539 | 9,688 |
| 长度中位数 | 20.232 cm | 20.066 cm |
| 全投影 front 覆盖率 | 94.364% | 94.371% |
| 全投影泄漏 | 36.039% | 36.119% |

增量门禁通过，长度中位数下降约 0.82%，泄漏增加约 0.080 个百分点。这只比较短连接前后，不能替代 OpenSpec 要求的同分辨率三组基线。完整根数作为统一分母保留，不仅比较幸存者。

全投影泄漏大量集中在脸部：它包含头后被遮挡发丝。新增 `scripts/vis/audit_volume_head_visibility.py` 用 `data/head_model.obj` 的首交点与发丝逐像素前深度比较，得到可见覆盖率 74.979%、可见泄漏 7.981%、被头模遮挡投影像素 50,379。头模是先验而非真实头部，该指标只作补充诊断，采用密集亚像素线段采样，与原 OpenCV 1 像素线宽栅格化也有差异；不得替换原门禁或直接将二者差值解释为精确遮挡率。

可视化：`coverage_comparison.png`（绿色为 seg 内、红色为 seg 外投影）、`root_status_overlay.png`（绿为短连接，红为超距，橙为归属模糊）、`head_visibility_overlay.png`（蓝为头模遮挡、绿为可见 seg 内、红为可见 seg 外）。

新增 5 项根连接测试覆盖身份保留、实体薄墙、空气/冲突、分区歧义及篡改连接拒绝；新增 1 项前后遮挡合成测试。此次新增实现为独立模块和诊断入口，未修改既有函数、生产默认配置或已有实验结果。后续仍需处理分区符号锚定、正式入口的根连接/同区 guide、固定三组对照及生产最终导出。

## HairLRM P0 输入契约

2026-09-06：完成来源清单、原始 front 保护和稳定逐视角 seed 的入口接线，8 项接口/推理替身测试通过。新增共同可见门禁与来源预算基础模块；历史四视图样例因来源和独立配准证据缺失，正式 accepted 均为 0，仅输出门禁前诊断。没有重新生成图谱或接入生产 PDE，详见 [观测输入契约](OBSERVATION_INPUT_CONTRACT.md)。

配准门禁已接入网格播种构建器和观测审计。样例 front 网格交点与 image-pixel 投影的 q95 误差约为 5.1e-13 px；配准通过不代表历史图谱来源已验证，也不解除 35 个无种子体素的组件拒绝。
