# HairLRM 思路迁移：几何域、表面约束与 front 反馈

样例：`0d285f5be7fa09c3dbbf1c9334047888`。日期：2026-09-06。

本轮实现了目标分辨率几何查询、头内域裁剪、内外表面分工、有限候选的 front 反馈对照，以及语义终止归因。覆盖和泄漏同时优于上一轮边界切向 PDE 基线，但额角缺口、未知语义区域和分区拓扑仍未解决。结果保留为可审查候选，不宣称完整 HairLRM 复现或全发型验收通过。

## 📊 对照结果

同一批 9,695 个根点、相同根身份与原有分区 ID，预算均为 192 mm。重建几何域只删除有头部内侧证据的节点，没有为未知域新增分区标签。

| 指标 | 上轮边界切向 PDE | 新几何域＋外表面约束 | 再加 front 反馈 |
| --- | ---: | ---: | ---: |
| 正面可见 seg 覆盖 | 70.85% | 74.14% | 72.15% |
| 正面可见 seg 外泄漏 | 6.68% | 5.61% | 2.99% |
| 有效长度中位数 | 163.50 mm | 184.50 mm | 190.88 mm |
| 跑满预算根数 | 3865 | 4629 | 4841 |
| 零长度根数 | 2 | 2 | 3 |
| 长度超过 50 mm 根数 | 8841 | 8813 | 8691 |
| 头部内侧超过 10 μm 的采样轨迹 | 0 | 0 | 0 |
| 最终体素审计非法线段 | 0 | 0 | 0 |

外表面约束组相比基线覆盖增加 3.29 个百分点、泄漏降低 1.07 个百分点。front 反馈组泄漏进一步降低，但覆盖比外表面组少约 1.99 个百分点，转向约束终止增加到 289 根。不能因为跑满预算数量最多，就认定 front 反馈组形态最好。

覆盖优先候选采用 `typed_outer`；`front_feedback` 保留为低泄漏对照。二者均通过本次几何与覆盖/泄漏非退化检查，但这个检查没有验证真实发型、方向误差和多视图质量。front 同时参与优化和评价，指标属于同一输入图像的拟合结果，不是独立测试集效果。

## 🖼️ 已生成的渲染

新几何域＋外表面约束，渲染全部 9,693 条非零前缀：

![覆盖优先候选](../results/multiview_data/0d285f5be7fa09c3dbbf1c9334047888/pde_governance/volume_partition_integration/typed_geometry_domain_feedback_20260906/typed_outer/render_all/front.png)

[左视图](../results/multiview_data/0d285f5be7fa09c3dbbf1c9334047888/pde_governance/volume_partition_integration/typed_geometry_domain_feedback_20260906/typed_outer/render_all/left.png)、[右视图](../results/multiview_data/0d285f5be7fa09c3dbbf1c9334047888/pde_governance/volume_partition_integration/typed_geometry_domain_feedback_20260906/typed_outer/render_all/right.png)、[后视图](../results/multiview_data/0d285f5be7fa09c3dbbf1c9334047888/pde_governance/volume_partition_integration/typed_geometry_domain_feedback_20260906/typed_outer/render_all/back.png)。

再加 front 反馈，渲染全部 9,692 条非零前缀：

![低泄漏对照](../results/multiview_data/0d285f5be7fa09c3dbbf1c9334047888/pde_governance/volume_partition_integration/typed_geometry_domain_feedback_20260906/front_feedback/render_all/front.png)

两组使用同一 Blender 模板和管径设置，不按终止状态筛选发丝。额角缺口在两组中仍然可见；部分发梢弯折和聚束也需要继续处理。

## 🔬 几何域与坐标检查

在 128³ 的每个节点直接查询头部、原始发体网格和水密外包络，共 2,097,152 个节点；没有放大 64³ 的旧 SDF。物理间距约为 3.223 / 3.255 / 2.886 mm。根点与头皮的最大局部法线侧距离约 5.72 μm，相机轴正交性和手性检查通过。

独立图像—网格配准对应点仍缺失。当前检查证明数值轴系一致，不能证明人物头模、原始图像和生成发体已达到真实精确配准。

旧可信域含 278,760 个节点，其中 80,221 个节点的头部法线侧距离小于负半体素对角线。按此几何候选判据，它们属于头部内侧较深的节点，占旧域约 28.8%。新实验域保留 198,539 个节点；9,695 个根点的对应体素全部保留，根点坐标没有移动。

头部网格不水密，使用的是最近三角形法线侧距离，以上数量是相对于该头模和此判据的诊断，不是精确解剖实体体积。水密外包络包住完整头脸，也不能单独定义头发语义。几何候选中另外 141,777 个节点没有旧可信分区支持，本轮未将它们自动纳入头发域。

为避免把头皮附近所有生长根删除，体素裁剪留出半体素对角线的不确定带，积分阶段继续用更细的直接头部距离检查。最终线段头部检查采样间距不超过 0.125 mm。开放网格与离散采样都不构成连续零碰撞证明。

域裁剪的单独消融也已保留：旧域上增加外表面约束得到覆盖 74.15%、泄漏 5.58%；裁剪后为 74.14%、5.61%。因此本轮可见收益主要来自表面方向约束，不能将覆盖提升归功于域裁剪本身。裁剪的价值是去除已识别的头内 PDE 自由度。

## 🛠️ 类型化约束与反馈

内部头皮沿用逐步距离守卫与离开头皮的根部过渡。外部发体表面只在离头皮超过 6 mm、距发体表面不足 8 mm 的带内加入距离衰减的切向惩罚，最大权重为 8。这样避免将根点方向全部强制压到切平面。

PDE 保留旧分区 ID 和同区耦合，固定局部影响带外的方向。外部表面法线来自不水密发体网格；惩罚使用法向分量平方，不依赖法线正负号，但仍受网格几何质量影响。

front 反馈候选使用原始 front 的 `seg`、`strand_map` 和现有标定相机。只在头部前方使用图像约束，遮挡区域不自动当作背景；屏幕方向沿已有三维场选择符号，保留相机深度分量。在轮廓窄带内增加向内的软方向修正。该方法是屏幕空间反馈实验，并非 HairLRM 的双交点 lifting 或学习到的三维先验。

目前是两个固定权重候选（0 和 1）的有界 solve→grow→render/score 对照；没有实现无限循环优化或从发丝重新编码 latent。保留所有输入 SHA256、求解报告、场、根身份、长度和可见性评价。

## 🔎 终止归因

新增可选语义信息后，原有调用仍可使用旧归因。只有在有相应证据时才分为实体体素、冲突、外包络出口、空气证据、未知语义或未分配头发。

| 终止原因 | 外表面约束 | front 反馈 |
| --- | ---: | ---: |
| 跑满预算 | 4629 | 4841 |
| 分区交界 | 1781 | 2171 |
| 未知语义 | 3238 | 2393 |
| 转向约束 | 47 | 289 |
| 头部表面碰撞 | 0 | 1 |

本次先前笼统记为无标签域的终止全部落在继承证据的 `unknown` 类。它们不是已验证空气，也不能自动认为是自然发梢。归因复跑两组轨迹与原结果逐坐标一致，最大差为 0，因此原渲染继续有效。

## ✅ 验证与使用

17 项测试通过，覆盖目标网格距离单位/符号、头部遮挡、语义归因、分区插值、沿墙生长、头部守卫及已有分区 PDE/RK4 不变量。两组各 2,481,920 条存储线段（包含重复填充点）通过体素审计。

```bash
PYTHONDONTWRITEBYTECODE=1 pixi run python -m pytest \
  tests/test_typed_hair_geometry.py tests/test_boundary_recovery.py \
  tests/test_head_guard_comparison.py tests/test_partitioned_weighted_poisson.py \
  tests/test_rk4_partition_invariant.py \
  -q -o cache_dir=/tmp/hairstream-typed-tests
```

重跑实验时使用新的输出目录，脚本会拒绝覆盖已有目录：

```bash
PYTHONDONTWRITEBYTECODE=1 pixi run python scripts/recon_3d/optimize_typed_hair_geometry.py \
  --data-dir results/multiview_data/0d285f5be7fa09c3dbbf1c9334047888 \
  --output-dir results/multiview_data/0d285f5be7fa09c3dbbf1c9334047888/pde_governance/volume_partition_integration/typed_geometry_repeat \
  --geometry-domain
```

[实验目录](../results/multiview_data/0d285f5be7fa09c3dbbf1c9334047888/pde_governance/volume_partition_integration/typed_geometry_domain_feedback_20260906/) 中 `geometry_128.npz` 保存直接查询的几何，`integration_domain.npz` 保存实验域。它不是包含全部版本字段的正式生产 bundle。`comparison.json`、`termination_attribution.json` 与各组 `report.json`、`visibility.json` 是上述数字来源。细分原因另存于 `typed_termination.npz`，历史 `all_root_prefixes.npz` 保留原始生成时的原因。

代码：[优化入口](../scripts/recon_3d/optimize_typed_hair_geometry.py)、[终止归因复核](../scripts/vis/audit_typed_termination.py)。本轮修改了实验归因函数及其积分调用，未替换默认生产链路。GitNexus 对这些实验符号返回 UNKNOWN，源码检查可见调用限于实验脚本和对应测试，无法用当前索引声称完整调用图分析通过。工作区其他既有修改保留，未提交。

## 📌 下一阶段边界

尚未实现或验收：独立配准验证、真实发缝与人工分区的可靠重分类、guide 引导的缺口补根、三维方向真值评价。当前仍使用固定 9,695 根，没有恢复最初被可信域拒绝的 305 根。额角缺口需要先定位其可见 guide 和域语义证据，继续增加积分长度或直接开放 unknown 域都不足以证明修复正确。
