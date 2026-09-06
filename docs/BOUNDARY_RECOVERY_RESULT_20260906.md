# 发丝边界中断修复与合并渲染结果

样例：`0d285f5be7fa09c3dbbf1c9334047888`；日期：2026-09-06。

本轮已经修复实验输出只显示“完整组”的问题，并通过同区连续插值、受限续长和 PDE 界面切向条件降低提前终止。头发可见覆盖有所改善，但额角缺口、贴界聚束和局部形态失真仍存在；本轮结果不作为完整发型重建验收通过或默认流水线替换。

## 📊 实测结果

三组固定同一批 9,695 个根点、根标签和域标签，长度预算均为 192 mm。表中“原几何修正”是上一轮 `head_guard_comparison_20260906/head_corrected.npz`，已经带头部守卫，因此与无头部约束的历史 trusted 分组不可混为一组。

| 指标 | 原几何修正 | 连续插值与受限续长 | 再加 PDE 界面切向条件 |
| --- | ---: | ---: | ---: |
| 非零轨迹 | 9555 | 9689 | 9693 |
| 零长度 | 140 | 6 | 2 |
| 跑满预算 | 2671 | 3642 | 3865 |
| 分区交界终止 | 2445 | 2279 | 1242 |
| 无标签或域外终止 | 4578 | 3729 | 4570 |
| 头部碰撞终止 | 1 | 0 | 0 |
| 转向约束终止 | 0 | 45 | 18 |
| 长度中位数（mm） | 131.99 | 156.00 | 163.50 |
| 长度超过 50 mm | 8014 | 8271 | 8841 |
| 头部内侧超过 1 mm 的根数 | 0 | 0 | 0 |
| 头部内侧超过 10 μm 的根数 | 0 | 0 | 0 |
| 正面可见 seg 覆盖率 | 65.53% | 70.79% | 70.85% |
| 正面可见 seg 外泄漏比例 | 5.28% | 6.83% | 6.68% |

最终组相比原几何修正：分区交界终止减少约 49.2%，长度中位数增加约 23.9%，可见覆盖增加 5.32 个百分点。泄漏增加 1.41 个百分点，表明增加长度并不等价于形状正确。最终仍有 5,830 根未跑满预算，但自然发梢可能早于统一预算，不能将这些轨迹全部判为失败发丝。

几何距离采用开放头部网格的最近三角形法线侧距离，加密线段采样间距不超过 0.125 mm；不是闭合实体的连续碰撞证明。最终组最小距离为 -4.75 μm，主要来自根点数值误差。最终 2,481,920 条存储线段（包括重复填充段）体素审计零违规。

## 🖼️ 合并渲染

所有非零前缀共同显示，不按终止状态过滤。零长度根继续保存在 NPZ 中，无法渲染成线段。原始 10,000 根中的 305 个域外根未在本轮凭空补回。

原几何修正的全部前缀：

![原几何修正全前缀](../results/multiview_data/0d285f5be7fa09c3dbbf1c9334047888/pde_governance/volume_partition_integration/boundary_recovery_20260906_v2/render_baseline/front.png)

最终 PDE 边界修正的全部前缀：

![最终前视图](../results/multiview_data/0d285f5be7fa09c3dbbf1c9334047888/pde_governance/volume_partition_integration/boundary_tangent_pde_20260906/render_all/front.png)

[左视图](../results/multiview_data/0d285f5be7fa09c3dbbf1c9334047888/pde_governance/volume_partition_integration/boundary_tangent_pde_20260906/render_all/left.png)、[右视图](../results/multiview_data/0d285f5be7fa09c3dbbf1c9334047888/pde_governance/volume_partition_integration/boundary_tangent_pde_20260906/render_all/right.png)、[后视图](../results/multiview_data/0d285f5be7fa09c3dbbf1c9334047888/pde_governance/volume_partition_integration/boundary_tangent_pde_20260906/render_all/back.png)。同一模板、管半径 0.3 mm、相同根梢 taper 和 32 samples，比较中不靠加粗发丝填洞。可见覆盖统计使用独立标定相机与头部遮挡检查，采用投影中心线，不是 Blender 管径像素统计。

## 🛠️ 实际修复

新增 [recover_boundary_strands.py](../scripts/recon_3d/recover_boundary_strands.py)：同分区三线性插值，不把其他分区相反方向平均进来；逐步中点积分，每个候选必须经过完整 DDA 和头部检查。候选失败后在局部参考方向约 60° 锥内搜索，限制相对上一段的转角为 45°，步长依次尝试 3/1.5/0.75 mm。加入历史复访检查，避免简单绕圈增加长度。最终浮点存储坐标参与守卫，避免 double 检查通过后 float32 落在非法边界。由于很短的预算末步受浮点误差影响，日志最大参考偏角为约 60.009°。

新增 [solve_boundary_tangent_recovery.py](../scripts/recon_3d/solve_boundary_tangent_recovery.py)：原求解器仅切断跨区邻接，并不要求方向不能指向分区墙。本轮在约两体素的界面带加入 `30 * (n·v)^2` 法向分量惩罚，同时保留方向保真项和同区平滑。带外固定旧场。标签不变，也没有给未知组件随机标签。PDE 在 25 次迭代收敛，残差 `7.27e-5`。

这相当于使方向场与既有不可跨越边界更一致，尚未证明这些边界都是正确发缝。表面种子传播生成的体内界面仍可能错误，因此不把此实验解释为真实拓扑已恢复。

新增 [render_all_prefixes.py](../scripts/render/render_all_prefixes.py)：默认共同渲染全部非零前缀，支持按需 front/left/right/back 组合；输出含 Blender 工程和根身份清单。既有 `complete` 分组文件保留，默认生产入口未更换。

## 🔎 剩余问题与证据

受限续长组有 3,729 根无标签终止，其中只有 96 根终点距头皮小于 4 mm，中位头皮距离为 43.3 mm。大多数不能仅凭终止类别认定是头皮域缺口，可能包含发体外轮廓和发梢出口。相反，2,279 根分区终止中有 1,257 根位于头皮 4 mm 内，说明分区界面条件是更直接的近头皮中断来源。

原候选域与可信域仅相差 280 个 128³ 体素，不能把数千根中断简单归咎于这 280 个被排除的体素。本轮没有扩大域，也没有把零标签自动改成头发。

左额角缺口和侧面的聚束在新渲染中仍可见。下一步若继续改善，应对照原 front 观测重建根部方向/发缝边界并评价可见误差，而不是继续放宽转角、加长全部发丝或取消所有分区隔离。当前保留实验状态；不能把“合并显示修复”表述为“秃头彻底修复”。

## 🧪 验证与产物

14 项测试通过：同区插值不混入对岸向量、方向锥范围、非法根不偷渡、合法根沿墙续长且保持分区、头部守卫，以及既有 RK4 和分区 PDE 测试。

```bash
PYTHONDONTWRITEBYTECODE=1 pixi run python -m pytest \
  tests/test_boundary_recovery.py tests/test_head_guard_comparison.py \
  tests/test_rk4_partition_invariant.py tests/test_partitioned_weighted_poisson.py \
  -q -o cache_dir=/tmp/hairstream-boundary-tests
```

[最终实验目录](../results/multiview_data/0d285f5be7fa09c3dbbf1c9334047888/pde_governance/volume_partition_integration/boundary_tangent_pde_20260906/) 包含 `manifest.json`、`solver.json`、`report.json`、`visibility.json`、`all_root_prefixes.npz`、`all_root_prefixes.ply` 和四视图 Blender 渲染。NPZ 保留全部 root labels、source indices、终止原因和步数。

首轮 `boundary_recovery_20260906` 存在浮点存储后的体素违规，属于失败诊断，不能使用其 PLY。`boundary_recovery_20260906_v2` 及最终 PDE 组通过审计。最终代码在审计失败时拒绝 PLY 导出，诊断数组仍保留。

GitNexus 对新增实验符号返回 UNKNOWN，未索引到对应函数，无法可靠列出调用范围。本轮修改仅限新增脚本内部和新测试，没有修改共享 PDE、守卫或旧生产入口，也未提交已有工作区内容。
