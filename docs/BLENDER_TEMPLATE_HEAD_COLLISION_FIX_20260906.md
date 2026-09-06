# Blender 模板预设头模穿入修复

本次修复针对 `assets/render_template.blend` 的 `head_smooth`，样例为 `0d285f5be7fa09c3dbbf1c9334047888`。输入采用上一轮 `typed_geometry_domain_feedback_20260906/typed_outer/all_root_prefixes.npz`。

## 🔎 原因

此前碰撞检查使用 `data/head_model.obj`，而 Blender 模板渲染使用另一份 `head_smooth`，并启用了细分修改器。虽然二者使用相同轴转换，形状并不一致。在旧头模外合法的发丝，仍可能位于模板头模内部。

本轮导出模板在 render 细分级别下求值后的世界网格，得到 72,306 个顶点、144,608 个三角形，水密检查通过。按照发丝渲染变换 `(x,y,z) → (x,-z,y)` 的逆变换，将其转换到发丝 Y-up 世界坐标，避免手工平移或缩放头模。

## 🛠️ 修复

新增 [模板头模导出](../scripts/render/export_template_head.py) 和 [发丝模板适配](../scripts/recon_3d/fit_strands_to_template_head.py)。前者绑定模板文件哈希、对象和修改器信息；后者按 0.5 mm 间距重采样有效前缀，将头内和过近的点向外修正，平滑位移后再次检查间隙。

目标中心线间隙为 0.5 mm，适用于当前 0.3 mm 管半径。输出前用不超过 0.125 mm 的线段采样检查至少 0.4 mm 间隙，否则拒绝导出。原输入与模板均未修改，适配结果保存到独立任务目录。

| 指标 | 适配前（相对于模板头模） | 适配后 |
| --- | ---: | ---: |
| 根身份数 | 9695 | 9695 |
| 非零前缀 | 9693 | 9693 |
| 头部内侧超过 1 mm 的根数 | 1551 | 0 |
| 头部内侧超过 3 mm 的根数 | 65 | 0 |
| 最小法线侧距离 | -13.741 mm | +0.495 mm |
| 有效长度中位数 | 184.500 mm | 184.677 mm |

根点位移中位数约 0.094 mm，90 分位约 1.843 mm，最大约 6.589 mm；全发丝最大点位移约 14.194 mm。根身份与 source indices 保留，原根坐标另存为 `source_roots_world`。原终止步数另存为 `source_termination_step`，重采样后有效点数存入 `valid_point_counts`。

这属于针对模板头模的渲染适配，坐标发生了变化，旧 PDE 分区审计不能直接套用于新发丝。它解决模板穿入，不代表原发型缺口、真实发缝和整体形状已修复。若未来更换模板、修改器级别或加大发丝半径，应重新导出头模并运行适配。

## 🖼️ 输出与验证

[结果目录](../results/multiview_data/0d285f5be7fa09c3dbbf1c9334047888/pde_governance/volume_partition_integration/template_head_collision_20260906/fitted/) 包含适配 NPZ、PLY、统计和 Blender 工程。

![修复后的正面渲染](../results/multiview_data/0d285f5be7fa09c3dbbf1c9334047888/pde_governance/volume_partition_integration/template_head_collision_20260906/fitted/render_all/front.png)

[左视图](../results/multiview_data/0d285f5be7fa09c3dbbf1c9334047888/pde_governance/volume_partition_integration/template_head_collision_20260906/fitted/render_all/left.png)、[右视图](../results/multiview_data/0d285f5be7fa09c3dbbf1c9334047888/pde_governance/volume_partition_integration/template_head_collision_20260906/fitted/render_all/right.png)、[后视图](../results/multiview_data/0d285f5be7fa09c3dbbf1c9334047888/pde_governance/volume_partition_integration/template_head_collision_20260906/fitted/render_all/back.png)。

两项专用测试通过，验证向外投影、管径间隙、原输入不变、根身份数量和零长度前缀保留。另有 `independent_sdf_audit.json` 记录水密头模的射线奇偶内外判定，与修正所用最近三角形法线判定分开：检查 3,742,270 个存储顶点（含填充点），头内点数为 0，最小带符号距离为 +0.500 mm。线段采用加密采样检查，未宣称解析连续碰撞证明。

复现命令（输出目录须未存在）：

```bash
experiment_base=results/multiview_data/0d285f5be7fa09c3dbbf1c9334047888/pde_governance/volume_partition_integration
blender -b assets/render_template.blend --python scripts/render/export_template_head.py -- \
  --output-dir "$experiment_base/template_head_repeat"
PYTHONDONTWRITEBYTECODE=1 pixi run python scripts/recon_3d/fit_strands_to_template_head.py \
  --input "$experiment_base/typed_geometry_domain_feedback_20260906/typed_outer/all_root_prefixes.npz" \
  --head "$experiment_base/template_head_repeat/template_head.npz" \
  --output-dir "$experiment_base/template_head_repeat/fitted"
blender -b --python scripts/render/render_all_prefixes.py -- \
  --input "$experiment_base/template_head_repeat/fitted/all_root_prefixes.npz" \
  --output-dir "$experiment_base/template_head_repeat/fitted/render_all" \
  --views front left right back
```
