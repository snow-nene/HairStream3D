# 原图守卫与发丝重建修复记录

对象为 `0a1ba3dbefc8934ab60577c5c91f66a0`。原图 `raw_img.png` 及其 front 特征仍为唯一真实观测，侧后视图只作为 Flux 先验。本轮保留 v27 原文件、原始 mesh、seg、相机和 Blender 模板，派生结果按 Image ID 保存。

## 已确认并修复的代码问题

原图守卫原先使用 `head_hit & depth_visible`：当发丝投影落在头模轮廓之外时，即使该像素不是头发，也不会被否决。修复后将无头模交点解释为无遮挡，并拒绝越出原图范围的轨迹。这个修复同时进入原有 `grow_visible_surface_gaps.py` 和新的批量生长路径。旧报告中的 373 个越界像素仅统计头模覆盖区，漏掉了大量轮廓外像素；完整统计为 **9,784 个可见中心线像素**。

方向场使用单位方向的二阶张量进行曲面延拓和空间插值，查询时沿已经接受的上一段选择符号，避免已求解向量在插值时正反抵消。对主轴不明确的交叉仍停止，不通过降低方向阈值强行穿过。补生长使用邻近有效 guide 的长度分布，取消统一 120 mm 终点；分支试探、正式积分和输出均有整段守卫及独立碰撞检查。

旧 guide 在 1 mm 弧长网格上平滑，根部采用渐消位移重连接，远端主体保持。重连接与补生长都受原图约束。最终模板绑定通过 SHA-256 校验，碰撞检查和 Blender 渲染使用同一套评估后头模几何。

另外修复了原图关键点拟合脚本强制使用 CUDA 的问题，使无 GPU 环境可用 CPU；拟合投影的像素约定改为与重建一致的 `image_size - 1`，避免半像素偏差。

## 没有通过验收的几何和相机候选

从约 95 万个完整 mesh 面中筛出 214,605 个具有多视图头发证据、且不受原图可见非头发区域否决的面，另存为开放的 `hair_outer_surface.obj`。它不是封闭 SDF 体积，未覆盖原来的 `hair_mesh_sdf.obj`。

头皮适配实验固定脸部等 56,219 个顶点，仅调整候选头发区域；候选仍封闭、无自交，固定区域位移为零。但这些几何检查不足以证明发型改善。第一次弱平滑适配造成局部凹凸及发量压缩，已排除。加强平滑并保留 guide 主体后的 v31，虽然原图语义与急折明显改善，侧后固定目标覆盖却退化：

| 固定目标区域的中心线缺口 | v27 | 适配头模 v31 |
| --- | ---: | ---: |
| 原图 front seg | 1.345% | 0.728% |
| Flux left seg | 19.756% | 31.646% |
| Flux right seg | 7.423% | 11.019% |
| Flux back seg | 1.338% | 4.407% |
| 固定旧头模冠部代理 | 3.078% | 3.304% |

因此，**不能把头皮适配宣称为几何根治，也不能把 v31 当成整体通过验收的版本**。适配后的 mesh 候选表面仍有相当部分处于头模内。当前自动入口默认保留原头模，`--adapt-head` 仅作为实验选项；不会全局缩小头模或静默修改现有模板。

原图 SO(3) 相机候选同样没有采用。每三个检测关键点留出一个，旧标定的留出平均误差为 6.057 px，候选为 8.723 px。留出仅针对本轮重拟合，旧标定可能用过这些点，检测关键点也不是人工真值；该实验支持拒绝这个候选，不等于证明旧标定完全正确。

## 入口与结果组织

[`run_subject_repair.py`](../scripts/recon_3d/run_subject_repair.py) 串联证据提取、派生模板导出、guide 重连接与重积分、合法缺口补生长和 Blender 渲染。默认只使用 front，可显式指定任意包含 front 的视角组合。以下为保留原头模的完整执行方式，输出目录必须不存在：

```bash
CASE_DATA=results/multiview_data/0a1ba3dbefc8934ab60577c5c91f66a0
CASE_WORK="$CASE_DATA/pde_governance/volume_partition_integration"

pixi run python scripts/recon_3d/run_subject_repair.py \
  --data-dir "$CASE_DATA" \
  --input "$CASE_WORK/nonrender_growth_v27_branches/all_root_prefixes.npz" \
  --head "$CASE_WORK/legacy_pde_template_render_20260906/template_head/template_head.npz" \
  --output-dir "$CASE_WORK/subject_repair_repeat" \
  --baseline-count 9289 --views front left right back
```

每个阶段成功后写入 `stages.json`；阶段输出依次为 `geometry/`、`regrow/`、`final/`、`render/`。所有结果均位于该 Image ID 下。该入口目前面向与现有标准模板绑定的重建结果，头皮中心和高度参数沿用本项目世界坐标尺度，不应直接用于任意坐标系的外部模型。

## 验证口径

固定各视角原 seg 比较覆盖，顶部使用固定的旧头模冠部投影区域，避免因换头模导致目标分母改变。中心线像素不包含发丝粗细，不能解释为真实露头皮面积。实际 Blender 对照保持材质、灯光、相机、32 samples 和 0.3 mm 发丝半径一致。

急折使用 2 mm 等弧长弦之间的夹角；身体间距跳过前 6 mm 后每 4 mm 取样。末端重复填充点不计入长度。回归测试覆盖头模固定边界、正反轴向方向、交叉歧义、连续符号、整段守卫、按根长度、拒绝步不提交状态，以及原图轮廓外漏检。

相关诊断工具为 [`compare_subject_repair.py`](../scripts/vis/compare_subject_repair.py)，此前的原因分析见 [`RECONSTRUCTION_CAUSE_AUDIT_20260909.md`](RECONSTRUCTION_CAUSE_AUDIT_20260909.md)。
