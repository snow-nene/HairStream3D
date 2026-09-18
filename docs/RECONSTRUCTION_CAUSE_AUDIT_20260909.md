# v27 残留缺口与发型形态排查

_对象：`0a1ba3dbefc8934ab60577c5c91f66a0`；日期：2026-09-09；固定 v27 的 12,040 根发丝，仅执行诊断。_

---

## 📋 排查结论

本轮证据把修复优先级指向**头发外包络与头模不兼容**，其次是冠部场插值的局部抵消，以及补生长的贴头皮和统一长度策略。网格非流形确实存在，但没有证据证明清理非流形边就能消除主要缺口。标定尚缺独立验证，现有候选也不能直接替换。

原图是唯一 front 证据，其他视角 seg 仅为 Flux 先验。所有诊断输出位于 [本轮实验目录][audit]；主要数值见 [几何与发丝报告][report]、[变换与相机对照][alignment]、[停止点重放][replay]。本轮新增两个诊断脚本和四项测试，未修改生产重建逻辑、mesh、相机、seg 或 v27 发丝。

```mermaid
flowchart LR
    accTitle: 重建残留问题的排查顺序
    accDescr: 先固定原图及已有轨迹，验证网格与头模及相机，再重放方向停止点，最后检查长度和离头模距离。
    fixed_input[固定原图与 v27] --> mesh_head[核对网格与头模]
    mesh_head --> cameras[核对变换与标定]
    cameras --> replay_stops[重放方向停止点]
    replay_stops --> strand_shape[测量长度与层次]
    strand_shape --> priorities[确定修复优先级]
```

## 📊 几何不兼容的直接证据

### 网格不是已经提取好的头发外包络

[`extract_hair_mesh_flame`][extract] 当前读取网格后直接写出，没有执行头发裁剪。磁盘上的 `hair_mesh_sdf.obj` 与 `hair_mesh_aligned_best.obj` 的顶点和面数组完全一致。两者都是对齐后的完整 Pixal3D 网格；不能仅依据文件名，把它当成经过头发语义提取和闭合修复的体积边界。

合并距离不超过 1 μm 的重合顶点、去除退化面和重复面后，网格有 10 个组件，最大组件占表面积 99.978%；边界边为 0，非流形边为 1,045。合并前的 24,465 个组件主要受重复顶点影响，不宜解释为数万个实际破洞。合并只发生在内存中，源 OBJ 未改变。

### 头发候选表面被标准头模包含

固定各视角相机，以 seg 内网格第一交点作为候选头发表面，测量其到绑定模板头模的有符号距离。负值表示落在模板内部；内部点计数使用小于 −0.4 mm 的阈值。

| 视角证据 | seg 内射线未命中网格 | 命中点位于模板内部 | 距模板的距离中位数 |
| --- | ---: | ---: | ---: |
| 原图 front | 30.31% | 55.38% | −1.19 mm |
| Flux left | 4.72% | 76.54% | −9.22 mm |
| Flux right | 14.61% | 83.61% | −9.98 mm |
| Flux back | 1.56% | 99.70% | −15.63 mm |

第二列以 seg 像素为分母；第三列仅以 seg 内命中网格的像素为分母。这里的“头发候选表面”由二维 seg 筛选，并未证明对应三角形本身属于头发。这些比例不是最终发丝穿模率。

网格最高处为世界 Y=1.875440 m，模板头模最高处为 1.886018 m，重建头模最高处为 1.887230 m。网格最高处甚至比两个光头头模低约 10.58 mm 和 11.79 mm。下面的中央切片显示，冠部和后脑的 Pixal3D 表面位于两个头模轮廓之内。

![对齐 Pixal3D 网格与重建头模、Blender 模板头模的中央切片][slices]
_绿色为完整 Pixal3D 网格，橙色为重建头模，红色为 Blender 模板头模；散点来自距切平面 2 mm 内的顶点，不能当作闭合截面。_

这一矛盾发生在发丝积分之前：外包络若落在实体头模内部，就没有可供该区域正常生长的头模外空间。后续将发丝推出头模并沿表面补根，可以改善覆盖，却不能恢复被几何关系破坏的发型体积。当前数据尚不能单独归责于 Pixal3D 原始形状、面部配准的尺度/位置，或标准头模形状；可以确认的是这三者组成的当前几何不兼容。

### 已排除“保存的坐标变换不匹配”这一简单解释

按 `glb_to_world.npz` 的 Umeyama 与 ICP 组合变换，将原始 GLB 顶点映射到世界坐标，与已对齐 OBJ 进行双向最近点核对。固定 seed=42，每个方向抽样 20,000 个顶点；最大误差分别为 0.005100 mm 和 0.005099 mm。

这表明已对齐 OBJ 与保存变换基本一致，不能证明 ICP 配准本身正确。现有对齐代码以面部关键点及局部 ICP 为约束，并不保证头发外包络在标准头模之外。

## 🔍 标定与视角约束对照

### 换回重建头模只能解除部分原图冲突

固定 v27 的缺口像素、模板上的候选根点及原图相机，只将原图遮挡头模换成 `data/head_model.obj`。原图头发蒙版保持不变。

| 指标 | left | right |
| --- | ---: | ---: |
| 固定缺口像素 | 4,988 | 1,683 |
| 模板遮挡下与原图冲突 | 3,781 | 919 |
| 重建头模遮挡下与原图冲突 | 2,754 | 866 |
| 原冲突被解除 | 1,028 | 237 |
| 新增冲突 | 1 | 184 |
| 原冲突中距 front seg 超过 10 px | 2,080 | 568 |

left 的冲突有约 27.2% 会因遮挡头模不同而解除，但大部分仍存在。原冲突点到原图头发区域的距离中位数为 11.40 px，最大为 58.25 px，不能将它们都解释成轮廓边缘的一两个像素误差。

重建头模不是封闭网格，本轮只使用它的可见交点和无符号距离，不判定其内外。某些“解除冲突”的根可能落在替代头模实体内，因此这不是可以直接采用的修复结果。该实验说明遮挡几何会影响守卫，而非支持关闭原图守卫。

### 现有相机候选不能直接替换

在同一原图 seg 和同一网格上，`front.npy` 对应 22,385 / 32,123 个 seg 像素命中网格，`glb_param.npy` 仅命中 9,068 个；后者命中的点又全部位于模板遮挡之后。本样例没有 `front_dense_silhouette.npy`。

`front.npy` 的两个图像轴夹角约 85.59°，并非严格正交相机。它是现有仿射投影的一部分，不能直接将矩阵正交化后宣布校准完成，也不能将耦合神经深度的第三行当作独立物理深度标定。需要原图上的独立对应或轮廓/关键点留出检验。目前未找到支持本样例独立验收的留出记录。

![原图头发蒙版与两个头模的同相机轮廓对照][contours]
_绿色为原图 seg；红色为模板头模；青色为重建头模。轮廓可视化展示空间关系，不提供真实隐藏头皮形状。_

## 📍 冠部方向停止点已经重现

v27 只有一个有效方向分区，场约束来自已有发丝。先检查邻近 guide 方向：在 23,285 个活动顶点中，未发现满足“有向均值范数 <0.3 且轴向二阶矩最大特征值 >0.85”的强正反抵消。因此此前“正反 guide 直接混合就是主因”的说法证据不足。

但重放保存的曲面场、原有分支试探及中点积分后，179 个 `unresolved_surface_direction` 的停止原因和积分长度全部与历史记录一致。进一步分解查询发现：179 次的加权输入向量模长平均值都不低于 0.05，而向量加权和模长降到 0.05 以下；失败发生在**已求解场的邻域插值抵消**阶段，尚未进行最后切向投影就已低于阈值。

这 179 个候选中，51 个停止前的合法前缀被原生长程序接受，128 个被拒绝。因此它们是 179 次方向终止，不是 179 根全部未生成的发丝。

| 停止点空间分组 | 候选数 |
| --- | ---: |
| 冠部区域 1 | 118 |
| 冠部区域 2 | 49 |
| 冠部区域 3 | 8 |
| 冠部区域 4 | 4 |

分组采用停止点之间 2 mm 邻接的连通组件，不意味着组件直径不超过 2 mm。前两组占 93.3%。它们是局部场方向抵消区域，不能解释为那里完全没有方向支持。活动顶点上仅 4 个保存向量模长小于 0.05，但插值仍能使更多查询点接近零，单看顶点数会低估影响。

局部 guide 一致性较高，并不证明它们符合真实发型；只有一个分区也不能证明已经恢复真实三维分缝。后续应检查分支方向与分区内延拓，避免让大量轨迹汇入这些抵消区域；单纯调低停止阈值或扩大支持半径不构成可靠修复。

## 📊 长度、折转与离头模距离

将前 9,289 根视为保留的 v12 基线，其后 2,751 根为增补组。使用固定弧长采样，避免重复填充点和不同采样密度扭曲比较。

| 指标 | 保留基线 | 增补组 |
| --- | ---: | ---: |
| 发丝数 | 9,289 | 2,751 |
| 长度中位数 | 78.16 mm | 120.00 mm |
| 长度在 120 mm ±0.01 mm 内 | 10 | 1,497 |
| 长度低于 14.99 mm | 533 | 0 |
| 存在 2 mm 步长折转超过 60° | 367 | 52 |
| 发丝身体距头模中位数 | 14.21 mm | 3.00 mm |
| 身体采样点距头模不足 4 mm | 3.77% | 63.04% |

长度使用去除末端重复点后的折线弧长。身体距离使用 4 mm 弧长网格，跳过最初 6 mm，以样本点计权；长发丝贡献更多点，不是等根权重。折转指标是在 2 mm 等弧长位置形成的相邻弦夹角，不等同于连续曲率。

增补组约 54.4% 的轨迹达到统一 12 cm 预算，说明长度被全局预算显著控制；这并不证明所有这些轨迹都应变短或变长，需要发梢区域和原图形态共同约束。约 63% 的增补身体点贴近头模，与曲面场约 3 mm 的间距控制一致，能解释为什么补覆盖后仍缺少体积层次。367 根具有明显局部折转的旧轨迹继续保留，追加新根无法修正它们。

## 🎯 修复优先级与验收边界

优先修复网格、头模与原图的联合几何关系：确认完整网格中的头发可见表面，核对面部约束与冠部/后脑空间是否兼容，建立一致的实体头模和头发外包络。不能根据本轮结果直接把头模全局缩小，或把网格整体外推；这两种做法都可能破坏已经对齐的脸部和原图轮廓。

随后处理已定位的冠部场抵消区域，建立可解释的分缝/根尖方向约束，重求解受影响轨迹。最后将局部发梢位置、长度分布及体积层次纳入质量目标，减少统一预算截断。非流形清理作为几何准备步骤保留，但不能把清理成功当作发型修复成功。

验收同时保留原图语义、头模碰撞、固定视角覆盖和实际 Blender 形态。当前 3.08% 冠部缺口仍是中心线像素代理，不是最终露头皮比例。本轮没有运行新的重建或渲染，也没有把上述诊断对照升级成生产参数。

## 🔧 复现与验证

诊断入口为 [`audit_reconstruction_causes.py`][cause_script] 和 [`audit_mesh_head_alignment.py`][alignment_script]。以下命令使用新的输出目录，避免覆盖本次证据。

```bash
CASE_DATA=results/multiview_data/0a1ba3dbefc8934ab60577c5c91f66a0
CASE_WORK="$CASE_DATA/pde_governance/volume_partition_integration"
CASE_RUN="$CASE_WORK/nonrender_growth_v27_branches"
CASE_HEAD="$CASE_WORK/legacy_pde_template_render_20260906/template_head/template_head.npz"
CASE_AUDIT="$CASE_WORK/cause_audit_v27_repeat"

pixi run python scripts/vis/audit_reconstruction_causes.py \
  --data-dir "$CASE_DATA" --run-dir "$CASE_RUN" --head "$CASE_HEAD" \
  --baseline-count 9289 --output-dir "$CASE_AUDIT"

pixi run python scripts/vis/audit_mesh_head_alignment.py \
  --data-dir "$CASE_DATA" --head "$CASE_HEAD" --output-dir "$CASE_AUDIT/alignment"

pixi run python scripts/vis/audit_reconstruction_causes.py \
  --data-dir "$CASE_DATA" --run-dir "$CASE_RUN" --head "$CASE_HEAD" \
  --baseline-count 9289 --replay-unresolved --output-dir "$CASE_AUDIT/stop_replay_detail"

pixi run python -m pytest -q tests/test_reconstruction_cause_audit.py \
  tests/test_surface_geometry_metrics.py tests/test_surface_guide_field.py
```

本轮 9 项测试通过，其中新增 4 项验证强方向抵消、弱场/无支持、切向投影损失和原图可见语义分类。四视图固定缺口计数、三个非 front 视角的原图冲突计数与既有 `nonrender_multiview_v27/report.json` 逐项一致。179 个方向停止点的原因和长度全部重现；全流程未导出新发丝。`git diff --check` 通过。

[audit]: ../results/multiview_data/0a1ba3dbefc8934ab60577c5c91f66a0/pde_governance/volume_partition_integration/cause_audit_v27_20260909_open_reference/
[report]: ../results/multiview_data/0a1ba3dbefc8934ab60577c5c91f66a0/pde_governance/volume_partition_integration/cause_audit_v27_20260909_open_reference/report.json
[alignment]: ../results/multiview_data/0a1ba3dbefc8934ab60577c5c91f66a0/pde_governance/volume_partition_integration/cause_audit_v27_20260909_open_reference/alignment/report.json
[replay]: ../results/multiview_data/0a1ba3dbefc8934ab60577c5c91f66a0/pde_governance/volume_partition_integration/cause_audit_v27_20260909_open_reference/stop_replay_detail/report.json
[slices]: ../results/multiview_data/0a1ba3dbefc8934ab60577c5c91f66a0/pde_governance/volume_partition_integration/cause_audit_v27_20260909_open_reference/alignment/mesh_head_slices.png
[contours]: ../results/multiview_data/0a1ba3dbefc8934ab60577c5c91f66a0/pde_governance/volume_partition_integration/cause_audit_v27_20260909_open_reference/front_head_contours.png
[extract]: ../scripts/recon_3d/extract_hair_mesh_flame.py
[cause_script]: ../scripts/vis/audit_reconstruction_causes.py
[alignment_script]: ../scripts/vis/audit_mesh_head_alignment.py
