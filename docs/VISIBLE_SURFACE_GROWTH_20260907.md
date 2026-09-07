# 可见表面补生长验证

样例 `0a1ba3dbefc8934ab60577c5c91f66a0`，2026-09-07。当前结果仍为候选实验，没有将全局秃头修复标为完成。

新增入口为 [grow_visible_surface_gaps.py](../scripts/recon_3d/grow_visible_surface_gaps.py)。该入口调用补生长执行器，在模板头皮新根点上用来源视角的轴向方向图重新积分，不复制旧轨迹，也不吸附到旧 guide 的位置。轴向符号使用图像向下的先验；方向图的突变与深度突变形成视角局部候选边界，不能当作经过验证的三维发缝身份。

第一轮使用来源后视相机选点，接受 66 根，但模板背面中央暗缺口视觉改善不足。第二轮将选点与评分改到真实模板透视相机，方向查询继续使用来源正交相机。两者分别投影，不再共用一个相机假设。模板相机导出入口为 [export_template_camera.py](../scripts/render/export_template_camera.py)，当前代码绑定模板文件哈希。

| 指标 | 第二轮补生长前 | 第二轮补生长后 |
| --- | ---: | ---: |
| 发丝数 | 9,220 | 9,420 |
| 模板目标区代理缺口像素 | 6,033 | 3,807 |
| 目标区像素 | 96,246 | 96,246 |
| Blender 可见头模像素 | 4,837 | 2,960 |

独立平色渲染确认第二轮目标区的露头模像素减少 1,877，即 38.8%。报告位于结果目录的 `visibility_audit_unlit/report.json`。这是固定后视目标区的改善，不代表其它视角或被候选边界排除区域的秃头已经解决。平色审计将发光材质的光源采样关闭，避免密集发丝作为大量面光源构建导致高内存占用；失败尝试目录保留。

第二轮合并结果经独立占据和距离审计：15,173,007 个采样点、头内采样为 0、最小距离 0.498595 mm。采样间距 0.125 mm，距离门限 0.4 mm，占据使用 11 条射线。该结果是离散采样保证。来源数据的全部旧轨迹保留，新轨迹另外通过 0.0625 mm 采样审计。

结果目录为 `results/multiview_data/0a1ba3dbefc8934ab60577c5c91f66a0/pde_governance/volume_partition_integration/visible_surface_growth_v2/`。其中 `report.json` 保存每个候选的终止、拒绝原因；`independent_audit/audit.json` 保存合并结果审计；`render/back.png` 和 `render/front.png` 为相同模板、半径和采样数的渲染。

代理指标使用头部深度测试后的中心线像素，不等同 Blender 曲线的真实可见覆盖。彩色渲染的中央暗缺口仍然明显，不能用代理下降 36.9% 宣称秃头消除。独立平色覆盖入口 [audit_visible_head_coverage.py](../scripts/render/audit_visible_head_coverage.py) 将头模渲染为红色、发丝渲染为绿色，直接测量固定目标区内的可见头模。补生长入口的 `--visibility-render` 接收该平色图，只在实际露头模区域提出新根点。

当前代码在导出合并 NPZ 之前重新审计全部已有和新增轨迹，未通过则只保留报告并拒绝导出。第二轮数据在这项门禁接入前启动，因此采用独立 `--audit-only` 命令复核，通过后才进行模板渲染。未启用默认生产链路，尚需固定目标区的平色对照以及其它视角的轮廓检查。

复现新版本补生长：

```bash
growth_data=results/multiview_data/0a1ba3dbefc8934ab60577c5c91f66a0
growth_base="$growth_data/pde_governance/volume_partition_integration"
growth_legacy="$growth_base/legacy_pde_template_render_20260906"
blender -b assets/render_template.blend -t 4 \
  --python scripts/render/export_template_camera.py -- \
  --output-dir "$growth_base/coverage_cameras_reproduce"
pixi run python scripts/recon_3d/grow_visible_surface_gaps.py \
  --data-dir "$growth_data" \
  --input "$growth_legacy/segment_corrected/all_root_prefixes.npz" \
  --head "$growth_legacy/template_head/template_head.npz" \
  --coverage-camera "$growth_base/coverage_cameras_reproduce/back.npz" \
  --output-dir "$growth_base/surface_growth_reproduce" --budget 300
```

输出目录必须尚不存在。当前相关回归测试 21 项通过，覆盖新增根间距、无 guide 拒绝、独立审计拒绝、整段/单点碰撞、透视与正交遮挡、同区检查、原模板细分投影、根部向外过渡以及旧采样数组精确保留。

第三轮 `visible_surface_growth_v3/` 仍从同一份 9,220 根基线开始，使用第二轮独立平色审计的基线红色像素选根，候选预算 600，根点保持 0.8 mm 初始间隙，表面方向场目标距离为 3 mm。接受 330 根，总数 9,550。已有采样和 padding 已逐点核对为完全保留；新增轨迹未复制。最终输出的重新审计见 `exact_output_audit/audit.json`：15,334,965 个采样点，头内为 0，最小距离 0.498595 mm。`report.json` 同时保留初次审计和最终导出审计，避免混淆恢复采样表示前后的检查。

复现第三轮时，在上述补生长命令中使用新的输出目录，将预算改为 600，并追加：

```bash
--surface-offset-m .003 \
--visibility-render "$growth_base/visible_surface_growth_v2/visibility_audit_unlit/before.png"
```

该平色图必须来自相同模板视角和相同基线。第三轮同时改变候选筛选、预算和表面过渡距离，不能从该组合实验中单独归因哪一个变量贡献了改善。

后续正视实验的修正：`v4_front_correct` 与 `v5_front_quality` 的新增发丝垂到脸部，不能作为合格修复。来源 seg 投影不能代表模板脸部与头皮的分界；终点最小间距也不能证明整条轨迹不聚束。此前用整幅红色头模作为正视评价目标还计入了正常裸露脸部，相关覆盖百分比不能解释为头发覆盖率。

`visible_surface_growth_v6_hairline` 从原始 9,220 根基线重新生成，排除 v5 的新增发丝。模板平色基线中绿色发丝区域经过半径 12 像素的闭运算，形成保守的可补范围；它只用于修补现有发型附近小缺口，并非解剖头皮或真实发际线标注。根点与每段积分均受该范围限制，越界记录为 `template_hairline_exit`。目标由 32,529 缩到 2,491 像素，避免把大面积脸部当作覆盖目标。

该轮 300 个候选中，248 个因模板边界终止；190 个有效长度不足，88 个未通过终点间距检查，7 个无覆盖增益，最终接受 15 根。合并 9,235 根最终采样审计通过，15,002,386 个采样点、头内 0、最小距离 0.498595 mm。代理缺口 2,098 → 1,906。这个结果不表示右侧缺口已消失；其余缺口需要可靠的模板头皮范围及三维方向约束，不能通过继续覆盖正常裸露脸部提升指标。

进一步核查发现正视相机接入错误：正式 `run_pde_multiview.py` 对原始正面 maps 使用 `maps/param/front.npy`，仅侧面与背面使用 Blender 相机。早期补生长入口却对 front 也调用 `load_blender_view_calibration`。因此 v4/v5/v6/v7 的正视来源投影存在错配；此前“前额方向语义不足”的归因不能成立，必须先修正相机再判断。已新增 `load_observation_camera` 并用测试禁止 front 路径调用 Blender 标定。

新增 `--direction-mode local_guides`：从已有轨迹按 2 mm 弧长采样，按来源候选分区建立局部切向量场，邻域半径 12 mm、至少 3 个支持、方向一致性至少 0.7；缺失或相反流向停止，保持三维方向符号。新根逐步查询该场，不复制或平移任何旧轨迹。这是局部插值重建，不是已经完成的全域 PDE 重求解，也不将候选分区声称为可信三维发缝。
