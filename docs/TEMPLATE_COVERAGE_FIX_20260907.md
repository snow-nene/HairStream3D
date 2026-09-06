# 模板头模穿入与覆盖诊断

样例：`0a1ba3dbefc8934ab60577c5c91f66a0`。输入为 `legacy_pde_template_render_20260906/bridge_source.npz`，共 9,220 根轨迹。

原始轨迹在模板头模上有 2,698 根至少一处进入头模超过 0.01 mm，1,630 根超过 1 mm，最小有符号法线距离为 -12.61 mm。缺口渲染使用全部非零轨迹，因此不是只显示 complete 组造成的。

本次修复在模板投影后重新细分连接段，并再次执行曲面间隙投影；碰撞审计按 0.125 mm 采样，使用 11 射线占据复核。修正结果保留全部 9,220 根，最小间隙为 0.498 mm，11 射线内部采样为 0。结果位于：

`results/multiview_data/0a1ba3dbefc8934ab60577c5c91f66a0/pde_governance/volume_partition_integration/legacy_pde_template_render_20260906/segment_corrected/`

头模穿入已经消除，但正面头顶中央仍有覆盖缺口。尝试从头模顶面复制 22 根补生长轨迹后，视觉收益不足且出现额前短发，因此该对照已删除。秃头问题的剩余原因是旧 PDE 方向场在冠部的覆盖不足，下一步应使用可见覆盖缺口驱动的同区 guide 重求解或正式 supplemental growth；不能把复制旧轨迹当作修复。
