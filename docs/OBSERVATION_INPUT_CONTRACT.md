# 多视图观测输入契约

_2026-09-06；对应 OpenSpec integrate-volume-aware-partitioned-pde 任务 8.1，8.4 部分基础接口。_

## 本轮结果

FLUX 与多视图 HairStep 提图入口新增来源清单。修复 `--views front` 时生成分支可能覆盖原始 front 图谱、合并数组重复 front 的问题；原始 front 与重绘 front 有不同身份。只指定 front 时不会消费生成视图。指定生成视图缺失时，在写图谱前失败。

本轮未重新执行实际 FLUX/HairStep 模型推理。测试使用现有入口和推理替身验证输入输出契约，未引入训练或 DOAE 依赖。旧图谱保持不变，不能依据文件名为其补造历史来源。

## 清单与来源

`flux_redrawn/generation_manifest.json` 保存原始图像、每视角渲染/生成图摘要、prompt、实际 seed、模型路径与配置摘要。`--seed` 默认为 42，每视角由视角名稳定派生，调整视角顺序不改变该视角 seed。固定 seed 不承诺跨硬件逐字节相同。模型权重目录未做全量哈希，清单明确记录 `model_identity_scope`，不宣称完整模型内容可验证。

`maps/observation_manifest.json` 保存指定视角、原始 front、HairStep checkpoint 摘要、原始图像与图谱摘要、复制或推理的派生方式、seg 及来源组。若生成清单存在，提图前检查其完成状态、原始图像和生成图是否匹配，保存关联清单摘要；缺失时显式记录 `unverified_legacy`。同源生成视图使用共同来源组。

两类清单均先写 `building`，完成后写 `complete`，采用临时文件替换。完成状态只表示该阶段写入结束，不自动证明历史生成来源或几何配准可靠。输入消费工具可校验所选视角图像及 strand/depth/seg 文件摘要，发现被替换或未完成运行即拒绝。

核心工具位于 `lib/multiview_observation.py`，只依赖 NumPy 与标准库；不会因 FLUX 独立环境导入该模块而加载重建策略、Torch 或 Taichi。

## 共同可见性与预算接口

`common_observation_gate` 显式接收 seg、RGB uint8 strand_map、depth_map、网格可见性与配准通过状态，返回共同有效区域及可重叠的逐原因拒绝掩码。缺失/无效像素记入 `unknown_or_conflict`，不生成空气证据。当前接口不自行证明相机配准或生成可见性，后续需接入任务 8.2 的独立检查。

`source_budget_weights` 输出用于采样或软观测的归一化质量。当前接口默认原始观测合计 1、生成观测合计 0.5；同源视图共享预算，增加来源组也不能突破生成总上限。这不是 PDE 权重标定结果，尚未接入生产 PDE，也不能授权生成观测覆盖原始硬锚。8.4 因共同可见检查、正式消费及权重接入尚未完成而保持未勾选。

## 当前图像诊断

`observation_contract_four_views/` 位于当前 image_id 的 `pde_governance/volume_partition_integration/`，由 `scripts/vis/audit_observation_contract.py` 对已有四视角网格交点缓存进行检查。

| 视角 | seg 像素 | 配准门禁前候选 |
| --- | --- | --- |
| front | 74,561 | 53,017 |
| left | 59,061 | 49,452 |
| right | 69,008 | 57,694 |
| back | 75,481 | 60,766 |

缓存只代表带轮廓裕量和入射过滤的头发网格命中，未新增头部遮挡检查。旧样例无来源清单，也缺本轮契约要求的独立配准证明，正式 `accepted` 均为 0；配准前候选及预算文件均带 `provisional` 标记，不能作为有效 PDE 输入。该拒绝不是此前神经深度校准失败的重复：现在使用网格几何，待验证的是图像与网格的配准及输入来源。

## 验证

8 项测试覆盖 front 身份、单 front 入口、重复生成视图和来源组预算、无效深度/几何/配准、资产更改、未完成清单、生成顺序与原始图保护、历史来源缺失。日志：`tests/outputs/volume_partition_integration/observation_all_tests.log`。两生产入口修改前的 GitNexus upstream impact 为 LOW，直接调用者各 1 个，未检索到上游受影响流程；新增独立模块尚未索引，其影响分析返回 UNKNOWN。

任务 8.2 已完成：`lib/multiview_registration.py` 将 HairStep/网格使用的 normalized camera 显式转换为 image-pixel projection，检查可见交点的像素重投影误差、图像内比例和矩阵有效性。样例 front 的 53,017 个缓存交点 q95 误差为约 5.1e-13 px，在 1 px 门禁下通过；轴翻转和尺度错误测试拒绝。四视图旧缓存仍没有生成清单或独立配准来源证明，observation audit 的正式 accepted 继续为 0。下一步完成 8.4 的真实共同可见掩码和生产权重接入，再送入播种/PDE；保持当前生成结果和正式默认路径隔离。

任务 9.1 已完成：`lib/multiview_direction_lifting.py` 沿 strand_map 的二维方向查询邻近可见交点，计算世界坐标差分方向。构建器在 mesh_raycast 模式优先使用连续交点差分；孤立像素可保留网格切平面作为低置信回退，但分区跳变、遮挡层跳变、缺失命中、出图和零长度都会保留原因码，不连接不同层。当前尚未完成中心差分与切平面夹角门禁（9.2）及三种提升模式固定样例消融（9.3）。

任务 9.2 已完成：差分结果与网格切平面结果通过轴向夹角门禁比较，默认最大夹角 30°；正负方向视为同一轴向方向。不兼容样本降级到切平面方向，并在报告中保留计数与中位角度。front 样例 44,619 个连续差分中 38,423 个兼容、6,196 个不兼容；6,276 个像素没有相邻可见交点。这里的“降级”不跨分区，也不证明三维方向真实正确；9.3 的模式消融仍待完成。

任务 9.3 已完成固定模式消融，产物位于 `lifting_ablation_front_64/`。相同 front 缓存和 2 px 步幅下，切平面参考、可见交点差分、一致性筛选的接受数为 53,017、46,513、25,673；一致性筛选接受样本轴向角 q95 为 28.47°。旧缓存没有保存网格法线，因此其切平面模式只是 image-ray tangent reference，不能替代正式法线切平面结果；消融仅用于固定输入下比较拒绝率和门限敏感性，不切换生产默认。

10.1/10.2 新增 `lib/direction_sampling.py`：在固定总预算内按分区选择高显著性样本，并用剩余预算补充均匀候选；方向分桶使用绝对点积，整体翻转不改变分桶或选择。样本权重按选中数归一化，重复同源样本不会增加总监督质量。当前采样器已通过合成测试，但尚未替换生产 PDE 锚点默认路径。
