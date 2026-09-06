## ADDED Requirements

### Requirement: 无向方向突变检测
系统 SHALL 将 `strand_map` 解码为无向双角线场，并在头发 `seg` 内计算不受方向正负号影响的局部方向突变分数。

#### Scenario: 方向符号翻转不产生伪边缘
- **WHEN** 相邻像素方向互为相反向量但表示同一条无向直线
- **THEN** 其轴向方向差必须接近零

#### Scenario: 不同线方向产生高分数
- **WHEN** 相邻有效像素的无向夹角显著不同
- **THEN** 方向突变分数必须随无向夹角增大

### Requirement: 深度突变检测
系统 SHALL 在头发 mask 内对 `depth_map` 做 mask-aware 平滑，并生成稳健归一化的内部深度跳变分数。

#### Scenario: 外轮廓不作为内部深度边缘
- **WHEN** 深度跳变只发生在 `seg` 外部或头发外轮廓
- **THEN** 内部有效区的深度边缘结果不得包含该跳变

### Requirement: 多证据融合与候选边界
系统 SHALL 独立保存方向分数、深度分数和融合分数，并按照预声明的多个分位数生成候选内部阻隔线。

#### Scenario: 检测不读取监督发缝
- **WHEN** 系统生成融合分数和候选阻隔线
- **THEN** 检测过程只能读取 `strand_map`、`depth_map`、`seg` 及显式数值参数

### Requirement: 可解释分区
系统 SHALL 从头发 mask 扣除候选阻隔线后计算连通分区，保存标签数组、彩色分区图、每区面积和有效分区数量。

#### Scenario: 边界不足以切开区域
- **WHEN** 候选阻隔线未形成有效拓扑切割
- **THEN** 系统必须报告单个有效分区，不得强制生成多个区域

### Requirement: 原图诊断与事后评估
系统 SHALL 将正面边缘和分区叠加在 `raw_img.png`，并在检测完成后可选地使用已有发缝 mask 计算邻域精度与召回率。

#### Scenario: 使用原始正面图
- **WHEN** 输出正面诊断图
- **THEN** 背景必须来自该 Image ID 的 `raw_img.png`，不得使用 Blender 渲染图

### Requirement: 隔离执行
系统 SHALL 通过 `scripts/vis/` 下的独立入口执行，并将生成结果写入 `results/multiview_data/<image_id>/pde_governance/` 下的专用目录。

#### Scenario: 第一阶段不改变主链路
- **WHEN** 执行突变检测与分区审计
- **THEN** 默认 `run_pde_multiview.py` 的代码、参数和输出均保持不变

### Requirement: 显式分区 PDE
系统 SHALL 支持把二维正面分区标签提升到三维体素标签，并在 screened-Poisson 离散算子中删除不同标签体素之间的扩散耦合。

#### Scenario: 跨分区邻接不传递方向
- **WHEN** 两个相邻 PDE 体素具有不同的正分区标签
- **THEN** 它们之间的有限差分通量权重必须为零

#### Scenario: 同分区保持原求解
- **WHEN** 相邻体素标签相同
- **THEN** 它们之间必须保留原有的加权扩散项

#### Scenario: 默认行为兼容
- **WHEN** 未提供分区标签
- **THEN** screened-Poisson 的算子、边界条件和输出必须与修改前一致
