## ADDED Requirements

### Requirement: 发缝必须改变离散求解拓扑
系统 SHALL 在根部薄层中删除或复制所有跨越发缝且位于端点 cap 之外的离散邻接，不得仅依赖有限权重 penalty 表示发缝。

#### Scenario: cap 外左右图册隔离
- **WHEN** 发缝曲线把测试曲面标注为左右两侧
- **THEN** cut graph 在 cap 外的跨缝邻接数必须为零，左右可达集必须不相交

### Requirement: 根部方向必须由真实曲面 PDE 生成
系统 SHALL 在 cut graph 上组装并求解带强边界的离散 Laplace 或 screened-Poisson 系统，并记录残差、迭代状态和强边界误差。

#### Scenario: 曲面 PDE 数值收敛
- **WHEN** 合成曲面具有有效 Dirichlet 锚点和连通图册
- **THEN** 求解器必须达到配置残差，且强边界误差低于容差

### Requirement: 根点和根部前缀必须共享曲面薄层坐标
系统 SHALL 从头皮表面坐标与连续离面高度曲线生成根点及前缀点，不得只对第零点做独立最近点吸附。

#### Scenario: 根部前缀贴附
- **WHEN** 从半球头皮生成 24 点测试发束
- **THEN** 根点距离目标壳层误差必须不超过 `0.5 mm`，前 6 点薄层误差必须不超过 `1 mm`

### Requirement: 发束侧别在端点前保持不变
系统 SHALL 为每根发束携带侧别标签，并确保根部 PDE 查询、积分和 guide 查找在端点 cap 前只访问同侧图册。

#### Scenario: 弯曲发缝零跨越
- **WHEN** 每侧各积分 128 根发束并经过弯曲发缝
- **THEN** cap 前侧别违反数量必须为零

### Requirement: 发缝端点必须有限且连续闭合
系统 SHALL 在有限公制长度的 cap 中恢复左右连通，并使用一阶连续门控避免硬墙或无限向后延伸。

#### Scenario: 端点自然闭合
- **WHEN** 发束经过显式后冠端点 cap
- **THEN** cap 外不得跨缝，cap 内方向变化必须满足配置转角上限

### Requirement: 默认重建行为保持不变
系统 MUST 通过默认关闭的配置启用新拓扑路径，未启用时 SHALL 保持现有重建行为。

#### Scenario: 未启用拓扑路径
- **WHEN** 用户不传入新配置
- **THEN** 主重建 SHALL 使用当前基线路径且不要求新的中间数据
