## ADDED Requirements

### Requirement: 合成冒烟必须先于真实数据运行
系统 SHALL 依次运行 S0 cut graph、S1 根部薄层、S2 端点 cap 和 S3 符号翻转测试；任一硬断言失败时 MUST 停止，不得继续真实数据冒烟。

#### Scenario: 合成冒烟失败
- **WHEN** S0–S3 任一测试报告跨缝、贴附超差、PDE 未收敛或符号不变性失败
- **THEN** 系统不得启动 S4，也不得通过调节 trim 或吸附参数掩盖失败

### Requirement: 真实冒烟必须限制规模
S4 SHALL 仅处理指定 Image ID 的冠部区域，最多使用 256 根、30 个采样点和 `64³` 体素或等价曲面规模。

#### Scenario: 运行真实冠部冒烟
- **WHEN** S0–S3 全部通过并启动 S4
- **THEN** 系统不得运行 `10,000` 根或 `128³` 全域重建

### Requirement: 核心冒烟必须关闭造型后处理
S0–S4 SHALL 关闭 KMeans 跨束聚集、散度噪声、Laplacian 平滑和最终发缝 trim。

#### Scenario: 核心不变量评估
- **WHEN** 测量拓扑和根部几何指标
- **THEN** 指标必须来自 PDE 与根部薄层本身，不得来自后处理删除或视觉遮盖

### Requirement: 冒烟必须输出结构化指标
系统 SHALL 输出 PDE 收敛、跨缝邻接、左右可达集、根部距离、薄层误差、侧别违反、端点转角、冻结和符号翻转轨迹差。

#### Scenario: 冒烟成功
- **WHEN** 一个测试完成
- **THEN** 对应输出目录必须包含可机器读取的 JSON 指标和清晰的通过状态

### Requirement: 输出必须遵守项目目录约束
合成测试输出 MUST 位于 `tests/outputs/parting_topology_smoke/`，真实数据输出 MUST 位于指定 Image ID 的 `results/multiview_data/.../pde_governance/parting_topology_smoke/`。

#### Scenario: 写入测试产物
- **WHEN** 冒烟生成 JSON、图像或几何文件
- **THEN** 项目根目录、`scripts/` 和测试源码目录不得出现生成产物
