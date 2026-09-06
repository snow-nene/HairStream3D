## Why

现有 PDE 发缝实验已经能在头皮根部形成左右分岸，但侧向参考场会在离开头皮后的前 30–50 mm 内持续把两岸发束推开，最终表现为宽阔的“头发连廊”，而不是真实的窄发缝。需要一个与全局 PDE 解耦的局部实验，验证只重建根区、保留 v30 外层发型能否稳定形成真实发缝。

## What Changes

- 新增默认关闭的局部发缝 groom layer 后处理能力，以 v30 发束作为外层造型，并只约束发缝附近的根段。
- 从已有发缝曲线与分岸根点中均衡采样两岸，将其平滑连接到 v30 的匹配导向发束。
- 在根部薄层内强制左右不跨岸、毫米级贴合头皮，并限制前 30 mm 的侧向扩张；交接区之后保持 v30 尾段不变。
- 输出独立 PLY、结构化质量指标和可渲染结果，便于与原始 v30 做定量和视觉对照。
- 新增合成单元测试和一个针对指定 Image ID 的可复现实验入口，不改变现有生产重建默认行为。

## Capabilities

### New Capabilities

- `local-parting-groom-layer`: 定义局部发缝根段的生成、平滑交接、质量门禁和结果输出行为。

### Modified Capabilities

无。

## Impact

- 新增 `lib/recon_strategy/` 下的局部发缝几何模块、`scripts/recon_3d/` 下的实验脚本以及 `tests/` 下的测试。
- 读取现有 v30 PLY、头部网格和 `parting_topology_full/root_prefixes.npz`，输出仅写入对应 Image ID 的 `results/multiview_data/<image_id>/pde_governance/` 子目录。
- 不修改现有 PDE 求解器、渲染器接口或默认重建链路；无破坏性 API 变更。
