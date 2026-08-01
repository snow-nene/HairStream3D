本项目使用 pixi 管理

seg为头发的蒙版

<!-- gitnexus:start -->
# GitNexus — Code Intelligence

This project is indexed by GitNexus as **HairStep** (16304 symbols, 32440 relationships, 300 execution flows). Use the GitNexus MCP tools to understand code, assess impact, and navigate safely.

> Index stale? Run `node .gitnexus/run.cjs analyze` from the project root — it auto-selects an available runner. No `.gitnexus/run.cjs` yet? `npx gitnexus analyze` (npm 11 crash → `npm i -g gitnexus`; #1939).

## Always Do

- **MUST run impact analysis before editing any symbol.** Before modifying a function, class, or method, run `impact({target: "symbolName", direction: "upstream"})` and report the blast radius (direct callers, affected processes, risk level) to the user.
- **MUST run `detect_changes()` before committing** to verify your changes only affect expected symbols and execution flows. For regression review, compare against the default branch: `detect_changes({scope: "compare", base_ref: "main"})`.
- **MUST warn the user** if impact analysis returns HIGH or CRITICAL risk before proceeding with edits.
- When exploring unfamiliar code, use `query({search_query: "concept"})` to find execution flows instead of grepping. It returns process-grouped results ranked by relevance.
- When you need full context on a specific symbol — callers, callees, which execution flows it participates in — use `context({name: "symbolName"})`.
- For security review, `explain({target: "fileOrSymbol"})` lists taint findings (source→sink flows; needs `analyze --pdg`).

## Never Do

- NEVER edit a function, class, or method without first running `impact` on it.
- NEVER ignore HIGH or CRITICAL risk warnings from impact analysis.
- NEVER rename symbols with find-and-replace — use `rename` which understands the call graph.
- NEVER commit changes without running `detect_changes()` to check affected scope.

## Resources

| Resource | Use for |
|----------|---------|
| `gitnexus://repo/HairStep/context` | Codebase overview, check index freshness |
| `gitnexus://repo/HairStep/clusters` | All functional areas |
| `gitnexus://repo/HairStep/processes` | All execution flows |
| `gitnexus://repo/HairStep/process/{name}` | Step-by-step execution trace |

## CLI

| Task | Read this skill file |
|------|---------------------|
| Understand architecture / "How does X work?" | `.claude/skills/gitnexus/gitnexus-exploring/SKILL.md` |
| Blast radius / "What breaks if I change X?" | `.claude/skills/gitnexus/gitnexus-impact-analysis/SKILL.md` |
| Trace bugs / "Why is X failing?" | `.claude/skills/gitnexus/gitnexus-debugging/SKILL.md` |
| Rename / extract / split / refactor | `.claude/skills/gitnexus/gitnexus-refactoring/SKILL.md` |
| Tools, resources, schema reference | `.claude/skills/gitnexus/gitnexus-guide/SKILL.md` |
| Index, status, clean, wiki CLI commands | `.claude/skills/gitnexus/gitnexus-cli/SKILL.md` |

<!-- gitnexus:end -->





## 目录布局与代码组织约束

**核心准则：请勿在项目根目录或其他已有目录中“胡乱”创建未分类的脚本或文件。必须严格遵守现有的目录分类结构！**

- **测试代码**: 所有的独立测试脚本必须放入 `tests/` 文件夹中。
- **执行脚本 (`scripts/`)**: `scripts/` 目录已按功能严格分类，新建脚本必须放入对应的子目录中，**绝对不得**直接放置在 `scripts/` 根目录下：
  - `scripts/train/`: 模型训练、伪标签生成等相关脚本
  - `scripts/test/`: 模型评估、Loss 测试及过拟合测试等相关脚本
  - `scripts/infer_2d/`: 2D 推理、遮罩生成、深度提取及 Flux 重绘相关脚本
  - `scripts/recon_3d/`: 3D 发丝重建、PDE 生成和 3D 网格提取脚本
  - `scripts/render/`: Blender 渲染、网格渲染及多视角贴图生成
  - `scripts/utils/`: 通用工具（如网格对齐、特征点提取、相机投影转换等）
  - `scripts/vis/`: 可视化排查及结果对比脚本
- **文档资料**: 所有的说明文档、计划和报告必须放入 `docs/` 文件夹中，并使用中文编写（代码字段除外）。

如果在开发中遇到上述分类无法涵盖的新功能，必须先询问用户是否需要建立新的子目录，严禁随意在现有目录中堆砌无分类文件。

## 输出目录约束

**所有脚本执行产生的输出文件（如日志、模型权重、中间数据、测试输出等）绝对不允许输出到代码源代码目录下，也严禁散落在根目录。**

1. **测试输出**: 所有位于 `tests/` 下的测试脚本或评估脚本产生的测试结果、图表、临时文件等，必须输出到 `tests/outputs/` 或指定的统一临时测试输出目录中，**绝对不得**与测试代码混放在一起。
2. **中间数据与生成结果**: 所有数据处理、渲染、推理产生的结果必须统一输出到 `results/` 下对应的任务分类子目录中（例如 `results/multiview_strand_depth/`）。
3. **模型权重**: 所有训练过程产生的模型权重与 Checkpoint 必须统一输出到 `checkpoints/` 目录下的对应子目录中。
4. **数据集与预处理文件**: 任何预处理后的数据集文件，必须集中保存在 `datasets/` 目录下。

**严禁**任何脚本执行后在当前运行目录（如项目根目录，或 `scripts/` 的各个子目录）遗留任何 `.png`, `.npy`, `.json`, `.obj`, `.log` 等生成性质的输出文件！
