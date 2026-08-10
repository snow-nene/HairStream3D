# Silhouette Guard：可见性驱动的轮廓硬约束

## 背景

多视角 PDE 重建（`scripts/recon_3d/run_pde_multiview.py`）曾出现重建轮廓与输入
front hair seg 不贴合的问题：发丝越过输入发型轮廓后继续生长，尤其向下外扩
（基线 front IoU ≈ 0.596，精度仅 ≈ 0.616）。根因分析确认这**不是相机标定问
题**（平移搜索最多把 IoU 提到 0.605），而是：

1. front seg 只用于筛选 guide 发根，RK4 生长过程本身不受轮廓约束；
2. 最近有效方向 fallback 无法区分「seg 内部小孔洞」（应继续）与「真实发型
   边界」（应停止），可跨越真实轮廓最多 12 步；
3. hair mesh 与输入 seg 不匹配，把轮廓误差传入 PDE 几何先验；
4. 保存阶段没有轮廓裁剪作为最后防线。

## 方案

新增 `lib/silhouette_guard.py` 的 `SilhouetteGuard`：把输入图像的头发 seg 作
为 RK4 生长的**硬约束**。对每个 RK4 候选点：

```text
候选 3D 点
  └─ 投影到主视角 front
       ├─ 清晰落在 seg 内（>= tolerance_px）    → OK，继续生长
       ├─ front 可见，刚越界 (0, tolerance_px]  → SOFT，进入 grace 期
       │    （允许最近有效方向 fallback 帮助回到轮廓内；回到内部重置 grace）
       ├─ front 可见，明显越过 seg              → HARD，立即停止该发丝
       └─ front 不可见（被头模遮挡）            → 交给 side/back seg 裁决
            ├─ 任一可见视角确认在 seg 内        → OK
            ├─ 仅在容差带内                     → SOFT
            ├─ 所有可见视角都明显越界           → HARD
            └─ 没有任何视角可见                 → 不约束
```

关键实现细节：

- **可见性**：与 `compute_root_head_visibility` 同一约定的正交射线遮挡测试
  （头模 RaycastingScene，构造一次复用）；
- **轮廓距离**：每个视角预先计算 seg 的符号距离场（EDT，像素单位，内正外
  负），运行时双线性采样；
- **front 是权威包络**：front 可见的点只由 front seg 裁决；side/back 只负责
  front 看不到的区域，且采用「任一可见视角确认即通过」的宽松策略，避免合成
  视角 seg 不一致导致误杀；
- **被停止的发丝冻结**在最后合法位置，不再推进。

## 防线布局

| 阶段 | 位置 | 作用 |
|------|------|------|
| RK4 生长约束 | `hair_synthesis_rk4`（guide 与全量各一次） | 越界即停，grace 带内允许 fallback 回补内部孔洞 |
| 保存前裁剪 | `lib/hair_util.py::trim_strands_by_silhouette` | 以更紧的 margin 截断越界点、压缩冻结重复尾部、丢弃残留短发丝 |

## 参数

| CLI 参数 | 默认 | 说明 |
|----------|------|------|
| `--silhouette_tolerance_px` | 3.0 | seg 外侧 grace 带宽（像素） |
| `--silhouette_grace_steps` | 3 | grace 带内最多徘徊的 RK4 步数 |
| `--silhouette_trim_margin_px` | 1.0 | 保存阶段裁剪容差（像素） |
| `--disable_silhouette_guard` | 关 | 消融开关：完全禁用约束 |

## 评测

`tests/test_silhouette_iou.py`：把 PLY 用真实 front 标定投影回输入图像（含头
模遮挡、按 3D 线段光栅化），计算 IoU / 召回 / 精度 / bbox，输出叠加图与报告
到 `tests/outputs/silhouette_iou/<img_id>/`。

img_id `0a1ba3dbefc8934ab60577c5c91f66a0`（4 视角）基线：
IoU=0.596，recall=0.948，precision=0.616，重建 bbox [91,15,391,346]
vs seg bbox [93,36,350,256]（向下外扩 90px）。

## 已知局限

- 约束只作用于「生长停止」，不会把已经外扩的 PDE 方向场拉回轮廓内；
- hair mesh 不贴合（根因 3）仍会通过切向/metric-band 先验影响方向场质量，
  需要在上游 mesh 对齐环节单独解决；
- 遮挡测试仅使用头模，不含头发自遮挡（与 seg-support volume 的权威包络语义
  一致）。
