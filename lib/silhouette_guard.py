"""可见性驱动的 silhouette 约束器（SilhouetteGuard）。

用于 RK4 发丝生长过程：把输入图像的头发 seg 作为**硬约束**，防止发丝
投影越过输入发型轮廓后继续生长。

判定流程（对每个 RK4 候选点）::

    候选 3D 点
      └─ 投影到主视角（front）
           ├─ 落在 seg 内                            → OK，继续生长
           ├─ front 可见 且 越界 < hard_px           → SOFT，进入 grace
           │    （允许短暂沿最近有效方向 fallback；连续越界步数超过
           │     grace 预算才停止，回到内部立即重置 grace）
           ├─ front 可见 且 越界 >= hard_px          → HARD，立即停止
           └─ front 不可见（被头模遮挡）             → 交给 side/back seg
                ├─ 任一可见视角确认在 seg 内          → OK
                ├─ 越界但 < hard_px                   → SOFT
                ├─ 所有可见视角都越界 >= hard_px      → HARD
                └─ 没有任何视角可见                   → 不约束（OK）

「深度 + 时长」双判据来自实测（img 0a1ba3db…，见 docs/SILHOUETTE_GUARD.md）：
发际线附近的合法发丝在生长初期会瞬态越界（中位深度 6px、最长 9 步）再回到
轮廓内，而真正过度生长的发丝会持续越界数十步或深达数十 px。

可见性用头模正交射线遮挡测试（与 ``compute_root_head_visibility`` 同一
约定）。每个视角的轮廓距离通过 seg 的符号距离场（像素单位）双线性采样，
内部为正、外部为负。
"""

import numpy as np

STATUS_OK = 0
STATUS_SOFT = 1
STATUS_HARD = 2


class SilhouetteGuard:
    """把多视角 hair seg 作为 RK4 生长的 silhouette 硬约束。"""

    def __init__(
        self,
        seg_masks,
        calibs,
        head_mesh_path="data/head_model.obj",
        hard_px=20.0,
        surface_tolerance=0.01,
        primary_view="front",
    ):
        """
        Args:
            seg_masks: dict view -> bool mask (H, W)， True = 头发。
            calibs:    dict view -> (calib[4,4], R) 或 calib[4,4]；
                       calib 把世界坐标映射到 NDC（左上原点，y 向下）。
            head_mesh_path: 用于遮挡测试的头模。
            hard_px: 立即停止的越界深度阈值（像素）。更浅的越界进入
                SOFT grace 流程，由调用方按连续越界步数裁决。
            surface_tolerance: 遮挡测试的表面容差（米）。
            primary_view: 权威视角（front）。其可见点只由它的 seg 判定。
        """
        import open3d as o3d
        from scipy.ndimage import distance_transform_edt

        self.hard_px = float(hard_px)
        self.surface_tolerance = float(surface_tolerance)
        self.views = [v for v in seg_masks if v in calibs]
        if not self.views:
            raise ValueError("SilhouetteGuard 需要至少一个有 calib 的视角")
        self.primary = primary_view if primary_view in self.views else self.views[0]
        self.side_views = [v for v in self.views if v != self.primary]

        self.calibs = {}
        self.masks = {}
        self.sd_fields = {}
        self.toward_camera = {}
        for view in self.views:
            calib = calibs[view]
            if isinstance(calib, (tuple, list)):
                calib = calib[0]
            calib = np.asarray(calib, dtype=np.float64)
            self.calibs[view] = calib
            mask = np.asarray(seg_masks[view]) > 0
            if mask.ndim == 3:
                mask = mask[:, :, 0] > 0
            self.masks[view] = mask
            inside = distance_transform_edt(mask)
            outside = distance_transform_edt(~mask)
            self.sd_fields[view] = (inside - outside).astype(np.float32)
            linear = calib[:3, :3]
            try:
                toward = np.linalg.solve(linear, np.array([0.0, 0.0, 1.0]))
            except np.linalg.LinAlgError:
                toward = linear[2].copy()
            self.toward_camera[view] = toward / (np.linalg.norm(toward) + 1e-8)

        head_mesh = o3d.io.read_triangle_mesh(str(head_mesh_path))
        if not head_mesh.has_vertices() or not head_mesh.has_triangles():
            raise ValueError(f"Head mesh is empty: {head_mesh_path}")
        ray_scene = o3d.t.geometry.RaycastingScene()
        ray_scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(head_mesh))
        self.scene = ray_scene
        bounds = head_mesh.get_axis_aligned_bounding_box()
        self.ray_length = max(float(np.linalg.norm(bounds.get_extent())) * 3.0, 1.0)

    # ------------------------------------------------------------------
    # 基础查询
    # ------------------------------------------------------------------

    def _project_px(self, view, points):
        """世界坐标 [N,3] -> 图像像素 (px_x, px_y)，左上原点。"""
        calib = self.calibs[view]
        homo = np.concatenate(
            [points, np.ones((len(points), 1), dtype=points.dtype)], axis=1
        )
        clip = homo @ calib.T
        ndc = clip[:, :2] / (clip[:, 3:4] + 1e-8)
        height, width = self.masks[view].shape
        px = (ndc[:, 0] + 1.0) * 0.5 * (width - 1)
        py = (ndc[:, 1] + 1.0) * 0.5 * (height - 1)
        return px, py

    def _sample_sd(self, view, px, py):
        """在符号距离场上双线性采样；画面外返回 NaN。"""
        sd = self.sd_fields[view]
        height, width = sd.shape
        in_img = (px >= 0) & (px <= width - 1) & (py >= 0) & (py <= height - 1)
        px_c = np.clip(px, 0, width - 1.001)
        py_c = np.clip(py, 0, height - 1.001)
        x0 = np.floor(px_c).astype(np.int64)
        y0 = np.floor(py_c).astype(np.int64)
        x1 = np.minimum(x0 + 1, width - 1)
        y1 = np.minimum(y0 + 1, height - 1)
        wx = (px_c - x0).astype(np.float32)
        wy = (py_c - y0).astype(np.float32)
        val = (
            sd[y0, x0] * (1 - wx) * (1 - wy)
            + sd[y0, x1] * wx * (1 - wy)
            + sd[y1, x0] * (1 - wx) * wy
            + sd[y1, x1] * wx * wy
        )
        val[~in_img] = np.nan
        return val, in_img

    def _visible(self, view, points):
        """正交遮挡测试：点是否在该视角可见（没有被头模遮挡）。"""
        import open3d as o3d

        if len(points) == 0:
            return np.zeros(0, dtype=bool)
        toward = self.toward_camera[view]
        length = self.ray_length
        origins = points + toward[None, :] * length
        directions = np.broadcast_to(-toward, points.shape).copy()
        rays = np.concatenate([origins, directions], axis=1).astype(np.float32)
        t_hit = self.scene.cast_rays(o3d.core.Tensor(rays))["t_hit"].numpy()
        return (~np.isfinite(t_hit)) | (
            t_hit >= length - self.surface_tolerance
        )

    # ------------------------------------------------------------------
    # 主判定
    # ------------------------------------------------------------------

    def classify(self, points, hard_px=None):
        """对 [N,3] 世界坐标点返回逐点状态 (int8)。

        STATUS_OK=0 在轮廓内；STATUS_SOFT=1 越界但深度 < hard_px（由调用方
        按连续越界步数 grace 裁决）；STATUS_HARD=2 越界深度 >= hard_px，
        应立即停止该发丝。

        hard_px 覆盖实例默认值（保存阶段裁剪会传更小的值）。
        """
        hp = self.hard_px if hard_px is None else float(hard_px)
        points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
        num = len(points)
        status = np.zeros(num, dtype=np.int8)
        if num == 0:
            return status

        primary = self.primary
        px, py = self._project_px(primary, points)
        sd, in_img = self._sample_sd(primary, px, py)

        clearly_inside = in_img & np.isfinite(sd) & (sd >= 0.0)
        need = ~clearly_inside
        if not need.any():
            return status

        idx = np.where(need)[0]
        sub = points[idx]
        sd_sub = sd[idx]
        in_sub = in_img[idx]
        vis = self._visible(primary, sub)

        local_hard = np.zeros(len(idx), dtype=bool)
        local_soft = np.zeros(len(idx), dtype=bool)

        # 主视角可见：由主视角 seg 单独裁决（front 是权威轮廓包络）
        far_out = (~in_sub) | ~np.isfinite(sd_sub) | (sd_sub <= -hp)
        near_out = in_sub & np.isfinite(sd_sub) & (sd_sub < 0.0) & (sd_sub > -hp)
        local_hard[vis] = far_out[vis]
        local_soft[vis] = near_out[vis]

        # 主视角被头模遮挡：交给 side/back seg（任一可见视角确认即通过）
        occluded = ~vis
        if occluded.any() and self.side_views:
            rest_pts = sub[occluded]
            side_ok = np.zeros(len(rest_pts), dtype=bool)
            side_near = np.zeros(len(rest_pts), dtype=bool)
            side_seen = np.zeros(len(rest_pts), dtype=bool)
            for view in self.side_views:
                vpx, vpy = self._project_px(view, rest_pts)
                vsd, v_in = self._sample_sd(view, vpx, vpy)
                v_vis = v_in & np.isfinite(vsd) & self._visible(view, rest_pts)
                side_seen |= v_vis
                side_ok |= v_vis & (vsd >= 0.0)
                side_near |= v_vis & (vsd > -hp)
            local_hard[occluded] = side_seen & ~side_near
            local_soft[occluded] = side_seen & side_near & ~side_ok
            # 没有任何视角可见的点：不约束（保持 OK）

        status[idx[local_hard]] = STATUS_HARD
        status[idx[local_soft & ~local_hard]] = STATUS_SOFT
        return status

    def summary(self):
        heights = [self.masks[v].shape[0] for v in self.views]
        widths = [self.masks[v].shape[1] for v in self.views]
        return (
            f"SilhouetteGuard(views={self.views}, primary={self.primary}, "
            f"hard_px={self.hard_px}, image={widths[0]}x{heights[0]})"
        )
