"""用原图边界距离和精确遮挡余量调整候选方向，不替代最终语义守卫。"""
import numpy as np
import open3d as o3d
from scipy.ndimage import distance_transform_edt, map_coordinates


class FrontBoundaryGuidance:
    def __init__(self, chart, band_px=4., max_angle_deg=15.):
        if band_px <= 0 or not 0 < max_angle_deg < 90:
            raise ValueError('边界带宽须为正，最大转角须在0至90度之间')
        if not np.allclose(chart.camera[3], [0,0,0,1]):
            raise ValueError('边界引导要求正交front相机')
        self.chart = chart
        self.band = band_px
        self.max_angle = np.deg2rad(max_angle_deg)
        mask = chart.hair_domain
        self.distance = distance_transform_edt(mask)-distance_transform_edt(~mask)
        self.gy, self.gx = np.gradient(self.distance)

    def __call__(self, points, directions, normals):
        if not len(points):
            return directions.copy()
        c = self.chart
        uv, _ = c.project(points)
        valid = ((uv >= 0) & (uv <= [c.w-1, c.h-1])).all(1)
        coords = uv[:, ::-1].T
        distance = map_coordinates(self.distance, coords, order=1, mode='nearest')
        gradient = np.column_stack([map_coordinates(g, coords, order=1, mode='nearest')
                                    for g in [self.gx, self.gy]])
        toward = c.inverse[:3, 2].copy()
        toward /= np.linalg.norm(toward)
        rays = np.c_[points+2*toward, np.broadcast_to(-toward, points.shape)]
        hit = c.scene.cast_rays(o3d.core.Tensor(rays.astype(np.float32)))['t_hit'].numpy()
        # 深度遮挡超过1mm时不干预；在轮廓内4px带内平滑增强引导。
        exposed = np.clip((hit-(2-.001))/.001, 0, 1)
        strength = valid*(distance > 0)*np.clip(1-distance/self.band, 0, 1)*exposed
        jacobian = c.camera[:2, :3]*np.array([c.w-1,c.h-1])[:,None]/2
        inward = gradient@jacobian
        inward -= normals*(inward*normals).sum(1)[:,None]
        inward /= np.maximum(np.linalg.norm(inward,axis=1,keepdims=True),1e-12)
        outward_component = np.minimum((directions*inward).sum(1), 0)
        proposal = directions-strength[:,None]*outward_component[:,None]*inward
        proposal /= np.maximum(np.linalg.norm(proposal,axis=1,keepdims=True),1e-12)
        angle = np.arccos(np.clip((proposal*directions).sum(1),-1,1))
        turn = proposal-directions*(proposal*directions).sum(1)[:,None]
        turn /= np.maximum(np.linalg.norm(turn,axis=1,keepdims=True),1e-12)
        limited = np.minimum(angle,self.max_angle)
        result = directions*np.cos(limited)[:,None]+turn*np.sin(limited)[:,None]
        return result/np.maximum(np.linalg.norm(result,axis=1,keepdims=True),1e-12)
