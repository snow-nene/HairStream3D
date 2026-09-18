"""用轴向二阶张量延拓方向；沿已接受步保持根尖符号，避免正反向量抵消。"""
import numpy as np
import open3d as o3d
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.spatial import cKDTree


def principal_tangent(tensor, normal, reference=None, minimum_gap=.12):
    projector = np.eye(3)-np.outer(normal, normal)
    tangent = projector@tensor@projector
    eigenvalues, vectors = np.linalg.eigh(tangent)
    if eigenvalues[-1] < .025:
        return np.zeros(3), 'insufficient_axial_support'
    if (eigenvalues[-1]-eigenvalues[-2])/eigenvalues[-1] < minimum_gap:
        return np.zeros(3), 'ambiguous_axial_crossing'
    axis = vectors[:, -1]
    if reference is not None and axis@reference < 0:
        axis = -axis
    return axis, None


class AxialSurfaceField:
    def __init__(self, mesh, local_field, chart, support_m=.05, offset_query=None):
        mesh.compute_vertex_normals()
        self.vertices = np.asarray(mesh.vertices)
        self.normals = np.asarray(mesh.vertex_normals)
        self.tree = cKDTree(self.vertices)
        self.chart, self.offset_query = chart, offset_query
        self.reference, self.sign = None, 1.
        self.local_fields = local_field.fields
        self.fields, self.report = {}, {}
        self.branch_counts = {'forward': 0, 'reverse': 0}
        faces = np.asarray(mesh.triangles)
        edges = np.unique(np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]],
                                                  faces[:, [2, 0]]]), axis=1), axis=0)
        for label, (tree, directions) in local_field.fields.items():
            distance, ids = tree.query(self.vertices, k=min(8, tree.n))
            distance = np.asarray(distance).reshape(len(self.vertices), -1)
            ids = np.asarray(ids).reshape(len(self.vertices), -1)
            weights = 1/np.maximum(distance, .001)**2
            weights /= weights.sum(1, keepdims=True)
            observed = np.einsum('nk,nki,nkj->nij', weights, directions[ids], directions[ids])
            normal = self.normals
            projector = np.eye(3)[None]-normal[:, :, None]*normal[:, None, :]
            observed = projector@observed@projector
            eigenvalues = np.linalg.eigvalsh(observed)
            active = distance[:, 0] < support_m
            anchor = (distance[:, 0] < .015) & (eigenvalues[:, -1] > .55)
            if not active.any() or not anchor.any():
                self.report[str(label)] = {'status': 'unsupported'}
                continue
            mapping = np.full(len(active), -1, int)
            mapping[active] = np.arange(active.sum())
            links = edges[active[edges].all(1)]
            a, b = mapping[links].T
            weight = 1/np.maximum(np.linalg.norm(self.vertices[links[:, 0]]-self.vertices[links[:, 1]], axis=1), .0001)
            adjacency = sparse.coo_matrix((np.r_[weight, weight], (np.r_[a, b], np.r_[b, a])),
                                         shape=(active.sum(),)*2).tocsr()
            lap = sparse.diags(np.asarray(adjacency.sum(1)).ravel())-adjacency
            strength = anchor[active].astype(float)*10000
            matrix = lap+sparse.diags(strength+1e-6)
            rhs = strength[:, None]*observed[active].reshape(-1, 9)
            solved = spsolve(matrix, rhs)
            full = np.zeros((len(self.vertices), 3, 3))
            full[active] = solved.reshape(-1, 3, 3)
            self.fields[label] = full
            self.report[str(label)] = {'active_vertices': int(active.sum()), 'anchors': int(anchor.sum()),
                'linear_relative_residual': float(np.linalg.norm(matrix@solved-rhs)/max(np.linalg.norm(rhs), 1))}

    def query_batch(self, points, references, targets, partition=1):
        if not len(points):
            return np.zeros_like(points), np.empty(0, np.int32)
        if partition not in self.fields:
            return np.zeros_like(points), np.full(len(points), 1, np.int32)
        distance, ids = self.tree.query(points, k=4)
        weights = 1/np.maximum(distance, .001)**2
        tensors = np.sum(self.fields[partition][ids]*weights[..., None, None], axis=1)/weights.sum(1)[:, None, None]
        closest = self.chart.scene.compute_closest_points(o3d.core.Tensor(points.astype(np.float32)))
        normals = closest['primitive_normals'].numpy()
        projector = np.eye(3)[None]-normals[:, :, None]*normals[:, None, :]
        eigenvalues, vectors = np.linalg.eigh(projector@tensors@projector)
        axes = vectors[:, :, -1]
        axes[(axes*references).sum(1) < 0] *= -1
        reasons = np.zeros(len(points), np.int32)
        reasons[eigenvalues[:, -1] < .025] = 1
        reasons[(reasons == 0) & ((eigenvalues[:, -1]-eigenvalues[:, -2])/np.maximum(eigenvalues[:, -1], 1e-12) < .12)] = 2
        separation = ((points-closest['points'].numpy())*normals).sum(1)
        if self.offset_query is not None:
            targets = np.maximum(targets, self.offset_query(points))
        axes += normals*np.clip((targets-separation)/.004, -.25, .5)[:, None]
        axes /= np.maximum(np.linalg.norm(axes, axis=1, keepdims=True), 1e-12)
        if getattr(self, 'boundary_guidance', None) is not None:
            axes = self.boundary_guidance(points, axes, normals)
        return axes, reasons

    def query(self, point, partition):
        if partition not in self.fields:
            return np.zeros(3), 'no_surface_region'
        distance, ids = self.tree.query(point, k=4)
        weights = 1/np.maximum(distance, .001)**2
        tensor = np.sum(self.fields[partition][ids]*weights[:, None, None], axis=0)/weights.sum()
        closest = self.chart.scene.compute_closest_points(o3d.core.Tensor(point[None].astype(np.float32)))
        normal = closest['primitive_normals'].numpy()[0]
        reference = self.reference
        if reference is None:
            tree, directions = self.local_fields[partition]
            _, nearest = tree.query(point, k=min(4, tree.n))
            reference = np.mean(directions[np.atleast_1d(nearest)], axis=0)*self.sign
        axis, reason = principal_tangent(tensor, normal, reference)
        if reason:
            return axis, reason
        separation = float((point-closest['points'].numpy()[0])@normal)
        target = float(self.offset_query(point[None])[0]) if self.offset_query is not None else self.chart.clearance
        axis += normal*np.clip((target-separation)/.004, -.25, .5)
        return axis, None

    def commit_step(self, start, end, partition):
        direction = end-start
        self.reference = direction/max(np.linalg.norm(direction), 1e-12)

    def begin_trajectory(self, root, partition, guard):
        lengths = []
        for sign in (1., -1.):
            self.sign, self.reference = sign, None
            point = np.asarray(root, float).copy()
            count = 0
            for _ in range(40):
                vector, reason = self.query(point, partition)
                if reason:
                    break
                end = point+.0005*vector/np.linalg.norm(vector)
                if guard(point, end, partition):
                    break
                self.commit_step(point, end, partition)
                point, count = end, count+1
            lengths.append(count)
        self.sign, self.reference = (1. if lengths[0] >= lengths[1] else -1.), None
        self.branch_counts['forward' if self.sign > 0 else 'reverse'] += 1
