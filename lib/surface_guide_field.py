"""在头模曲面重求解带 guide 约束的切向场，查询后重新积分新根。"""
import numpy as np
from scipy.spatial import cKDTree
from scipy import sparse
from scipy.sparse.linalg import spsolve
import open3d as o3d


class SurfaceGuideField:
    def __init__(self, mesh, local_field, chart, support_m=.03):
        self.chart = chart
        self.sign = 1.
        self.branch_counts={'forward':0,'reverse':0}
        mesh.compute_vertex_normals()
        self.vertices = np.asarray(mesh.vertices)
        self.normals = np.asarray(mesh.vertex_normals)
        self.tree = cKDTree(self.vertices)
        faces = np.asarray(mesh.triangles)
        edges = np.unique(np.sort(np.concatenate([faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]]]),axis=1),axis=0)
        self.fields = {}
        self.report = {}
        for label,(guide_tree,directions) in local_field.fields.items():
            distance, ids = guide_tree.query(self.vertices,k=min(8,guide_tree.n))
            distance, ids = np.asarray(distance).reshape(len(self.vertices),-1),np.asarray(ids).reshape(len(self.vertices),-1)
            weights = 1/np.maximum(distance,.001)**2
            observed = (directions[ids]*weights[...,None]).sum(1)/weights.sum(1)[:,None]
            observed -= self.normals*(observed*self.normals).sum(1)[:,None]
            magnitude = np.linalg.norm(observed,axis=1)
            anchor = (distance[:,0]<.006)&(magnitude>.7)
            observed /= np.maximum(magnitude[:,None],1e-12)
            active = distance[:,0]<support_m
            if not active.any() or not anchor.any():
                self.report[str(label)] = {'active_vertices':int(active.sum()),'anchors':int(anchor.sum()),'status':'unsupported'}
                continue
            mapping = np.full(len(active),-1,int)
            mapping[active] = np.arange(active.sum())
            links = edges[active[edges].all(1)]
            a,b = mapping[links].T
            weight = 1/np.maximum(np.linalg.norm(self.vertices[links[:,0]]-self.vertices[links[:,1]],axis=1),.0001)
            adjacency = sparse.coo_matrix((np.r_[weight,weight],(np.r_[a,b],np.r_[b,a])),shape=(active.sum(),)*2).tocsr()
            lap = sparse.diags(np.asarray(adjacency.sum(1)).ravel())-adjacency
            constraint = anchor[active].astype(float)*10000
            matrix = lap+sparse.diags(constraint+1e-6)
            solved = spsolve(matrix,constraint[:,None]*observed[active])
            residual = float(np.linalg.norm(matrix@solved-constraint[:,None]*observed[active])/max(1,np.linalg.norm(constraint[:,None]*observed[active])))
            normals = self.normals[active]
            solved -= normals*(solved*normals).sum(1)[:,None]
            full = np.zeros_like(self.vertices)
            full[active] = solved
            self.fields[label] = full
            self.report[str(label)] = {'active_vertices':int(active.sum()),'anchors':int(anchor.sum()),
                'relative_residual_before_tangent_projection':residual}

    def query(self,point,partition):
        if partition not in self.fields:
            return np.zeros(3),'no_surface_region'
        distance,ids = self.tree.query(point,k=min(4,self.tree.n))
        weights = 1/np.maximum(distance,.001)**2
        vector = (self.fields[partition][ids]*weights[:,None]).sum(0)/weights.sum()
        closest = self.chart.scene.compute_closest_points(o3d.core.Tensor(point[None].astype(np.float32)))
        normal = closest['primitive_normals'].numpy()[0]
        separation = float((point-closest['points'].numpy()[0])@normal)
        vector -= normal*(vector@normal)
        magnitude = np.linalg.norm(vector)
        if magnitude<.05:
            return np.zeros(3),'unresolved_surface_direction'
        vector *= self.sign/magnitude
        vector += normal*np.clip((self.chart.clearance-separation)/.002,-.25,.5)
        return vector,None

    def begin_trajectory(self,root,partition,guard):
        """根尖符号通过两个短分支的完整守卫检验选择；不移动根点。"""
        lengths=[]
        for sign in (1.,-1.):
            self.sign=sign
            point=np.asarray(root,float).copy()
            steps=0
            for _ in range(40):
                vector,reason=self.query(point,partition)
                if reason:
                    break
                end=point+.0005*vector/np.linalg.norm(vector)
                if guard(point,end,partition):
                    break
                point=end
                steps+=1
            lengths.append(steps)
        self.sign=1. if lengths[0]>=lengths[1] else -1.
        self.branch_counts['forward' if self.sign>0 else 'reverse']+=1
