"""
从 Pixal3D output.glb 中提取头发 Mesh。

算法:
  1. 加载 GLB 几何数据（跳过材质）
  2. 坐标对齐: GLB 坐标 -> HairStep 世界坐标
  3. 用 calib 正交投影每个顶点到 2D 像素坐标 (u, v)
  4. 用头发 seg mask 判定每个顶点是否属于头发区域
  5. 保留三个顶点全部落在头发 mask 内的三角面
  6. 输出 hair_mesh.obj

用法:
  pixi run python scripts/extract_hair_mesh.py \
      --glb    results/test_pixal3d/output.glb \
      --param  results/real_imgs/param/0d285f5be7fa09c3dbbf1c9334047888.npy \
      --seg    results/real_imgs/seg/0d285f5be7fa09c3dbbf1c9334047888.png \
      --out    results/test_pixal3d/hair_mesh_new.obj
"""

import sys, os, struct, json
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import numpy as np
import trimesh
import cv2


def load_glb_geometry(path):
    """
    从 GLB 文件中只提取几何数据（顶点 + 面），跳过材质/纹理。
    trimesh 对含 WebP 纹理的 GLB 会报 KeyError，这里手动解析 glTF binary。
    """
    # 先尝试 trimesh
    try:
        scene = trimesh.load(path, force='scene')
        mesh = trimesh.util.concatenate(list(scene.geometry.values()))
        return mesh
    except (KeyError, Exception):
        pass

    # 手动解析 GLB
    with open(path, 'rb') as f:
        magic, version, length = struct.unpack('<III', f.read(12))
        assert magic == 0x46546C67, "Not a valid GLB file"

        # Chunk 0: JSON
        chunk_len, chunk_type = struct.unpack('<II', f.read(8))
        json_data = json.loads(f.read(chunk_len).decode('utf-8'))

        # Chunk 1: BIN
        chunk_len, chunk_type = struct.unpack('<II', f.read(8))
        bin_data = f.read(chunk_len)

    accessors = json_data['accessors']
    buffer_views = json_data['bufferViews']

    def read_accessor(idx):
        acc = accessors[idx]
        bv = buffer_views[acc['bufferView']]
        offset = bv.get('byteOffset', 0) + acc.get('byteOffset', 0)
        count = acc['count']
        comp_type = acc['componentType']
        acc_type = acc['type']

        dtype_map = {5120: np.int8, 5121: np.uint8, 5122: np.int16,
                     5123: np.uint16, 5125: np.uint32, 5126: np.float32}
        dtype = dtype_map[comp_type]
        n_comp = {'SCALAR': 1, 'VEC2': 2, 'VEC3': 3, 'VEC4': 4}[acc_type]

        data = np.frombuffer(bin_data, dtype=dtype, count=count * n_comp, offset=offset)
        if n_comp > 1:
            data = data.reshape(count, n_comp)
        return data

    all_verts = []
    all_faces = []
    vert_offset = 0

    for mesh_entry in json_data['meshes']:
        for prim in mesh_entry['primitives']:
            pos_idx = prim['attributes']['POSITION']
            verts = read_accessor(pos_idx).astype(np.float32)

            if 'indices' in prim:
                faces = read_accessor(prim['indices']).reshape(-1, 3).astype(np.int64)
            else:
                n = len(verts)
                faces = np.arange(n).reshape(-1, 3)

            all_verts.append(verts)
            all_faces.append(faces + vert_offset)
            vert_offset += len(verts)

    verts = np.concatenate(all_verts, axis=0)
    faces = np.concatenate(all_faces, axis=0)
    return trimesh.Trimesh(vertices=verts, faces=faces, process=False)


def load_calib(calib_path, loadSize=1024):
    param = np.load(calib_path, allow_pickle=True).item()
    ortho_ratio = param['ortho_ratio']
    scale       = param['scale']
    center      = param['center']
    R           = param['R']

    translate = -np.matmul(R, center).reshape(3, 1)
    extrinsic = np.concatenate([R, translate], axis=1)
    extrinsic = np.concatenate([extrinsic, np.array([[0, 0, 0, 1]])], axis=0)

    scale_intrinsic = np.eye(4)
    scale_intrinsic[0, 0] =  scale / ortho_ratio
    scale_intrinsic[1, 1] = -scale / ortho_ratio
    scale_intrinsic[2, 2] =  scale / ortho_ratio

    uv_intrinsic = np.eye(4)
    uv_intrinsic[0, 0] = 1.0 / float(loadSize // 2)
    uv_intrinsic[1, 1] = 1.0 / float(loadSize // 2)
    uv_intrinsic[2, 2] = 1.0 / float(loadSize // 2)

    intrinsic = uv_intrinsic @ scale_intrinsic
    calib = intrinsic @ extrinsic
    return calib.astype(np.float32)


def align_glb_to_hairstep(verts):
    """GLB 坐标 -> HairStep 坐标: 归一化到 [-0.5, 0.5] 然后平移到 Y=1.7"""
    v_min = verts.min(axis=0)
    v_max = verts.max(axis=0)
    center = (v_min + v_max) / 2.0
    scale  = (v_max - v_min).max()
    verts_norm = (verts - center) / scale
    hairstep_center = np.array([0.0, 1.7, 0.0])
    return (verts_norm + hairstep_center).astype(np.float32)


def project_vertices(verts, calib, img_size):
    """正交投影: 3D 顶点 -> 2D 像素坐标 (u, v)"""
    N = verts.shape[0]
    ones = np.ones((N, 1), dtype=np.float32)
    verts_h = np.concatenate([verts, ones], axis=1)  # (N, 4)
    proj = (calib @ verts_h.T).T                      # (N, 4)
    u = proj[:, 0] * (img_size / 2) + (img_size / 2)
    v = proj[:, 1] * (img_size / 2) + (img_size / 2)
    return np.stack([u, v], axis=1)                   # (N, 2)


def extract_hair_faces(mesh, uv, proj, hair_mask, depth_map, img_size):
    """
    判定算法:
      1. 正交投影后，一个顶点必须落在 2D hair_mask 的白色区域内。
      2. 深度过滤：因为是正交投影，后脑勺的顶点也会投影到正面的头发区域。
         我们需要比较顶点的相机空间 Z 坐标 (proj[:, 2]) 和 depth_map 中的可见表面深度。
         如果顶点的深度比表面的深度大太多，说明它在后脑勺，予以剔除。
      3. 三角面的三个顶点必须全部满足上述条件。
    """
    u = np.clip(uv[:, 0].astype(int), 0, img_size - 1)
    v = np.clip(uv[:, 1].astype(int), 0, img_size - 1)
    
    in_mask = hair_mask[v, u] > 128  # (N,) bool
    
    cam_z = proj[:, 2]
    surf_depth = depth_map[v, u]
    
    # 深度阈值，允许一定的厚度
    margin = 0.3
    in_hair = in_mask & (cam_z < surf_depth + margin)
    
    print(f"  [Depth Filter] 剔除了 {in_mask.sum() - in_hair.sum()} 个后脑勺顶点")

    faces = mesh.faces
    face_mask = in_hair[faces[:, 0]] & in_hair[faces[:, 1]] & in_hair[faces[:, 2]]

    n_total = len(faces)
    n_hair  = face_mask.sum()
    print(f"  总面数: {n_total}，头发面数: {n_hair} ({100*n_hair/max(n_total,1):.1f}%)")

    if n_hair == 0:
        print("  没有找到头发面。")
        return None, None

    hair_faces = faces[face_mask]
    used = np.unique(hair_faces)
    old2new = np.full(len(mesh.vertices), -1, dtype=np.int64)
    old2new[used] = np.arange(len(used))
    new_faces = old2new[hair_faces]
    new_verts = mesh.vertices[used]
    
    return new_verts, new_faces


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--glb",   required=True, help="Pixal3D output.glb")
    parser.add_argument("--param", required=True, help="param.npy (calib)")
    parser.add_argument("--seg",   required=True, help="头发 seg mask (白=头发)")
    parser.add_argument("--out",   required=True, help="输出 hair_mesh.obj")
    parser.add_argument("--img_size", type=int, default=512)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    # 1. 加载 GLB
    print(f"[1/4] 加载 GLB: {args.glb}")
    raw_mesh = load_glb_geometry(args.glb)
    print(f"  顶点: {len(raw_mesh.vertices)}  面: {len(raw_mesh.faces)}")
    print(f"  bounds: {raw_mesh.bounds}")

    # 2. 坐标对齐
    print("[2/4] 坐标对齐 (GLB -> HairStep)")
    verts = align_glb_to_hairstep(raw_mesh.vertices.astype(np.float32))
    mesh = trimesh.Trimesh(vertices=verts, faces=raw_mesh.faces, process=False)
    print(f"  对齐后 bounds: {mesh.bounds}")

    # 3. 投影
    print("[3/4] 正交投影到 2D")
    calib = load_calib(args.param, loadSize=1024)
    N = verts.shape[0]
    ones = np.ones((N, 1), dtype=np.float32)
    verts_h = np.concatenate([verts, ones], axis=1)  # (N, 4)
    proj = (calib @ verts_h.T).T                      # (N, 4)
    
    uv = project_vertices(verts, calib, img_size=args.img_size)
    print(f"  UV 范围: u=[{uv[:,0].min():.1f}, {uv[:,0].max():.1f}]"
          f"  v=[{uv[:,1].min():.1f}, {uv[:,1].max():.1f}]")

    # 4. 用 seg mask 提取头发面
    print("[4/4] 提取头发面")
    hair_mask = cv2.imread(args.seg, cv2.IMREAD_GRAYSCALE)
    hair_mask = cv2.resize(hair_mask, (args.img_size, args.img_size))
    print(f"  mask 头发像素: {(hair_mask > 128).sum()}")
    
    depth_path = args.seg.replace('seg', 'depth_map').replace('.png', '.npy')
    depth_map = np.load(depth_path)

    new_verts, new_faces = extract_hair_faces(mesh, uv, proj, hair_mask, depth_map, img_size=args.img_size)
    if new_verts is None:
        sys.exit(1)

    with open(args.out, 'w') as f:
        for v3 in new_verts:
            f.write(f'v {v3[0]:.6f} {v3[1]:.6f} {v3[2]:.6f}\n')
        for face in new_faces:
            f.write(f'f {face[0]+1} {face[1]+1} {face[2]+1}\n')

    print(f"\n输出: {args.out}")
    print(f"  顶点: {len(new_verts)}  面: {len(new_faces)}")

if __name__ == '__main__':
    main()
