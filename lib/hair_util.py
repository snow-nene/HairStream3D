import torch
import numpy as np
import open3d as o3d
import os
import trimesh
from .mesh_util import load_obj_mesh

def hair_synthesis(net, cuda, root_tensor, calib_tensor, num_sample=100, hair_unit=0.006):
    #root:[3, 1024]
    num_strand = root_tensor.shape[2]
    hair_strands = torch.zeros(num_sample, 3, num_strand).to(device=cuda)
    
    curr_node = root_tensor.squeeze()
    hair_strands[0] = curr_node
    for i in range(1,num_sample-1):
        curr_node_orien = net.query(curr_node.unsqueeze(0), calib_tensor)
        curr_node_orien = curr_node_orien.squeeze()
        hair_strands[i] = hair_strands[i-1] + hair_unit * curr_node_orien
        curr_node = hair_strands[i]

    return hair_strands.permute(2, 0, 1).cpu().detach().numpy()

def hair_synthesis_DSH(net, cuda, root_tensor, calib_tensor, num_sample=100, hair_unit=0.006, threshold=[60,150]):
    #growing algorithm in DeepSketchHair
    #root:[3, 1024]
    num_strand = root_tensor.shape[2]
    hair_strands = torch.zeros(num_sample, 3, num_strand).to(device=cuda)
    
    curr_node = root_tensor.squeeze()
    last_node_orien = 0
    hair_strands[0] = curr_node
    for i in range(1,num_sample-1):
        curr_node_orien = net.query(curr_node.unsqueeze(0), calib_tensor).squeeze()

        if i>1:
            len_cd = torch.norm(curr_node_orien,p=2,dim=0)
            len_pd = torch.norm(last_node_orien,p=2,dim=0)
            in_prod = torch.sum(curr_node_orien * last_node_orien, dim=0)
            theta = torch.acos( in_prod/ (len_cd*len_pd))*180/np.pi

            idx_big_theta = theta > threshold[1]
            idx_mid_theta = ((theta > threshold[0]).float() - idx_big_theta.float()).bool().unsqueeze(0)#60<theta<150
            idx_stop = (idx_big_theta + torch.isnan(theta).float()).bool().unsqueeze(0)#orien=0 or theta>150

            idx_stop = torch.cat((idx_stop,idx_stop,idx_stop),dim=0)
            idx_mid_theta = torch.cat((idx_mid_theta,idx_mid_theta,idx_mid_theta),dim=0)
            

            half_node_orien = (curr_node_orien + last_node_orien)/2

            orien_zeros = torch.zeros_like(curr_node_orien).float().to(device=cuda)
            curr_node_orien = torch.where(idx_stop, orien_zeros, curr_node_orien)
            curr_node_orien = torch.where(idx_mid_theta, half_node_orien, curr_node_orien)

        hair_strands[i] = hair_strands[i-1] + hair_unit * curr_node_orien
        curr_node = hair_strands[i]
        last_node_orien = curr_node_orien

    return hair_strands.permute(2, 0, 1).cpu().detach().numpy()

def save_strands_with_mesh(strands, mesh_path, outputpath, err=0.3, is_eval=False, min_len=0.0):
    """
    保存生成的 3D 发丝 PLY 模型。
    min_len (float): 发丝最小物理长度限制（单位：米，默认 0.0m = 不修剪），低于此长度的发丝将被修剪/滤除。
    """
    lst_pc_all_valid = []
    lst_num_pt = []
    pc_all_valid = []
    lines = []
    sline = 0

    for i in range(strands.shape[0]):
        current_pc_all_valid = []
        if np.dot(strands[i, 0], strands[i, 0]) < 0.001 or np.dot(strands[i, 1], strands[i, 1]) < 0.001:
            continue
            
        # 计算整根发丝的总物理长度 (Euclidean length)
        pts = strands[i]  # [num_sample, 3]
        segment_lengths = np.linalg.norm(pts[1:] - pts[:-1], axis=1)
        total_length = np.sum(segment_lengths)
        
        # 过滤/修剪过短的发丝（如物理长度 < min_len）
        if min_len > 0.0 and total_length < min_len:
            continue

        num_pt = 0
        for j in range(strands.shape[1]):
            current_pc_all_valid.append(strands[i][j])
            num_pt += 1
            
        lst_pc_all_valid.append(current_pc_all_valid)
        lst_num_pt.append(num_pt)

    for i in range(len(lst_pc_all_valid)):
        pc_all_valid.extend(lst_pc_all_valid[i])
        for j in range(lst_num_pt[i] - 1):
            lines.append([sline + j, sline + j + 1])
        sline += lst_num_pt[i]

    import open3d as o3d
    line_set = o3d.geometry.LineSet(points=o3d.utility.Vector3dVector(np.asarray(pc_all_valid)), lines=o3d.utility.Vector2iVector(lines))
    o3d.io.write_line_set(outputpath, line_set)
    print(f"[Strand Pruning] Filtered out short strands (< {min_len*100:.1f} cm). Kept {len(lst_pc_all_valid)} / {strands.shape[0]} strands.")


def save_polyline_strands(curves, outputpath):
    """Save variable-length 3D polyline curves as an Open3D LineSet."""
    points = []
    lines = []
    for curve in curves:
        curve = np.asarray(curve, dtype=np.float64)
        if len(curve) < 2:
            continue
        offset = len(points)
        points.extend(curve)
        lines.extend(
            [offset + index, offset + index + 1]
            for index in range(len(curve) - 1)
        )
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64)),
        lines=o3d.utility.Vector2iVector(np.asarray(lines, dtype=np.int32)),
    )
    if not o3d.io.write_line_set(str(outputpath), line_set):
        raise RuntimeError(f"Failed to write polyline strands: {outputpath}")
    print(
        f"[Polyline Strands] Saved {len(curves)} curves "
        f"({len(points)} points) to {outputpath}"
    )

def get_hair_root(filepath='./data/roots10k.obj'):
    from lib.mesh_util import load_obj_mesh
    root, _ = load_obj_mesh(filepath)
    return root.T

def export_hair_real(net, cuda, data, mesh_path, save_path):
    image_tensor = data['hairstep'].to(device=cuda).unsqueeze(0)
    calib_tensor = data['calib'].to(device=cuda).unsqueeze(0)
    root_tensor = torch.from_numpy(get_hair_root()).to(device=cuda).float().unsqueeze(0)

    print('Evaluating filter pass for orien_net...')
    net.filter(image_tensor)

    print('Starting hair synthesis...')
    strands = hair_synthesis(net, cuda, root_tensor, calib_tensor, num_sample=100, hair_unit=0.006)
    print('Hair synthesis finished.')
    print('Starting save_strands_with_mesh...')
    save_strands_with_mesh(strands, mesh_path, save_path, 0.3)
    print('save_strands_with_mesh finished.')

def export_hair_strategy(strategy, cuda, data, mesh_path, save_path,
                         num_sample=100, hair_unit=0.006):
    """
    Strategy-pattern version of export_hair_real.
    Works with any BaseReconStrategy (NeuralPIFuStrategy or LaplacePDEStrategy).

    Assumes strategy.filter(data) has already been called.

    Args:
        strategy:   BaseReconStrategy instance
        cuda:       torch.device
        data:       dict with 'hairstep' [4,H,W] and 'calib' [4,4]
        mesh_path:  path to the coarse hair mesh .obj (used to clip strands)
        save_path:  output .ply path for 3D hair strands
        num_sample: number of integration steps per strand (default 100)
        hair_unit:  step size per integration step in world units (default 0.006)
    """
    calib_tensor = data['calib'].to(device=cuda).unsqueeze(0)   # [1, 4, 4]
    root_tensor  = torch.from_numpy(get_hair_root()).to(device=cuda).float().unsqueeze(0)

    # Switch to orientation query mode
    strategy.set_query_mode('orien')

    strands = hair_synthesis(strategy, cuda, root_tensor, calib_tensor,
                             num_sample=num_sample, hair_unit=hair_unit)
    print('Starting save_strands_with_mesh...')
    save_strands_with_mesh(strands, mesh_path, save_path, 0.3)
    print('save_strands_with_mesh finished.')
