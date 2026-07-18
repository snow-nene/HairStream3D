import os
import numpy as np
from PIL import Image
import cv2
from scipy import ndimage
from compare_180deg import custom_180deg_colorwheel, extract_theta_from_strand_map

def circular_smooth(angle: np.ndarray, weight: np.ndarray, sigma: float) -> np.ndarray:
    """Step 5: 双倍角映射法 (Double-angle Filtering) 平滑"""
    a2 = 2.0 * angle
    c = ndimage.gaussian_filter(weight * np.cos(a2), sigma, mode="reflect")
    s = ndimage.gaussian_filter(weight * np.sin(a2), sigma, mode="reflect")
    wsum = ndimage.gaussian_filter(weight, sigma, mode="reflect") + 1e-6
    c /= wsum
    s /= wsum
    return np.mod(0.5 * np.arctan2(s, c), np.pi).astype(np.float32)

def standard_structure_tensor_pipeline(rgb_img, mask, sigma_integration=5.0):
    # ==========================================
    # Step 1: 预处理 (Preprocessing)
    # ==========================================
    # 1. 灰度转换 & 归一化到 [0, 1]
    gray = cv2.cvtColor((rgb_img * 255.0).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    
    # 2. 高斯平滑 (减少传感器噪声)
    gray_smoothed = cv2.GaussianBlur(gray, (3, 3), sigmaX=1.0)
    
    # ==========================================
    # Step 2: 梯度场计算 (Gradient Field Computation)
    # ==========================================
    # 使用推荐的 Scharr 算子
    Ix = cv2.Scharr(gray_smoothed, cv2.CV_32F, 1, 0)
    Iy = cv2.Scharr(gray_smoothed, cv2.CV_32F, 0, 1)
    
    # ==========================================
    # Step 3: 结构张量矩阵构建 (Tensor Construction)
    # ==========================================
    Ixx = Ix * Ix
    Iyy = Iy * Iy
    Ixy = Ix * Iy
    
    # 高斯加权窗积分 (Gaussian Weighted Window)
    ksize = int(round(6 * sigma_integration)) | 1
    Jxx = cv2.GaussianBlur(Ixx, (ksize, ksize), sigma_integration)
    Jyy = cv2.GaussianBlur(Iyy, (ksize, ksize), sigma_integration)
    Jxy = cv2.GaussianBlur(Ixy, (ksize, ksize), sigma_integration)
    
    # ==========================================
    # Step 4: 特征值与特征向量分解 (Eigendecomposition)
    # ==========================================
    # 解析求解特征值和特征向量
    trace = Jxx + Jyy
    det = Jxx * Jyy - Jxy * Jxy
    sqrt_term = np.sqrt(np.maximum(0, trace * trace - 4.0 * det))
    
    lambda1 = 0.5 * (trace + sqrt_term)  # 较大特征值
    lambda2 = 0.5 * (trace - sqrt_term)  # 较小特征值
    
    # 特征向量 v2 的角度 (对应较小特征值 lambda2 的方向，即头发切线)
    theta = 0.5 * np.arctan2(2.0 * Jxy, Jxx - Jyy) + np.pi / 2.0
    theta = theta % np.pi
    
    # 计算相干性指标 (Coherence)
    C = np.zeros_like(trace)
    valid = (lambda1 + lambda2) > 1e-6
    C[valid] = (lambda1[valid] - lambda2[valid]) / (lambda1[valid] + lambda2[valid])
    
    # ==========================================
    # Step 5: 后处理与平滑 (Post-processing)
    # ==========================================
    # 使用双倍角映射法 (Double-angle) 平滑，并以 Coherence (C) 作为权重掩码
    theta_smoothed = theta.copy()
    for _ in range(3): # 迭代 3 次平滑
        theta_smoothed = circular_smooth(theta_smoothed, C, sigma=2.0)
        # 刷新相干性掩码权重
        C = ndimage.gaussian_filter(C, sigma=2.0, mode="reflect")
        C = np.clip(C, 0.0, 1.0)
        
    return theta_smoothed, C

def main():
    raw_img_path = "./HiSa_HiDa/img/002c5a69cd4546957b02da0f762ed929.png"
    gt_path = "./HiSa_HiDa/strand_map/002c5a69cd4546957b02da0f762ed929.png"
    mask_path = "./HiSa_HiDa/seg/002c5a69cd4546957b02da0f762ed929.png"
    size = (512, 512)
    
    print("Loading data...")
    raw_img = Image.open(raw_img_path).convert("RGB").resize(size, Image.Resampling.LANCZOS)
    rgb = np.asarray(raw_img, dtype=np.float32) / 255.0
    mask = np.array(Image.open(mask_path).convert("L").resize(size, Image.Resampling.NEAREST)) > 127
    theta_gt = extract_theta_from_strand_map(gt_path)
    
    print("Running Standard Structure Tensor Pipeline...")
    theta_pipeline, coherence = standard_structure_tensor_pipeline(rgb, mask, sigma_integration=4.0)
    
    print("Generating color maps...")
    rgb_fused = custom_180deg_colorwheel(theta_pipeline)
    rgb_gt = custom_180deg_colorwheel(theta_gt)
    
    rgb_fused[~mask] = [0, 0, 0]
    rgb_gt[~mask] = [0, 0, 0]
    
    img_fused = Image.fromarray(rgb_fused.astype(np.uint8))
    img_gt = Image.fromarray(rgb_gt.astype(np.uint8))
    
    w, h = img_fused.size
    tiled = Image.new("RGB", (w * 2 + 10, h), (40, 40, 40))
    tiled.paste(img_gt, (0, 0))
    tiled.paste(img_fused, (w + 10, 0))
    
    out_dir = "./outputs_gabor"
    os.makedirs(out_dir, exist_ok=True)
    tiled_path = f"{out_dir}/compare_standard_pipeline_180deg.png"
    tiled.save(tiled_path)
    print(f"Saved standard pipeline comparison to {tiled_path}")
    
    # Save Coherence (Confidence) Map
    coherence_map = (coherence * 255.0).astype(np.uint8)
    coherence_map[~mask] = 0 # mask out background
    coherence_path = f"{out_dir}/standard_coherence.png"
    Image.fromarray(coherence_map).save(coherence_path)
    print(f"Saved coherence (confidence) map to {coherence_path}")
    
    # Metrics
    diff = np.abs(theta_pipeline - theta_gt)
    err = np.minimum(diff, np.pi - diff)
    err_deg = np.degrees(err)[mask]
    
    print(f"--- Standard Textbook Pipeline vs GT ---")
    print(f"Mean Error: {np.mean(err_deg):.2f} deg")

if __name__ == "__main__":
    main()
