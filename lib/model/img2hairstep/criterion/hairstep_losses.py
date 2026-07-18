"""
Comprehensive loss functions for HairStep direction field training.

Includes:
  1. Cosine Similarity Loss (基础回归损失)
  2. Total Variation Loss (空间一致性 / 平滑项)
  3. Gradient/Laplacian Structural Similarity Loss (结构相似性损失)
  4. Multi-Scale Auxiliary Supervision Loss (多尺度辅助监督)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 1. Cosine Similarity Loss
# ---------------------------------------------------------------------------
def cosine_similarity_loss(pred, target, mask=None, eps=1e-8):
    """
    L_cos = 1 - (pred · target) / (||pred|| * ||target||)

    pred, target: (B, 2, H, W) – each channel pair encodes (cosθ, sinθ)
    mask:         (B, 1, H, W) – optional hair-region mask
    """
    # Normalize both vectors to unit length
    pred_norm = F.normalize(pred, p=2, dim=1, eps=eps)
    target_norm = F.normalize(target, p=2, dim=1, eps=eps)

    # Cosine similarity per pixel (B, 1, H, W)
    cos_sim = (pred_norm * target_norm).sum(dim=1, keepdim=True)

    # Loss: 1 - cos_sim  → range [0, 2], 0 = perfect alignment
    loss = 1.0 - cos_sim  # (B, 1, H, W)

    if mask is not None:
        loss = loss * mask
        return loss.sum() / mask.sum().clamp_min(1.0)
    return loss.mean()


# ---------------------------------------------------------------------------
# 2. TV Loss (Total Variation) – spatial smoothness
# ---------------------------------------------------------------------------
def tv_loss(pred, mask=None, reduction='mean'):
    """
    L_tv = sum_{i,j} ( ||∇_x v||^2 + ||∇_y v||^2 )

    pred: (B, 2, H, W)
    """
    # Gradient in x-direction (horizontal differences)
    dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]        # (B, 2, H, W-1)
    # Gradient in y-direction (vertical differences)
    dy = pred[:, :, 1:, :] - pred[:, :, :-1, :]        # (B, 2, H-1, W)

    tv_x = (dx ** 2).sum(dim=1)  # (B, H, W-1)
    tv_y = (dy ** 2).sum(dim=1)  # (B, H-1, W)

    if mask is not None:
        # Align mask with gradient fields
        mask_x = mask[:, 0, :, 1:]  # (B, H, W-1)
        mask_y = mask[:, 0, 1:, :]  # (B, H-1, W)
        tv_x = tv_x * mask_x
        tv_y = tv_y * mask_y
        loss = tv_x.sum() + tv_y.sum()
        # TV is divided by (number of valid gradient positions * channels)
        count = mask_x.sum() + mask_y.sum()
        return loss / count.clamp_min(1.0)

    if reduction == 'mean':
        return tv_x.mean() + tv_y.mean()
    return tv_x.sum() + tv_y.sum()


# ---------------------------------------------------------------------------
# 3. Gradient / Laplacian Structural Similarity Loss
# ---------------------------------------------------------------------------
def _normalize_vectors(vec, eps=1e-8):
    """vec: (B, 2, H, W) -> normalized same shape"""
    return F.normalize(vec, p=2, dim=1, eps=eps)


def gradient_consistency_loss(pred, target, mask=None, eps=1e-8):
    """
    L_grad: Compute spatial gradients of the predicted and target vector fields,
    then measure L1/L2 difference between them. This enforces structural
    alignment of hair flow geometry.

    Uses first-order spatial derivatives (sobel-like) on each channel.
    """
    # Sobel kernels for x and y gradients
    kx = torch.tensor([[-1., 0., 1.],
                        [-2., 0., 2.],
                        [-1., 0., 1.]], device=pred.device).view(1, 1, 3, 3)
    ky = torch.tensor([[-1., -2., -1.],
                        [ 0.,  0.,  0.],
                        [ 1.,  2.,  1.]], device=pred.device).view(1, 1, 3, 3)

    B, C, H, W = pred.shape
    loss = 0.0

    for c in range(C):
        pred_c = pred[:, c:c+1, :, :]
        target_c = target[:, c:c+1, :, :]

        # Gradients for predicted field
        gx_pred = F.conv2d(pred_c, kx, padding=1)
        gy_pred = F.conv2d(pred_c, ky, padding=1)
        # Gradients for target field
        gx_target = F.conv2d(target_c, kx, padding=1)
        gy_target = F.conv2d(target_c, ky, padding=1)

        # L2 difference of gradients
        ch_loss = (gx_pred - gx_target) ** 2 + (gy_pred - gy_target) ** 2  # (B, 1, H, W)

        if mask is not None:
            ch_loss = ch_loss * mask

        loss += ch_loss.mean()

    return loss / C


def laplacian_consistency_loss(pred, target, mask=None):
    """
    L_lap: Compute Laplacian of both predicted and target vector fields,
    then measure their L1/L2 consistency. This preserves the second-order
    geometric structure of hair flow.
    """
    # Laplacian kernel (discrete approximation)
    lap_kernel = torch.tensor([[0., 1., 0.],
                                [1., -4., 1.],
                                [0., 1., 0.]], device=pred.device).view(1, 1, 3, 3)

    B, C, H, W = pred.shape
    loss = 0.0

    for c in range(C):
        pred_c = pred[:, c:c+1, :, :]
        target_c = target[:, c:c+1, :, :]

        lap_pred = F.conv2d(pred_c, lap_kernel, padding=1)
        lap_target = F.conv2d(target_c, lap_kernel, padding=1)

        ch_loss = (lap_pred - lap_target) ** 2  # (B, 1, H, W)

        if mask is not None:
            ch_loss = ch_loss * mask

        loss += ch_loss.mean()

    return loss / C


def structural_similarity_loss(pred, target, mask=None):
    """
    Combined gradient + laplacian structural similarity loss.
    """
    l_grad = gradient_consistency_loss(pred, target, mask)
    l_lap = laplacian_consistency_loss(pred, target, mask)
    return l_grad + l_lap


# ---------------------------------------------------------------------------
# 4. Multi-Scale Auxiliary Supervision Loss
# ---------------------------------------------------------------------------
def multi_scale_aux_loss(aux_preds, target, mask=None, base_loss_fn=None):
    """
    Apply the same loss function to each auxiliary prediction from
    different HRNet branches, weighted by resolution fidelity.

    aux_preds: list of (B, 2, H, W) tensors from intermediate branches
    target:    (B, 2, H, W) ground truth
    mask:      (B, 1, H, W)

    Returns: scalar loss averaged over all auxiliary heads
    """
    if aux_preds is None or len(aux_preds) == 0:
        return torch.tensor(0.0, device=target.device)

    if base_loss_fn is None:
        base_loss_fn = cosine_similarity_loss

    total_loss = 0.0
    n = len(aux_preds)
    for i, aux_pred in enumerate(aux_preds):
        # Higher-resolution features get linearly higher weight
        weight = 0.5 + 0.5 * (i / max(1, n - 1))  # [0.5, 1.0]
        total_loss += weight * base_loss_fn(aux_pred, target, mask)

    return total_loss / n


# ---------------------------------------------------------------------------
# Combined Loss Class – convenient wrapper for training
# ---------------------------------------------------------------------------
class HairStepLoss(nn.Module):
    """
    Comprehensive loss combining:
      - L1 (masked MAE) loss       (w_l1)   — HairStep paper Eq. (2)
      - Cosine similarity loss     (w_cos)
      - TV smoothness loss         (w_tv)
      - Structural similarity      (w_struct)
      - Multi-scale auxiliary      (w_aux)

    The L1 loss follows the HairStep paper formulation:
        L_strand = (1 / (C * sum(M))) * ||O_hat - O||_1
    where C = 2 (strand-map channels) and M is the hair-region mask.

    Usage:
        criterion = HairStepLoss(w_l1=1.0, w_cos=0.0, w_tv=0.0, w_struct=0.0, w_aux=0.0)
        total_loss, loss_dict = criterion(pred_main, aux_preds, target, mask)
    """

    def __init__(self, w_l1=0.0, w_cos=1.0, w_tv=0.1, w_struct=0.05, w_aux=0.3):
        super().__init__()
        self.w_l1 = w_l1
        self.w_cos = w_cos
        self.w_tv = w_tv
        self.w_struct = w_struct
        self.w_aux = w_aux
        self.l1_loss = nn.L1Loss(reduction='none')

    def forward(self, pred_main, aux_preds, target, mask):
        """
        pred_main:  (B, 2, H, W) – main network prediction
        aux_preds:  list of (B, 2, H, W) or None – auxiliary branch predictions
        target:     (B, 2, H, W) – ground truth direction field
        mask:       (B, 1, H, W) – hair-region mask

        Set any weight to 0.0 to skip that loss entirely (faster training).
        """
        C = pred_main.shape[1]  # number of channels (2 for strand map)
        losses = {}
        total = torch.tensor(0.0, device=pred_main.device)

        # 0) L1 loss — HairStep paper Eq. (2): masked MAE
        if self.w_l1 != 0.0:
            l1_per_pixel = self.l1_loss(pred_main * mask, target * mask)
            # L_strand = sum(|pred-gt| * mask) / (C * sum(mask))
            l_l1 = l1_per_pixel.sum() / (mask.sum() * C).clamp_min(1.0)
            losses['l1'] = l_l1
            total = total + self.w_l1 * l_l1
        else:
            losses['l1'] = torch.tensor(0.0, device=pred_main.device)

        # 1) Cosine similarity loss on main output
        if self.w_cos != 0.0:
            l_cos = cosine_similarity_loss(pred_main, target, mask)
            losses['cos'] = l_cos
            total = total + self.w_cos * l_cos
        else:
            losses['cos'] = torch.tensor(0.0, device=pred_main.device)

        # 2) TV loss on main output (spatial smoothness)
        if self.w_tv != 0.0:
            l_tv = tv_loss(pred_main, mask)
            losses['tv'] = l_tv
            total = total + self.w_tv * l_tv
        else:
            losses['tv'] = torch.tensor(0.0, device=pred_main.device)

        # 3) Structural similarity loss (gradient + laplacian)
        if self.w_struct != 0.0:
            l_struct = structural_similarity_loss(pred_main, target, mask)
            losses['struct'] = l_struct
            total = total + self.w_struct * l_struct
        else:
            losses['struct'] = torch.tensor(0.0, device=pred_main.device)

        # 4) Multi-scale auxiliary supervision
        if self.w_aux != 0.0 and aux_preds is not None:
            l_aux = multi_scale_aux_loss(aux_preds, target, mask)
            losses['aux'] = l_aux
            total = total + self.w_aux * l_aux
        else:
            losses['aux'] = torch.tensor(0.0, device=pred_main.device)

        losses['total'] = total
        return total, losses


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Testing HairStepLoss on {device}')

    B, C, H, W = 2, 2, 64, 64
    pred = torch.randn(B, C, H, W, device=device, requires_grad=True)
    target = torch.randn(B, C, H, W, device=device)
    mask = torch.ones(B, 1, H, W, device=device)
    aux = [torch.randn(B, C, H, W, device=device, requires_grad=True) for _ in range(3)]

    crit = HairStepLoss()
    total, losses = crit(pred, aux, target, mask)
    total.backward()

    print('Losses:')
    for k, v in losses.items():
        print(f'  {k}: {v.item():.6f}')
    print(f'Grad exists: pred={pred.grad is not None}')