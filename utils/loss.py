"""
Flexible Pyramid Chamfer Distance Loss.

Supports variable number of predicted point clouds (2, 3, or 4+ stages).
Replaces the separate get_loss_clamp and get_loss_mvp functions.
"""

import torch
from Chamfer3D.dist_chamfer_3D import chamfer_3DDist
from models.utils import fps_subsample

chamfer_dist = chamfer_3DDist()


def chamfer_l2(p1, p2):
    """Chamfer Distance L2 (squared)."""
    d1, d2, _, _ = chamfer_dist(p1, p2)
    return torch.mean(d1) + torch.mean(d2)


def chamfer_l1(p1, p2):
    """Chamfer Distance L1 (sqrt, clamped for stability)."""
    d1, d2, _, _ = chamfer_dist(p1, p2)
    d1 = torch.clamp(d1, min=1e-9)
    d2 = torch.clamp(d2, min=1e-9)
    d1 = torch.mean(torch.sqrt(d1))
    d2 = torch.mean(torch.sqrt(d2))
    return (d1 + d2) / 2


def chamfer_single_side(pcd1, pcd2, sqrt=False):
    """One-directional Chamfer Distance (pcd1 -> pcd2)."""
    d1, d2, _, _ = chamfer_dist(pcd1, pcd2)
    if sqrt:
        d1 = torch.clamp(d1, min=1e-9)
        d1 = torch.mean(torch.sqrt(d1))
    else:
        d1 = torch.mean(d1)
    return d1


def pyramid_loss(pcds_pred, partial, gt, sqrt=True):
    """
    Flexible pyramid Chamfer Distance loss for any number of prediction stages.

    Automatically creates ground truth point clouds at matching resolutions
    using FPS downsampling.

    Args:
        pcds_pred: tuple/list of predicted point clouds, ordered coarse to fine.
                   Each element is (B, N_i, 3).
        partial:   Partial input point cloud (B, N_in, 3).
        gt:        Ground truth complete point cloud (B, N_gt, 3).
        sqrt:      If True, use L1 (sqrt) Chamfer. If False, use L2.

    Returns:
        loss_total: Scalar total loss (x1e3 scale).
        losses:     List of per-stage CD losses [cd_0, cd_1, ..., cd_K, partial_matching].
        gts:        List of ground truth point clouds at each stage resolution.
    """
    CD = chamfer_l1 if sqrt else chamfer_l2
    num_stages = len(pcds_pred)

    # Build ground truth hierarchy by downsampling from fine to coarse
    gts = [None] * num_stages
    gts[-1] = gt  # finest stage matches original gt

    for i in range(num_stages - 2, -1, -1):
        target_n = pcds_pred[i].shape[1]
        source = gts[i + 1]
        if source.shape[1] > target_n:
            gts[i] = fps_subsample(source, target_n)
        else:
            gts[i] = source

    # Compute per-stage losses
    stage_losses = []
    loss_total = 0
    for i in range(num_stages):
        cd_i = CD(pcds_pred[i], gts[i])
        stage_losses.append(cd_i)
        loss_total += cd_i

    # Partial matching loss (coarse prediction)
    partial_matching = stage_losses[0]

    # Assemble losses list: [cd_0, cd_1, ..., cd_fine, partial_matching]
    # For backward compatibility, pad to at least 4 stage losses + partial_matching
    losses = list(stage_losses)
    while len(losses) < 4:
        losses.append(losses[-1])  # duplicate last
    losses.append(partial_matching)

    loss_total = loss_total * 1e3

    return loss_total, losses, gts
