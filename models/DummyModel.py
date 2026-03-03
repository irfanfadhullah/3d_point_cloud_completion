"""
Dummy Model for Pipeline Verification.

A minimal model registered in the MODELS registry for testing that the
entire train/test pipeline works (data loading -> forward -> loss -> backward).

Usage:
    python train.py --config configs/dummy.yaml --dummy
    python train.py --config configs/dummy.yaml --dummy --epochs 3
"""

import torch
import torch.nn as nn
from models.build import MODELS


@MODELS.register_module()
class DummyModel(nn.Module):
    """
    Minimal point cloud completion model for pipeline testing.

    Takes (B, N, 3) partial cloud and returns a tuple of predicted
    point clouds at different resolutions, matching the interface
    expected by the unified train/test scripts.

    Config options:
        config.num_pred:   Number of points in fine output (default: 16384)
        config.num_coarse: Number of points in coarse output (default: 1024)
        config.feat_dim:   Feature dimension (default: 256)
    """

    def __init__(self, config):
        super().__init__()
        self.num_coarse = getattr(config, 'num_coarse', 1024)
        self.num_pred = getattr(config, 'num_pred', 16384)
        feat_dim = getattr(config, 'feat_dim', 256)

        # Simple encoder: PointNet-style
        self.encoder = nn.Sequential(
            nn.Conv1d(3, 64, 1),
            nn.ReLU(),
            nn.Conv1d(64, 128, 1),
            nn.ReLU(),
            nn.Conv1d(128, feat_dim, 1),
        )

        # Coarse decoder
        self.coarse_decoder = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(),
            nn.Linear(feat_dim, self.num_coarse * 3),
        )

        # Fine decoder (from coarse)
        self.fine_decoder = nn.Sequential(
            nn.Conv1d(3 + feat_dim, 128, 1),
            nn.ReLU(),
            nn.Conv1d(128, 64, 1),
            nn.ReLU(),
            nn.Conv1d(64, 3, 1),
        )

        self.upsample_factor = self.num_pred // self.num_coarse

    def forward(self, xyz):
        """
        Args:
            xyz: (B, N, 3) partial point cloud

        Returns:
            tuple: (coarse, fine) predicted point clouds
                   coarse: (B, num_coarse, 3)
                   fine:   (B, num_pred, 3)
        """
        B = xyz.shape[0]

        # Encode: (B, N, 3) -> (B, 3, N) -> (B, feat, N) -> (B, feat)
        x = xyz.transpose(1, 2).contiguous()          # (B, 3, N)
        feat = self.encoder(x)                         # (B, feat, N)
        global_feat = torch.max(feat, dim=2)[0]        # (B, feat)

        # Coarse prediction
        coarse = self.coarse_decoder(global_feat)      # (B, num_coarse*3)
        coarse = coarse.view(B, self.num_coarse, 3)    # (B, num_coarse, 3)

        # Upsample coarse to fine resolution
        # Repeat each coarse point and decode with features
        coarse_up = coarse.unsqueeze(2).repeat(1, 1, self.upsample_factor, 1)
        coarse_up = coarse_up.view(B, self.num_pred, 3)  # (B, num_pred, 3)

        # Concat global feature with upsampled coarse
        feat_expand = global_feat.unsqueeze(2).repeat(1, 1, self.num_pred)  # (B, feat, num_pred)
        coarse_t = coarse_up.transpose(1, 2).contiguous()                    # (B, 3, num_pred)
        fine_input = torch.cat([coarse_t, feat_expand], dim=1)               # (B, 3+feat, num_pred)

        # Fine prediction = coarse + delta
        delta = self.fine_decoder(fine_input)           # (B, 3, num_pred)
        fine = coarse_up + delta.transpose(1, 2).contiguous()  # (B, num_pred, 3)

        return (coarse, fine)
