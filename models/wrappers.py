"""
Wrapper classes for models that don't use the MODELS registry.
These thin wrappers adapt CRAPCN, MSN, PFNet to accept a config object
and register them so the unified train/test scripts can instantiate them by name.
"""

import torch
import torch.nn as nn
from easydict import EasyDict as edict
from .build import MODELS


# =====================================================================
# CRA-PCN Wrappers
# =====================================================================

@MODELS.register_module()
class CRAPCN_Wrapper(nn.Module):
    """
    Registry wrapper for CRA-PCN.
    
    Config options:
        config.variant: 'pcn' | 'sn55' | 'mvp'  (default: 'pcn')
        config.use_deconv: bool  (default: False, use deconv-based seed generator)
    """
    def __init__(self, config):
        super().__init__()
        from .crapcn import CRAPCN, CRAPCN_d, CRAPCN_sn55, CRAPCN_sn55_d, CRAPCN_mvp, CRAPCN_mvp_d
        
        variant = getattr(config, 'variant', 'pcn')
        use_deconv = getattr(config, 'use_deconv', False)
        
        model_map = {
            ('pcn', False): CRAPCN,
            ('pcn', True): CRAPCN_d,
            ('sn55', False): CRAPCN_sn55,
            ('sn55', True): CRAPCN_sn55_d,
            ('mvp', False): CRAPCN_mvp,
            ('mvp', True): CRAPCN_mvp_d,
        }
        
        model_cls = model_map.get((variant, use_deconv))
        if model_cls is None:
            raise ValueError(f"Unknown CRAPCN variant: {variant} (deconv={use_deconv})")
        
        self.model = model_cls()
    
    def forward(self, xyz):
        return self.model(xyz)
    
    def get_loss(self, ret, gt, epoch=0):
        """CRAPCN uses the default pyramid CD loss from the trainer."""
        raise NotImplementedError("Use default pyramid CD loss")


# =====================================================================
# SeedFormer Wrapper (not registered in original code)
# =====================================================================

@MODELS.register_module()
class SeedFormer_Wrapper(nn.Module):
    """
    Registry wrapper for SeedFormer.
    
    Config options:
        config.feat_dim: int (default: 512)
        config.embed_dim: int (default: 128)
        config.num_p0: int (default: 512)
        config.n_knn: int (default: 20)
        config.radius: int (default: 1)
        config.up_factors: list (default: [1, 2, 2])
        config.seed_factor: int (default: 2)
        config.interpolate: str (default: 'three')
        config.attn_channel: bool (default: True)
    """
    def __init__(self, config):
        super().__init__()
        from .SeedFormer import SeedFormer
        
        self.model = SeedFormer(
            feat_dim=getattr(config, 'feat_dim', 512),
            embed_dim=getattr(config, 'embed_dim', 128),
            num_p0=getattr(config, 'num_p0', 512),
            n_knn=getattr(config, 'n_knn', 20),
            radius=getattr(config, 'radius', 1),
            up_factors=getattr(config, 'up_factors', [1, 2, 2]),
            seed_factor=getattr(config, 'seed_factor', 2),
            interpolate=getattr(config, 'interpolate', 'three'),
            attn_channel=getattr(config, 'attn_channel', True),
        )
    
    def forward(self, xyz):
        return self.model(xyz)


# =====================================================================
# MSN Wrapper
# =====================================================================

@MODELS.register_module()
class MSN_Wrapper(nn.Module):
    """
    Registry wrapper for MSN (Morphing and Sampling Network).
    
    NOTE: MSN.forward() returns (out1, out2, loss_mst) — the expansion
    penalty loss is computed internally. The trainer should handle this.
    
    Config options:
        config.num_points: int (default: 8192)
        config.bottleneck_size: int (default: 1024)
        config.n_primitives: int (default: 16)
    """
    def __init__(self, config):
        super().__init__()
        from .MSN import MSN
        
        self.model = MSN(
            num_points=getattr(config, 'num_points', 8192),
            bottleneck_size=getattr(config, 'bottleneck_size', 1024),
            n_primitives=getattr(config, 'n_primitives', 16),
        )
        self.has_internal_loss = True
    
    def forward(self, xyz):
        """
        MSN expects input (B, 3, N), so we transpose.
        Returns (coarse, fine) for compatibility.
        The expansion loss is stored as self.last_expansion_loss.
        """
        # MSN expects (B, 3, N)
        xyz_t = xyz.transpose(1, 2).contiguous()
        out1, out2, loss_mst = self.model(xyz_t)
        self.last_expansion_loss = loss_mst
        return (out1, out2)
    
    def get_loss(self, ret, gt, epoch=0):
        """MSN loss = CD(coarse, gt) + CD(fine, gt) + expansion_penalty."""
        from Chamfer3D.dist_chamfer_3D import chamfer_3DDist
        chamfer_dist = chamfer_3DDist()
        
        d1, d2, _, _ = chamfer_dist(ret[0], gt)
        cd_coarse = torch.mean(d1) + torch.mean(d2)
        
        d1, d2, _, _ = chamfer_dist(ret[1], gt)
        cd_fine = torch.mean(d1) + torch.mean(d2)
        
        loss = cd_coarse + cd_fine + 0.1 * self.last_expansion_loss
        return loss


# =====================================================================
# PFNet Wrapper
# =====================================================================

@MODELS.register_module()
class PFNet_Wrapper(nn.Module):
    """
    Registry wrapper for PFNet (Point Fractal Network).
    
    NOTE: PFNet originally uses a GAN-based training paradigm with a
    discriminator. This wrapper only wraps the generator (_netG).
    For full adversarial training, refer to PFNet's original code.
    
    Config options:
        config.num_scales: int (default: 3)
        config.each_scales_size: int (default: 1)
        config.point_scales_list: list (default: [2048, 512, 256])
        config.crop_point_num: int (default: 512)
    """
    def __init__(self, config):
        super().__init__()
        from .model_PFNet import _netG
        
        self.model = _netG(
            num_scales=getattr(config, 'num_scales', 3),
            each_scales_size=getattr(config, 'each_scales_size', 1),
            point_scales_list=getattr(config, 'point_scales_list', [2048, 512, 256]),
            crop_point_num=getattr(config, 'crop_point_num', 512),
        )
        self.crop_point_num = getattr(config, 'crop_point_num', 512)
    
    def forward(self, xyz):
        """
        PFNet generator expects a list of multi-scale inputs.
        We create them from the single input for compatibility.
        Returns (coarse, fine) point clouds.
        """
        from models.utils import fps_subsample
        
        # Create multi-scale inputs — PFNet expects (B, N, 3) format
        B, N, _ = xyz.shape
        x1 = xyz  # (B, N, 3)
        x2 = fps_subsample(xyz, 512)   # (B, 512, 3)
        x3 = fps_subsample(xyz, 256)   # (B, 256, 3)
        
        inputs = [x1, x2, x3]
        outputs = self.model(inputs)
        
        # outputs is typically a list of point clouds at different scales
        # Return as tuple for consistency
        if isinstance(outputs, (list, tuple)):
            # Ensure (B, N, 3) format
            results = []
            for o in outputs:
                if o.dim() == 3 and o.shape[1] == 3:
                    results.append(o.transpose(1, 2).contiguous())
                else:
                    results.append(o)
            return tuple(results) if len(results) > 1 else (results[0], results[0])
        else:
            if outputs.dim() == 3 and outputs.shape[1] == 3:
                outputs = outputs.transpose(1, 2).contiguous()
            return (outputs, outputs)
