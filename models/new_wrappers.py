"""
Wrappers for 10 new models adopted from new_unmerge repository.
Each wrapper registers the model with MODELS registry and normalizes
the forward interface to: input (B,N,3) → output tuple of (B,M,3) tensors.

Models are grouped by input type:
  Point-cloud-only: DSPF, SDT, MPGLNet, LEMA
  Multi-modal (image + pc): BiMPRNet, IAET, MAENet, GeoFormer
  Special input: FDANet (section points), TEETHM4T (gt in forward)
"""

import torch
import torch.nn as nn
from models.build import MODELS

# ============================================================================
# 1. DSPF — simple encoder-decoder, input (B,N,3), output single (B,M,3)
# ============================================================================
@MODELS.register_module()
class DSPF(nn.Module):
    """
    Wrapper for DSPF (Dual Structure Prior Fusion).
    Original: our_model in models/DSPF/our_models.py
    Input:  partial (B, N, 3)
    Output: (pred,)  where pred is (B, 2048, 3)
    """
    def __init__(self, config):
        super().__init__()
        from models.DSPF.our_models import our_model
        self.model = our_model()
        self.num_pred = getattr(config, 'num_pred', 2048)

    def forward(self, partial):
        # our_model expects (B, N, 3), returns (B, M, 3)
        pred = self.model(partial)
        return (pred,)


# ============================================================================
# 2. SDT — Selective Dense Transformer
# ============================================================================
@MODELS.register_module()
class SDT(nn.Module):
    """
    Wrapper for SDT.
    Original: Model in models/SDT/model.py
    Input:  partial (B, N, 3)
    Output: (coarse, fine) where coarse (B, 1024, 3), fine (B, 2048, 3)
    Note: Original input is (B, 3, N), wrapper transposes.
    """
    def __init__(self, config):
        super().__init__()
        num_coarse = getattr(config, 'num_coarse', 1024)
        num_fine = getattr(config, 'num_fine', 2048)
        num_input = getattr(config, 'num_input', 2048)
        from models.SDT.model import Model as SDTModel
        self.model = SDTModel(
            num_coarse=num_coarse,
            num_fine=num_fine,
            num_input=num_input,
        )

    def forward(self, partial):
        # partial: (B, N, 3) → need (B, 3, N) for original model
        x = partial.transpose(1, 2).contiguous()
        coarse, fine = self.model(x)
        # outputs are already (B, N, 3)
        return (coarse, fine)


# MPGLNet already has @MODELS.register_module() in its source file.
# We import it so it gets registered.
try:
    from models.MPGLNet_src.MPGLNet import MPGLNet  # triggers @MODELS.register_module()
except Exception as _mpgl_err:
    import warnings
    warnings.warn(f"MPGLNet could not be imported: {_mpgl_err}")


# ============================================================================
# 4. LEMA — GAN-based, like PFNet
# ============================================================================
@MODELS.register_module()
class LEMA(nn.Module):
    """
    Wrapper for LEMA (Loss-Edge-Merging-with-Attention).
    Original: _netG (generator) in models/LEMA/model_main.py
    GAN-based model with generator and discriminator.
    Input:  partial (B, N, 3) — wrapper creates multi-scale input
    Output: (coarse, fine)
    """
    def __init__(self, config):
        super().__init__()
        from models.LEMA.model_main import _netG, _netlocalD
        crop_point_num = getattr(config, 'crop_point_num', 512)
        self.generator = _netG(
            num_scales=3,
            each_scales_size=1,
            point_scales_list=[2048, 1024, 512],
            crop_point_num=crop_point_num,
        )
        # Discriminator for GAN training (optional)
        self.discriminator = _netlocalD(crop_point_num=crop_point_num)
        self.crop_point_num = crop_point_num

    def forward(self, partial):
        # _netG expects list of multi-scale inputs [full, mid, small] + midu
        B, N, _ = partial.shape
        x1 = partial  # (B, 2048, 3)
        # Simple random sampling for multi-scale (avoids CUDA FPS issues)
        idx_1024 = torch.randperm(N, device=partial.device)[:min(1024, N)]
        x2 = partial[:, idx_1024, :]  # (B, 1024, 3)
        idx_512 = torch.randperm(N, device=partial.device)[:min(512, N)]
        x3 = partial[:, idx_512, :]   # (B, 512, 3)

        input_list = [x1, x2, x3]
        # midu: intermediate feature that flattens to 240 elements
        # fc1 expects 3184 = 2944 (latent) + 240 (midu flat)
        # Use (B, 80, 3) subsample of partial
        idx_80 = torch.randperm(N, device=partial.device)[:min(80, N)]
        midu = partial[:, idx_80, :]  # (B, 80, 3)
        pc1_xyz, pc2_xyz, pc3_xyz = self.generator(input_list, midu)
        # pc1_xyz: (B, 64, 3) coarse, pc3_xyz: (B, crop_point_num, 3) fine
        return (pc1_xyz, pc3_xyz)


# ============================================================================
# 5. BiMPRNet — multi-modal (image + point cloud)
# ============================================================================
@MODELS.register_module()
class BiMPRNet(nn.Module):
    """
    Wrapper for BiMPR-Net.
    Original: Network in models/BiMPRNet/model.py
    Input:  partial (B, N, 3), optionally image (B, 3, 224, 224)
    Output: (pred,)
    Note: Requires image input for full functionality. Without image,
          uses a zero tensor as placeholder.
    """
    def __init__(self, config):
        super().__init__()
        from models.BiMPRNet.model import Network
        self.model = Network()
        self.use_image = getattr(config, 'use_image', False)

    def forward(self, partial, image=None):
        # Original expects (B, 3, N) for pc and (B, 3, 224, 224) for image
        x = partial.transpose(1, 2).contiguous()
        if image is None:
            # Create dummy image if not provided
            B = partial.shape[0]
            device = partial.device
            image = torch.zeros(B, 3, 224, 224, device=device)
        pred = self.model(x, image)
        return (pred,)


# ============================================================================
# 6. IAET — multi-modal (image + point cloud)
# ============================================================================
@MODELS.register_module()
class IAET(nn.Module):
    """
    Wrapper for IAET (Interlaced Attention Enhancement Transformer).
    Original: IAET in models/IAET_src/IAET.py
    Input:  partial (B, N, 3), optionally image (B, 3, 224, 224)
    Output: list of (B, M_i, 3) from coarse to fine
    """
    def __init__(self, config):
        super().__init__()
        dim_feat = getattr(config, 'dim_feat', 256)
        num_points = getattr(config, 'num_points', 256)
        up_factors = getattr(config, 'up_factors', [2, 2, 2])
        if isinstance(up_factors, list):
            up_factors = tuple(up_factors)
        from models.IAET_src.IAET import IAET as IAETModel
        self.model = IAETModel(
            dim_feat=dim_feat,
            num_points=num_points,
            up_factors=up_factors,
        )
        self.use_image = getattr(config, 'use_image', False)

    def forward(self, partial, image=None):
        if image is None:
            B = partial.shape[0]
            device = partial.device
            image = torch.zeros(B, 3, 224, 224, device=device)
        pcd_list = self.model(partial, image)
        return tuple(pcd_list)


# ============================================================================
# 7. MAENet — multi-modal (image + point cloud), multi-output
# ============================================================================
@MODELS.register_module()
class MAENet(nn.Module):
    """
    Wrapper for MAENet.
    Original: Network in models/MAENet/model.py
    Input:  partial (B, N, 3), optionally image (B, 3, 224, 224)
    Output: (complete, final, pc_generate, pred_pcds)
    """
    def __init__(self, config):
        super().__init__()
        from models.MAENet.model import Network
        self.model = Network()
        self.use_image = getattr(config, 'use_image', False)

    def forward(self, partial, image=None):
        # Original expects (B, 3, N) for pc
        x = partial.transpose(1, 2).contiguous()
        if image is None:
            B = partial.shape[0]
            device = partial.device
            image = torch.zeros(B, 3, 224, 224, device=device)
        complete, final, pc_generate, pred_pcds = self.model(x, image)
        return (final, complete)  # Return (coarse, fine) order


# ============================================================================
# 8. GeoFormer (SVDFormer) — multi-modal (depth + point cloud)
# ============================================================================
@MODELS.register_module()
class GeoFormer(nn.Module):
    """
    Wrapper for GeoFormer/SVDFormer.
    Original: Model in models/GeoFormer/SVDFormer.py
    Input:  partial (B, N, 3), optionally depth image
    Output: multi-stage predictions
    """
    def __init__(self, config):
        super().__init__()
        from easydict import EasyDict as edict
        # Build config expected by SVDFormer
        cfg = edict()
        cfg.NETWORK = edict()
        cfg.NETWORK.view_distance = getattr(config, 'view_distance', 0.7)
        cfg.NETWORK.step1 = getattr(config, 'step1', 4)
        cfg.NETWORK.step2 = getattr(config, 'step2', 8)
        cfg.NETWORK.merge_points = getattr(config, 'merge_points', 512)
        cfg.NETWORK.local_points = getattr(config, 'local_points', 512)
        cfg.DATASET = edict()
        cfg.DATASET.TEST_DATASET = getattr(config, 'dataset_name', 'ShapeNet')
        from models.GeoFormer.SVDFormer import Model as GeoModel
        self.model = GeoModel(cfg)
        self.use_depth = getattr(config, 'use_depth', False)

    def forward(self, partial, depth=None):
        if depth is None:
            B = partial.shape[0]
            device = partial.device
            depth = torch.zeros(B, 3, 224, 224, device=device)
        results = self.model(partial, depth)
        if isinstance(results, (tuple, list)):
            return tuple(results)
        return (results,)


# ============================================================================
# 9. FDANet — needs section/skeleton points
# ============================================================================
@MODELS.register_module()
class FDANet(nn.Module):
    """
    Wrapper for FDANet.
    Original: mymodel in models/FDANet/Model.py
    Input:  partial (B, N, 3), optionally section pts S (B, 36, 3)
    Output: (pred,)
    Note: Input pc expected as (B, 3, N). Section points needed for full use.
    """
    def __init__(self, config):
        super().__init__()
        num_pred = getattr(config, 'num_pred', 1024)
        k = getattr(config, 'k', 8)
        N_in = getattr(config, 'num_input', 2048)
        N_x = [N_in, N_in // 2, N_in // 4]
        from models.FDANet.Model import mymodel
        self.model = mymodel(
            point_resolution=N_x,
            k=k,
            FE_inchans=[32, 64, 128, 256],
            p_num=num_pred,
        )

    def forward(self, partial, section_pts=None):
        # mymodel expects (B, 3, N) for x and (B, 36, 3) for S
        x = partial.transpose(1, 2).contiguous()
        pred = self.model(x, section_pts)
        # Output is (B, N, 3) already
        return (pred,)


# ============================================================================
# 10. TEETHM4T — PoinTr-style, needs gt in forward during training
# ============================================================================
@MODELS.register_module()
class TEETHM4T_Wrapper(nn.Module):
    """
    Wrapper for TEETHM4T (Memory-Guided Point Cloud Completion).
    Original: TEETHM4T in models/TEETHM4T/TEETHM4T.py
    Input:  partial (B, N, 3), gt (B, M, 3) [gt needed during training]
    Output: (coarse, fine) during eval, (pred_coarse, denoised_coarse, denoised_fine, pred_fine) during training
    """
    def __init__(self, config):
        super().__init__()
        from models.TEETHM4T.TEETHM4T import TEETHM4T
        self.model = TEETHM4T(config)

    def forward(self, partial, gt=None):
        if gt is None and self.training:
            # During training without gt, create dummy gt
            B, N, _ = partial.shape
            gt = torch.randn(B, 2048, 3, device=partial.device)
        elif gt is None:
            gt = torch.randn(partial.shape[0], 2048, 3, device=partial.device)

        ret, lfa = self.model(partial, gt)

        if self.training:
            # ret = (pred_coarse, denoised_coarse, denoised_fine, pred_fine)
            return ret
        else:
            # ret = (coarse_point_cloud, rebuild_points)
            return ret

    def get_loss(self, pcds_pred, gt, epoch=1, **kwargs):
        """Delegate to inner model's loss."""
        return self.model.get_loss(pcds_pred, gt, epoch)

    def build_loss_func(self):
        self.model.build_loss_func()
