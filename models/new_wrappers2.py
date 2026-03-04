import sys
import os
import torch
import torch.nn as nn
from models.build import MODELS

# Helper to automatically resolve sys.path dynamically
curr_dir = os.path.dirname(os.path.abspath(__file__))

# ============================================================================
# 1. FBNet
# ============================================================================
@MODELS.register_module()
class FBNetWrapper(nn.Module):
    def __init__(self, config):
        super().__init__()
        from models.FBNet import Model as FBNetModel
        self.model = FBNetModel()

    def forward(self, partial):
        # partial: (B, N, 3)
        # FBNet expects (B, 3, N)
        x = partial.transpose(1, 2).contiguous()
        # outputs a tuple or tuple-like. Let's return just the prediction
        # Output of FBNet is coarse, fine, etc. Let's see later.
        outputs = self.model(x)
        # Convert outputs from (B, 3, M) to (B, M, 3) if needed
        # Or if FBNet returns (B, M, 3), we leave it
        def ensure_format(tensor):
            if tensor.shape[1] == 3 and tensor.shape[2] != 3:
                return tensor.transpose(1, 2).contiguous()
            return tensor

        if isinstance(outputs, (tuple, list)):
            return tuple(ensure_format(o) for o in outputs)
        return (ensure_format(outputs),)

# ============================================================================
# 2. FoldingNet
# ============================================================================
@MODELS.register_module()
class FoldingNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        from models.FoldingNet_src.model import AutoEncoder
        self.model = AutoEncoder()

    def forward(self, partial):
        # expected X: (B, 3, N)
        x = partial.transpose(1, 2).contiguous()
        recon2 = self.model(x)
        # recon2: (B, 3, M), transpose to (B, M, 3)
        return (recon2.transpose(1, 2).contiguous(),)

# ============================================================================
# 3. FFSC
# ============================================================================
sys.path.append(os.path.join(curr_dir, 'FSC_src'))
@MODELS.register_module()
class FFSC(nn.Module):
    def __init__(self, config):
        super().__init__()
        from easydict import EasyDict as edict
        # dummy config compatible with FSC models
        cfg = edict()
        cfg.NETWORK = edict()
        cfg.NETWORK.merge_points = 512
        cfg.NETWORK.local_points = 512
        cfg.NETWORK.step1 = 4
        cfg.NETWORK.step2 = 8
        cfg.NETWORK.view_distance = 0.7
        cfg.DATASET = edict()
        cfg.DATASET.TEST_DATASET = 'ShapeNet'

        from models.FSC_src.models.FFSC import Model as FFSCModel
        self.model = FFSCModel(cfg)

    def forward(self, partial):
        # partial: (B, N, 3)
        outputs = self.model(partial)
        return tuple(outputs)

# ============================================================================
# 4. FSCSVD
# ============================================================================
@MODELS.register_module()
class FSCSVD(nn.Module):
    def __init__(self, config):
        super().__init__()
        from easydict import EasyDict as edict
        cfg = edict()
        cfg.NETWORK = edict()
        cfg.NETWORK.merge_points = 512
        cfg.NETWORK.local_points = 512
        cfg.NETWORK.step1 = 4
        cfg.NETWORK.step2 = 8
        cfg.NETWORK.view_distance = 0.7
        cfg.DATASET = edict()
        cfg.DATASET.TEST_DATASET = 'ShapeNet'

        from models.FSC_src.models.FSCSVD import Model as FSCSVDModel
        self.model = FSCSVDModel(cfg)

    def forward(self, partial):
        # partial: (B, N, 3)
        outputs = self.model(partial)
        return tuple(outputs)

# ============================================================================
# 5. PMPNet
# ============================================================================
sys.path.append(os.path.join(curr_dir, 'PMPNet_src'))
@MODELS.register_module()
class PMPNetWrapper(nn.Module):
    def __init__(self, config):
        super().__init__()
        from .PMPNet_src.models.model import PMPNet
        self.model = PMPNet(dataset='ShapeNet')

    def forward(self, partial):
        # outputs: (list of point clouds, list of deltas)
        outputs = self.model(partial)
        if isinstance(outputs, tuple) and len(outputs) == 2 and isinstance(outputs[0], list):
            return tuple(outputs[0])
        if isinstance(outputs, list):
            return tuple(outputs)
        return (outputs,)

# ============================================================================
# 6. PMPNetPlus
# ============================================================================
@MODELS.register_module()
class PMPNetPlusWrapper(nn.Module):
    def __init__(self, config):
        super().__init__()
        from .PMPNet_src.models.model import PMPNetPlus
        self.model = PMPNetPlus(dataset='ShapeNet')

    def forward(self, partial):
        outputs = self.model(partial)
        if isinstance(outputs, tuple) and len(outputs) == 2 and isinstance(outputs[0], list):
            return tuple(outputs[0])
        if isinstance(outputs, list):
            return tuple(outputs)
        return (outputs,)

# ============================================================================
# 7. SVDFormer New
# ============================================================================
sys.path.append(os.path.join(curr_dir, 'SVDFormer_src'))
@MODELS.register_module()
class SVDFormerNew(nn.Module):
    def __init__(self, config):
        super().__init__()
        from easydict import EasyDict as edict
        cfg = edict()
        cfg.NETWORK = edict()
        cfg.NETWORK.view_distance = 0.7
        cfg.NETWORK.step1 = 4
        cfg.NETWORK.step2 = 8
        cfg.NETWORK.merge_points = 512
        cfg.NETWORK.local_points = 512
        cfg.DATASET = edict()
        cfg.DATASET.TEST_DATASET = 'ShapeNet'

        from SVDFormer import Model as SVDFormerModel
        self.model = SVDFormerModel(cfg)

    def forward(self, partial, depth=None):
        if depth is None:
            # 3 views, 1 channel each
            depth = torch.zeros(partial.shape[0] * 3, 1, 224, 224, device=partial.device)
        outputs = self.model(partial, depth)
        return tuple(outputs) if isinstance(outputs, (list, tuple)) else (outputs,)

# ============================================================================
# 8. PointSea
# ============================================================================
sys.path.append(os.path.join(curr_dir, 'PointSea_src'))
@MODELS.register_module()
class PointSeaWrapper(nn.Module):
    def __init__(self, config):
        super().__init__()
        from easydict import EasyDict as edict
        cfg = edict()
        cfg.NETWORK = edict()
        cfg.NETWORK.view_distance = 0.7
        cfg.NETWORK.step1 = 4
        cfg.NETWORK.step2 = 8
        cfg.NETWORK.merge_points = 512
        cfg.NETWORK.local_points = 512
        cfg.DATASET = edict()
        cfg.DATASET.TEST_DATASET = 'ShapeNet'

        from PointSea import Model as PointSeaModel
        self.model = PointSeaModel(cfg)

    def forward(self, partial, depth=None):
        if depth is None:
            # 3 views, 3 channels each for ResEncoder
            depth = torch.zeros(partial.shape[0] * 3, 3, 224, 224, device=partial.device)
        outputs = self.model(partial, depth)
        return tuple(outputs) if isinstance(outputs, (list, tuple)) else (outputs,)

# ============================================================================
# 9. PCTMA
# ============================================================================
sys.path.append(os.path.join(curr_dir, 'PCTMA_src'))
@MODELS.register_module()
class PCTMAWrapper(nn.Module):
    def __init__(self, config):
        super().__init__()
        from models.PCTMA_src.pctma.pctMa_network import PCTMA_Net
        parameter = {
            "use_atlas": True,
            "use_pointModule": 1,
            "n_primitives": 32,
            "fine_factor": 4,
            "grid_scale": 0.05,
            "Num_Encoder": 4,
            "num_head": 8,
            "dropout": 0.1,
            "d_ff": 2048,
            "d_model": 1024,
            "src_vocab": 3,
            "tgt_vocab": 2048,
            "epochs": 300,
            "De_numpoints": 2048,
            "En_channels": [1024],
            "De_channels": [1024, 2048, 2048],
            "train_pt": False,
            "train_gt": False,
            "train_pt_to_gt": True,
            "use_emd": True,
            "use_cd": False,
            "use_consistence": False,
            "use_dgcnn": False,
            "use_cmlp": False,
            "w_cd": 10000,
            "w_emd": 0.00001,
            "gene_file": False,
            "down_sampling": False,
            "combined_pc": True,
            "ppd_loss": False
        }
        # The PCTMA model includes optimizers and logging out-of-the-box. We just want the model
        # Try to initialize or mock it
        self.model = PCTMA_Net(parameter)

    def forward(self, partial):
        # PCTMA returns: pc3_xyz, pc2_xyz, pc1_xyz, z_hat
        outputs = self.model(partial)
        if isinstance(outputs, tuple) and len(outputs) >= 3:
            # We only want point cloud tensors, not the latent vector z_hat.
            return tuple(x for x in outputs[:-1] if x is not None)
        return tuple(outputs) if isinstance(outputs, (list, tuple)) else (outputs,)
