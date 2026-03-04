import sys
sys.path.append('../pointnet2_ops_lib')
sys.path.append('..')

# Import all registered models so they appear in the MODELS registry
from .build import MODELS, build_model_from_cfg

# Models with @MODELS.register_module() decorator
from . import PCN
from . import TopNet
from . import PoinTr
from . import AdaPoinTr
from . import AnchorFormer
from . import GRNet
from . import SnowFlakeNet
from . import SymmCompletion

# Wrapper-registered models (CRAPCN, SeedFormer, MSN, PFNet)
from . import wrappers

# Dummy model for pipeline testing
from . import DummyModel

# New models adopted from new_unmerge repository
# (DSPF, SDT, MPGLNet, LEMA, BiMPRNet, IAET, MAENet, GeoFormer, FDANet, TEETHM4T)
from . import new_wrappers

# 8 more new models integrated recently
from . import new_wrappers2