# Point Cloud Completion Boilerplate

A unified framework for point cloud completion research with 22 supported models.

## Supported Models

### Point-Cloud-Only Models
| Model | Config | Registry Name |
|-------|--------|---------------|
| AdaPoinTr | `configs/adapointr.yaml` | `AdaPoinTr` |
| AnchorFormer | `configs/anchorformer.yaml` | `AnchorFormer` |
| CRA-PCN | `configs/crapcn.yaml` | `CRAPCN` |
| DSPF | `configs/dspf.yaml` | `DSPF` |
| GRNet | `configs/grnet.yaml` | `GRNet` |
| LEMA | `configs/lema.yaml` | `LEMA` |
| MPGLNet | `configs/mpglnet.yaml` | `MPGLNet` |
| MSN | `configs/msn.yaml` | `MSN` |
| PCN | `configs/pcn.yaml` | `PCN` |
| PFNet | `configs/pfnet.yaml` | `PFNet` |
| PoinTr | `configs/pointr.yaml` | `PoinTr` |
| SDT | `configs/sdt.yaml` | `SDT` |
| SeedFormer | `configs/seedformer.yaml` | `SeedFormer` |
| SnowFlakeNet | `configs/snowflakenet.yaml` | `SnowFlakeNet` |
| SymmCompletion | `configs/symmcompletion.yaml` | `SymmCompletion` |
| TopNet | `configs/topnet.yaml` | `TopNet` |

### Multi-Modal Models (image/depth + point cloud)
| Model | Config | Registry Name | Extra Input |
|-------|--------|---------------|-------------|
| BiMPR-Net | `configs/bimprnet.yaml` | `BiMPRNet` | RGB image (224×224) |
| GeoFormer | `configs/geoformer.yaml` | `GeoFormer` | Depth image |
| IAET | `configs/iaet.yaml` | `IAET` | RGB image (224×224) |
| MAENet | `configs/maenet.yaml` | `MAENet` | RGB image (224×224) |

### Special-Input Models
| Model | Config | Registry Name | Extra Input |
|-------|--------|---------------|-------------|
| FDANet | `configs/fdanet.yaml` | `FDANet` | Section/skeleton pts |
| TEETHM4T | `configs/teethm4t.yaml` | `TEETHM4T_Wrapper` | GT during training |

## Quick Start

### Training
```bash
# Train any model using its config
python train.py --config configs/pcn.yaml
python train.py --config configs/seedformer.yaml --batch_size 16 --gpu 0,1
python train.py --config configs/snowflakenet.yaml --epochs 200

# Resume from checkpoint
python train.py --config configs/pcn.yaml --resume results/pcn/checkpoints/ckpt-best.pth
```

### Testing
```bash
# Evaluate a trained model
python test.py --config configs/pcn.yaml --checkpoint path/to/ckpt-best.pth
python test.py --config configs/pcn.yaml --checkpoint ckpt.pth --output

# ShapeNet55 multi-viewpoint evaluation
python test.py --config configs/seedformer.yaml --checkpoint ckpt.pth --mode median
python test.py --config configs/seedformer.yaml --checkpoint ckpt.pth --mode hard
```

### Pipeline Verification
Test that everything works without real data or checkpoints.  
Dummy mode requires no CUDA extensions, while model verification may require CUDA/custom ops:
```bash
# Verify the dummy model pipeline (fastest — no CUDA extensions needed)
python verify_pipeline.py

# Verify a specific model (needs CUDA extensions for that model)
python verify_pipeline.py --config configs/pcn.yaml

# Verify ALL models (default: isolated subprocess per config)
python verify_pipeline.py --all

# Legacy behavior: run all models in one process
python verify_pipeline.py --all --no-isolate

# Skip selected configs by basename
python verify_pipeline.py --all --skip-configs grnet.yaml,iaet.yaml

# Skip configs from a quarantine file (one basename per line)
python verify_pipeline.py --all --quarantine-file configs/quarantine.txt

# Disable debug retry (default is enabled)
python verify_pipeline.py --all --no-debug-retry

# Override per-model timeout for isolated workers (seconds)
python verify_pipeline.py --all --timeout 300

# Dummy training (runs 3 epochs with random data)
python train.py --config configs/dummy.yaml --dummy
python train.py --config configs/pcn.yaml --dummy --epochs 2

# Dummy testing (random weights + random data)
python test.py --config configs/dummy.yaml --dummy
```

`--all` exit codes:
- Non-zero only when non-skipped models fail.
- Skipped/quarantined models are reported but do not fail CI.

## Adding a New Model

1. Create your model file in `models/` (use `models/BaseModel.py` as template):

```python
from models.BaseModel import BaseModel
from models.build import MODELS

@MODELS.register_module()
class MyNewModel(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        # Build your network layers...
        self.build_loss_func()

    def forward(self, xyz):
        # xyz: (B, N, 3) partial point cloud
        coarse = ...
        fine = ...
        return (coarse, fine)  # tuple of (B, M_i, 3) predictions

    def build_loss_func(self):
        from extensions.chamfer_dist import ChamferDistanceL2
        self.loss_func = ChamferDistanceL2()

    def get_loss(self, ret, gt, epoch=0):
        return self.loss_func(ret[0], gt) + self.loss_func(ret[1], gt)
```

2. Import it in `models/__init__.py`:
```python
from . import my_new_model
```

3. Create a config in `configs/my_model.yaml`:
```yaml
_base_: default.yaml
model:
  name: "MyNewModel"
  config:
    my_param: 512
```

4. Train: `python train.py --config configs/my_model.yaml`

## Adding a New Dataset

Use `dataset/BaseDataset.py` as your template. Your dataset must return:
```python
(taxonomy_id, model_id, {'partial_cloud': Tensor(N,3), 'gtcloud': Tensor(M,3)})
```

See `dataset/BaseDataset.py` for a complete `PairedFileDataset` example.

## Configuration System

Configs use YAML with inheritance (`_base_: default.yaml`).
CLI arguments override YAML values:

```bash
python train.py --config configs/pcn.yaml --batch_size 64 --lr 0.0005 --gpu 0,1,2
```

## Project Structure

```
├── configs/                  # YAML configuration files
│   ├── default.yaml          # Base config with defaults
│   ├── config.py             # Config loader with inheritance
│   └── *.yaml                # Per-model configs (22 models)
├── models/                   # Model implementations
│   ├── BaseModel.py          # Template for new models
│   ├── build.py              # Model registry
│   ├── wrappers.py           # Wrappers for CRAPCN/MSN/PFNet/SeedFormer
│   ├── new_wrappers.py       # Wrappers for 10 new models
│   ├── DSPF/                 # DSPF source files
│   ├── SDT/                  # SDT source files
│   ├── MPGLNet_src/          # MPGLNet source files
│   ├── LEMA/                 # LEMA source files
│   ├── BiMPRNet/             # BiMPR-Net source + encoders/decoders
│   ├── IAET_src/             # IAET source files
│   ├── MAENet/               # MAENet source + encoders/decoders
│   ├── GeoFormer/            # GeoFormer/SVDFormer source files
│   ├── FDANet/               # FDANet source files
│   ├── TEETHM4T/             # TEETHM4T source files
│   └── *.py                  # Original model files
├── dataset/                  # Dataset implementations
│   └── BaseDataset.py        # Template for new datasets
├── utils/                    # Utility functions
│   ├── loss.py               # Unified pyramid CD loss
│   ├── data_loaders.py       # Dataset loaders
│   └── ...
├── train.py                  # ← Unified training (all models)
├── test.py                   # ← Unified testing (all models)
├── verify_pipeline.py        # ← Pipeline verification tool
└── train_pcn.py, etc.        # Legacy per-model scripts (kept for reference)
```
