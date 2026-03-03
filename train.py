"""
=======================================================================
Unified Training Script for Point Cloud Completion
=======================================================================

Single entry point that trains any registered model on any dataset.

Usage:
    python train.py --config configs/pcn.yaml
    python train.py --config configs/seedformer.yaml --batch_size 16
    python train.py --config configs/snowflakenet.yaml --gpu 0,1
    python train.py --config configs/pcn.yaml --resume results/pcn/ckpt-best.pth

    # Dummy mode — test pipeline without real data or GPU:
    python train.py --config configs/pcn.yaml --dummy
    python train.py --config configs/dummy.yaml --dummy --epochs 2

All model/dataset/training parameters are specified in the YAML config.
CLI arguments override YAML values.
"""

import os
import argparse
import json
import time
import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
from easydict import EasyDict as edict
from pprint import pprint

import utils.data_loaders
import utils.helpers
from utils.average_meter import AverageMeter
from utils.metrics import Metrics
from utils.schedular import GradualWarmupScheduler
from utils.loss import pyramid_loss
from configs.config import load_config, cfg_from_args


# =====================================================================
# Model Builder
# =====================================================================

def build_model(cfg):
    """
    Build a model from the registry by name.

    The model name in the config maps to registered models:
        - Registry models: PCN, TopNet, PoinTr, AdaPoinTr, AnchorFormer,
          GRNet, SnowFlakeNet, SymmCompletion
        - Wrapper models: CRAPCN_Wrapper, SeedFormer_Wrapper,
          MSN_Wrapper, PFNet_Wrapper

    For convenience, common aliases are mapped:
        'CRAPCN' -> 'CRAPCN_Wrapper'
        'SeedFormer' -> 'SeedFormer_Wrapper'
        'MSN' -> 'MSN_Wrapper'
        'PFNet' -> 'PFNet_Wrapper'
    """
    # Ensure models are registered
    import models  # noqa: triggers __init__.py imports

    from models.build import MODELS

    model_name = cfg.model.name
    model_config = edict(cfg.model.config) if hasattr(cfg.model, 'config') else edict()

    # Alias mapping for convenience
    aliases = {
        'CRAPCN': 'CRAPCN_Wrapper',
        'CRA-PCN': 'CRAPCN_Wrapper',
        'SeedFormer': 'SeedFormer_Wrapper',
        'MSN': 'MSN_Wrapper',
        'PFNet': 'PFNet_Wrapper',
    }
    registry_name = aliases.get(model_name, model_name)

    print(f'Building model: {model_name} (registry: {registry_name})')
    model = MODELS.build(edict(NAME=registry_name, **model_config))

    return model


# =====================================================================
# Dataset Builder
# =====================================================================

def build_data_loaders(cfg, dummy=False):
    """
    Build train and validation data loaders.

    Supports all existing dataset loaders via DATASET_LOADER_MAPPING.
    Also supports MVP dataset with its custom format.
    If dummy=True, uses DummyDataset with random point clouds.
    """
    # Dummy mode: random data, no files needed
    if dummy:
        return _build_dummy_loaders(cfg)

    dataset_name = cfg.dataset.train
    test_dataset_name = getattr(cfg.dataset, 'test', dataset_name)

    # Build legacy-compatible config for data loaders
    legacy_cfg = _build_legacy_cfg(cfg)

    if dataset_name == 'MVP':
        return _build_mvp_loaders(cfg)

    # Use existing DATASET_LOADER_MAPPING
    train_loader_cls = utils.data_loaders.DATASET_LOADER_MAPPING.get(dataset_name)
    val_loader_cls = utils.data_loaders.DATASET_LOADER_MAPPING.get(test_dataset_name)

    if train_loader_cls is None:
        raise ValueError(
            f"Unknown dataset: {dataset_name}. "
            f"Available: {list(utils.data_loaders.DATASET_LOADER_MAPPING.keys())}"
        )

    train_dataset_loader = train_loader_cls(legacy_cfg)
    val_dataset_loader = val_loader_cls(legacy_cfg)

    train_data_loader = torch.utils.data.DataLoader(
        dataset=train_dataset_loader.get_dataset(utils.data_loaders.DatasetSubset.TRAIN),
        batch_size=cfg.train.batch_size,
        num_workers=cfg.train.num_workers,
        collate_fn=utils.data_loaders.collate_fn,
        pin_memory=True,
        shuffle=True,
        drop_last=False,
    )
    val_data_loader = torch.utils.data.DataLoader(
        dataset=val_dataset_loader.get_dataset(utils.data_loaders.DatasetSubset.TEST),
        batch_size=cfg.train.batch_size,
        num_workers=max(1, cfg.train.num_workers // 2),
        collate_fn=utils.data_loaders.collate_fn,
        pin_memory=True,
        shuffle=False,
    )

    return train_data_loader, val_data_loader


def _build_dummy_loaders(cfg):
    """Build dummy data loaders with random point clouds for pipeline testing."""
    from dataset.DummyDataset import DummyDataset

    n_gt = 16384  # default
    if hasattr(cfg.model, 'config'):
        n_gt = getattr(cfg.model.config, 'num_pred', n_gt)

    train_dataset = DummyDataset(n_samples=50, n_input=2048, n_gt=n_gt)
    val_dataset = DummyDataset(n_samples=10, n_input=2048, n_gt=n_gt)

    train_data_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=min(cfg.train.batch_size, 4),  # small for speed
        num_workers=0,
        collate_fn=utils.data_loaders.collate_fn,
        pin_memory=False,
        shuffle=True,
    )
    val_data_loader = torch.utils.data.DataLoader(
        dataset=val_dataset,
        batch_size=min(cfg.train.batch_size, 4),
        num_workers=0,
        collate_fn=utils.data_loaders.collate_fn,
        pin_memory=False,
        shuffle=False,
    )

    return train_data_loader, val_data_loader


def _build_mvp_loaders(cfg):
    """Build MVP dataset loaders (custom format, not using DATASET_LOADER_MAPPING)."""
    import h5py
    import torch.utils.data as data

    class MVP_Dataset(data.Dataset):
        def __init__(self, file_path):
            input_file = h5py.File(file_path, 'r')
            self.input_data = np.array(input_file['incomplete_pcds'][()])
            self.gt_data = np.array(input_file['complete_pcds'][()])
            self.labels = np.array(input_file['labels'][()])
            input_file.close()

        def __len__(self):
            return self.input_data.shape[0]

        def __getitem__(self, index):
            partial = torch.from_numpy(self.input_data[index]).float()
            gt = torch.from_numpy(self.gt_data[index // 26]).float()
            label = self.labels[index]
            return label, partial, gt

    mvp_cfg = cfg.datasets.mvp
    train_dataset = MVP_Dataset(mvp_cfg.train_path)
    val_dataset = MVP_Dataset(mvp_cfg.val_path)

    train_data_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=cfg.train.batch_size,
        num_workers=cfg.train.num_workers,
        pin_memory=True,
        shuffle=True,
        drop_last=False,
    )
    val_data_loader = torch.utils.data.DataLoader(
        dataset=val_dataset,
        batch_size=cfg.train.batch_size,
        num_workers=max(1, cfg.train.num_workers // 2),
        pin_memory=True,
        shuffle=False,
    )

    return train_data_loader, val_data_loader


def _build_legacy_cfg(cfg):
    """
    Convert new YAML config to legacy edict format expected by existing
    data_loaders.py (ShapeNetDataLoader, etc.).
    """
    legacy = edict()

    # Datasets section
    legacy.DATASETS = edict()

    legacy.DATASETS.SHAPENET = edict()
    if hasattr(cfg, 'datasets') and hasattr(cfg.datasets, 'shapenet'):
        sn = cfg.datasets.shapenet
        legacy.DATASETS.SHAPENET.CATEGORY_FILE_PATH = sn.category_file_path
        legacy.DATASETS.SHAPENET.N_RENDERINGS = getattr(sn, 'n_renderings', 8)
        legacy.DATASETS.SHAPENET.N_POINTS = getattr(sn, 'n_points', 2048)
        legacy.DATASETS.SHAPENET.PARTIAL_POINTS_PATH = sn.partial_points_path
        legacy.DATASETS.SHAPENET.COMPLETE_POINTS_PATH = sn.complete_points_path

    legacy.DATASETS.COMPLETION3D = edict()
    if hasattr(cfg, 'datasets') and hasattr(cfg.datasets, 'completion3d'):
        c3d = cfg.datasets.completion3d
        legacy.DATASETS.COMPLETION3D.CATEGORY_FILE_PATH = c3d.category_file_path
        legacy.DATASETS.COMPLETION3D.PARTIAL_POINTS_PATH = c3d.partial_points_path
        legacy.DATASETS.COMPLETION3D.COMPLETE_POINTS_PATH = c3d.complete_points_path

    if hasattr(cfg, 'datasets') and hasattr(cfg.datasets, 'shapenet55'):
        legacy.DATASETS.SHAPENET55 = edict()
        sn55 = cfg.datasets.shapenet55
        legacy.DATASETS.SHAPENET55.CATEGORY_FILE_PATH = sn55.category_file_path
        legacy.DATASETS.SHAPENET55.N_POINTS = getattr(sn55, 'n_points', 2048)
        legacy.DATASETS.SHAPENET55.COMPLETE_POINTS_PATH = sn55.complete_points_path

    # Dataset selection
    legacy.DATASET = edict()
    legacy.DATASET.TRAIN_DATASET = cfg.dataset.train
    legacy.DATASET.TEST_DATASET = getattr(cfg.dataset, 'test', cfg.dataset.train)

    # Constants
    legacy.CONST = edict()
    legacy.CONST.NUM_WORKERS = cfg.train.num_workers
    legacy.CONST.N_INPUT_POINTS = getattr(cfg.dataset, 'n_input_points', 2048)

    return legacy


# =====================================================================
# Unified Data Unpacker
# =====================================================================

def unpack_batch(batch, dataset_name):
    """
    Unpack a batch from the data loader into (partial, gt) tensors.

    Different datasets return different batch formats:
        - ShapeNet/Completion3D: (taxonomy_ids, model_ids, data_dict)
        - ShapeNet55: (taxonomy_ids, model_ids, data_dict) - gt only, partial generated online
        - MVP: (labels, partial, gt)
    """
    if dataset_name == 'MVP':
        labels, partial, gt = batch
        partial = partial.float().cuda()
        gt = gt.float().cuda()
        return partial, gt, None, None

    taxonomy_ids, model_ids, data = batch
    for k, v in data.items():
        data[k] = utils.helpers.var_or_cuda(v)

    if dataset_name in ('ShapeNet55', 'ShapeNet34'):
        # Generate partial data online
        gt = data['gtcloud']
        _, npoints, _ = gt.shape
        partial, _ = utils.helpers.seprate_point_cloud(
            gt, npoints,
            [int(npoints * 1/4), int(npoints * 3/4)],
            fixed_points=None
        )
    else:
        partial = data['partial_cloud']
        gt = data['gtcloud']

    return partial, gt, taxonomy_ids, model_ids


# =====================================================================
# Training
# =====================================================================

def train(model, train_data_loader, val_data_loader, cfg, resume_path=None):
    """
    Unified training loop for all models.
    """
    dataset_name = cfg.dataset.train

    # Optimizer
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg.train.learning_rate,
        weight_decay=cfg.train.weight_decay,
        betas=tuple(cfg.train.betas),
    )

    # LR Scheduler: warmup + step decay
    scheduler_steplr = StepLR(optimizer, step_size=1, gamma=0.1 ** (1 / cfg.train.lr_decay))
    lr_scheduler = GradualWarmupScheduler(
        optimizer, multiplier=1,
        total_epoch=cfg.train.warmup_epochs,
        after_scheduler=scheduler_steplr,
    )

    # Resume
    init_epoch = 0
    best_metrics = float('inf')
    best_epoch = 0

    if resume_path and os.path.exists(resume_path):
        print(f'Resuming from {resume_path} ...')
        checkpoint = torch.load(resume_path)
        model.load_state_dict(checkpoint['model'])
        init_epoch = checkpoint.get('epoch_index', 0)
        best_metrics = checkpoint.get('best_metrics', float('inf'))
        print(f'Resumed at epoch {init_epoch}, best_metrics = {best_metrics}')

    # Setup output directories
    timestr = time.strftime('_Log_%Y_%m_%d_%H_%M_%S', time.gmtime())
    out_path = os.path.join(cfg.output.out_path, cfg.model.name + timestr)
    ckpt_dir = os.path.join(out_path, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)

    # Save config
    config_path = os.path.join(out_path, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(dict(cfg), f, indent=4, sort_keys=True, default=str)

    # Log files
    train_log = open(os.path.join(out_path, 'training.txt'), 'w')
    test_log = open(os.path.join(out_path, 'testing.txt'), 'w')

    def log(msg, log_file=None, show=True):
        if show:
            print(msg)
        if log_file:
            log_file.write(msg + '\n')
            log_file.flush()

    log('n_itr, cd_pc, cd_p1, cd_p2, cd_p3, partial_matching', train_log, show=False)
    log('#epoch cdc cd1 cd2 partial_matching | cd_fine | #best_epoch best_metrics', test_log)

    # Check if model has its own loss function
    has_model_loss = hasattr(model, 'module') and hasattr(model.module, 'get_loss')
    if not has_model_loss:
        has_model_loss = hasattr(model, 'get_loss')

    # =================== Training Loop ===================
    print(f'\n{"="*60}')
    print(f' Training {cfg.model.name} on {dataset_name}')
    print(f' Epochs: {cfg.train.n_epochs}, Batch: {cfg.train.batch_size}, LR: {cfg.train.learning_rate}')
    print(f' Output: {out_path}')
    print(f'{"="*60}\n')

    for epoch_idx in range(init_epoch + 1, cfg.train.n_epochs + 1):
        epoch_start = time.time()
        model.train()
        lr_scheduler.step()

        total_loss = 0
        total_cd_fine = 0
        n_batches = len(train_data_loader)
        lr = optimizer.param_groups[0]['lr']

        for batch_idx, batch in enumerate(train_data_loader):
            partial, gt, _, _ = unpack_batch(batch, dataset_name)

            # Forward
            pcds_pred = model(partial)

            # Compute loss
            try:
                # Try model's own loss function first
                model_ref = model.module if hasattr(model, 'module') else model
                loss = model_ref.get_loss(pcds_pred, gt, epoch_idx)
                if isinstance(loss, tuple):
                    loss_total = loss[0]
                else:
                    loss_total = loss
            except (NotImplementedError, AttributeError):
                # Fall back to default pyramid CD loss
                loss_total, losses, _ = pyramid_loss(pcds_pred, partial, gt, sqrt=True)

            # Backward
            optimizer.zero_grad()
            loss_total.backward()
            optimizer.step()

            total_loss += loss_total.item()
            n_itr = (epoch_idx - 1) * n_batches + batch_idx

        # End of epoch
        avg_loss = total_loss / n_batches
        epoch_time = time.time() - epoch_start

        log(f'[Epoch {epoch_idx}/{cfg.train.n_epochs}] LR={lr:.6f} '
            f'Time={epoch_time:.1f}s Loss={avg_loss:.4f}', train_log)

        # Validate
        cd_eval = validate(model, val_data_loader, cfg, dataset_name)
        log(f'  Val CD = {cd_eval:.4f} (best: {best_metrics:.4f} @ epoch {best_epoch})', test_log)

        # Save checkpoint
        if cd_eval < best_metrics:
            best_metrics = cd_eval
            best_epoch = epoch_idx
            output_path = os.path.join(ckpt_dir, 'ckpt-best.pth')
            torch.save({
                'epoch_index': epoch_idx,
                'best_metrics': best_metrics,
                'model': model.state_dict()
            }, output_path)
            print(f'  ★ New best! Saved to {output_path}')

        # Periodic save
        if epoch_idx % cfg.train.save_freq == 0:
            output_path = os.path.join(ckpt_dir, f'ckpt-epoch-{epoch_idx:03d}.pth')
            torch.save({
                'epoch_index': epoch_idx,
                'best_metrics': best_metrics,
                'model': model.state_dict()
            }, output_path)

    train_log.close()
    test_log.close()
    print(f'\nTraining complete. Best CD = {best_metrics:.4f} at epoch {best_epoch}')


# =====================================================================
# Validation
# =====================================================================

def validate(model, val_data_loader, cfg, dataset_name):
    """Validate and return the fine-level CD metric."""
    model.eval()
    test_losses = AverageMeter(['cd_fine'])

    with torch.no_grad():
        for batch in val_data_loader:
            partial, gt, _, _ = unpack_batch(batch, dataset_name)

            pcds_pred = model(partial.contiguous())

            # Compute CD for finest prediction
            loss_total, losses, _ = pyramid_loss(pcds_pred, partial, gt, sqrt=True)

            # losses[-2] is cd_fine (last stage before partial_matching)
            cd_fine = losses[min(len(pcds_pred) - 1, 3)].item() * 1e3
            test_losses.update([cd_fine])

    model.train()
    return test_losses.avg(0)


# =====================================================================
# CLI
# =====================================================================

def parse_args():
    parser = argparse.ArgumentParser(description='Unified Point Cloud Completion Training')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to YAML config file (e.g., configs/pcn.yaml)')
    parser.add_argument('--model_name', type=str, default=None,
                        help='Override model name from config')
    parser.add_argument('--dataset', type=str, default=None,
                        help='Override dataset name from config')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Override batch size')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Override number of epochs')
    parser.add_argument('--lr', type=float, default=None,
                        help='Override learning rate')
    parser.add_argument('--num_workers', type=int, default=None,
                        help='Override number of data workers')
    parser.add_argument('--gpu', type=str, default=None,
                        help='GPU device IDs (e.g., "0,1")')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--weights', type=str, default=None,
                        help='Path to pretrained weights (no resume, just load weights)')
    parser.add_argument('--dummy', action='store_true',
                        help='Use dummy random data for pipeline testing (no real dataset needed)')
    return parser.parse_args()


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def main():
    args = parse_args()

    # Load config
    cfg = cfg_from_args(args, args.config)

    # Set GPU
    gpu = getattr(cfg.general, 'gpu', '0')
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu

    # Set seed
    seed = getattr(cfg.general, 'seed', 1128)
    set_seed(seed)

    print(f'CUDA available: {torch.cuda.is_available()}')
    print(f'GPU devices: {gpu}')

    # Build model
    model = build_model(cfg)
    if torch.cuda.is_available():
        model = nn.DataParallel(model).cuda()
    print(f'Model parameters: {sum(p.numel() for p in model.parameters()):,}')

    # Load pretrained weights (optional, without resume)
    if args.weights:
        print(f'Loading pretrained weights from {args.weights} ...')
        checkpoint = torch.load(args.weights)
        model.load_state_dict(checkpoint['model'])

    # Build data loaders
    train_data_loader, val_data_loader = build_data_loaders(cfg, dummy=args.dummy)
    print(f'Train samples: {len(train_data_loader.dataset)}, '
          f'Val samples: {len(val_data_loader.dataset)}')
    if args.dummy:
        print('⚡ DUMMY MODE: using random point cloud data for pipeline testing')

    # Train
    torch.backends.cudnn.benchmark = True
    train(model, train_data_loader, val_data_loader, cfg, resume_path=args.resume)


if __name__ == '__main__':
    main()
