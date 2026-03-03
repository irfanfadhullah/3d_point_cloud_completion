"""
=======================================================================
Unified Testing Script for Point Cloud Completion
=======================================================================

Single entry point that evaluates any registered model on any dataset.

Usage:
    python test.py --config configs/pcn.yaml --checkpoint path/to/ckpt-best.pth
    python test.py --config configs/seedformer.yaml --checkpoint ckpt.pth --output
    python test.py --config configs/snowflakenet.yaml --checkpoint ckpt.pth --mode median

    # Dummy mode — test pipeline without real data or checkpoint:
    python test.py --config configs/dummy.yaml --dummy

All model/dataset parameters are specified in the YAML config.
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from easydict import EasyDict as edict
from PIL import Image

import utils.data_loaders
import utils.helpers
from utils.average_meter import AverageMeter
from utils.metrics import Metrics
from utils.loss import pyramid_loss
from utils.ply import write_ply
from configs.config import load_config, cfg_from_args

# Reuse builders from train.py
from train import build_model, build_data_loaders, unpack_batch, _build_legacy_cfg


# =====================================================================
# Testing
# =====================================================================

def test(model, test_data_loader, cfg, outdir=None):
    """
    Unified testing loop for all models.

    Computes per-category metrics, saves PLY output files and images
    if outdir is specified.
    """
    torch.backends.cudnn.benchmark = True
    model.eval()

    dataset_name = getattr(cfg.dataset, 'test', cfg.dataset.train)
    mode = getattr(cfg.test, 'mode', 'median')

    # ShapeNet55/34 uses multi-viewpoint evaluation
    if dataset_name in ('ShapeNet55', 'ShapeNet34'):
        return test_shapenet55(model, test_data_loader, cfg, outdir, mode)

    # Standard evaluation
    n_samples = len(test_data_loader)
    test_losses = AverageMeter(['cdc', 'cd1', 'cd2', 'cd3', 'partial_matching'])
    test_metrics = AverageMeter(Metrics.names())
    mclass_metrics = AverageMeter(Metrics.names())
    category_metrics = dict()

    print(f'\nEvaluating {cfg.model.name} on {dataset_name} ({n_samples} batches)...\n')

    for model_idx, batch in enumerate(test_data_loader):
        partial, gt, taxonomy_ids, model_ids = unpack_batch(batch, dataset_name)

        if taxonomy_ids is not None:
            taxonomy_id = taxonomy_ids[0] if isinstance(taxonomy_ids[0], str) else taxonomy_ids[0]
        else:
            taxonomy_id = 'unknown'

        with torch.no_grad():
            pcds_pred = model(partial.contiguous())
            loss_total, losses, _ = pyramid_loss(pcds_pred, partial, gt, sqrt=True)

            # Per-stage losses
            cdc = losses[0].item() * 1e3
            cd1 = losses[1].item() * 1e3
            cd2 = losses[2].item() * 1e3
            cd3 = losses[3].item() * 1e3
            partial_matching = losses[4].item() * 1e3

            test_losses.update([cdc, cd1, cd2, cd3, partial_matching])

            # Full metrics
            _metrics = Metrics.get(pcds_pred[-1], gt)
            test_metrics.update(_metrics)

            if taxonomy_id not in category_metrics:
                category_metrics[taxonomy_id] = AverageMeter(Metrics.names())
            category_metrics[taxonomy_id].update(_metrics)

            # Output files
            if outdir and model_ids is not None:
                _save_outputs(outdir, taxonomy_id, model_ids, pcds_pred, gt, partial)

    # Print results
    _print_results(test_metrics, mclass_metrics, category_metrics, test_losses)

    return test_losses.avg(3)  # cd_fine


def test_shapenet55(model, test_data_loader, cfg, outdir=None, mode='median'):
    """
    Testing for ShapeNet-55/34 with multi-viewpoint evaluation.

    Generates partial clouds online from 8 fixed viewpoints.
    """
    from models.utils import fps_subsample

    torch.backends.cudnn.benchmark = True
    model.eval()

    crop_ratio = {'easy': 1/4, 'median': 1/2, 'hard': 3/4}
    choice = [
        torch.Tensor([1, 1, 1]), torch.Tensor([1, 1, -1]),
        torch.Tensor([1, -1, 1]), torch.Tensor([-1, 1, 1]),
        torch.Tensor([-1, -1, 1]), torch.Tensor([-1, 1, -1]),
        torch.Tensor([1, -1, -1]), torch.Tensor([-1, -1, -1])
    ]

    n_samples = len(test_data_loader)
    test_losses = AverageMeter(['cdc', 'cd1', 'cd2', 'cd3', 'partial_matching'])
    test_metrics = AverageMeter(Metrics.names())
    mclass_metrics = AverageMeter(Metrics.names())
    category_metrics = dict()

    print(f'\nEvaluating {cfg.model.name} on ShapeNet55 (mode: {mode}, '
          f'{n_samples} batches, 8 viewpoints each)...\n')

    for model_idx, batch in enumerate(test_data_loader):
        taxonomy_ids, model_ids, data = batch

        taxonomy_id = taxonomy_ids[0] if isinstance(taxonomy_ids[0], str) else taxonomy_ids[0]

        with torch.no_grad():
            for k, v in data.items():
                data[k] = utils.helpers.var_or_cuda(v)

            gt = data['gtcloud']
            _, npoints, _ = gt.shape
            num_crop = int(npoints * crop_ratio[mode])

            for partial_id, item in enumerate(choice):
                partial, _ = utils.helpers.seprate_point_cloud(
                    gt, npoints, num_crop, fixed_points=item
                )
                partial = fps_subsample(partial, 2048)

                pcds_pred = model(partial.contiguous())
                loss_total, losses, _ = pyramid_loss(pcds_pred, partial, gt, sqrt=False)

                cdc = losses[0].item() * 1e3
                cd1 = losses[1].item() * 1e3
                cd2 = losses[2].item() * 1e3
                cd3 = losses[3].item() * 1e3
                partial_matching = losses[4].item() * 1e3
                test_losses.update([cdc, cd1, cd2, cd3, partial_matching])

                _metrics = Metrics.get(pcds_pred[-1], gt)
                test_metrics.update(_metrics)

                if taxonomy_id not in category_metrics:
                    category_metrics[taxonomy_id] = AverageMeter(Metrics.names())
                category_metrics[taxonomy_id].update(_metrics)

                # Save outputs
                if outdir and model_ids is not None:
                    pred = pcds_pred[-1]
                    for mm, model_name in enumerate(model_ids):
                        cat_dir = os.path.join(outdir, taxonomy_id)
                        os.makedirs(cat_dir, exist_ok=True)
                        output_file = os.path.join(cat_dir, f'{model_name}_{partial_id:02d}')
                        write_ply(output_file + '_pred.ply',
                                  pred[mm].detach().cpu().numpy(), ['x', 'y', 'z'])
                        write_ply(output_file + '_gt.ply',
                                  gt[mm].detach().cpu().numpy(), ['x', 'y', 'z'])
                        write_ply(output_file + '_partial.ply',
                                  partial[mm].detach().cpu().numpy(), ['x', 'y', 'z'])

    _print_results(test_metrics, mclass_metrics, category_metrics, test_losses)

    return test_losses.avg(3)


# =====================================================================
# Output Helpers
# =====================================================================

def _save_outputs(outdir, taxonomy_id, model_ids, pcds_pred, gt, partial):
    """Save predicted PLY files and optional images."""
    cat_dir = os.path.join(outdir, taxonomy_id)
    os.makedirs(cat_dir, exist_ok=True)

    pred = pcds_pred[-1]
    for mm, model_name in enumerate(model_ids):
        output_file = os.path.join(cat_dir, model_name)
        write_ply(output_file + '_pred.ply',
                  pred[mm].detach().cpu().numpy(), ['x', 'y', 'z'])
        write_ply(output_file + '_gt.ply',
                  gt[mm].detach().cpu().numpy(), ['x', 'y', 'z'])
        write_ply(output_file + '_partial.ply',
                  partial[mm].detach().cpu().numpy(), ['x', 'y', 'z'])

        # Optional: render 3-view image
        try:
            import pointnet_utils.pc_util as pc_util
            img_dir = os.path.join(outdir, taxonomy_id + '_images')
            os.makedirs(img_dir, exist_ok=True)
            img_filename = os.path.join(img_dir, model_name + '.jpg')
            output_img = pc_util.point_cloud_three_views(
                pred[mm].detach().cpu().numpy(), diameter=7
            )
            output_img = (output_img * 255).astype('uint8')
            im = Image.fromarray(output_img)
            im.save(img_filename)
        except (ImportError, Exception):
            pass  # Skip image rendering if pc_util not available


def _print_results(test_metrics, mclass_metrics, category_metrics, test_losses):
    """Print formatted test results."""
    print('\n' + '=' * 70)
    print(' TEST RESULTS')
    print('=' * 70)

    header = 'Taxonomy\t#Sample\t' + '\t'.join(test_metrics.items)
    print(header)

    for taxonomy_id in sorted(category_metrics.keys()):
        cat = category_metrics[taxonomy_id]
        msg = f'{taxonomy_id}\t{cat.count(0)}\t'
        msg += '\t'.join([f'{v:.4f}' for v in cat.avg()])
        mclass_metrics.update(cat.avg())
        print(msg)

    print(f'Overall\t{test_metrics.count(0)}\t'
          + '\t'.join([f'{v:.4f}' for v in test_metrics.avg()]))
    print(f'MeanClass\t\t'
          + '\t'.join([f'{v:.4f}' for v in mclass_metrics.avg()]))

    print(f'\nPer-stage losses: CDC={test_losses.avg(0):.4f} '
          f'CD1={test_losses.avg(1):.4f} CD2={test_losses.avg(2):.4f} '
          f'CD_fine={test_losses.avg(3):.4f}')
    print('=' * 70)


# =====================================================================
# CLI
# =====================================================================

def parse_args():
    parser = argparse.ArgumentParser(description='Unified Point Cloud Completion Testing')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to YAML config file')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to model checkpoint (.pth), not needed with --dummy')
    parser.add_argument('--model_name', type=str, default=None,
                        help='Override model name')
    parser.add_argument('--dataset', type=str, default=None,
                        help='Override test dataset name')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Override test batch size')
    parser.add_argument('--num_workers', type=int, default=None,
                        help='Override number of data workers')
    parser.add_argument('--gpu', type=str, default=None,
                        help='GPU device IDs')
    parser.add_argument('--mode', type=str, default=None,
                        help='ShapeNet55 eval mode: easy/median/hard')
    parser.add_argument('--output', action='store_true',
                        help='Save output PLY files and images')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Custom output directory for results')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed')
    parser.add_argument('--dummy', action='store_true',
                        help='Use dummy random data for pipeline testing (no checkpoint needed)')
    return parser.parse_args()


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    args = parse_args()

    # Validate args
    if not args.dummy and not args.checkpoint:
        print('ERROR: --checkpoint is required unless using --dummy mode')
        return

    # Load config
    cfg = cfg_from_args(args, args.config)

    # Override test-specific settings
    if args.mode:
        cfg.test.mode = args.mode
    if args.batch_size:
        cfg.test.batch_size = args.batch_size

    # Set GPU
    gpu = getattr(cfg.general, 'gpu', '0')
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu

    # Set seed
    seed = getattr(cfg.general, 'seed', 1128)
    set_seed(seed)

    print(f'CUDA available: {torch.cuda.is_available()}')

    # Build model
    model = build_model(cfg)
    if torch.cuda.is_available():
        model = nn.DataParallel(model).cuda()

    # Load checkpoint (skip in dummy mode)
    if args.checkpoint and not args.dummy:
        print(f'Loading checkpoint: {args.checkpoint}')
        checkpoint = torch.load(args.checkpoint)
        model.load_state_dict(checkpoint['model'])
        epoch = checkpoint.get('epoch_index', 0)
        best_metrics = checkpoint.get('best_metrics', 'N/A')
        print(f'Checkpoint epoch: {epoch}, best_metrics: {best_metrics}')
    elif args.dummy:
        print('⚡ DUMMY MODE: skipping checkpoint, using random weights')

    # Build test data loader
    test_batch_size = getattr(cfg.test, 'batch_size', 1)
    cfg.train.batch_size = test_batch_size  # data loader builder reads from train

    _, test_data_loader = build_data_loaders(cfg, dummy=args.dummy)

    if args.dummy:
        print(f'⚡ DUMMY MODE: using random data ({len(test_data_loader.dataset)} samples)')

    # Output directory
    outdir = None
    if args.output:
        outdir = args.output_dir or os.path.join(
            cfg.output.get('test_path', 'test'),
            cfg.model.name, 'results'
        )
        os.makedirs(outdir, exist_ok=True)
        print(f'Saving output to: {outdir}')

    # Test
    cd_result = test(model, test_data_loader, cfg, outdir=outdir)
    print(f'\nFinal CD (fine): {cd_result:.4f}')


if __name__ == '__main__':
    main()
