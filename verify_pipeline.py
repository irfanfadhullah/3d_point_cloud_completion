#!/usr/bin/env python
"""
=======================================================================
Pipeline Verification Script
=======================================================================

Tests that the train/test pipeline works for any model by running
a short dummy training loop with random data.
Dummy mode needs no GPU/extensions; real model verification may.

Usage:
    # Test a specific model:
    python verify_pipeline.py --config configs/pcn.yaml

    # Test the dummy model (fastest, no CUDA extensions needed):
    python verify_pipeline.py --config configs/dummy.yaml

    # Test all models with isolated subprocesses (default):
    python verify_pipeline.py --all

    # Test all models in legacy single-process mode:
    python verify_pipeline.py --all --no-isolate

    # Test with custom number of iterations:
    python verify_pipeline.py --config configs/pcn.yaml --iters 5
"""

import argparse
import gc
import json
import os
import subprocess
import sys
import time
import traceback

import torch

from configs.config import load_config
from train import build_model

RESULT_PREFIX = '__RESULT_JSON__='


def _new_result(config_path):
    return {
        'config': os.path.basename(config_path),
        'model_name': 'unknown',
        'success': False,
        'status': 'FAIL',
        'n_params': 0,
        'time': 0.0,
        'error': None,
        'attempts': 1,
        'debug_error': None,
        'skipped': False,
        'output_shapes': [],
    }


def _make_skip_result(config_path, reason):
    result = _new_result(config_path)
    result['status'] = 'SKIP'
    result['skipped'] = True
    result['success'] = False
    result['error'] = reason
    try:
        cfg = load_config(config_path)
        result['model_name'] = cfg.model.name
    except Exception:
        result['model_name'] = '-'
    return result


def _finalize_status(result):
    if result.get('skipped'):
        result['status'] = 'SKIP'
    else:
        result['status'] = 'PASS' if result.get('success') else 'FAIL'
    return result


def verify_single_model(config_path, n_iters=3, verbose=True):
    """
    Verify that a model config works end-to-end:
        1. Config loads correctly
        2. Model instantiates from registry
        3. Forward pass produces valid output shapes
        4. Loss computation succeeds
        5. Backward pass + optimizer step succeeds

    Args:
        config_path: Path to the model YAML config.
        n_iters: Number of training iterations to simulate.
        verbose: Print detailed progress.

    Returns:
        dict with fields including success/model_name/n_params/time/error.
    """
    result = _new_result(config_path)
    start = time.time()

    try:
        # 1. Load config
        cfg = load_config(config_path)
        result['model_name'] = cfg.model.name
        if verbose:
            print(f'\n{"="*50}')
            print(f' Verifying: {cfg.model.name} ({os.path.basename(config_path)})')
            print(f'{"="*50}')

        # 2. Build model
        if verbose:
            print('  [1/5] Building model...', end=' ')
        model = build_model(cfg)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        n_params = sum(p.numel() for p in model.parameters())
        result['n_params'] = n_params
        if verbose:
            print(f'OK ({n_params:,} params, device={device})')

        # 3. Create dummy input
        if verbose:
            print('  [2/5] Forward pass...', end=' ')
        B = 2
        N_in = 2048
        n_gt = getattr(cfg.model.config, 'num_pred', 16384)
        partial = torch.randn(B, N_in, 3, device=device)
        gt = torch.randn(B, n_gt, 3, device=device)

        # Forward (with OOM retry at reduced batch size)
        model.train()
        try:
            pcds_pred = model(partial)
        except torch.cuda.OutOfMemoryError:
            # Retry with batch_size=1 for memory-heavy models
            torch.cuda.empty_cache()
            B = 1
            partial = torch.randn(B, N_in, 3, device=device)
            gt = torch.randn(B, n_gt, 3, device=device)
            if verbose:
                print('(OOM, retrying B=1)...', end=' ')
            pcds_pred = model(partial)

        # Validate output
        if not isinstance(pcds_pred, (tuple, list)):
            pcds_pred = (pcds_pred,)

        output_shapes = [p.shape for p in pcds_pred]
        result['output_shapes'] = [str(s) for s in output_shapes]

        # Check all outputs are (B, M, 3)
        for i, p in enumerate(pcds_pred):
            assert p.dim() == 3, f'Output {i} has {p.dim()} dims, expected 3'
            assert p.shape[0] == B, f'Output {i} batch size {p.shape[0]} != {B}'
            assert p.shape[2] == 3, f'Output {i} has {p.shape[2]} channels, expected 3'

        if verbose:
            shapes_str = ', '.join([f'{s}' for s in output_shapes])
            print(f'OK -> [{shapes_str}]')

        # 4. Loss computation
        if verbose:
            print('  [3/5] Loss computation...', end=' ')

        # Adjust gt size to match finest prediction
        finest = pcds_pred[-1]
        if gt.shape[1] != finest.shape[1]:
            if gt.shape[1] > finest.shape[1]:
                gt = gt[:, :finest.shape[1], :]
            else:
                gt = torch.randn(B, finest.shape[1], 3, device=device)
        gt = gt.contiguous()

        # Try model's own loss
        try:
            model_ref = model
            loss = model_ref.get_loss(pcds_pred, gt, epoch=1)
            if isinstance(loss, tuple):
                loss_val = loss[0]
            else:
                loss_val = loss
            if verbose:
                print(f'OK (model loss = {loss_val.item():.4f})')
        except (NotImplementedError, AttributeError):
            # Fall back to simple CD
            from Chamfer3D.dist_chamfer_3D import chamfer_3DDist
            chamfer_dist = chamfer_3DDist()
            d1, d2, _, _ = chamfer_dist(finest, gt)
            loss_val = (torch.mean(d1) + torch.mean(d2)) * 1e3
            if verbose:
                print(f'OK (default CD = {loss_val.item():.4f})')

        # 5. Backward + optimizer
        if verbose:
            print('  [4/5] Backward pass...', end=' ')
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        optimizer.zero_grad()
        loss_val.backward()
        optimizer.step()
        if verbose:
            print('OK')

        # Multiple iterations
        if verbose:
            print(f'  [5/5] Running {n_iters} iterations...', end=' ')

        for _ in range(n_iters):
            partial = torch.randn(B, N_in, 3, device=device)
            gt = torch.randn(B, pcds_pred[-1].shape[1], 3, device=device)

            pcds_pred = model(partial)

            try:
                model_ref = model
                loss = model_ref.get_loss(pcds_pred, gt, epoch=1)
                loss_val = loss[0] if isinstance(loss, tuple) else loss
            except (NotImplementedError, AttributeError):
                d1, d2, _, _ = chamfer_dist(pcds_pred[-1], gt)
                loss_val = (torch.mean(d1) + torch.mean(d2)) * 1e3

            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()

        if verbose:
            print(f'OK (final loss = {loss_val.item():.4f})')

        result['success'] = True

    except Exception as e:
        result['error'] = str(e)
        if verbose:
            print(f'\n  x FAILED: {e}')
            traceback.print_exc()
    finally:
        # GPU cleanup for single-run safety.
        try:
            if 'model' in locals():
                del model
            if 'optimizer' in locals():
                del optimizer
            if 'pcds_pred' in locals():
                del pcds_pred
            if 'partial' in locals():
                del partial
            if 'gt' in locals():
                del gt
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    result['time'] = time.time() - start
    return _finalize_status(result)


def _parse_result_json(stdout):
    for line in reversed(stdout.splitlines()):
        if line.startswith(RESULT_PREFIX):
            payload = line[len(RESULT_PREFIX):]
            return json.loads(payload)
    raise ValueError('Worker did not emit JSON result marker')


def _run_single_isolated(config_path, n_iters, gpu, timeout, extra_env=None):
    cmd = [
        sys.executable,
        os.path.abspath(__file__),
        '--config',
        config_path,
        '--iters',
        str(n_iters),
        '--gpu',
        str(gpu),
        '--json-result',
    ]
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu)
    if extra_env:
        env.update(extra_env)

    start = time.time()
    try:
        proc = subprocess.run(
            cmd,
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        elapsed = time.time() - start

        try:
            result = _parse_result_json(proc.stdout)
        except Exception as parse_err:
            result = _new_result(config_path)
            result['time'] = elapsed
            result['error'] = (
                f'Failed to parse worker result (rc={proc.returncode}): {parse_err}. '
                f'stdout_tail={proc.stdout[-500:]!r} stderr_tail={proc.stderr[-500:]!r}'
            )
            return _finalize_status(result)

        # Preserve worker timing when available; fallback to parent timing.
        if not result.get('time'):
            result['time'] = elapsed
        return _finalize_status(result)

    except subprocess.TimeoutExpired as exc:
        result = _new_result(config_path)
        result['time'] = time.time() - start
        result['error'] = f'Timed out after {timeout}s while verifying {os.path.basename(config_path)}'
        if exc.stdout or exc.stderr:
            out_tail = (exc.stdout or '')[-300:]
            err_tail = (exc.stderr or '')[-300:]
            result['error'] += f'. stdout_tail={out_tail!r} stderr_tail={err_tail!r}'
        return _finalize_status(result)
    except Exception as e:
        result = _new_result(config_path)
        result['time'] = time.time() - start
        result['error'] = f'Failed to launch worker subprocess: {e}'
        return _finalize_status(result)


def _load_skip_set(skip_configs_arg, quarantine_file):
    skip_set = set()

    if skip_configs_arg:
        parts = [p.strip() for p in skip_configs_arg.split(',') if p.strip()]
        skip_set.update(parts)

    if quarantine_file:
        if not os.path.isfile(quarantine_file):
            raise FileNotFoundError(f'Quarantine file not found: {quarantine_file}')
        with open(quarantine_file, 'r', encoding='utf-8') as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith('#'):
                    continue
                skip_set.add(line)

    return skip_set


def _print_summary(results):
    print(f'\n\n{"="*70}')
    print(' VERIFICATION SUMMARY')
    print(f'{"="*70}')
    print(f'{"Config":<25} {"Model":<20} {"Status":<10} {"Params":<15} {"Time":<8}')
    print('-' * 70)

    passed = 0
    skipped = 0
    for r in results:
        status = {
            'PASS': 'PASS',
            'FAIL': 'FAIL',
            'SKIP': 'SKIP',
        }.get(r.get('status', 'FAIL'), 'FAIL')
        params = f'{r["n_params"]:,}' if r.get('n_params') else '-'
        print(f'{r["config"]:<25} {r.get("model_name", "unknown"):<20} {status:<10} {params:<15} {r.get("time", 0.0):.1f}s')

        if r.get('status') == 'PASS':
            passed += 1
        if r.get('status') == 'SKIP':
            skipped += 1

    total = len(results)
    ran = total - skipped
    failed = ran - passed

    print(f'\n{passed}/{ran} models passed verification ({failed} failed, {skipped} skipped, {total} total configs)')

    if failed > 0:
        print('\nFailed models:')
        for r in results:
            if r.get('status') == 'FAIL':
                error = r.get('error') or 'Unknown error'
                if r.get('debug_error'):
                    error = f'{error} | debug retry: {r["debug_error"]}'
                print(f'  {r.get("model_name", "unknown")}: {error}')

    if skipped > 0:
        print('\nSkipped models:')
        for r in results:
            if r.get('status') == 'SKIP':
                print(f'  {r.get("config")}: {r.get("error", "Skipped")}')

    return failed


def _run_all_legacy(config_paths, n_iters, skip_set):
    results = []
    for config_path in config_paths:
        base = os.path.basename(config_path)
        if base in skip_set:
            results.append(_make_skip_result(config_path, 'Skipped by user configuration'))
            continue
        result = verify_single_model(config_path, n_iters=n_iters)
        results.append(_finalize_status(result))
    return results


def _run_all_isolated(config_paths, args, skip_set):
    results = []

    for config_path in config_paths:
        base = os.path.basename(config_path)
        if base in skip_set:
            results.append(_make_skip_result(config_path, 'Skipped by user configuration'))
            continue

        primary = _run_single_isolated(
            config_path=config_path,
            n_iters=args.iters,
            gpu=args.gpu,
            timeout=args.timeout,
        )
        primary['attempts'] = 1

        if primary.get('status') == 'FAIL' and args.debug_retry:
            debug = _run_single_isolated(
                config_path=config_path,
                n_iters=args.iters,
                gpu=args.gpu,
                timeout=args.timeout,
                extra_env={'CUDA_LAUNCH_BLOCKING': '1'},
            )
            primary['attempts'] = 2
            if debug.get('status') == 'FAIL':
                primary['debug_error'] = debug.get('error')

        results.append(_finalize_status(primary))

    return results


def main():
    parser = argparse.ArgumentParser(description='Pipeline Verification')
    parser.add_argument('--config', type=str, default=None,
                        help='Verify a single model config')
    parser.add_argument('--all', action='store_true',
                        help='Verify ALL model configs in configs/')
    parser.add_argument('--iters', type=int, default=3,
                        help='Number of training iterations per model')
    parser.add_argument('--gpu', type=str, default='0',
                        help='GPU device ID')
    parser.add_argument('--no-isolate', action='store_true',
                        help='Use legacy single-process --all behavior')
    parser.add_argument('--skip-configs', type=str, default='',
                        help='Comma-separated config basenames to skip (e.g. grnet.yaml,iaet.yaml)')
    parser.add_argument('--quarantine-file', type=str, default=None,
                        help='Path to file listing config basenames to skip (one per line)')
    parser.add_argument('--timeout', type=int, default=180,
                        help='Per-model timeout in seconds for isolated workers')
    parser.add_argument('--debug-retry', dest='debug_retry', action='store_true', default=True,
                        help='On isolated failure, retry once with CUDA_LAUNCH_BLOCKING=1 (default: enabled)')
    parser.add_argument('--no-debug-retry', dest='debug_retry', action='store_false',
                        help='Disable isolated debug retry')
    parser.add_argument('--json-result', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    if args.config:
        result = verify_single_model(args.config, n_iters=args.iters, verbose=not args.json_result)

        if args.json_result:
            print(RESULT_PREFIX + json.dumps(result))
            sys.exit(0 if result.get('success') else 1)

        status = 'PASS' if result['success'] else 'FAIL'
        print(f'\n{status}: {result["model_name"]} ({result["n_params"]:,} params, {result["time"]:.1f}s)')
        if result['error']:
            print(f'  Error: {result["error"]}')
        return

    if args.all:
        configs_dir = os.path.join(os.path.dirname(__file__), 'configs')
        configs = sorted([
            f for f in os.listdir(configs_dir)
            if f.endswith('.yaml') and f != 'default.yaml'
        ])
        config_paths = [os.path.join(configs_dir, cfg) for cfg in configs]

        try:
            skip_set = _load_skip_set(args.skip_configs, args.quarantine_file)
        except Exception as e:
            print(f'Error loading skip/quarantine configuration: {e}')
            sys.exit(2)

        if args.no_isolate:
            print('Running in legacy single-process mode (--no-isolate).')
            results = _run_all_legacy(config_paths, args.iters, skip_set)
        else:
            results = _run_all_isolated(config_paths, args, skip_set)

        failed = _print_summary(results)
        sys.exit(1 if failed > 0 else 0)

    # Default: verify dummy model
    print('No --config or --all specified. Verifying DummyModel...')
    dummy_config = os.path.join(os.path.dirname(__file__), 'configs', 'dummy.yaml')
    result = verify_single_model(dummy_config, n_iters=args.iters)
    status = 'PASS' if result['success'] else 'FAIL'
    print(f'\n{status}: {result["model_name"]} ({result["n_params"]:,} params, {result["time"]:.1f}s)')

    if result['success']:
        print('\nDummy model works! To test actual models:')
        print('  python verify_pipeline.py --config configs/pcn.yaml')
        print('  python verify_pipeline.py --all')


if __name__ == '__main__':
    main()
