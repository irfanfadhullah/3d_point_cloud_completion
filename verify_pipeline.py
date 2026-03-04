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
    
    # Test with profiling (FLOPs, Max Batch Size, VRAM, Big O, params):
    python verify_pipeline.py --config configs/dummy.yaml --profile

    # Profiling with mixed precision:
    python verify_pipeline.py --config configs/dummy.yaml --profile --precision fp16
    python verify_pipeline.py --config configs/dummy.yaml --profile --precision bf16
    python verify_pipeline.py --config configs/dummy.yaml --profile --precision tf32

    # Backward compatible --amp flag (alias for --precision fp16):
    python verify_pipeline.py --config configs/dummy.yaml --profile --amp
"""

import argparse
import gc
import json
import math
import os
import subprocess
import sys
import time
import traceback
from contextlib import nullcontext

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
        'trainable_params': 0,
        'frozen_params': 0,
        'param_size_mb': 0.0,
        'top_modules': [],
        'flops': '-',
        'complexity': '-',
        'max_bs_train': '-',
        'max_bs_eval': '-',
        'vram_train': '-',
        'vram_eval': '-',
        'est_vram_train': '-',
        'est_vram_eval': '-',
        'gpu_name': '-',
        'gpu_vram_total': '-',
        'precision': 'fp32',
        'requested_precision': 'fp32',
        'effective_precision': 'fp32',
        'precision_fallback_used': False,
        'precision_fallback_from': None,
        'precision_fallback_to': None,
        'precision_fallback_reason': None,
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


def _detect_gpu_info(verbose=True):
    """Detect and display GPU information."""
    info = {'gpu_name': '-', 'gpu_vram_total': '-', 'gpu_count': 0}
    if not torch.cuda.is_available():
        if verbose:
            print('  GPU: Not available (running on CPU)')
        return info

    info['gpu_count'] = torch.cuda.device_count()
    dev = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(dev)
    info['gpu_name'] = props.name
    info['gpu_vram_total'] = f"{props.total_memory / 1024**3:.1f}GB"

    if verbose:
        print(f'\n  {"─"*48}')
        print(f'  GPU Information')
        print(f'  {"─"*48}')
        print(f'  Name          : {props.name}')
        print(f'  VRAM          : {props.total_memory / 1024**3:.1f} GB')
        print(f'  Compute Cap.  : {props.major}.{props.minor}')
        print(f'  GPU Count     : {info["gpu_count"]}')
        print(f'  CUDA Version  : {torch.version.cuda}')
        print(f'  PyTorch       : {torch.__version__}')
        try:
            bf16_ok = torch.cuda.is_bf16_supported()
        except Exception:
            bf16_ok = False
        print(f'  BF16 Support  : {"Yes" if bf16_ok else "No"}')
        print(f'  TF32 Support  : {"Yes" if props.major >= 8 else "No"}')
        print(f'  {"─"*48}')

    return info


def _get_amp_settings(precision):
    """Map precision string to (use_amp, amp_dtype). Also sets TF32 flags."""
    if precision == 'fp16':
        return True, torch.float16
    elif precision == 'bf16':
        return True, torch.bfloat16
    elif precision == 'tf32':
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        return False, torch.float32
    else:  # fp32
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        return False, torch.float32


def _autocast_context(device, amp_enabled, amp_dtype):
    if not amp_enabled or device.type != 'cuda':
        return nullcontext()
    return torch.amp.autocast(device_type='cuda', dtype=amp_dtype)


def _make_grad_scaler(device, amp_enabled, amp_dtype):
    use_scaler = device.type == 'cuda' and amp_enabled and amp_dtype == torch.float16
    return torch.cuda.amp.GradScaler(enabled=use_scaler)


def _is_precision_mismatch_error(error, precision):
    if not error or precision not in ('fp16', 'bf16'):
        return False

    txt = error.lower()
    if precision == 'fp16':
        return (
            ('half' in txt and 'float' in txt and ('expected' in txt or 'type' in txt))
            or "not implemented for 'half'" in txt
            or 'cuda.halftensor' in txt
        )
    return (
        ('bfloat16' in txt and 'float' in txt and ('expected' in txt or 'type' in txt))
        or "not implemented for 'bfloat16'" in txt
        or 'cuda.bfloattensor' in txt
    )


def _format_precision_display(result):
    req = str(result.get('requested_precision') or result.get('precision') or 'fp32')
    eff = str(result.get('effective_precision') or result.get('precision') or req)
    return req if req == eff else f'{req}->{eff}'


def _short_error(error, max_len=140):
    if not error:
        return 'Unknown error'
    first = str(error).splitlines()[0].strip()
    if len(first) > max_len:
        return f'{first[:max_len - 3]}...'
    return first


def _profile_parameters(model, verbose=True):
    """Detailed parameter breakdown: trainable, frozen, top modules."""
    total = 0
    trainable = 0
    frozen = 0
    module_params = {}

    for name, p in model.named_parameters():
        n = p.numel()
        total += n
        if p.requires_grad:
            trainable += n
        else:
            frozen += n
        top_mod = name.split('.')[0]
        module_params[top_mod] = module_params.get(top_mod, 0) + n

    sorted_mods = sorted(module_params.items(), key=lambda x: -x[1])[:5]
    param_size_mb = total * 4 / (1024**2)

    result = {
        'total': total,
        'trainable': trainable,
        'frozen': frozen,
        'param_size_mb': round(param_size_mb, 2),
        'top_modules': sorted_mods,
    }

    if verbose:
        print(f'\n  {"─"*48}')
        print(f'  Parameter Breakdown')
        print(f'  {"─"*48}')
        print(f'  Total       : {total:>12,}')
        print(f'  Trainable   : {trainable:>12,}  ({100*trainable/max(total,1):.1f}%)')
        print(f'  Frozen      : {frozen:>12,}  ({100*frozen/max(total,1):.1f}%)')
        print(f'  Size (FP32) : {param_size_mb:>10.2f} MB')
        print(f'  {"─"*48}')
        print(f'  Top Modules by Parameters:')
        for mod_name, cnt in sorted_mods:
            pct = 100 * cnt / max(total, 1)
            print(f'    {mod_name:<20s} {cnt:>12,}  ({pct:.1f}%)')
        print(f'  {"─"*48}')

    return result


def _estimate_complexity(model, N_in, device, amp_enabled, amp_dtype):
    """Estimate Big O by comparing FLOPs at N vs 2N input."""
    try:
        from fvcore.nn import FlopCountAnalysis
        import warnings

        model.eval()
        flops_list = []

        for sz in [N_in, N_in * 2]:
            inp = torch.randn(1, sz, 3, device=device)
            with warnings.catch_warnings(), torch.no_grad():
                warnings.simplefilter('ignore')
                if amp_enabled and device.type == 'cuda':
                    with _autocast_context(device, amp_enabled, amp_dtype):
                        fa = FlopCountAnalysis(model, inp)
                        flops_list.append(fa.total())
                else:
                    fa = FlopCountAnalysis(model, inp)
                    flops_list.append(fa.total())
            del inp
            torch.cuda.empty_cache()

        if flops_list[0] == 0:
            return '?'

        ratio = flops_list[1] / max(flops_list[0], 1)
        if ratio < 1.5:
            return '~O(1)'
        elif ratio < 2.3:
            return '~O(N)'
        elif ratio < 3.0:
            return '~O(N log N)'
        elif ratio < 5.0:
            return '~O(N²)'
        elif ratio < 9.0:
            return '~O(N³)'
        else:
            return f'~O(N^{math.log2(ratio):.1f})'
    except Exception as e:
        return f'Err: {e}'


def _estimate_vram(model, N_in, batch_size, precision, verbose=True):
    """Analytical VRAM estimate (no trial run)."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    bpp = 2 if precision in ('fp16', 'bf16') else 4
    param_mem = total_params * bpp
    grad_mem = trainable_params * bpp
    optim_mem = trainable_params * 4 * 2  # Adam fp32 states
    act_mem = batch_size * total_params * bpp * 2

    train_total = param_mem + grad_mem + optim_mem + act_mem
    eval_total = param_mem + int(act_mem * 0.3)

    train_gb = train_total / (1024**3)
    eval_gb = eval_total / (1024**3)

    if verbose:
        print(f'\n  {"─"*48}')
        print(f'  Estimated VRAM (analytical, BS={batch_size})')
        print(f'  {"─"*48}')
        print(f'  Params ({precision:>4s}) : {param_mem / 1024**2:>8.1f} MB')
        print(f'  Gradients      : {grad_mem / 1024**2:>8.1f} MB')
        print(f'  Optimizer      : {optim_mem / 1024**2:>8.1f} MB')
        print(f'  Activations    : {act_mem / 1024**2:>8.1f} MB')
        print(f'  {"─"*48}')
        print(f'  Est. Train     : {train_gb:>8.2f} GB')
        print(f'  Est. Eval      : {eval_gb:>8.2f} GB')
        print(f'  {"─"*48}')

    return f'{train_gb:.2f}GB', f'{eval_gb:.2f}GB'


def _profile_flops(model, N_in, device, amp_enabled, amp_dtype):
    try:
        from fvcore.nn import FlopCountAnalysis
        import warnings
        
        model.eval()
        dummy_input = torch.randn(1, N_in, 3, device=device)
        with warnings.catch_warnings(), torch.no_grad(), _autocast_context(device, amp_enabled, amp_dtype):
            warnings.simplefilter('ignore')
            flops = FlopCountAnalysis(model, dummy_input)
            total_flops = flops.total()
            
            # Format (e.g. 10.5G)
            if total_flops > 1e9:
                return f"{total_flops / 1e9:.1f}G"
            elif total_flops > 1e6:
                return f"{total_flops / 1e6:.1f}M"
            else:
                return f"{total_flops:,}"
    except Exception as e:
        return f"Err: {e}"


def _profile_max_batch_size(model, N_in, n_gt, device, is_train, amp_enabled, amp_dtype):
    def check_bs(b):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        try:
            partial = torch.randn(b, N_in, 3, device=device)
            gt = torch.randn(b, n_gt, 3, device=device)
            
            if is_train:
                model.train()
                optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
                scaler = _make_grad_scaler(device, amp_enabled, amp_dtype)
                optimizer.zero_grad()
                with _autocast_context(device, amp_enabled, amp_dtype):
                    out = model(partial)
                    if not isinstance(out, (tuple, list)):
                        out = (out,)
                    loss = out[-1].sum()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                model.eval()
                with torch.no_grad(), _autocast_context(device, amp_enabled, amp_dtype):
                    out = model(partial)
            return True, torch.cuda.max_memory_allocated() / (1024**3)
        except torch.cuda.OutOfMemoryError:
            return False, 0.0
        except Exception:
            # Other errors (e.g., batch size too large for some ops)
            return False, 0.0
        finally:
            if 'optimizer' in locals():
                optimizer.zero_grad(set_to_none=True)
                del optimizer
            if 'out' in locals(): del out
            if 'loss' in locals(): del loss
            if 'partial' in locals(): del partial
            if 'gt' in locals(): del gt
            torch.cuda.empty_cache()

    bs = 1
    max_bs = 0
    vram = 0.0
    
    # 1. Doubling phase
    while True:
        success, v = check_bs(bs)
        if success:
            max_bs = bs
            vram = v
            bs *= 2
        else:
            break
            
    if max_bs == 0:
        return 0, 0.0

    # 2. Binary search phase between max_bs and bs
    low = max_bs + 1
    high = bs - 1
    while low <= high:
        mid = (low + high) // 2
        success, v = check_bs(mid)
        if success:
            max_bs = mid
            vram = v
            low = mid + 1
        else:
            high = mid - 1
            
    torch.cuda.empty_cache()
    return max_bs, vram


def _verify_single_model_once(config_path, n_iters=3, verbose=True, use_amp=False, run_profile=False, precision='fp32'):
    # Handle backward compat: --amp flag sets precision to fp16
    if use_amp and precision == 'fp32':
        precision = 'fp16'
    amp_enabled, amp_dtype = _get_amp_settings(precision)

    result = _new_result(config_path)
    result['precision'] = precision
    result['requested_precision'] = precision
    result['effective_precision'] = precision
    start = time.time()

    try:
        # 1. Load config
        cfg = load_config(config_path)
        result['model_name'] = cfg.model.name
        if verbose:
            print(f'\n{"="*60}')
            print(f' Verifying: {cfg.model.name} ({os.path.basename(config_path)})')
            print(f' Precision: {precision.upper()}' + (' (AMP)' if amp_enabled else ''))
            if run_profile:
                print(' Profiling: FLOPs, Params, Big O, Max BS, VRAM')
            print(f'{"="*60}')

        # GPU info (always show when profiling)
        if run_profile and verbose:
            gpu_info = _detect_gpu_info(verbose=True)
        else:
            gpu_info = _detect_gpu_info(verbose=False)
        result['gpu_name'] = gpu_info['gpu_name']
        result['gpu_vram_total'] = gpu_info['gpu_vram_total']

        # 2. Build model
        if verbose:
            print('  [1/6] Building model...', end=' ', flush=True)
        model = build_model(cfg)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        n_params = sum(p.numel() for p in model.parameters())
        result['n_params'] = n_params
        if verbose:
            print(f'OK ({n_params:,} params, device={device})')

        N_in = 2048
        n_gt = getattr(cfg.model.config, 'num_pred', 16384)

        # Parameter breakdown
        if run_profile:
            if verbose:
                print('  [1.1/6] Parameter breakdown...')
            pinfo = _profile_parameters(model, verbose=verbose)
            result['trainable_params'] = pinfo['trainable']
            result['frozen_params'] = pinfo['frozen']
            result['param_size_mb'] = pinfo['param_size_mb']
            result['top_modules'] = [(n, c) for n, c in pinfo['top_modules']]

        if run_profile and device.type == 'cuda':
            if verbose:
                print('  [2.1/6] Profiling FLOPs...', end=' ', flush=True)
            result['flops'] = _profile_flops(model, N_in, device, amp_enabled, amp_dtype)
            if verbose:
                print(result['flops'])

            if verbose:
                print('  [2.2/6] Estimating Big O complexity...', end=' ', flush=True)
            result['complexity'] = _estimate_complexity(model, N_in, device, amp_enabled, amp_dtype)
            if verbose:
                print(result['complexity'])

            if verbose:
                print('  [2.3/6] Profiling max batch size & VRAM (Eval)...', end=' ', flush=True)
            eval_bs, eval_vram = _profile_max_batch_size(
                model, N_in, n_gt, device, is_train=False, amp_enabled=amp_enabled, amp_dtype=amp_dtype
            )
            result['max_bs_eval'] = eval_bs
            result['vram_eval'] = f"{eval_vram:.2f}GB"
            if verbose:
                print(f"BS={eval_bs}, VRAM={eval_vram:.2f}GB")

            if verbose:
                print('  [2.4/6] Profiling max batch size & VRAM (Train)...', end=' ', flush=True)

            train_bs, train_vram = _profile_max_batch_size(
                model, N_in, n_gt, device, is_train=True, amp_enabled=amp_enabled, amp_dtype=amp_dtype
            )
            result['max_bs_train'] = train_bs
            result['vram_train'] = f"{train_vram:.2f}GB"
            if verbose:
                print(f"BS={train_bs}, VRAM={train_vram:.2f}GB")

            # Estimated VRAM (analytical)
            cfg_bs = getattr(cfg.train, 'batch_size', 32)
            if verbose:
                print(f'  [2.5/6] Estimated VRAM (analytical, config BS={cfg_bs})...')
            est_train, est_eval = _estimate_vram(model, N_in, cfg_bs, precision, verbose=verbose)
            result['est_vram_train'] = est_train
            result['est_vram_eval'] = est_eval

        # 3. Create dummy input
        if verbose:
            print('  [3/6] Forward pass...', end=' ', flush=True)
        B = 2
        partial = torch.randn(B, N_in, 3, device=device)
        gt = torch.randn(B, n_gt, 3, device=device)

        # Forward (with OOM retry at reduced batch size)
        model.train()
        try:
            with _autocast_context(device, amp_enabled, amp_dtype):
                pcds_pred = model(partial)
        except torch.cuda.OutOfMemoryError:
            # Retry with batch_size=1 for memory-heavy models
            torch.cuda.empty_cache()
            B = 1
            partial = torch.randn(B, N_in, 3, device=device)
            gt = torch.randn(B, n_gt, 3, device=device)
            if verbose:
                print('(OOM, retrying B=1)...', end=' ')
            with _autocast_context(device, amp_enabled, amp_dtype):
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
            print('  [4/6] Loss computation...', end=' ', flush=True)

        # Adjust gt size to match finest prediction
        finest = pcds_pred[-1]
        if gt.shape[1] != finest.shape[1]:
            if gt.shape[1] > finest.shape[1]:
                gt = gt[:, :finest.shape[1], :]
            else:
                gt = torch.randn(B, finest.shape[1], 3, device=device)
        gt = gt.contiguous()

        chamfer_dist = None
        scaler = _make_grad_scaler(device, amp_enabled, amp_dtype)
        try:
            model_ref = model
            with _autocast_context(device, amp_enabled, amp_dtype):
                loss = model_ref.get_loss(pcds_pred, gt, epoch=1)
                loss_val = loss[0] if isinstance(loss, tuple) else loss
            if verbose:
                print(f'OK (model loss = {loss_val.item():.4f})')
        except (NotImplementedError, AttributeError):
            # Fall back to simple CD
            from Chamfer3D.dist_chamfer_3D import chamfer_3DDist
            chamfer_dist = chamfer_3DDist()
            with _autocast_context(device, False, torch.float32):
                d1, d2, _, _ = chamfer_dist(finest.float(), gt.float())
                loss_val = (torch.mean(d1) + torch.mean(d2)) * 1e3
            if verbose:
                print(f'OK (default CD = {loss_val.item():.4f})')

        # 5. Backward + optimizer
        if verbose:
            print('  [5/6] Backward pass...', end=' ', flush=True)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        optimizer.zero_grad()
        scaler.scale(loss_val).backward()
        scaler.step(optimizer)
        scaler.update()
        if verbose:
            print('OK')

        # Multiple iterations
        if verbose:
            print(f'  [6/6] Running {n_iters} iterations...', end=' ', flush=True)

        for _ in range(n_iters):
            partial = torch.randn(B, N_in, 3, device=device)
            gt = torch.randn(B, pcds_pred[-1].shape[1], 3, device=device)

            with _autocast_context(device, amp_enabled, amp_dtype):
                pcds_pred = model(partial)
                if not isinstance(pcds_pred, (tuple, list)):
                    pcds_pred = (pcds_pred,)
                try:
                    model_ref = model
                    loss = model_ref.get_loss(pcds_pred, gt, epoch=1)
                    loss_val = loss[0] if isinstance(loss, tuple) else loss
                except (NotImplementedError, AttributeError):
                    if chamfer_dist is None:
                        from Chamfer3D.dist_chamfer_3D import chamfer_3DDist
                        chamfer_dist = chamfer_3DDist()
                    with _autocast_context(device, False, torch.float32):
                        d1, d2, _, _ = chamfer_dist(pcds_pred[-1].float(), gt.float())
                        loss_val = (torch.mean(d1) + torch.mean(d2)) * 1e3

            optimizer.zero_grad()
            scaler.scale(loss_val).backward()
            scaler.step(optimizer)
            scaler.update()

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


def verify_single_model(
    config_path,
    n_iters=3,
    verbose=True,
    use_amp=False,
    run_profile=False,
    precision='fp32',
    allow_precision_fallback=True,
):
    """
    Verify that a model config works end-to-end.
    Adds optional precision fallback (e.g. fp16->fp32) on AMP type mismatch errors.
    """
    if use_amp and precision == 'fp32':
        precision = 'fp16'
    requested_precision = precision

    primary = _verify_single_model_once(
        config_path=config_path,
        n_iters=n_iters,
        verbose=verbose,
        use_amp=use_amp,
        run_profile=run_profile,
        precision=requested_precision,
    )
    primary['requested_precision'] = requested_precision
    primary['effective_precision'] = primary.get('precision', requested_precision)

    if primary.get('success'):
        return _finalize_status(primary)

    should_try_fallback = (
        allow_precision_fallback
        and requested_precision in ('fp16', 'bf16')
        and _is_precision_mismatch_error(primary.get('error'), requested_precision)
    )
    if not should_try_fallback:
        return _finalize_status(primary)

    if verbose:
        print(f'  ! Precision fallback: {requested_precision} failed with a dtype mismatch; retrying with FP32...')

    fallback = _verify_single_model_once(
        config_path=config_path,
        n_iters=n_iters,
        verbose=verbose,
        use_amp=False,
        run_profile=run_profile,
        precision='fp32',
    )
    fallback['requested_precision'] = requested_precision
    fallback['effective_precision'] = fallback.get('precision', 'fp32')
    fallback['precision_fallback_from'] = requested_precision
    fallback['precision_fallback_to'] = 'fp32'
    fallback['precision_fallback_reason'] = primary.get('error')
    fallback['time'] = primary.get('time', 0.0) + fallback.get('time', 0.0)

    if fallback.get('success'):
        fallback['precision_fallback_used'] = True
        fallback['error'] = None
        return _finalize_status(fallback)

    fallback['precision_fallback_used'] = False
    primary_err = primary.get('error') or 'Unknown error'
    fallback_err = fallback.get('error') or 'Unknown error'
    fallback['error'] = f'{fallback_err} | initial {requested_precision} failure: {primary_err}'
    return _finalize_status(fallback)


def _parse_result_json(stdout):
    for line in reversed(stdout.splitlines()):
        if line.startswith(RESULT_PREFIX):
            payload = line[len(RESULT_PREFIX):]
            return json.loads(payload)
    raise ValueError('Worker did not emit JSON result marker')


def _run_single_isolated(
    config_path,
    n_iters,
    gpu,
    timeout,
    use_amp=False,
    run_profile=False,
    precision='fp32',
    allow_precision_fallback=True,
    extra_env=None,
):
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
        '--precision',
        precision,
    ]
    if use_amp:
        cmd.append('--amp')
    if run_profile:
        cmd.append('--profile')
    if not allow_precision_fallback:
        cmd.append('--no-precision-fallback')
        
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


def _print_summary(results, run_profile=False):
    print(f'\n\n{"="*146}')
    print(' VERIFICATION SUMMARY')
    print(f'{"="*146}')
    
    if run_profile:
        hdr = (f'{"Config":<15} {"Model":<15} {"St":<5} {"Prec":<11} '
               f'{"Params":<12} {"FLOPs":<10} {"Big O":<12} '
               f'{"MaxBS T/E":<11} {"VRAM T/E":<16} {"Time":<6}')
        print(hdr)
        print('-' * 146)
    else:
        print(f'{"Config":<25} {"Model":<20} {"Status":<10} {"Params":<15} {"Time":<8}')
        print('-' * 90)

    passed = 0
    skipped = 0
    for r in results:
        status = {
            'PASS': 'PASS',
            'FAIL': 'FAIL',
            'SKIP': 'SKIP',
        }.get(r.get('status', 'FAIL'), 'FAIL')
        params = f'{r["n_params"]:,}' if r.get('n_params') else '-'
        
        if run_profile:
            flops = str(r.get('flops', '-'))
            bigo = str(r.get('complexity', '-'))
            prec = _format_precision_display(r)
            bs_t = str(r.get('max_bs_train', '-'))
            bs_e = str(r.get('max_bs_eval', '-'))
            vr_t = str(r.get('vram_train', '-')).replace('GB', '')
            vr_e = str(r.get('vram_eval', '-')).replace('GB', '')
            
            bs_str = f"{bs_t}/{bs_e}"
            vr_str = f"{vr_t}/{vr_e}"
            
            print(f'{r["config"][:14]:<15} {r.get("model_name", "unknown")[:14]:<15} '
                  f'{status:<5} {prec[:10]:<11} {params[:11]:<12} {flops[:9]:<10} '
                  f'{bigo[:11]:<12} {bs_str[:10]:<11} {vr_str[:15]:<16} '
                  f'{r.get("time", 0.0):.1f}s')
        else:
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

    fallback_used = [r for r in results if r.get('precision_fallback_used')]
    if fallback_used:
        print('\nPrecision fallbacks used:')
        for r in fallback_used:
            reason = _short_error(r.get('precision_fallback_reason') or 'AMP dtype mismatch')
            print(
                f'  {r.get("model_name", "unknown")}: '
                f'{r.get("precision_fallback_from", "?")} -> {r.get("precision_fallback_to", "?")} '
                f'({reason})'
            )

    if skipped > 0:
        print('\nSkipped models:')
        for r in results:
            if r.get('status') == 'SKIP':
                print(f'  {r.get("config")}: {r.get("error", "Skipped")}')

    return failed


def _run_all_legacy(
    config_paths,
    n_iters,
    skip_set,
    use_amp=False,
    run_profile=False,
    precision='fp32',
    allow_precision_fallback=True,
):
    results = []
    for config_path in config_paths:
        base = os.path.basename(config_path)
        if base in skip_set:
            results.append(_make_skip_result(config_path, 'Skipped by user configuration'))
            continue
        result = verify_single_model(
            config_path,
            n_iters=n_iters,
            use_amp=use_amp,
            run_profile=run_profile,
            precision=precision,
            allow_precision_fallback=allow_precision_fallback,
        )
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
            use_amp=args.amp,
            run_profile=args.profile,
            precision=args.precision,
            allow_precision_fallback=args.precision_fallback,
        )
        primary['attempts'] = 1

        if primary.get('status') == 'FAIL' and args.debug_retry:
            debug = _run_single_isolated(
                config_path=config_path,
                n_iters=args.iters,
                gpu=args.gpu,
                timeout=args.timeout,
                use_amp=args.amp,
                run_profile=args.profile,
                precision=args.precision,
                allow_precision_fallback=args.precision_fallback,
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
    
    # New profiling options
    parser.add_argument('--amp', '--fp16', action='store_true', dest='amp',
                        help='Enable Automatic Mixed Precision (FP16) [backward compat alias for --precision fp16]')
    parser.add_argument('--precision', type=str, default='fp32',
                        choices=['fp32', 'fp16', 'bf16', 'tf32'],
                        help='Precision mode: fp32 (default), fp16, bf16, tf32')
    parser.add_argument('--precision-fallback', dest='precision_fallback', action='store_true', default=True,
                        help='Retry fp16/bf16 failures in fp32 when failure looks like AMP dtype mismatch (default: enabled)')
    parser.add_argument('--no-precision-fallback', dest='precision_fallback', action='store_false',
                        help='Disable fp16/bf16 -> fp32 fallback')
    parser.add_argument('--profile', action='store_true',
                        help='Profile FLOPs, params, Big O, Max Batch Size, and VRAM')
                        
    args = parser.parse_args()

    # --amp is a backward compat alias for --precision fp16
    if args.amp and args.precision == 'fp32':
        args.precision = 'fp16'

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    if args.config:
        result = verify_single_model(
            args.config, 
            n_iters=args.iters, 
            verbose=not args.json_result, 
            use_amp=args.amp, 
            run_profile=args.profile,
            precision=args.precision,
            allow_precision_fallback=args.precision_fallback,
        )

        if args.json_result:
            # Serialize top_modules tuples to list for JSON
            if result.get('top_modules'):
                result['top_modules'] = [[n, c] for n, c in result['top_modules']]
            print(RESULT_PREFIX + json.dumps(result))
            sys.exit(0 if result.get('success') else 1)

        status = 'PASS' if result['success'] else 'FAIL'
        extra = f", precision={_format_precision_display(result)}"
        if args.profile:
            extra += f", FLOPs={result.get('flops')}, Big O={result.get('complexity')}"
        print(f'\n{status}: {result["model_name"]} ({result["n_params"]:,} params{extra}, {result["time"]:.1f}s)')
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
            results = _run_all_legacy(
                config_paths,
                args.iters,
                skip_set,
                use_amp=args.amp,
                run_profile=args.profile,
                precision=args.precision,
                allow_precision_fallback=args.precision_fallback,
            )
        else:
            results = _run_all_isolated(config_paths, args, skip_set)

        failed = _print_summary(results, run_profile=args.profile)
        sys.exit(1 if failed > 0 else 0)

    # Default: verify dummy model
    print('No --config or --all specified. Verifying DummyModel...')
    dummy_config = os.path.join(os.path.dirname(__file__), 'configs', 'dummy.yaml')
    result = verify_single_model(
        dummy_config,
        n_iters=args.iters,
        use_amp=args.amp,
        run_profile=args.profile,
        precision=args.precision,
        allow_precision_fallback=args.precision_fallback,
    )
    status = 'PASS' if result['success'] else 'FAIL'
    extra = f", precision={_format_precision_display(result)}"
    if args.profile:
        extra += f", FLOPs={result.get('flops')}, Big O={result.get('complexity')}"
    print(f'\n{status}: {result["model_name"]} ({result["n_params"]:,} params{extra}, {result["time"]:.1f}s)')


    if result['success']:
        print('\nDummy model works! To test actual models:')
        print('  python verify_pipeline.py --config configs/pcn.yaml')
        print('  python verify_pipeline.py --all')


if __name__ == '__main__':
    main()
