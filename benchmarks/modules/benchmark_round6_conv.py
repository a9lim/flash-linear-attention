# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Paired dense fused-convolution benchmark against a frozen Git kernel."""

import argparse
import importlib.util
import json
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path

import torch
import triton

from fla.modules.conv.triton import ops
from fla.utils import assert_close


def load_baseline(base, directory):
    path = Path(directory) / 'round6_conv_baseline.py'
    source = subprocess.check_output(
        ['git', 'show', f'{base}:fla/modules/conv/triton/kernels.py'], text=True,
    )
    path.write_text(source)
    spec = importlib.util.spec_from_file_location('round6_conv_baseline', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.causal_conv1d_bwd_l2norm_kernel


def summarize(samples):
    ordered = sorted(samples)
    return {
        'median_ms': statistics.median(samples),
        'mean_ms': statistics.mean(samples),
        'std_ms': statistics.pstdev(samples),
        'min_ms': min(samples),
        'p10_ms': ordered[int(0.1 * (len(ordered) - 1))],
        'p90_ms': ordered[int(0.9 * (len(ordered) - 1))],
        'samples_ms': samples,
    }


def difference(reference, actual):
    ref, result = reference.float(), actual.float()
    return {
        'relative_l2': ((ref - result).norm() / ref.norm().clamp_min(1e-30)).item(),
        'max_abs': (ref - result).abs().max().item(),
        'norm_ratio': (result.norm() / ref.norm().clamp_min(1e-30)).item(),
    }


def capture(body):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            body()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        outputs = body()
    return graph, outputs


def paired(graphs, rounds, replays):
    samples = [[], []]
    for round_index in range(rounds + 2):
        for index in ([0, 1] if round_index % 2 == 0 else [1, 0]):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(replays):
                graphs[index].replay()
            end.record()
            end.synchronize()
            if round_index >= 2:
                samples[index].append(start.elapsed_time(end) / replays)
    return samples


def benchmark(args, baseline):
    torch.manual_seed(42)
    torch.set_float32_matmul_precision('high')
    B, T, HD, H, W = args.batch, args.length, args.head_dim, args.heads, args.width
    D = H * HD
    widths = (D // 3,) * 3
    assert H % 3 == 0
    x = torch.randn(B, T, D, device='cuda', dtype=torch.bfloat16)
    weight = torch.randn(D, W, device='cuda') * 0.2
    bias = torch.randn(D, device='cuda')
    dys = tuple(torch.randn(B, T, width, device='cuda', dtype=torch.bfloat16) for width in widths)
    common = dict(
        x=x, weight=weight, bias=bias, residual=None, activation='silu',
        l2norm_head_dim=HD, l2norm_channels=2 * D // 3, split_outputs=widths,
    )
    candidate = ops.causal_conv1d_bwd_l2norm_kernel
    print(json.dumps({
        'base': args.base,
        'candidate': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'torch': torch.__version__, 'triton': triton.__version__,
        'device': str(torch.cuda.get_device_properties(0)),
        'shape': [B, T, D], 'W': W, 'head_dim': HD, 'slabs': widths,
        'dtype': str(x.dtype), 'matmul_precision': torch.get_float32_matmul_precision(),
    }), flush=True)
    try:
        for BT in args.tiles:
            baseline_tile = args.baseline_tile or BT

            def full_body(kernel, mode, backward_tile):
                ops.causal_conv1d_bwd_l2norm_kernel = kernel
                forward = ops.causal_conv1d_fwd(**common, BT=args.forward_tile) if mode != 'backward' else None
                backward = ops.causal_conv1d_bwd(**common, dy=dys, dht=None, BT=backward_tile) if mode != 'forward' else None
                return forward, backward

            reference = full_body(baseline, 'forward_backward', baseline_tile)
            actual = full_body(candidate, 'forward_backward', BT)
            errors = {}
            for index, (ref, result) in enumerate(zip(reference[0][0], actual[0][0], strict=True)):
                errors[f'output[{index}]'] = difference(ref, result)
                assert_close(f'output[{index}]', ref, result, 1e-6)
            for name, ref, result in zip(('dx', 'dw', 'db', 'dr', 'dh0'), reference[1], actual[1], strict=True):
                if ref is not None:
                    errors[name] = difference(ref, result)
                    assert_close(name, ref, result, 1e-6)
                else:
                    assert result is None
                    errors[name] = None
            print(json.dumps({'BT': BT, 'baseline_BT': baseline_tile, 'errors': errors}), flush=True)
            del reference, actual

            def kernel_body(kernel, buffers, tile):
                dx, dw, db = buffers
                offset = 0
                compiled = []
                for dy, width in zip(dys, widths, strict=True):
                    result = kernel[(width // HD, triton.cdiv(T, tile), B)](
                        x=x, weight=weight, bias=bias, dy=dy, dx=dx, dw=dw, db=db, T=T,
                        stride_x_n=x.stride(0), stride_x_t=x.stride(1), stride_x_d=x.stride(2),
                        stride_dx_n=dx.stride(0), stride_dx_t=dx.stride(1), stride_dx_d=dx.stride(2),
                        stride_dy_n=dy.stride(0), stride_dy_t=dy.stride(1), stride_dy_d=dy.stride(2),
                        Y_OFF=offset, D=D, W=W, BT=tile, BW=triton.next_power_of_2(W), BD=HD,
                        NORM_D=2 * D // 3, EPS=1e-6, ACTIVATION='silu',
                    )
                    compiled.append(result)
                    offset += width
                return compiled

            for mode in ('kernel_backward', 'forward', 'backward', 'forward_backward'):
                graph_outputs, buffers, resources = [], [], []
                for kernel, tile in ((baseline, baseline_tile), (candidate, BT)):
                    if mode == 'kernel_backward':
                        storage = (
                            torch.empty_like(x),
                            torch.empty(B * triton.cdiv(T, tile), D, W, device='cuda'),
                            torch.empty(B * triton.cdiv(T, tile), D, device='cuda'),
                        )
                        buffers.append(storage)
                        compiled = kernel_body(kernel, storage, tile)
                        resources.append([
                            {
                                'registers': getattr(item, 'n_regs', None),
                                'spills': getattr(item, 'n_spills', None),
                                'shared_bytes': getattr(getattr(item, 'metadata', None), 'shared', None),
                            }
                            for item in compiled
                        ])
                        def body(kernel=kernel, storage=storage, tile=tile):
                            return kernel_body(kernel, storage, tile)
                    else:
                        def body(kernel=kernel, mode=mode, tile=tile):
                            return full_body(kernel, mode, tile)
                    graph_outputs.append(capture(body))
                samples = paired([item[0] for item in graph_outputs], args.rounds, args.replays)
                base_stats, candidate_stats = map(summarize, samples)
                print(json.dumps({
                    'BT': BT, 'baseline_BT': baseline_tile, 'forward_BT': args.forward_tile,
                    'mode': mode, 'baseline': base_stats, 'candidate': candidate_stats,
                    'speedup': base_stats['median_ms'] / candidate_stats['median_ms'],
                    'resources': resources,
                    'allocated_mib': torch.cuda.memory_allocated() / 2**20,
                    'reserved_mib': torch.cuda.memory_reserved() / 2**20,
                }), flush=True)
                del graph_outputs, buffers
    finally:
        ops.causal_conv1d_bwd_l2norm_kernel = candidate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', default='34848cb2632e82a7756b9f83dd85067554a0b0be')
    parser.add_argument('--batch', type=int, default=4)
    parser.add_argument('--length', type=int, default=1024)
    parser.add_argument('--heads', type=int, default=30)
    parser.add_argument('--head-dim', type=int, default=128)
    parser.add_argument('--width', type=int, default=4)
    parser.add_argument('--tiles', type=int, nargs='+', default=[64, 32])
    parser.add_argument('--baseline-tile', type=int, default=None)
    parser.add_argument('--forward-tile', type=int, default=64)
    parser.add_argument('--rounds', type=int, default=12)
    parser.add_argument('--replays', type=int, default=30)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix='round6-conv-') as directory:
        benchmark(args, load_baseline(args.base, directory))


if __name__ == '__main__':
    main()
