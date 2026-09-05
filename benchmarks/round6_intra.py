# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Compare guarded intra backward with a frozen Git version on identical inputs."""

import argparse
import importlib.util
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import torch
import triton.testing

from fla.ops.precond_kda.chunk_intra import chunk_precond_kda_bwd_intra
from fla.utils import assert_close


def load_baseline(ref, directory):
    root = Path(__file__).resolve().parents[1]
    source = subprocess.check_output(
        ['git', 'show', f'{ref}:fla/ops/precond_kda/chunk_intra.py'], cwd=root, text=True,
    )
    path = Path(directory) / 'round6_intra_baseline.py'
    path.write_text(source)
    spec = importlib.util.spec_from_file_location('round6_intra_baseline', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.chunk_precond_kda_bwd_intra


def make_inputs(batch=4, length=1024, heads=10, width=128, regime='mild', varlen=False):
    torch.manual_seed(42)
    shape = (batch, length, heads, width)
    values = {
        name: torch.randn(shape, device='cuda', dtype=torch.bfloat16) * 0.1
        for name in ('q', 'k', 'k_precond', 'dq', 'dk', 'dk_precond')
    }
    values['beta'] = torch.rand(shape[:-1], device='cuda', dtype=torch.bfloat16)
    values['db'] = torch.randn(shape[:-1], device='cuda') * 0.1
    values['dg'] = torch.randn(shape, device='cuda') * 0.1
    magnitudes = {'mild': 0.25, 'mixed': 3.0, 'extreme': 32.0}
    increments = -(0.5 + torch.rand(shape, device='cuda')) * magnitudes[regime]
    values['g'] = torch.empty_like(increments)
    lengths = [length] * batch
    if varlen:
        assert batch == 1 and length >= 65
        lengths = [17, length - 17]
        values['cu_seqlens'] = torch.tensor([0, 17, length], device='cuda', dtype=torch.int32)
    row = torch.arange(length, device='cuda')
    local_rows = torch.empty((batch, length), device='cuda', dtype=torch.long)
    if varlen:
        offset = 0
        for size in lengths:
            for start in range(0, size, 64):
                end = min(start + 64, size)
                values['g'][:, offset + start:offset + end] = increments[:, offset + start:offset + end].cumsum(1)
            local_rows[:, offset:offset + size] = torch.arange(size, device='cuda') % 64
            offset += size
    else:
        for start in range(0, length, 64):
            values['g'][:, start:start + 64] = increments[:, start:start + 64].cumsum(1)
        local_rows[:] = row % 64
    columns = torch.arange(64, device='cuda')
    for name, strict in (('dAqk', False), ('dAkk', True)):
        tensor = torch.randn((batch, length, heads, 64), device='cuda') * 0.1
        mask = columns[None, None, None, :] < (local_rows[:, :, None, None] + (not strict))
        values[name] = tensor.masked_fill(~mask, 0)
    values['defer_reverse_cumsum'] = True
    return values


def coverage(values, block):
    g = values['g'].float()
    batch, length, heads, width = g.shape
    boundaries = values.get('cu_seqlens')
    spans = []
    for b in range(batch):
        cuts = boundaries.cpu().tolist() if boundaries is not None else [0, length]
        for lo, hi in zip(cuts[:-1], cuts[1:]):
            for start in range(lo, hi, block):
                tile = g[b, start:min(start + block, hi)]
                span = tile.amax(0) - tile.amin(0)
                for k in range(0, width, 32):
                    spans.append((span[:, k:k + 32].amax(-1) <= 64).float())
    return torch.cat(spans).mean().item()


def compare(baseline, values, enforce=True):
    expected = baseline(**values)
    actual = chunk_precond_kda_bwd_intra(**values)
    errors = {}
    for name, reference, result in zip(('dq', 'dk', 'dk_precond', 'dbeta', 'dg'), expected, actual):
        if enforce:
            assert_close(name, reference, result, 1e-4)
        delta = (reference.float() - result.float())
        errors[name] = {
            'relative_l2': (delta.norm() / reference.float().norm().clamp_min(1e-12)).item(),
            'max_abs': delta.abs().max().item(),
        }
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', default='34848cb2')
    parser.add_argument('--inputs', type=Path, help='Optional torch-saved intra wrapper kwargs from a trained layer')
    parser.add_argument('--regime', choices=['mild', 'mixed', 'extreme'], default='mild')
    parser.add_argument('--parity-only', action='store_true')
    parser.add_argument('--measure-drift', action='store_true', help='Report drift without enforcing the supplemental 1e-4 baseline comparison')
    args = parser.parse_args()
    values = torch.load(args.inputs, map_location='cuda', weights_only=True) if args.inputs else make_inputs(regime=args.regime)
    with tempfile.TemporaryDirectory(prefix='round6_intra_') as directory:
        baseline = load_baseline(args.base, directory)
        result = {'regime': args.regime, 'errors': compare(baseline, values, enforce=not args.measure_drift)}
        result['fast_fraction'] = {str(block): coverage(values, block) for block in (16, 32)}
        if not args.parity_only:
            # alternate complete captured calls; both include dbeta reduction
            times = {'baseline': [], 'candidate': []}
            for _ in range(3):
                for name, function in (('baseline', baseline), ('candidate', chunk_precond_kda_bwd_intra)):
                    times[name].append(triton.testing.do_bench_cudagraph(lambda: function(**values), rep=500))
            result['milliseconds'] = times
        print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
