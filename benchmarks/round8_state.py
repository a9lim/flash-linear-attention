# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Sweep state-kernel launch configurations and compare complete PKDA gradients."""

import argparse
import gc
import importlib
import importlib.util
import json
import statistics
import subprocess
import sys
import tempfile
from contextlib import ExitStack, contextmanager
from functools import partial
from pathlib import Path

import torch
import torch.nn.functional as F
import triton
from triton.compiler.errors import CompilationError
from triton.runtime.errors import OutOfResources

import fla.ops.common.chunk_delta_h as state_ops
import fla.ops.utils.cache as cache_ops
from fla.ops.precond_kda import chunk_precond_kda
from fla.utils import assert_close


def load_baseline(ref, directory):
    root = Path(__file__).resolve().parents[1]
    source = subprocess.check_output(
        ['git', 'show', f'{ref}:fla/ops/common/chunk_delta_h.py'], cwd=root, text=True,
    )
    path = Path(directory) / 'round8_state_baseline.py'
    path.write_text(source)
    spec = importlib.util.spec_from_file_location('round8_state_baseline', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def autotuner(kernel):
    while not hasattr(kernel, 'configs'):
        kernel = kernel.fn
    return kernel


@contextmanager
def force_config(kernel, config):
    tuner = autotuner(kernel)
    configs, cache = tuner.configs, tuner.cache
    old_best = getattr(tuner, 'best_config', None)
    cache_mode = cache_ops.FLA_CACHE_MODE
    tuner.configs, tuner.cache = [config], {}
    cache_ops.FLA_CACHE_MODE = cache_ops.FlaCacheMode.DISABLED
    try:
        yield
        actual = getattr(tuner, 'best_config', None)
        if actual is None or config_dict(actual) != config_dict(config):
            raise RuntimeError(f'Forced configuration was not selected: {config_dict(config)}')
    finally:
        tuner.configs, tuner.cache = configs, cache
        cache_ops.FLA_CACHE_MODE = cache_mode
        if old_best is not None:
            tuner.best_config = old_best
        elif hasattr(tuner, 'best_config'):
            del tuner.best_config


def config_dict(config):
    return dict(BV=config.kwargs['BV'], num_warps=config.num_warps, num_stages=config.num_stages)


def state_kernel(module, direction):
    name = ('chunk_gated_delta_rule_fwd_kernel_h_blockdim64' if direction == 'forward'
            else 'chunk_gated_delta_rule_bwd_kernel_dhu_blockdim64')
    return getattr(module, name)


def tuning_result(kernel):
    tuner = autotuner(kernel)
    selected = getattr(tuner, 'best_config', None)
    return dict(
        selected=config_dict(selected) if selected is not None else None,
        key_fields=list(tuner.keys),
        cached_keys=[list(key) for key in tuner.cache],
    )


def parse_config(value):
    bv, warps, stages = map(int, value.split(','))
    return triton.Config({'BV': bv}, num_warps=warps, num_stages=stages)


def make_state_inputs(batch, length):
    torch.manual_seed(42)
    shape = (batch, length, 10, 128)
    q = F.normalize(torch.randn(shape, device='cuda'), dim=-1).bfloat16()
    k = F.normalize(torch.randn(shape, device='cuda'), dim=-1).bfloat16()
    increments = -(torch.rand(shape, device='cuda') * .02 + .01)
    gk = torch.empty_like(increments)
    for start in range(0, length, 64):
        gk[:, start:start + 64] = increments[:, start:start + 64].cumsum(1)
    return dict(
        q=(q.float() * gk.exp2()).bfloat16(),
        k=k,
        w=(k.float() * .2 * gk.exp2()).bfloat16(),
        u=torch.randn(shape, device='cuda', dtype=torch.bfloat16) * .1,
        do=torch.randn(shape, device='cuda', dtype=torch.bfloat16) * .01,
        dv=torch.randn(shape, device='cuda', dtype=torch.bfloat16) * .01,
        gk=gk,
    )


def state_call(module, direction, values):
    if direction == 'forward':
        return module.chunk_gated_delta_rule_fwd_h(
            k=values['k'], w=values['w'], u=values['u'], gk=values['gk'],
            chunk_size=64, output_final_state=False, state_v_first=False,
        )
    return module.chunk_gated_delta_rule_bwd_dhu(
        q=values['q'], k=values['k'], w=values['w'], do=values['do'], dv=values['dv'],
        gk=values['gk'], chunk_size=64, scale=128**-.5, state_v_first=False,
    )


def errors(reference, candidate, names, tolerance=.006):
    result = {}
    for name, ref, actual in zip(names, reference, candidate, strict=True):
        if ref is None:
            assert actual is None
            continue
        assert torch.isfinite(ref).all() and torch.isfinite(actual).all(), name
        ref, actual = ref.float(), actual.float()
        diff = actual - ref
        result[name] = dict(
            relative_l2=float(diff.norm() / ref.norm().clamp_min(1e-30)),
            max_abs=float(diff.abs().max()),
        )
        assert_close(name, ref, actual, tolerance)
    return result


def graph_time(fn, replays, samples):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    gc.collect()
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            retained = fn()
    finally:
        if was_enabled:
            gc.enable()
    measured = []
    for _ in range(samples):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(replays):
            graph.replay()
        end.record()
        end.synchronize()
        measured.append(start.elapsed_time(end) / replays)
    del graph, retained
    return dict(median_ms=statistics.median(measured), samples_ms=measured)


def sweep(baseline, args):
    values = make_state_inputs(args.batch, args.length)
    kernels = {
        'forward': state_ops.chunk_gated_delta_rule_fwd_kernel_h_blockdim64,
        'backward': state_ops.chunk_gated_delta_rule_bwd_kernel_dhu_blockdim64,
    }
    results = {}
    for direction, kernel in kernels.items():
        names = ('h', 'v_new', 'ht') if direction == 'forward' else ('dh', 'dh0', 'dv')
        baseline_call = partial(state_call, baseline, direction, values)
        candidate_call = partial(state_call, state_ops, direction, values)
        expected = tuple(t.clone() if t is not None else None for t in baseline_call())
        baseline_time = graph_time(baseline_call, args.replays, args.samples)
        print(json.dumps(dict(
            direction=direction, variant='baseline', **baseline_time,
            tuning=tuning_result(state_kernel(baseline, direction)),
        )), flush=True)
        default_differences = errors(expected, candidate_call(), names)
        default_time = graph_time(candidate_call, args.replays, args.samples)
        print(json.dumps(dict(
            direction=direction, variant='candidate_default', **default_time,
            tuning=tuning_result(kernel), differences=default_differences,
        )), flush=True)
        configs = list(autotuner(kernel).configs)
        results[direction] = []
        for config in configs:
            try:
                with force_config(kernel, config), torch.no_grad():
                    differences = errors(expected, candidate_call(), names)
                    timing = graph_time(candidate_call, args.replays, args.samples)
                row = dict(direction=direction, config=config_dict(config), **timing, differences=differences)
                results[direction].append(row)
            except (CompilationError, OutOfResources) as error:
                row = dict(direction=direction, config=config_dict(config), compile_error=str(error)[:600])
            print(json.dumps(row), flush=True)
        if not results[direction]:
            raise RuntimeError(f'No valid {direction} configurations')
        best = min(results[direction], key=lambda row: row['median_ms'])
        with force_config(kernel, parse_config(','.join(str(best['config'][k]) for k in ('BV', 'num_warps', 'num_stages')))):
            repeats = [
                dict(baseline=graph_time(baseline_call, args.replays, args.samples),
                     candidate=graph_time(candidate_call, args.replays, args.samples))
                for _ in range(2)
            ]
        print(json.dumps(dict(direction=direction, best=best['config'], paired_recheck=repeats)), flush=True)
    return results


def make_pkda_inputs(batch, length):
    torch.manual_seed(42)
    shape = (batch, length, 10, 128)
    return dict(
        q=F.normalize(torch.randn(shape, device='cuda'), dim=-1).bfloat16(),
        k=F.normalize(torch.randn(shape, device='cuda'), dim=-1).bfloat16(),
        v=torch.randn(shape, device='cuda', dtype=torch.bfloat16) * .1,
        g=torch.randn(shape, device='cuda', dtype=torch.bfloat16) * .25,
        g_atk=-(torch.rand(shape[:-1], device='cuda') * .1 + .01),
        beta_atk=torch.rand(shape[:-1], device='cuda', dtype=torch.bfloat16) * .5 + .25,
        beta=torch.rand(shape[:-1], device='cuda', dtype=torch.bfloat16) * .5 + .25,
        A_log=torch.linspace(1, 16, shape[2], device='cuda').log(),
        dt_bias=torch.zeros(shape[2] * shape[3], device='cuda'),
        log_atk_scale=torch.full((shape[2],), -.2, device='cuda'),
    )


@contextmanager
def pkda_state_operators(module):
    chunk = importlib.import_module('fla.ops.precond_kda.chunk')
    names = ('chunk_gated_delta_rule_fwd_h', 'chunk_gated_delta_rule_bwd_dhu')
    original = {name: getattr(chunk, name) for name in names}
    try:
        for name in names:
            setattr(chunk, name, getattr(module, name))
        yield
    finally:
        for name, function in original.items():
            setattr(chunk, name, function)


def pkda_call(module, values, do, *, prepared=False):
    inputs = values if prepared else {name: value.detach().clone().requires_grad_() for name, value in values.items()}
    with pkda_state_operators(module), torch.enable_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        output, _, _ = chunk_precond_kda(
            **inputs, scale=128**-.5, output_final_state=False,
            use_qk_l2norm_in_kernel=False, use_gate_in_kernel=True, disable_recompute=True,
        )
        gradients = torch.autograd.grad(output, tuple(inputs.values()), grad_outputs=do)
    return (output.detach(), *(gradient.detach() for gradient in gradients))


def compare_pkda(baseline, args):
    values = make_pkda_inputs(args.batch, args.length)
    do = torch.randn_like(values['v']) * .01
    expected = pkda_call(baseline, values, do)
    baseline_tuning = {direction: tuning_result(state_kernel(baseline, direction)) for direction in ('forward', 'backward')}
    baseline_inputs = {name: value.detach().clone().requires_grad_() for name, value in values.items()}
    candidate_inputs = {name: value.detach().clone().requires_grad_() for name, value in values.items()}
    runners = {
        'baseline': partial(pkda_call, baseline, baseline_inputs, do, prepared=True),
        'candidate': partial(pkda_call, state_ops, candidate_inputs, do, prepared=True),
    }
    with ExitStack() as stack:
        if args.fwd_config:
            stack.enter_context(force_config(state_ops.chunk_gated_delta_rule_fwd_kernel_h_blockdim64, args.fwd_config))
        if args.bwd_config:
            stack.enter_context(force_config(state_ops.chunk_gated_delta_rule_bwd_kernel_dhu_blockdim64, args.bwd_config))
        actual = pkda_call(state_ops, values, do)
        candidate_tuning = {direction: tuning_result(state_kernel(state_ops, direction)) for direction in ('forward', 'backward')}
        differences = errors(expected, actual, ('output', *('d_' + name for name in values)))
        print(json.dumps(dict(
            mode='full_pkda_parity', batch=args.batch, length=args.length, differences=differences,
            tuning=dict(baseline=baseline_tuning, candidate=candidate_tuning),
        )), flush=True)
        del expected, actual
        for order in (('baseline', 'candidate'), ('candidate', 'baseline')):
            timings = {variant: graph_time(runners[variant], args.replays, args.samples) for variant in order}
            print(json.dumps(dict(
                mode='full_pkda_runtime', batch=args.batch, length=args.length,
                measured='forward_and_all_input_gradients_without_input_clones', order=order, **timings,
            )), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', required=True, help='Frozen FLA Git commit containing the baseline state kernels')
    parser.add_argument('--mode', choices=('sweep', 'pkda', 'both'), default='both')
    parser.add_argument('--batch', type=int, choices=(1, 4), default=1)
    parser.add_argument('--length', type=int)
    parser.add_argument('--replays', type=int, default=100)
    parser.add_argument('--samples', type=int, default=5)
    parser.add_argument('--fwd-config', type=parse_config, help='Optional PKDA forward BV,warps,stages')
    parser.add_argument('--bwd-config', type=parse_config, help='Optional PKDA backward BV,warps,stages')
    args = parser.parse_args()
    args.length = args.length or 4096 // args.batch
    torch.set_float32_matmul_precision('high')
    print(json.dumps(dict(
        device=torch.cuda.get_device_name(), torch=torch.__version__, triton=triton.__version__,
        base=args.base, batch=args.batch, length=args.length, heads=10, width=128, synthetic=True,
        fla_cache_mode=cache_ops.FLA_CACHE_MODE.value,
    )), flush=True)
    with tempfile.TemporaryDirectory(prefix='round8-state-baseline-') as directory:
        baseline = load_baseline(args.base, directory)
        if args.mode in ('sweep', 'both'):
            sweep(baseline, args)
        if args.mode in ('pkda', 'both'):
            compare_pkda(baseline, args)


if __name__ == '__main__':
    main()
