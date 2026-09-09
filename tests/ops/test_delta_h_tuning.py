# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

from contextlib import contextmanager

import pytest
import torch
import torch.nn.functional as F
import triton

from fla.ops.common.chunk_delta_h import (
    chunk_gated_delta_rule_bwd_kernel_dhu_blockdim64,
    chunk_gated_delta_rule_fwd_kernel_h_blockdim64,
)
from fla.ops.precond_kda import chunk_precond_kda
from fla.ops.utils.cache import AutotuneKey
from fla.utils import IS_NVIDIA, assert_close, device


def _autotuner(kernel):
    while not hasattr(kernel, 'configs'):
        kernel = kernel.fn
    return kernel


@pytest.mark.parametrize('kernel', [
    chunk_gated_delta_rule_fwd_kernel_h_blockdim64,
    chunk_gated_delta_rule_bwd_kernel_dhu_blockdim64,
], ids=['forward', 'backward'])
def test_delta_h_tuning_separates_sequence_geometry(kernel):
    tuner = _autotuner(kernel)
    kwargs = dict(H=10, HV=10, K=128, V=128, BT=64, STATE_V_FIRST=False, USE_G=False)
    keys = [
        AutotuneKey.build(tuner.arg_names, tuner.keys, (), dict(kwargs, N=n, T=t)).autotune_key
        for n, t in [(1, 4096), (4, 1024), (1, 1024), (4, 4096)]
    ]
    assert len(set(keys)) == 4


@contextmanager
def _state_tile(bv):
    kernels = (chunk_gated_delta_rule_fwd_kernel_h_blockdim64, chunk_gated_delta_rule_bwd_kernel_dhu_blockdim64)
    previous = []
    try:
        for kernel in kernels:
            tuner = _autotuner(kernel)
            previous.append((tuner, tuner.configs, tuner.cache))
            tuner.configs = [triton.Config({'BV': bv}, num_warps=2, num_stages=2)]
            tuner.cache = {}
        yield
    finally:
        for tuner, configs, cache in previous:
            tuner.configs, tuner.cache = configs, cache


@pytest.mark.skipif(not IS_NVIDIA or torch.cuda.get_device_capability() != (8, 9), reason='Ada state tile')
@pytest.mark.parametrize('length', [64, 129], ids=['aligned', 'partial-tail'])
def test_pkda_state_bv16_outputs_and_gradients(length):
    torch.manual_seed(42)
    shape = (1, length, 2, 128)
    values = dict(
        q=F.normalize(torch.randn(shape, device=device), dim=-1).bfloat16(),
        k=F.normalize(torch.randn(shape, device=device), dim=-1).bfloat16(),
        v=torch.randn(shape, device=device, dtype=torch.bfloat16) * .1,
        g=-torch.rand(shape, device=device) * .1,
        g_atk=-torch.rand(shape[:-1], device=device) * .1,
        beta_atk=torch.rand(shape[:-1], device=device, dtype=torch.bfloat16) * .5,
        beta=torch.rand(shape[:-1], device=device, dtype=torch.bfloat16) * .5,
        log_atk_scale=torch.full((2,), -.2, device=device),
        initial_state=torch.randn(1, 2, 128, 128, device=device) * .01,
        initial_A_state=torch.rand(1, 2, 128, device=device) * .1,
    )
    douts = (
        torch.randn_like(values['v']),
        torch.randn_like(values['initial_state']),
        torch.randn_like(values['initial_A_state']),
    )
    results = []
    for bv in (32, 16):
        inputs = {name: value.detach().clone().requires_grad_() for name, value in values.items()}
        with _state_tile(bv):
            outputs = chunk_precond_kda(**inputs, output_final_state=True, disable_recompute=True)
            gradients = torch.autograd.grad(outputs, tuple(inputs.values()), grad_outputs=douts)
        results.append(tuple(t.detach().clone() for t in (*outputs, *gradients)))
    names = ('output', 'final_state', 'final_A_state', *('d_' + name for name in values))
    for name, ref, actual in zip(names, *results, strict=True):
        assert torch.isfinite(ref).all() and torch.isfinite(actual).all(), name
        assert_close(name, ref.float(), actual.float(), .006)
