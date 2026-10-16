# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import math

import pytest
import torch
import torch.nn.functional as F

from fla.ops.atk.chunk_atk_bwd import chunk_atk_bwd
from fla.ops.atk.chunk_atk_fwd import chunk_atk_fwd
from fla.utils import assert_close, device


def _reference(k, beta, g, center, initial):
    state = initial.float()
    outputs = []
    for index in range(k.shape[1]):
        key = k[:, index].float()
        state = g[:, index].exp().unsqueeze(-1) * state + beta[:, index].float().unsqueeze(-1) * key.square()
        deviation = (state + 1e-6).log() - center[None, :, None]
        multiplier = (-math.log(1.5) * deviation / (1 + deviation.abs())).exp()
        outputs.append(key * multiplier)
    return torch.stack(outputs, dim=1).to(k.dtype), state


@pytest.mark.parametrize('length', [128, 129, 4096], ids=['aligned', 'partial-tail', 'long'])
def test_atk_scan_outputs_and_gradients(length):
    """Both chunk scans walk one state slice per program; outputs, the FP32 final
    state, and every gradient match the token-by-token reference."""
    torch.manual_seed(42)
    shape = (2, length, 2, 128)
    values = (
        F.normalize(torch.randn(shape, device=device), dim=-1).bfloat16(),
        (torch.rand(shape[:-1], device=device) * .5).bfloat16(),
        -torch.rand(shape[:-1], device=device) * .1,
        torch.full((2,), -.2, device=device),
        torch.rand(2, 2, 128, device=device) * .1,
    )
    douts = (torch.randn_like(values[0]), torch.randn_like(values[4]))
    leaves = tuple(value.detach().clone().requires_grad_() for value in values)
    expected_outputs = _reference(*leaves)
    expected_gradients = torch.autograd.grad(expected_outputs, leaves, grad_outputs=douts)
    k, beta, g, center, initial = values
    output, ac, a, sa, final = chunk_atk_fwd(
        k=k,
        beta=beta,
        log_g=g,
        initial_A_state=initial,
        output_final_state=True,
        x=1.5,
        eps=1e-6,
        log_atk_scale=center,
    )
    for buffer in (ac, a, sa, final):
        assert buffer.dtype == torch.float32 and torch.isfinite(buffer).all()
    dk, dbeta, dg, dcenter, dinitial = chunk_atk_bwd(
        k=k,
        g_raw=g,
        beta=beta,
        dk_precond=douts[0],
        ac=ac,
        a=a,
        sa=sa,
        initial_A_state=initial,
        x=1.5,
        eps=1e-6,
        log_atk_scale=center,
        dat=douts[1],
    )
    names = ('output', 'final_state', 'dk', 'dbeta', 'dg', 'dcenter', 'dinitial')
    # same reference budgets as the existing PKDA state/gradient tests
    tolerances = (.005, .005, .008, .02, .02, .02, .02)
    reference = (*expected_outputs, *expected_gradients)
    actual = (output, final, dk, dbeta, dg, dcenter, dinitial)
    for name, expected, value, tolerance in zip(names, reference, actual, tolerances, strict=True):
        assert torch.isfinite(value).all(), name
        assert_close(name, expected.float(), value.float(), tolerance)
