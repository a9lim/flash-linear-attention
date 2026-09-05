# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Compare every intra-backward result with direct FP64 pairwise equations."""

import pytest
import torch

from fla.ops.precond_kda.chunk_intra import chunk_precond_kda_bwd_intra
from fla.utils import assert_close, device


def intra_reference(values):
    q, k, kp, g = (values[name].double() for name in ('q', 'k', 'k_precond', 'g'))
    batch, length, heads, width = q.shape
    dq = torch.zeros_like(q)
    dk = torch.zeros_like(k)
    dkp = torch.zeros_like(kp)
    db = torch.zeros((batch, length, heads), device=q.device, dtype=torch.float64)
    bounds = values.get('cu_seqlens')
    bounds = bounds.cpu().tolist() if bounds is not None else [0, length]
    for b in range(batch):
        for lo, hi in zip(bounds[:-1], bounds[1:]):
            for start in range(lo, hi, 64):
                stop = min(start + 64, hi)
                n = stop - start
                qt, kt, kpt, gt = (tensor[b, start:stop].permute(1, 0, 2) for tensor in (q, k, kp, g))
                beta = values['beta'][b, start:stop].double().T
                aq = values['dAqk'][b, start:stop, :, :n].double().permute(1, 0, 2)
                ak = values['dAkk'][b, start:stop, :, :n].double().permute(1, 0, 2)
                lower = torch.ones((n, n), device=q.device, dtype=torch.bool).tril()
                delta = (gt[:, :, None, :] - gt[:, None, :, :]).masked_fill(~lower[None, :, :, None], 0)
                decay = delta.exp2().masked_fill(~lower[None, :, :, None], 0)
                dq_local = (aq[:, :, :, None] * kpt[:, None, :, :] * decay).sum(2)
                dk_read = (ak[:, :, :, None] * kpt[:, None, :, :] * decay).sum(2)
                dk_local = dk_read * beta[:, :, None]
                dkp_local = ((aq[:, :, :, None] * qt[:, :, None, :]
                              + ak[:, :, :, None] * kt[:, :, None, :] * beta[:, :, None, None]) * decay).sum(1)
                dq[b, start:stop] = dq_local.permute(1, 0, 2)
                dk[b, start:stop] = dk_local.permute(1, 0, 2)
                dkp[b, start:stop] = dkp_local.permute(1, 0, 2)
                db[b, start:stop] = (dk_read * kt).sum(-1).T
    dg = values['dg'].double() + q * dq + k * dk - kp * dkp
    return (
        dq + values['dq'].double(),
        dk + values['dk'].double(),
        dkp + values['dk_precond'].double(),
        db + values['db'].double(),
        dg,
    )


def make_inputs(length, width, regime, varlen):
    torch.manual_seed(42)
    shape = (1, length, 3, width)
    values = {name: torch.randn(shape, device=device, dtype=torch.bfloat16) * 0.1
              for name in ('q', 'k', 'k_precond', 'dq', 'dk', 'dk_precond')}
    values['beta'] = torch.rand(shape[:-1], device=device, dtype=torch.bfloat16)
    values['db'] = torch.randn(shape[:-1], device=device) * 0.1
    values['dg'] = torch.randn(shape, device=device) * 0.1
    increments = -(0.5 + torch.rand(shape, device=device)) * {'mild': 0.25, 'mixed': 3.0, 'extreme': 32.0}[regime]
    values['g'] = torch.empty_like(increments)
    bounds = [0, 17, length] if varlen else [0, length]
    if varlen:
        values['cu_seqlens'] = torch.tensor(bounds, device=device, dtype=torch.int32)
    for name in ('dAqk', 'dAkk'):
        values[name] = torch.zeros((1, length, 3, 64), device=device)
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        for start in range(lo, hi, 64):
            stop = min(start + 64, hi)
            values['g'][:, start:stop] = increments[:, start:stop].cumsum(1)
            for name, diagonal in (('dAqk', 0), ('dAkk', -1)):
                triangle = torch.randn((3, stop - start, 64), device=device).tril(diagonal) * 0.1
                values[name][:, start:stop] = triangle.permute(1, 0, 2)
    values['defer_reverse_cumsum'] = True
    return values


@pytest.mark.parametrize('regime', ['mild', 'mixed', 'extreme'])
@pytest.mark.parametrize('length,width,varlen', [(65, 32, False), (128, 64, True), (1024, 128, False)])
def test_round6_intra_reference(regime, length, width, varlen):
    values = make_inputs(length, width, regime, varlen)
    expected = intra_reference(values)
    actual = chunk_precond_kda_bwd_intra(**values)
    # use the original full operator's gradient budgets against an independent FP64 reference
    for name, reference, result, budget in zip(
        ('dq', 'dk', 'dk_precond', 'dbeta', 'dg'), expected, actual, (0.008, 0.008, 0.008, 0.02, 0.02),
    ):
        assert_close(name, reference, result, budget)
