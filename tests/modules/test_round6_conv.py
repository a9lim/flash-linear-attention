# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import pytest
import torch
import torch.nn.functional as F

from fla.modules.conv import causal_conv1d
from fla.utils import assert_close, device


def _reference(x, weight, bias, residual, head_dim, norm_channels, activation):
    batch, length, channels = x.shape
    width = weight.shape[1]
    result = torch.zeros_like(x)
    for tap in range(width):
        delay = width - tap - 1
        shifted = F.pad(x[:, :max(length - delay, 0)], (0, 0, delay, 0))[:, :length]
        result = result + shifted * weight[:, tap]
    if bias is not None:
        result = result + bias
    if activation is not None:
        result = result * result.sigmoid()
    if norm_channels:
        heads = result[:, :, :norm_channels].reshape(batch, length, -1, head_dim)
        normalized = heads / (heads.square().sum(-1, keepdim=True) + 1e-6).sqrt()
        result = torch.cat((normalized.reshape(batch, length, norm_channels), result[:, :, norm_channels:]), -1)
    if residual is not None:
        result = result + residual
    return result


@pytest.mark.parametrize(
    ('B', 'T', 'H', 'HD', 'W', 'dtype', 'activation', 'bias', 'split', 'residual', 'strided', 'norm_heads'),
    [
        pytest.param(1, 1, 3, 32, 1, torch.float32, None, False, False, False, False, 3, id='single_tap_token'),
        pytest.param(2, 3, 3, 32, 7, torch.bfloat16, 'silu', True, True, False, True, 2, id='sequence_shorter_than_filter'),
        pytest.param(1, 31, 3, 64, 2, torch.float16, 'swish', True, True, False, True, 2, id='fp16_partial'),
        pytest.param(2, 63, 3, 128, 4, torch.bfloat16, 'silu', True, True, False, False, 2, id='partial_63'),
        pytest.param(1, 65, 3, 128, 4, torch.bfloat16, 'silu', False, True, False, True, 2, id='partial_65_strided'),
        pytest.param(2, 127, 3, 64, 7, torch.float32, None, True, False, True, True, 3, id='residual_fp32'),
        pytest.param(1, 129, 3, 64, 3, torch.bfloat16, 'silu', True, False, False, True, 0, id='no_normalized_channels'),
        pytest.param(4, 1024, 30, 128, 4, torch.bfloat16, 'silu', True, True, False, False, 20, id='production_three_slabs'),
    ],
)
def test_causal_conv1d_l2norm(B, T, H, HD, W, dtype, activation, bias, split, residual, strided, norm_heads):
    torch.manual_seed(42)
    D = H * HD
    x = torch.randn(B, T, D + HD if strided else D, device=device, dtype=dtype)[:, :, :D].detach().requires_grad_()
    weight = (torch.randn(D, W, device=device, dtype=torch.float32) * 0.2).requires_grad_()
    bias_tensor = torch.randn(D, device=device, dtype=torch.float32).requires_grad_() if bias else None
    residual_tensor = torch.randn_like(x).requires_grad_() if residual else None
    widths = (D // 3,) * 3 if split else None
    actual, state = causal_conv1d(
        x=x,
        weight=weight,
        bias=bias_tensor,
        residual=residual_tensor,
        activation=activation,
        l2norm_head_dim=HD,
        l2norm_channels=norm_heads * HD,
        split_outputs=widths,
    )
    assert state is None
    actuals = actual if split else (actual,)
    dy = torch.randn(B, T, D + HD if strided else D, device=device, dtype=dtype)[:, :, :D]
    dys = dy.split(widths, -1) if split else (dy,)
    inputs = [x, weight] + ([bias_tensor] if bias else []) + ([residual_tensor] if residual else [])
    gradients = torch.autograd.grad(actuals, inputs, dys)

    refs = [tensor.detach().float().requires_grad_() for tensor in inputs]
    x_ref, weight_ref = refs[:2]
    bias_ref = refs[2] if bias else None
    residual_ref = refs[-1] if residual else None
    reference = _reference(x_ref, weight_ref, bias_ref, residual_ref, HD, norm_heads * HD, activation).to(dtype)
    references = reference.split(widths, -1) if split else (reference,)
    reference_gradients = torch.autograd.grad(references, refs, dys)
    for index, (expected, result) in enumerate(zip(references, actuals, strict=True)):
        assert_close(f'output[{index}]', expected, result, 0.006)
    names = ['dx', 'dw'] + (['db'] if bias else []) + (['dr'] if residual else [])
    for name, expected, result in zip(names, reference_gradients, gradients, strict=True):
        assert_close(name, expected.to(result), result, 0.006)


@pytest.mark.parametrize('invalid', ['cache', 'varlen', 'final_state', 'split_residual', 'head_width'])
def test_causal_conv1d_l2norm_rejections(invalid):
    x = torch.zeros(1, 5, 96, device=device)
    weight = torch.ones(96, 4, device=device)
    kwargs = dict(l2norm_head_dim=32)
    if invalid == 'cache':
        kwargs['initial_state'] = torch.zeros(1, 96, 4, device=device)
    elif invalid == 'varlen':
        kwargs['cu_seqlens'] = torch.tensor([0, 5], dtype=torch.int32, device=device)
    elif invalid == 'final_state':
        kwargs['output_final_state'] = True
    elif invalid == 'split_residual':
        kwargs.update(split_outputs=(32, 32, 32), residual=torch.zeros_like(x))
    else:
        kwargs['l2norm_head_dim'] = 24
    with pytest.raises(ValueError):
        causal_conv1d(x=x, weight=weight, **kwargs)
