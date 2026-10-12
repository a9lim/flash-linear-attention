# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import os
import warnings

import torch
import torch.nn as nn
import triton
import triton.language as tl
from einops import rearrange

from fla.backends import dispatch
from fla.ops.cp import FLACPContext
from fla.ops.utils import prepare_chunk_indices
from fla.ops.utils.cache import fla_cache_autotune
from fla.utils import IS_AMD, IS_NVIDIA, autotune_cache_kwargs, deprecate_kwarg, get_device_capability, input_guard

try:
    from causal_conv1d import causal_conv1d_fn as causal_conv1d_fn_cuda
    from causal_conv1d import causal_conv1d_update as causal_conv1d_update_cuda
except ImportError:
    causal_conv1d_fn_cuda = None
    causal_conv1d_update_cuda = None

NUM_WARPS_AUTOTUNE = [2, 4, 8, 16] if IS_AMD else [4, 8, 16, 32]
STATIC_WARPS = 32 if not IS_AMD else 16


@triton.heuristics({
    'HAS_WEIGHT': lambda args: args['weight'] is not None,
    'HAS_BIAS': lambda args: args['bias'] is not None,
    'HAS_RESIDUAL': lambda args: args['residual'] is not None,
    'USE_INITIAL_STATE': lambda args: args['initial_state'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@fla_cache_autotune(
    configs=[
        triton.Config({'BD': BD}, num_warps=num_warps)
        for BD in [16, 32, 64, 128]
        for num_warps in NUM_WARPS_AUTOTUNE
    ],
    key=['D', 'W'],
    **autotune_cache_kwargs,
)
@triton.jit
def causal_conv1d_fwd_kernel(
    x,
    y,
    weight,
    bias,
    residual,
    cu_seqlens,
    initial_state,
    chunk_indices,
    B,
    T,
    stride_x_n,
    stride_x_t,
    stride_x_d,
    D: tl.constexpr,
    W: tl.constexpr,
    BT: tl.constexpr,
    BW: tl.constexpr,
    BD: tl.constexpr,
    NB: tl.constexpr,
    ACTIVATION: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    HAS_RESIDUAL: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_d, i_t, i_b = tl.program_id(0), tl.program_id(1), tl.program_id(2)

    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = eos - bos
        p_x = x + bos * stride_x_t
    else:
        i_n = i_b
        bos, eos = (i_b * T).to(tl.int64), (i_b * T + T).to(tl.int64)
        p_x = x + tl.cast(i_b, tl.int64) * stride_x_n

    o_d = i_d * BD + tl.arange(0, BD)
    o_t = i_t.to(tl.int64) * BT + tl.arange(0, BT)
    o_w = tl.arange(0, BW) + W - BW
    m_d = o_d < D
    m_w = o_w >= 0
    m_y = (o_t < T)[:, None] & m_d[None, :]

    if HAS_WEIGHT:
        # [BD, BW]
        b_w = tl.load(weight + o_d[:, None] * W + o_w, mask=m_d[:, None] & m_w, other=0).to(tl.float32)

    b_y = tl.zeros((BT, BD), dtype=tl.float32)
    if not USE_INITIAL_STATE:
        for i_w in tl.static_range(-W + 1, 1):
            o_x = o_t + i_w
            p_yi = p_x + o_x[:, None] * stride_x_t + o_d[None, :] * stride_x_d
            # [BT, BD]
            b_yi = tl.load(p_yi, mask=((o_x >= 0) & (o_x < T))[:, None] & m_d[None, :], other=0.0).to(tl.float32)
            if HAS_WEIGHT:
                b_yi *= tl.sum(b_w * (o_w == (i_w + W - 1)), 1)
            b_y += b_yi
    elif i_t * BT >= W:
        # separate branches avoid a Triton compiler issue.
        for i_w in tl.static_range(-W + 1, 1):
            o_x = o_t + i_w
            p_yi = p_x + o_x[:, None] * stride_x_t + o_d[None, :] * stride_x_d
            # [BT, BD]
            b_yi = tl.load(p_yi, mask=((o_x >= 0) & (o_x < T))[:, None] & m_d[None, :], other=0.0).to(tl.float32)
            if HAS_WEIGHT:
                b_yi *= tl.sum(b_w * (o_w == (i_w + W - 1)), 1)
            b_y += b_yi
    else:
        o_t = i_t.to(tl.int64) * BT + tl.arange(0, BT)
        for i_w in tl.static_range(-W + 1, 1):
            o_x = o_t + i_w
            m_x = ((o_x >= 0) & (o_x < T))[:, None] & m_d
            m_c = ((o_x + W >= 0) & (o_x < 0))[:, None] & m_d

            b_yi = tl.load(p_x + o_x[:, None] * stride_x_t + o_d * stride_x_d, mask=m_x, other=0).to(tl.float32)

            b_yi += tl.load(initial_state + i_n * D*W + o_d * W + (o_x + W)[:, None], mask=m_c, other=0).to(tl.float32)

            if HAS_WEIGHT:
                b_yi *= tl.sum(b_w * (o_w == (i_w + W - 1)), 1)
            b_y += b_yi

    if HAS_BIAS:
        b_y += tl.load(bias + o_d, mask=m_d).to(tl.float32)

    if ACTIVATION == 'swish' or ACTIVATION == 'silu':
        b_y = b_y * tl.sigmoid(b_y)

    if HAS_RESIDUAL:
        p_residual = residual + bos * D + o_t[:, None] * D + o_d[None, :]
        b_residual = tl.load(p_residual, mask=m_y, other=0.0)
        b_y += b_residual

    p_y = y + bos * D + o_t[:, None] * D + o_d[None, :]
    tl.store(p_y, tl.cast(b_y, dtype=p_y.dtype.element_ty, fp_downcast_rounding='rtne'), mask=m_y)


@triton.heuristics({
    'HAS_WEIGHT': lambda args: args['dw'] is not None,
    'HAS_BIAS': lambda args: args['db'] is not None,
    'USE_INITIAL_STATE': lambda args: args['initial_state'] is not None,
    'USE_FINAL_STATE': lambda args: args['dht'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@fla_cache_autotune(
    configs=[
        triton.Config({'BD': BD}, num_warps=num_warps)
        for BD in [16, 32, 64, 128]
        for num_warps in NUM_WARPS_AUTOTUNE
    ],
    key=['D', 'W'],
    **autotune_cache_kwargs,
)
@triton.jit
def causal_conv1d_bwd_kernel(
    x,
    y,
    weight,
    initial_state,
    dht,
    dy,
    dx,
    dw,
    db,
    cu_seqlens,
    chunk_indices,
    B,
    T,
    stride_x_n,
    stride_x_t,
    stride_x_d,
    stride_dx_n,
    stride_dx_t,
    stride_dx_d,
    stride_dy_n,
    stride_dy_t,
    stride_dy_d,
    D: tl.constexpr,
    W: tl.constexpr,
    BT: tl.constexpr,
    BW: tl.constexpr,
    BD: tl.constexpr,
    NB: tl.constexpr,
    ACTIVATION: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    USE_FINAL_STATE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_d, i_t, i_b = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    if IS_VARLEN:
        i_tg = i_t
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = eos - bos
        p_x = x + bos * stride_x_t
    else:
        i_tg = i_b * tl.num_programs(1) + i_t
        i_n = i_b
        bos, eos = (i_b * T).to(tl.int64), (i_b * T + T).to(tl.int64)
        p_x = x + tl.cast(i_b, tl.int64) * stride_x_n

    if IS_VARLEN:
        p_dy = dy + bos * stride_dy_t
    else:
        p_dy = dy + tl.cast(i_b, tl.int64) * stride_dy_n

    o_d = i_d * BD + tl.arange(0, BD)
    o_t = i_t.to(tl.int64) * BT + tl.arange(0, BT)
    o_w = tl.arange(0, BW) + W - BW
    m_d = o_d < D
    m_w = o_w >= 0
    m_x = (o_t < T)[:, None] & m_d[None, :]

    if HAS_WEIGHT:
        p_x = p_x + o_t[:, None] * stride_x_t + o_d[None, :] * stride_x_d
        b_x = tl.load(p_x, mask=m_x, other=0.0)
        # [BD, BW]
        b_w = tl.load(weight + o_d[:, None] * W + o_w, mask=m_d[:, None] & m_w, other=0)

    b_dx = tl.zeros((BT, BD), dtype=tl.float32)
    if HAS_BIAS:
        b_db = tl.zeros((BD,), dtype=tl.float32)

    if not USE_FINAL_STATE and not USE_INITIAL_STATE:
        for i_w in tl.static_range(0, W):
            o_dy = o_t + i_w
            p_dy_blk = p_dy + o_dy[:, None] * stride_dy_t + o_d[None, :] * stride_dy_d
            # [BT, BD]
            b_dy = tl.load(p_dy_blk, mask=((o_dy < T)[:, None] & m_d[None, :]), other=0.0).to(tl.float32)
            if ACTIVATION == 'swish' or ACTIVATION == 'silu':
                p_y = y + bos * D + o_dy[:, None] * D + o_d[None, :]
                b_y = tl.load(p_y, mask=((o_dy < T)[:, None] & m_d[None, :]), other=0.0).to(tl.float32)
                b_ys = tl.sigmoid(b_y)
                b_dy = b_dy * b_ys * (1 + b_y * (1 - b_ys))
            b_wdy = b_dy
            if HAS_WEIGHT:
                # [BT, BD]
                b_wdy = b_wdy * tl.sum(b_w * (o_w == (W - i_w - 1)), 1)
                # [BD]
                b_dw = tl.sum(b_dy * b_x, 0)
                tl.store(dw + i_tg * D*W + o_d * W + W - i_w - 1, b_dw.to(dw.dtype.element_ty), mask=m_d)
            if HAS_BIAS and i_w == 0:
                b_db += tl.sum(b_dy, 0)
            b_dx += b_wdy
    elif i_t * BT >= W:
        # separate branches avoid a Triton compiler issue.
        for i_w in tl.static_range(0, W):
            o_dy = o_t + i_w
            p_dy_blk = p_dy + o_dy[:, None] * stride_dy_t + o_d[None, :] * stride_dy_d
            # [BT, BD]
            b_dy = tl.load(p_dy_blk, mask=((o_dy < T)[:, None] & m_d[None, :]), other=0.0).to(tl.float32)
            if ACTIVATION == 'swish' or ACTIVATION == 'silu':
                p_y = y + bos * D + o_dy[:, None] * D + o_d[None, :]
                b_y = tl.load(p_y, mask=((o_dy < T)[:, None] & m_d[None, :]), other=0.0).to(tl.float32)
                b_ys = tl.sigmoid(b_y)
                b_dy = b_dy * b_ys * (1 + b_y * (1 - b_ys))
            b_wdy = b_dy
            if HAS_WEIGHT:
                # [BT, BD]
                b_wdy = b_wdy * tl.sum(b_w * (o_w == (W - i_w - 1)), 1)
                # [BD]
                b_dw = tl.sum(b_dy * b_x, 0)
                tl.store(dw + i_tg * D*W + o_d * W + W - i_w - 1, b_dw.to(dw.dtype.element_ty), mask=m_d)
            if HAS_BIAS and i_w == 0:
                b_db += tl.sum(b_dy, 0)
            b_dx += b_wdy
    else:
        o_t = i_t.to(tl.int64) * BT + tl.arange(0, BT)
        for i_w in tl.static_range(0, W):
            o_dy = o_t + i_w
            p_dy_blk = p_dy + o_dy[:, None] * stride_dy_t + o_d[None, :] * stride_dy_d
            b_dy_shift = tl.load(p_dy_blk, mask=((o_dy < T)[:, None] & m_d[None, :]), other=0.0).to(tl.float32)
            if ACTIVATION == 'swish' or ACTIVATION == 'silu':
                p_y = y + bos * D + o_dy[:, None] * D + o_d[None, :]
                b_y_shift = tl.load(p_y, mask=((o_dy < T)[:, None] & m_d[None, :]), other=0.0).to(tl.float32)
                b_ys = tl.sigmoid(b_y_shift)
                b_dy_shift = b_dy_shift * b_ys * (1 + b_y_shift * (1 - b_ys))
            if HAS_WEIGHT:
                b_dw = tl.sum(b_dy_shift * b_x, 0)
                if USE_INITIAL_STATE:
                    mask_head_rows = (o_t < i_w) & (o_t < T)
                    b_dy_head = tl.load(
                        p_dy + o_t[:, None] * stride_dy_t + o_d * stride_dy_d,
                        mask=mask_head_rows[:, None] & m_d[None, :],
                        other=0.0,
                    ).to(tl.float32)
                    if ACTIVATION == 'swish' or ACTIVATION == 'silu':
                        # cache gradients use the unshifted preactivation y[t].
                        b_y_head = tl.load(
                            y + bos * D + o_t[:, None] * D + o_d,
                            mask=mask_head_rows[:, None] & m_d[None, :],
                            other=0.0,
                        ).to(tl.float32)
                        b_ys_head = tl.sigmoid(b_y_head)
                        b_dy_head = b_dy_head * b_ys_head * (1 + b_y_head * (1 - b_ys_head))
                    o_c = W - i_w + o_t
                    # index 0 is padding 0
                    mask_c = (mask_head_rows & (o_c >= 1) & (o_c < W))
                    b_xc = tl.load(
                        initial_state + i_n * D * W + o_d[None, :] * W + o_c[:, None],
                        mask=mask_c[:, None] & m_d[None, :],
                        other=0.0,
                    ).to(tl.float32)
                    b_dw += tl.sum(b_dy_head * b_xc, 0)
                tl.store(dw + i_tg * D * W + o_d * W + W - i_w - 1, b_dw.to(dw.dtype.element_ty), mask=m_d)

            if HAS_BIAS and i_w == 0:
                b_db += tl.sum(b_dy_shift, 0)
            b_wdy = b_dy_shift if not HAS_WEIGHT else (b_dy_shift * tl.sum(b_w * (o_w == (W - i_w - 1)), 1))
            b_dx += b_wdy

    if HAS_BIAS:
        b_db = tl.cast(b_db, dtype=db.dtype.element_ty, fp_downcast_rounding='rtne')
        tl.store(db + i_tg * D + o_d, b_db, mask=m_d)

    if USE_FINAL_STATE:
        if i_t * BT + BT >= T-W:
            start_tok = max(0, T - (W - 1))
            offset = i_t * BT + tl.arange(0, BT)
            mask = (offset >= start_tok) & (offset < T)
            # state slot w is token T - W + w, so token t takes dht[:, t + W - T].
            # start_tok is clamped, which only matters for the mask (offset >= 0 anyway),
            # so the slot index has to come from the unclamped expression.
            w_idx = offset + (W - T)
            dht_off = i_n * D * W + o_d[None, :] * W + w_idx[:, None]
            b_dht = tl.load(dht + dht_off, mask=mask[:, None] & m_d[None, :], other=0.).to(tl.float32)
            b_dx += b_dht

    if IS_VARLEN:
        p_dx = dx + bos * stride_dx_t
    else:
        p_dx = dx + tl.cast(i_b, tl.int64) * stride_dx_n

    p_dx = p_dx + o_t[:, None] * stride_dx_t + o_d[None, :] * stride_dx_d
    tl.store(p_dx, tl.cast(b_dx, dtype=p_dx.dtype.element_ty, fp_downcast_rounding='rtne'), mask=m_x)


@triton.heuristics({
    'HAS_BIAS': lambda args: args['bias'] is not None,
    'HAS_RESIDUAL': lambda args: args['residual'] is not None,
})
@fla_cache_autotune(
    configs=[triton.Config({}, num_warps=num_warps) for num_warps in [2, 4, 8]],
    key=['D', 'W', 'BT', 'BD', 'NORM_D'],
    **autotune_cache_kwargs,
)
@triton.jit
def causal_conv1d_fwd_l2norm_kernel(
    x,
    y,
    weight,
    bias,
    residual,
    T,
    stride_x_n,
    stride_x_t,
    stride_x_d,
    Y_OFF,
    YD,
    D: tl.constexpr,
    W: tl.constexpr,
    BT: tl.constexpr,
    BW: tl.constexpr,
    BD: tl.constexpr,
    NORM_D: tl.constexpr,
    EPS: tl.constexpr,
    ACTIVATION: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    HAS_RESIDUAL: tl.constexpr,
):
    """Dense fixed-length causal conv with the activation and a per-head L2 norm fused.

    Every program owns one head (``BD`` channels) of ``BT`` rows; channels below
    ``NORM_D`` are normalized over the head after the activation. No cache state.

    ``y`` is a ``[B, T, YD]`` slab holding the input channels
    ``[Y_OFF, Y_OFF + YD)``; the whole output is one slab with ``Y_OFF=0`` and
    ``YD=D``.
    """
    i_d, i_t, i_b = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    bos = (i_b * T).to(tl.int64)
    p_x = x + tl.cast(i_b, tl.int64) * stride_x_n

    o_dl = i_d * BD + tl.arange(0, BD)
    o_d = Y_OFF + o_dl
    o_t = i_t.to(tl.int64) * BT + tl.arange(0, BT)
    o_w = tl.arange(0, BW) + W - BW
    m_d = o_d < D
    m_w = o_w >= 0
    m_y = (o_t < T)[:, None] & m_d[None, :]

    # [BD, BW]
    b_w = tl.load(weight + o_d[:, None] * W + o_w, mask=m_d[:, None] & m_w, other=0).to(tl.float32)

    b_y = tl.zeros((BT, BD), dtype=tl.float32)
    for i_w in tl.static_range(-W + 1, 1):
        o_x = o_t + i_w
        p_yi = p_x + o_x[:, None] * stride_x_t + o_d[None, :] * stride_x_d
        b_yi = tl.load(p_yi, mask=((o_x >= 0) & (o_x < T))[:, None] & m_d[None, :], other=0.0).to(tl.float32)
        b_yi *= tl.sum(b_w * (o_w == (i_w + W - 1)), 1)
        b_y += b_yi

    if HAS_BIAS:
        b_y += tl.load(bias + o_d, mask=m_d).to(tl.float32)

    if ACTIVATION == 'swish' or ACTIVATION == 'silu':
        b_y = b_y * tl.sigmoid(b_y)

    if Y_OFF + i_d * BD < NORM_D:
        b_rstd = 1 / tl.sqrt(tl.sum(b_y * b_y, 1) + EPS)
        b_y = b_y * b_rstd[:, None]

    if HAS_RESIDUAL:
        p_residual = residual + bos * D + o_t[:, None] * D + o_d[None, :]
        b_y += tl.load(p_residual, mask=m_y, other=0.0)

    p_y = y + bos * YD + o_t[:, None] * YD + o_dl[None, :]
    tl.store(p_y, tl.cast(b_y, dtype=p_y.dtype.element_ty, fp_downcast_rounding='rtne'), mask=m_y)


@triton.jit
def _causal_conv1d_l2norm_derivative(
    p_x,
    p_dy,
    b_w,
    b_bias,
    o_w,
    o_d,
    o_dl,
    o_dy,
    normalize,
    T,
    stride_x_t,
    stride_x_d,
    stride_dy_t,
    stride_dy_d,
    D: tl.constexpr,
    W: tl.constexpr,
    BD: tl.constexpr,
    EPS: tl.constexpr,
    ACTIVATION: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    m_d = o_d < D
    m_dy = (o_dy < T)[:, None] & m_d[None, :]
    b_dz = tl.load(p_dy + o_dy[:, None] * stride_dy_t + o_dl[None, :] * stride_dy_d, mask=m_dy, other=0.0).to(tl.float32)
    b_y = tl.full((o_dy.shape[0], BD), 0, dtype=tl.float32)
    for j_w in tl.static_range(-W + 1, 1):
        o_x = o_dy + j_w
        b_xi = tl.load(
            p_x + o_x[:, None] * stride_x_t + o_d[None, :] * stride_x_d,
            mask=((o_x >= 0) & (o_x < T))[:, None] & m_d[None, :],
            other=0.0,
        ).to(tl.float32)
        b_xi *= tl.sum(b_w * (o_w == (j_w + W - 1)), 1)
        b_y += b_xi
    if HAS_BIAS:
        b_y += b_bias[None, :]

    if ACTIVATION == 'swish' or ACTIVATION == 'silu':
        b_ys = tl.sigmoid(b_y)
        b_s = b_y * b_ys
    else:
        b_s = b_y

    if normalize:
        b_rstd = 1 / tl.sqrt(tl.sum(b_s * b_s, 1) + EPS)
        b_z = b_s * b_rstd[:, None]
        b_dz = (b_dz - b_z * tl.sum(b_dz * b_z, 1)[:, None]) * b_rstd[:, None]

    if ACTIVATION == 'swish' or ACTIVATION == 'silu':
        return b_dz * b_ys * (1 + b_y * (1 - b_ys))
    return b_dz


@triton.heuristics({
    'HAS_BIAS': lambda args: args['db'] is not None,
})
@fla_cache_autotune(
    configs=[triton.Config({}, num_warps=num_warps) for num_warps in [2, 4, 8]],
    key=['D', 'W', 'BT', 'BD', 'NORM_D'],
    **autotune_cache_kwargs,
)
@triton.jit
def causal_conv1d_bwd_l2norm_kernel(
    x,
    weight,
    bias,
    dy,
    dx,
    dw,
    db,
    T,
    stride_x_n,
    stride_x_t,
    stride_x_d,
    stride_dx_n,
    stride_dx_t,
    stride_dx_d,
    stride_dy_n,
    stride_dy_t,
    stride_dy_d,
    Y_OFF,
    D: tl.constexpr,
    W: tl.constexpr,
    BT: tl.constexpr,
    BW: tl.constexpr,
    BD: tl.constexpr,
    NORM_D: tl.constexpr,
    EPS: tl.constexpr,
    ACTIVATION: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    """Backward of ``causal_conv1d_fwd_l2norm_kernel``.

    ``dy`` is the gradient of one output slab covering input channels
    ``[Y_OFF, Y_OFF + BD * num_programs(0))``; ``dx`` stays full width and
    ``dw``/``db`` are per-tile partials, ``dw`` transposed to ``[tile, W, D]``
    so a tap's channel row stores as one contiguous run.

    The nonlinear derivative is recomputed once for the input tile and once for
    the rows ahead of it, and a tap reaches the rows it needs by shifting both
    through shared memory; the untapped position needs no shift at all. Heads
    narrower than 32 channels cannot lower that gather in Triton and evaluate
    the derivative again per tap instead, which on an H100 PCIe at the screen
    shape would cost 0.900 ms against 0.500 ms.
    """
    i_d, i_t, i_b = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_tg = i_b * tl.num_programs(1) + i_t
    p_x = x + tl.cast(i_b, tl.int64) * stride_x_n
    p_dy = dy + tl.cast(i_b, tl.int64) * stride_dy_n

    o_dl = i_d * BD + tl.arange(0, BD)
    o_d = Y_OFF + o_dl
    o_t = i_t.to(tl.int64) * BT + tl.arange(0, BT)
    o_w = tl.arange(0, BW) + W - BW
    m_d = o_d < D
    m_w = o_w >= 0
    m_x = (o_t < T)[:, None] & m_d[None, :]

    b_x = tl.load(p_x + o_t[:, None] * stride_x_t + o_d[None, :] * stride_x_d, mask=m_x, other=0.0)
    # [BD, BW]
    b_w = tl.load(weight + o_d[:, None] * W + o_w, mask=m_d[:, None] & m_w, other=0).to(tl.float32)
    if HAS_BIAS:
        b_bias = tl.load(bias + o_d, mask=m_d, other=0.0).to(tl.float32)
        b_db = tl.zeros((BD,), dtype=tl.float32)

    if not HAS_BIAS:
        b_bias = tl.full((BD,), 0, dtype=tl.float32)
    normalize = Y_OFF + i_d * BD < NORM_D
    if BD >= 32:
        b_derivative = _causal_conv1d_l2norm_derivative(
            p_x=p_x,
            p_dy=p_dy,
            b_w=b_w,
            b_bias=b_bias,
            o_w=o_w,
            o_d=o_d,
            o_dl=o_dl,
            o_dy=o_t,
            normalize=normalize,
            T=T,
            stride_x_t=stride_x_t,
            stride_x_d=stride_x_d,
            stride_dy_t=stride_dy_t,
            stride_dy_d=stride_dy_d,
            D=D,
            W=W,
            BD=BD,
            EPS=EPS,
            ACTIVATION=ACTIVATION,
            HAS_BIAS=HAS_BIAS,
        )
        if W > 1:
            # four rows also keep one- and two-row gathers out of Triton's unsupported layout conversion
            BH: tl.constexpr = 4 if W <= 4 else triton.next_power_of_2(W - 1)
            o_halo = i_t.to(tl.int64) * BT + BT + tl.arange(0, BH)
            b_halo = _causal_conv1d_l2norm_derivative(
                p_x=p_x,
                p_dy=p_dy,
                b_w=b_w,
                b_bias=b_bias,
                o_w=o_w,
                o_d=o_d,
                o_dl=o_dl,
                o_dy=o_halo,
                normalize=normalize,
                T=T,
                stride_x_t=stride_x_t,
                stride_x_d=stride_x_d,
                stride_dy_t=stride_dy_t,
                stride_dy_d=stride_dy_d,
                D=D,
                W=W,
                BD=BD,
                EPS=EPS,
                ACTIVATION=ACTIVATION,
                HAS_BIAS=HAS_BIAS,
            )

    b_dx = tl.zeros((BT, BD), dtype=tl.float32)
    for i_w in tl.static_range(0, W):
        if BD >= 32:
            if i_w == 0:
                b_dy = b_derivative
            else:
                o_shift = tl.arange(0, BT) + i_w
                b_dy = tl.gather(b_derivative, tl.broadcast_to((o_shift % BT)[:, None], (BT, BD)), axis=0)
                o_tail = tl.maximum(o_shift - BT, 0)
                b_tail = tl.gather(b_halo, tl.broadcast_to(o_tail[:, None], (BT, BD)), axis=0)
                b_dy = tl.where((o_shift < BT)[:, None], b_dy, b_tail)
        else:
            b_dy = _causal_conv1d_l2norm_derivative(
                p_x=p_x,
                p_dy=p_dy,
                b_w=b_w,
                b_bias=b_bias,
                o_w=o_w,
                o_d=o_d,
                o_dl=o_dl,
                o_dy=o_t + i_w,
                normalize=normalize,
                T=T,
                stride_x_t=stride_x_t,
                stride_x_d=stride_x_d,
                stride_dy_t=stride_dy_t,
                stride_dy_d=stride_dy_d,
                D=D,
                W=W,
                BD=BD,
                EPS=EPS,
                ACTIVATION=ACTIVATION,
                HAS_BIAS=HAS_BIAS,
            )

        b_dw = tl.sum(b_dy * b_x, 0)
        tl.store(dw + i_tg * W*D + (W - i_w - 1) * D + o_d, b_dw.to(dw.dtype.element_ty), mask=m_d)
        if HAS_BIAS and i_w == 0:
            b_db += tl.sum(b_dy, 0)
        b_dx += b_dy * tl.sum(b_w * (o_w == (W - i_w - 1)), 1)

    if HAS_BIAS:
        tl.store(db + i_tg * D + o_d, tl.cast(b_db, dtype=db.dtype.element_ty, fp_downcast_rounding='rtne'), mask=m_d)

    p_dx = dx + tl.cast(i_b, tl.int64) * stride_dx_n + o_t[:, None] * stride_dx_t + o_d[None, :] * stride_dx_d
    tl.store(p_dx, tl.cast(b_dx, dtype=p_dx.dtype.element_ty, fp_downcast_rounding='rtne'), mask=m_x)


@triton.heuristics({
    'USE_INITIAL_STATE': lambda args: args['cache'] is not None,
    'HAS_WEIGHT': lambda args: args['weight'] is not None,
    'HAS_BIAS': lambda args: args['bias'] is not None,
    'HAS_RESIDUAL': lambda args: args['residual'] is not None,
})
@fla_cache_autotune(
    configs=[
        triton.Config({'BD': BD}, num_warps=num_warps)
        for BD in [8, 16, 32, 64, 128, 256]
        for num_warps in NUM_WARPS_AUTOTUNE
    ],
    key=['D', 'W'],
    restore_value=['cache'],
    **autotune_cache_kwargs,
)
@triton.jit
def causal_conv1d_update_kernel(
    x,
    cache,
    residual,
    y,
    weight,
    bias,
    stride_x_n,
    stride_x_d,
    stride_y_n,
    stride_y_d,
    D: tl.constexpr,
    W: tl.constexpr,
    BD: tl.constexpr,
    BW: tl.constexpr,
    ACTIVATION: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    HAS_RESIDUAL: tl.constexpr,
):
    pid = tl.program_id(0)
    ND = tl.cdiv(D, BD)
    i_d, i_n = pid % ND, (pid // ND).to(tl.int64)

    o_d = i_d * BD + tl.arange(0, BD)
    o_w = tl.arange(0, BW)
    m_d = o_d < D
    m_w = o_w < W

    # [BD]
    b_x = tl.load(x + i_n * stride_x_n + o_d * stride_x_d, mask=m_d, other=0).to(tl.float32)

    b_cache = tl.zeros((BD, BW), dtype=tl.float32)

    if USE_INITIAL_STATE:
        p_cache_read = cache + i_n * D*W + o_d[:, None] * W + (o_w + 1)[None, :]
        b_cache = tl.load(p_cache_read, mask=m_d[:, None] & ((o_w + 1) < W)[None, :], other=0.0).to(tl.float32)

        m_update = o_w == (W - 1)
        b_cache = tl.where(m_update[None, :], b_x[:, None], b_cache)

    if HAS_WEIGHT:
        b_w = tl.load(weight + o_d[:, None] * W + o_w, mask=m_d[:, None] & m_w, other=0)
        b_y = tl.sum(b_cache * b_w, 1)
    else:
        b_y = tl.sum(b_cache, 1)

    if HAS_BIAS:
        b_y += tl.load(bias + o_d, mask=m_d)

    if ACTIVATION == 'swish' or ACTIVATION == 'silu':
        b_y = b_y * tl.sigmoid(b_y)

    if HAS_RESIDUAL:
        b_y += tl.load(residual + i_n * D + o_d, mask=m_d, other=0)

    tl.store(
        y + i_n * stride_y_n + o_d * stride_y_d,
        tl.cast(b_y, dtype=y.dtype.element_ty, fp_downcast_rounding='rtne'),
        mask=m_d,
    )

    if USE_INITIAL_STATE:
        p_cache_write = cache + i_n * D*W + o_d[:, None] * W + o_w[None, :]
        tl.store(
            p_cache_write,
            tl.cast(b_cache, dtype=cache.dtype.element_ty, fp_downcast_rounding='rtne'),
            mask=m_d[:, None] & m_w[None, :],
        )


@triton.heuristics({
    'USE_ACTIVATION': lambda args: args['y'] is not None,
    'USE_FINAL_STATE': lambda args: args['dht'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.jit
def compute_dh0_kernel(
    dy,
    y,
    weight,
    dh0,
    dht,
    cu_seqlens,
    stride_dy_n,
    stride_dy_t,
    stride_dy_d,
    T,
    D: tl.constexpr,
    W: tl.constexpr,
    BD: tl.constexpr,
    USE_ACTIVATION: tl.constexpr,
    USE_FINAL_STATE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    """
    Compute dh0 (gradient w.r.t. initial_state) in a separate kernel.
    This avoids Triton compiler bugs on some architectures (e.g., GB200).

    Grid: (cdiv(D, BD) * N,)
    """
    pid = tl.program_id(0)
    ND = tl.cdiv(D, BD)
    i_d, i_n = pid % ND, (pid // ND).to(tl.int64)

    if IS_VARLEN:
        bos = tl.load(cu_seqlens + i_n).to(tl.int64)
        eos = tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        seq_len = eos - bos
        # for varlen, dy is [1, total_T, D], offset by bos
        dy_base = dy + bos * stride_dy_t
    else:
        seq_len = T
        # for non-varlen, dy is [B, T, D], offset by i_n * stride_dy_n
        dy_base = dy + tl.cast(i_n, tl.int64) * stride_dy_n

    o_d = i_d * BD + tl.arange(0, BD)
    m_d = o_d < D

    for i_w in tl.static_range(1, W):
        b_dh0 = tl.zeros([BD], dtype=tl.float32)

        if USE_FINAL_STATE:
            # short sequences retain initial_state[:, i_w] in final_state[:, i_w - seq_len],
            # so that slot's gradient passes straight through.
            if i_w >= seq_len:
                p_dht = dht + i_n * D * W + o_d * W + (i_w - seq_len)
                b_dh0 += tl.load(p_dht, mask=m_d, other=0).to(tl.float32)

        for t in tl.static_range(0, W - 1):
            if t < i_w:
                w_idx = i_w - 1 - t

                p_dy = dy_base + t * stride_dy_t + o_d * stride_dy_d
                m_t = (t < seq_len) & m_d
                b_dy = tl.load(p_dy, mask=m_t, other=0).to(tl.float32)

                if USE_ACTIVATION:
                    # `y` is recomputed contiguous [*, T, D]; only `dy` may carry other strides
                    if IS_VARLEN:
                        p_y = y + bos * D + t * D + o_d
                    else:
                        p_y = y + tl.cast(i_n, tl.int64) * T * D + t * D + o_d
                    b_y = tl.load(p_y, mask=m_t, other=0).to(tl.float32)
                    b_ys = tl.sigmoid(b_y)
                    b_dy = b_dy * b_ys * (1 + b_y * (1 - b_ys))

                b_w_col = tl.load(weight + o_d * W + w_idx, mask=m_d, other=0).to(tl.float32)

                b_dh0 += tl.where(m_t, b_dy * b_w_col, 0)

        p_dh0 = dh0 + i_n * D * W + o_d * W + i_w
        tl.store(p_dh0, b_dh0.to(dh0.dtype.element_ty), mask=m_d)


@triton.heuristics({
    'USE_INITIAL_STATE': lambda args: args['initial_state'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.jit
def causal_conv1d_states_fwd_kernel(
    x,
    initial_state,
    final_state,
    cu_seqlens,
    T,
    D,
    W,
    stride_x_n,
    stride_x_t,
    stride_x_d,
    BD: tl.constexpr,
    BW: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    pid = tl.program_id(0)
    ND = tl.cdiv(D, BD)
    i_d, i_n = pid % ND, (pid // ND).to(tl.int64)

    # [BD]
    o_d = i_d * BD + tl.arange(0, BD)
    m_d = o_d < D

    if IS_VARLEN:
        bos = tl.load(cu_seqlens + i_n).to(tl.int64)
        eos = tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        seq_len = (eos - bos).to(tl.int32)
        p_x = x + bos * stride_x_t
    else:
        seq_len = T
        p_x = x + tl.cast(i_n, tl.int64) * stride_x_n

    o_x = tl.cast(seq_len, tl.int64) - BW + tl.arange(0, BW)
    p_x = p_x + o_x[:, None] * stride_x_t + o_d[None, :] * stride_x_d

    # [BW, BD]
    b_x = tl.load(p_x, mask=((o_x >= 0) & (o_x < seq_len))[:, None] & m_d[None, :], other=0.0).to(tl.float32)

    if USE_INITIAL_STATE:
        if seq_len < BW:
            o_c = W - (BW - seq_len) + tl.arange(0, BW)
            m_c = (o_c >= 0) & (o_c < W)

            p_init = initial_state + i_n * D*W + o_d[None, :] * W + o_c[:, None]
            mask_init = m_d[None, :] & m_c[:, None]

            b_cache = tl.load(p_init, mask=mask_init, other=0)
            b_x += b_cache

    # final_state: [N, D, W]
    # [BW]
    o_w = W - BW + tl.arange(0, BW)

    # o_d[:, None] -> [BD, 1]
    # o_w[None, :] -> [1, BW]
    # [BD, BW]
    p_final = final_state + tl.cast(i_n, tl.int64) * D*W + o_d[:, None] * W + o_w[None, :]

    # [BD, BW]
    m_final = m_d[:, None] & (o_w[None, :] >= 0)

    tl.store(p_final, tl.trans(b_x).to(final_state.dtype.element_ty), mask=m_final)


def _has_non_standard_layout(x: torch.Tensor) -> bool:
    """QKV-style views (stride_t != D) break triton-ascend masked kernels."""
    if x.dtype not in (torch.float16, torch.bfloat16):
        return False
    if x.dim() != 3:
        return not x.is_contiguous()
    _, stride_t, stride_d = x.stride()
    return stride_d == 1 and stride_t != x.shape[-1]


def _split_widths(
    split_outputs: tuple[int, ...] | None,
    D: int,
    BD: int,
    residual: torch.Tensor | None,
) -> tuple[int, ...]:
    """Validate ``split_outputs`` and return the per-slab channel widths."""
    if split_outputs is None:
        return (D,)
    if residual is not None:
        raise ValueError("split_outputs and residual cannot be combined")
    if sum(split_outputs) != D:
        raise ValueError(f"split_outputs={split_outputs} must sum to D={D}")
    if any(w <= 0 or w % BD for w in split_outputs):
        raise ValueError(f"every split_outputs width must be a positive multiple of the channel tile {BD}")
    return tuple(split_outputs)


def _fused_kwargs(
    l2norm_head_dim: int | None,
    l2norm_channels: int | None,
    l2norm_eps: float,
    split_outputs: tuple[int, ...] | None,
) -> dict:
    """Keywords for the fused L2-norm/split path, empty when the call is a plain convolution."""
    if l2norm_head_dim is None and split_outputs is None:
        return {}
    return {
        "l2norm_head_dim": l2norm_head_dim,
        "l2norm_channels": l2norm_channels,
        "l2norm_eps": l2norm_eps,
        "split_outputs": split_outputs,
    }


def _l2norm_geometry(
    D: int,
    l2norm_head_dim: int,
    l2norm_channels: int | None,
    initial_state: torch.Tensor | None,
    cu_seqlens: torch.Tensor | None,
) -> tuple[int, int]:
    """Validate the fused conv+L2-norm request and return ``(BD, NORM_D)``."""
    if initial_state is not None or cu_seqlens is not None:
        raise ValueError("the fused conv+L2-norm path supports dense fixed-length inputs without cache state")
    norm_d = D if l2norm_channels is None else l2norm_channels
    if l2norm_head_dim & (l2norm_head_dim - 1) or D % l2norm_head_dim or norm_d % l2norm_head_dim:
        raise ValueError(
            f"l2norm_head_dim={l2norm_head_dim} must be a power of two dividing D={D} and "
            f"l2norm_channels={norm_d}"
        )
    return l2norm_head_dim, norm_d


@deprecate_kwarg('BT', version='0.7.0', new_name='chunk_size')
@dispatch
@input_guard(no_guard_contiguous=["x"])
def causal_conv1d_fwd(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    residual: torch.Tensor,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    activation: str | None = None,
    cu_seqlens: torch.LongTensor | None = None,
    cu_seqlens_cpu: torch.LongTensor | None = None,
    chunk_indices: torch.LongTensor | None = None,
    chunk_size: int = 64,
    layout_fallback: bool = False,
    l2norm_head_dim: int | None = None,
    l2norm_channels: int | None = None,
    l2norm_eps: float = 1e-6,
    split_outputs: tuple[int, ...] | None = None,
) -> torch.Tensor:
    BT = chunk_size
    shape = x.shape
    if x.shape[-1] != weight.shape[0]:
        x = rearrange(x, 'b t ... -> b t (...)')
    B, T, D = x.shape[0], x.shape[1], weight.shape[0]
    W = weight.shape[1]
    stride_x_n, stride_x_t, stride_x_d = x.stride()

    BW = triton.next_power_of_2(W)
    if l2norm_head_dim is not None:
        if output_final_state:
            raise ValueError("the fused conv+L2-norm path does not produce a final state")
        BD, NORM_D = _l2norm_geometry(D, l2norm_head_dim, l2norm_channels, initial_state, cu_seqlens)
        widths = _split_widths(split_outputs, D, BD, residual)
        ys = [torch.empty(B, T, w, dtype=x.dtype, device=x.device) for w in widths]
        off = 0
        for y, w in zip(ys, widths):
            causal_conv1d_fwd_l2norm_kernel[(w // BD, triton.cdiv(T, BT), B)](
                x=x,
                y=y,
                weight=weight,
                bias=bias,
                residual=residual,
                T=T,
                stride_x_n=stride_x_n,
                stride_x_t=stride_x_t,
                stride_x_d=stride_x_d,
                Y_OFF=off,
                YD=w,
                D=D,
                W=W,
                BT=BT,
                BW=BW,
                BD=BD,
                NORM_D=NORM_D,
                EPS=l2norm_eps,
                ACTIVATION=activation,
            )
            off += w
        if split_outputs is not None:
            return tuple(ys), None
        return ys[0].view(shape), None
    if cu_seqlens is not None and chunk_indices is None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT, cu_seqlens_cpu=cu_seqlens_cpu)
    NT = len(chunk_indices) if cu_seqlens is not None else triton.cdiv(T, BT)
    NB = triton.cdiv(B*T, 1024)

    y = torch.empty_like(x, memory_format=torch.contiguous_format)

    def grid(meta) -> tuple[int, ...]:

        return (triton.cdiv(D, meta['BD']), NT, B)
    causal_conv1d_fwd_kernel[grid](
        x=x,
        y=y,
        weight=weight,
        bias=bias,
        residual=residual,
        cu_seqlens=cu_seqlens,
        initial_state=initial_state,
        chunk_indices=chunk_indices,
        B=B,
        T=T,
        stride_x_n=stride_x_n,
        stride_x_t=stride_x_t,
        stride_x_d=stride_x_d,
        D=D,
        W=W,
        BT=BT,
        BW=BW,
        NB=NB,
        ACTIVATION=activation,
    )
    final_state = None
    if output_final_state:
        final_state = causal_conv1d_update_states(x=x, state_len=W, initial_state=initial_state, cu_seqlens=cu_seqlens)
    return y.view(shape), final_state


@dispatch
def compute_dh0_triton(
    dy: torch.Tensor,
    y: torch.Tensor | None,
    weight: torch.Tensor,
    initial_state: torch.Tensor,
    activation: str | None,
    cu_seqlens: torch.Tensor | None,
    dht: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Compute dh0 (gradient w.r.t. initial_state) using a separate Triton kernel.
    This is a workaround for Triton compiler bugs on some architectures (e.g., GB200).
    """
    D, W = weight.shape
    N = initial_state.shape[0]
    T = dy.shape[1]

    dh0 = torch.zeros_like(initial_state)

    BD = 32
    grid = (triton.cdiv(D, BD) * N,)

    y_to_pass = y if activation in ('swish', 'silu') else None
    # dy is [B, T, D] but may be a strided view; `y` is always contiguous
    stride_dy_n, stride_dy_t, stride_dy_d = dy.stride()

    compute_dh0_kernel[grid](
        dy=dy,
        y=y_to_pass,
        weight=weight,
        dh0=dh0,
        dht=dht,
        cu_seqlens=cu_seqlens,
        stride_dy_n=stride_dy_n,
        stride_dy_t=stride_dy_t,
        stride_dy_d=stride_dy_d,
        T=T,
        D=D,
        W=W,
        BD=BD,
    )

    return dh0


@deprecate_kwarg('BT', version='0.7.0', new_name='chunk_size')
@dispatch
def causal_conv1d_bwd(
    x: torch.Tensor,
    dy: torch.Tensor,
    dht: torch.Tensor,
    weight: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
    residual: torch.Tensor | None = None,
    initial_state: torch.Tensor | None = None,
    activation: str | None = None,
    cu_seqlens: torch.Tensor | None = None,
    cu_seqlens_cpu: torch.LongTensor | None = None,
    chunk_indices: torch.LongTensor | None = None,
    chunk_size: int = 64,
    layout_fallback: bool = False,
    l2norm_head_dim: int | None = None,
    l2norm_channels: int | None = None,
    l2norm_eps: float = 1e-6,
    split_outputs: tuple[int, ...] | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    BT = chunk_size
    shape = x.shape
    if x.shape[-1] != weight.shape[0]:
        x = rearrange(x, 'b t ... -> b t (...)')
    B, T, D = x.shape
    W = weight.shape[1] if weight is not None else None

    stride_x_n, stride_x_t, stride_x_d = x.stride()

    BW = triton.next_power_of_2(W)
    if l2norm_head_dim is not None:
        if dht is not None:
            raise ValueError("the fused conv+L2-norm path carries no final state")
        BD, NORM_D = _l2norm_geometry(D, l2norm_head_dim, l2norm_channels, initial_state, cu_seqlens)
        widths = _split_widths(split_outputs, D, BD, residual if split_outputs is not None else None)
        dys = list(dy) if split_outputs is not None else [dy]
        # the smaller Ada tile keeps compact halo gathers spill-free on the dense Q/K/V workload
        if (
            BT == 64 and (B, T, D, W, BD, NORM_D) == (4, 1024, 3840, 4, 128, 2560)
            and widths == (1280, 1280, 1280) and x.dtype == torch.bfloat16
            and IS_NVIDIA and get_device_capability(x.device.index) == (8, 9)
        ):
            BT = 32
        NT = triton.cdiv(T, BT)
        dx = torch.empty_like(x)
        # [tile, W, D]: a tap's channel row is one contiguous store.
        dw = weight.new_empty(B*NT, W, D, dtype=torch.float)
        db = bias.new_empty(B*NT, *bias.shape, dtype=torch.float) if bias is not None else None
        stride_dx_n, stride_dx_t, stride_dx_d = dx.stride()
        off = 0
        for dy_i, w in zip(dys, widths):
            stride_dy_n, stride_dy_t, stride_dy_d = dy_i.stride()
            causal_conv1d_bwd_l2norm_kernel[(w // BD, NT, B)](
                x=x,
                weight=weight,
                bias=bias,
                dy=dy_i,
                dx=dx,
                dw=dw,
                db=db,
                T=T,
                stride_x_n=stride_x_n,
                stride_x_t=stride_x_t,
                stride_x_d=stride_x_d,
                stride_dx_n=stride_dx_n,
                stride_dx_t=stride_dx_t,
                stride_dx_d=stride_dx_d,
                stride_dy_n=stride_dy_n,
                stride_dy_t=stride_dy_t,
                stride_dy_d=stride_dy_d,
                Y_OFF=off,
                D=D,
                W=W,
                BT=BT,
                BW=BW,
                BD=BD,
                NORM_D=NORM_D,
                EPS=l2norm_eps,
                ACTIVATION=activation,
            )
            off += w
        dw = dw.sum(0).transpose(0, 1).contiguous().to(weight)
        if db is not None:
            db = db.sum(0).to(bias)
        dr = dy if residual is not None else None
        return dx.view(shape), dw, db, dr, None
    stride_dy_n, stride_dy_t, stride_dy_d = dy.stride()
    if cu_seqlens is not None and chunk_indices is None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT, cu_seqlens_cpu=cu_seqlens_cpu)
    NT = len(chunk_indices) if cu_seqlens is not None else triton.cdiv(T, BT)
    NB = triton.cdiv(B*T, 1024)

    y = None
    if activation is not None:
        y, _ = causal_conv1d_fwd(
            x=x,
            weight=weight,
            bias=bias,
            residual=None,
            initial_state=initial_state,
            output_final_state=False,
            activation=None,
            cu_seqlens=cu_seqlens,
            cu_seqlens_cpu=cu_seqlens_cpu,
            chunk_indices=chunk_indices,
        )
    dx = torch.empty_like(x)
    dw = weight.new_empty(B*NT, *weight.shape, dtype=torch.float) if weight is not None else None
    db = bias.new_empty(B*NT, *bias.shape, dtype=torch.float) if bias is not None else None
    dr = dy if residual is not None else None

    stride_dx_n, stride_dx_t, stride_dx_d = dx.stride()

    def grid(meta) -> tuple[int, ...]:

        return (triton.cdiv(D, meta['BD']), NT, B)
    causal_conv1d_bwd_kernel[grid](
        x=x,
        y=y,
        weight=weight,
        initial_state=initial_state,
        dht=dht,
        dy=dy,
        dx=dx,
        dw=dw,
        db=db,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        B=B,
        T=T,
        stride_x_n=stride_x_n,
        stride_x_t=stride_x_t,
        stride_x_d=stride_x_d,
        stride_dx_n=stride_dx_n,
        stride_dx_t=stride_dx_t,
        stride_dx_d=stride_dx_d,
        stride_dy_n=stride_dy_n,
        stride_dy_t=stride_dy_t,
        stride_dy_d=stride_dy_d,
        D=D,
        W=W,
        BT=BT,
        BW=BW,
        NB=NB,
        ACTIVATION=activation,
    )
    if weight is not None:
        dw = dw.sum(0).to(weight)
    if bias is not None:
        db = db.sum(0).to(bias)

    # compute dh0 using separate Triton kernel to avoid compiler bugs on some architectures (e.g., GB200)
    dh0 = None
    if initial_state is not None:
        dh0 = compute_dh0_triton(
            dy=dy,
            y=y,
            weight=weight,
            initial_state=initial_state,
            activation=activation,
            cu_seqlens=cu_seqlens,
            dht=dht,
        )

    return dx.view(shape), dw, db, dr, dh0


@dispatch
@input_guard(no_guard_contiguous=["x"])
def causal_conv1d_update_states(
    x: torch.Tensor,
    state_len: int,
    initial_state: torch.Tensor | None = None,
    cu_seqlens: torch.Tensor | None = None,
) -> torch.Tensor:
    if cu_seqlens is not None:
        N = len(cu_seqlens) - 1
        if x.dim() == 2:
            stride_x_n = 0
            stride_x_t, stride_x_d = x.stride()
            T = x.shape[0]
        else:
            stride_x_n = x.stride(0)
            stride_x_t, stride_x_d = x.stride(1), x.stride(2)
            T = x.shape[1]
        D = x.shape[-1]
    else:
        B, T, D = x.shape
        N = B
        stride_x_n, stride_x_t, stride_x_d = x.stride()

    W = state_len
    final_state = torch.empty(N, D, W, dtype=x.dtype, device=x.device)

    BD = min(triton.next_power_of_2(D), 256)
    BW = triton.next_power_of_2(W)

    grid = (triton.cdiv(D, BD) * N,)

    causal_conv1d_states_fwd_kernel[grid](
        x=x,
        initial_state=initial_state,
        final_state=final_state,
        cu_seqlens=cu_seqlens,
        T=T,
        D=D,
        W=W,
        stride_x_n=stride_x_n,
        stride_x_t=stride_x_t,
        stride_x_d=stride_x_d,
        BD=BD,
        BW=BW,
    )
    return final_state


@dispatch
@input_guard(no_guard_contiguous=["x"])
def causal_conv1d_update(
    x: torch.Tensor,
    cache: torch.Tensor,
    residual: torch.Tensor | None = None,
    weight: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
    activation: str | None = None,
) -> torch.Tensor:
    shape = x.shape
    if weight is not None and x.shape[-1] != weight.shape[0]:
        x = rearrange(x, 'b t ... -> b t (...)')

    D = x.shape[-1]
    N = x.numel() // D
    W = weight.shape[1] if weight is not None else None
    BW = triton.next_power_of_2(W)

    if x.dim() == 2:
        # case: (N, D)
        stride_x_n = x.stride(0)
        stride_x_d = x.stride(1)
    elif x.dim() == 3 and x.shape[0] == 1:
        # [1, N, D]: one token per sequence
        stride_x_n = x.stride(1)
        stride_x_d = x.stride(2)
    elif x.dim() == 3:
        # [N, 1, D]: one token per sequence
        stride_x_n = x.stride(0)
        stride_x_d = x.stride(2)
    else:
        raise ValueError(f"Unsupported input shape: {x.shape}")

    y = torch.empty_like(x, memory_format=torch.contiguous_format)

    if y.dim() == 2:
        stride_y_n, stride_y_d = y.stride(0), y.stride(1)
    elif y.dim() == 3 and y.shape[0] == 1:
        stride_y_n, stride_y_d = y.stride(1), y.stride(2)
    elif y.dim() == 3:
        stride_y_n, stride_y_d = y.stride(0), y.stride(2)

    def grid(meta) -> tuple[int, ...]:

        return (triton.cdiv(D, meta['BD']) * N,)

    causal_conv1d_update_kernel[grid](
        x=x,
        cache=cache,
        residual=residual,
        y=y,
        weight=weight,
        bias=bias,
        stride_x_n=stride_x_n,
        stride_x_d=stride_x_d,
        stride_y_n=stride_y_n,
        stride_y_d=stride_y_d,
        D=D,
        W=W,
        BW=BW,
        ACTIVATION=activation,
    )
    return y.view(shape), cache


class CausalConv1dFunction(torch.autograd.Function):

    @staticmethod
    @input_guard(no_guard_contiguous=["x"])
    def forward(
        ctx,
        x: torch.Tensor,
        weight: torch.Tensor | None = None,
        bias: torch.Tensor | None = None,
        residual: torch.Tensor | None = None,
        initial_state: torch.Tensor | None = None,
        output_final_state: bool | None = False,
        activation: str | None = None,
        cu_seqlens: torch.Tensor | None = None,
        cu_seqlens_cpu: torch.LongTensor | None = None,
        chunk_indices: torch.LongTensor | None = None,
        chunk_size: int = 64,
        l2norm_head_dim: int | None = None,
        l2norm_channels: int | None = None,
        l2norm_eps: float = 1e-6,
        split_outputs: tuple[int, ...] | None = None,
    ) -> tuple[torch.Tensor, ...]:
        if cu_seqlens is not None and chunk_indices is None:
            chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size, cu_seqlens_cpu=cu_seqlens_cpu)
        ctx.activation = activation
        ctx.cu_seqlens = cu_seqlens
        ctx.cu_seqlens_cpu = cu_seqlens_cpu
        ctx.chunk_indices = chunk_indices
        ctx.layout_fallback = _has_non_standard_layout(x=x)
        # Only fused calls carry the fused keywords, so plain calls keep upstream's
        # signature and stay eligible for every registered backend.
        ctx.fused = _fused_kwargs(l2norm_head_dim, l2norm_channels, l2norm_eps, split_outputs)
        ctx.split_outputs = split_outputs
        ctx.save_for_backward(x, weight, bias, residual, initial_state)
        y, final_state = causal_conv1d_fwd(
            x=x,
            weight=weight,
            bias=bias,
            residual=residual,
            initial_state=initial_state,
            output_final_state=output_final_state,
            activation=activation,
            cu_seqlens=cu_seqlens,
            cu_seqlens_cpu=cu_seqlens_cpu,
            chunk_indices=chunk_indices,
            chunk_size=chunk_size,
            layout_fallback=ctx.layout_fallback,
            **ctx.fused,
        )
        if split_outputs is not None:
            return (*y, final_state)
        return y, final_state

    @staticmethod
    @input_guard(no_guard_contiguous=True)
    def backward(ctx, *grads: torch.Tensor | None) -> tuple[torch.Tensor | None, ...]:
        if ctx.split_outputs is not None:
            n = len(ctx.split_outputs)
            dy, dht = [g.contiguous() for g in grads[:n]], grads[n] if len(grads) > n else None
        else:
            dy, dht = grads[0], grads[1] if len(grads) > 1 else None
            dht = dht.contiguous() if isinstance(dht, torch.Tensor) else dht
        x, weight, bias, residual, initial_state = ctx.saved_tensors
        dx, dw, db, dr, dh0 = causal_conv1d_bwd(
            x=x,
            dy=dy,
            dht=dht,
            weight=weight,
            bias=bias,
            residual=residual,
            initial_state=initial_state,
            activation=ctx.activation,
            cu_seqlens=ctx.cu_seqlens,
            cu_seqlens_cpu=ctx.cu_seqlens_cpu,
            chunk_indices=ctx.chunk_indices,
            layout_fallback=ctx.layout_fallback,
            **ctx.fused,
        )
        return dx, dw, db, dr, dh0, None, None, None, None, None, None, None, None, None, None


@input_guard(no_guard_contiguous=["x"])
def causal_conv1d(
    x: torch.Tensor,
    weight: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
    residual: torch.Tensor | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool | None = False,
    activation: str | None = None,
    backend: str | None = 'triton',
    cu_seqlens: torch.Tensor | None = None,
    cu_seqlens_cpu: torch.LongTensor | None = None,
    chunk_indices: torch.LongTensor | None = None,
    cp_context: FLACPContext | None = None,
    l2norm_head_dim: int | None = None,
    l2norm_channels: int | None = None,
    l2norm_eps: float = 1e-6,
    split_outputs: tuple[int, ...] | None = None,
    **kwargs,
) -> tuple[torch.Tensor | tuple[torch.Tensor, ...], torch.Tensor | None]:
    """
    A causal 1D convolution implementation that powers Mamba/Mamba2 and DeltaNet architectures.

    With a residual connection, this implements the Canon operation:
    https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5240330.

    Args:
        x (torch.Tensor):
            Input tensor of shape [B, T, D].
        weight (torch.Tensor, Optional):
            Weight tensor of shape [D, W]. Default: `None`.
        bias (torch.Tensor, Optional):
            Bias tensor of shape [D]. Default: `None`.
        residual (torch.Tensor, Optional):
            Residual tensor of shape [B, T, D]. Default: `None`.
        initial_state (torch.Tensor, Optional):
            Initial state tensor of shape [N, D, W],
            where `N` is the number of sequences in the batch and `W` is the kernel size.
            If provided, the initial state is used to initialize the cache. Default: `None`.
        output_final_state (bool, Optional):
            Whether to output the final state of shape [N, D, W]. Default: `False`.
        activation (str, Optional):
            Activations applied to output, only `swish`/`silu` or `None` (i.e., no activation) are supported.
            Default: `None`.
        backend (str, Optional):
            Specifies the backend to use for the convolution operation. Supported values are `'cuda'`, `'triton'`, and `'mix'`.
            Default: `'triton'`.
        cu_seqlens (torch.Tensor, Optional):
            Cumulative sequence lengths. Default: `None`.
        cu_seqlens_cpu (torch.LongTensor, Optional):
            CPU copy of the cumulative sequence lengths. Default: `None`.
        chunk_indices (torch.Tensor, Optional):
            Chunk indices for variable-length sequences. Default: `None`.
        cp_context (FLACPContext, Optional):
            Context-parallel sequence metadata. Default: `None`.
        l2norm_head_dim (int, Optional):
            When set, the Triton backend fuses a per-head L2 normalization (heads of
            this many channels, applied after the activation) into the convolution
            for the first `l2norm_channels` channels (all channels by default).
            Dense fixed-length training only: no cache state, no `cu_seqlens`. Default: `None`.
        l2norm_channels (int, Optional):
            Number of leading channels the fused L2 normalization covers. Default: `None`.
        l2norm_eps (float):
            Epsilon inside the fused normalization's square root. Default: `1e-6`.
        split_outputs (tuple[int, ...], Optional):
            Channel widths of separate outputs to write, in order. Each width must
            be a positive multiple of `l2norm_head_dim` and they must sum to `D`.
            The convolution then writes one contiguous `[B, T, w_i]` tensor per
            width instead of a single `[B, T, D]` output, and its backward takes
            one gradient per width. Requires the fused conv+L2-norm triton path
            and no residual. Default: `None`.

    Returns:
        tuple[torch.Tensor | tuple[torch.Tensor, ...], torch.Tensor | None]:
            Output and final state, where the output is a tuple of tensors when
            `split_outputs` is given. If `output_final_state` is `False`, the final state is `None`.
    """
    # import here to avoid circular dependencies
    from fla.modules.causal_conv1d.backends.cuda import causal_conv1d_cuda, fast_causal_conv1d_fn
    from fla.modules.causal_conv1d.cp import causal_conv1d_cp

    if l2norm_head_dim is not None and (backend != 'triton' or cp_context is not None):
        raise ValueError("the fused conv+L2-norm path is implemented by the triton backend only")
    if split_outputs is not None and l2norm_head_dim is None:
        raise ValueError("split_outputs is implemented by the fused conv+L2-norm path only")

    if cp_context is not None:
        assert initial_state is None, "Initial state is not supported for CP"
        assert output_final_state is False, "Output final state is not supported for CP"
        output = causal_conv1d_cp(
            x=x,
            weight=weight,
            bias=bias,
            activation=activation,
            chunk_indices=chunk_indices,
            cp_context=cp_context,
            residual=residual,
        )
        return output, None

    if backend == 'triton':
        out = CausalConv1dFunction.apply(
            x,
            weight,
            bias,
            residual,
            initial_state,
            output_final_state,
            activation,
            cu_seqlens,
            cu_seqlens_cpu,
            chunk_indices,
            64,
            l2norm_head_dim,
            l2norm_channels,
            l2norm_eps,
            split_outputs,
        )
        if split_outputs is not None:
            return out[:-1], out[-1]
        return out[0], out[1]
    elif backend == 'mix':
        seq_idx = kwargs.get('seq_idx')
        return fast_causal_conv1d_fn(
            x=x,
            weight=weight,
            bias=bias,
            residual=residual,
            initial_state=initial_state,
            output_final_state=output_final_state,
            activation=activation,
            cu_seqlens=cu_seqlens,
            cu_seqlens_cpu=cu_seqlens_cpu,
            chunk_indices=chunk_indices,
            seq_idx=seq_idx,
        )
    elif backend == 'cuda':
        return causal_conv1d_cuda(
            x=x,
            weight=weight,
            bias=bias,
            residual=residual,
            initial_state=initial_state,
            output_final_state=output_final_state,
            activation=activation,
            cu_seqlens=cu_seqlens,
            cu_seqlens_cpu=cu_seqlens_cpu,
            **kwargs,
        )
    else:
        raise ValueError(f"Unsupported backend: {backend}")


class ShortConvolution(nn.Conv1d):
    """Depthwise causal convolution with optional activation and state caching.

    Args:
        hidden_size (int):
            Number of input and output channels.
        kernel_size (int):
            Width of the convolution kernel.
        bias (bool, Optional):
            Whether to use a channel bias. Default: `False`.
        activation (str, Optional):
            Activation applied to the convolution output. Default: `"silu"`.
        backend (str, Optional):
            Backend selected unless overridden by `FLA_CONV_BACKEND`. Default: `"triton"`.
        device (torch.device, Optional):
            Device for the parameters. Default: `None`.
        dtype (torch.dtype, Optional):
            Parameter dtype. Default: `None`.
        **kwargs:
            Accepts the deprecated `use_fast_conv1d` option.
    """

    def __init__(
        self,
        hidden_size: int,
        kernel_size: int,
        bias: bool = False,
        activation: str | None = 'silu',
        backend: str | None = 'triton',
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        **kwargs,
    ) -> None:
        super().__init__(
            in_channels=hidden_size,
            out_channels=hidden_size,
            kernel_size=kernel_size,
            padding=kernel_size - 1,
            groups=hidden_size,
            bias=bias,
            device=device,
            dtype=dtype,
        )

        self.hidden_size = hidden_size
        self.activation = None

        if activation is not None:
            assert activation in ['silu', 'swish'], f"Activation `{activation}` not supported yet."
            self.activation = activation

        if 'use_fast_conv1d' in kwargs:
            warnings.warn(
                "The `use_fast_conv1d` parameter is deprecated and will be ignored. "
                "Please use the `backend` parameter instead.",
            )
        self.backend = os.environ.get('FLA_CONV_BACKEND', backend)
        if self.backend not in ['cuda', 'triton']:
            raise ValueError(f"Invalid backend: {self.backend}, must be one of ['cuda', 'triton']")
        if self.backend == 'cuda':
            if causal_conv1d_fn_cuda is None:
                warnings.warn(
                    "The `backend` parameter is set to `cuda`, but `causal_conv1d_fn` is not available. "
                    "Switching to the Triton implementation instead. "
                    "Consider installing `causal_conv1d` to enable the CUDA backend.",
                )
                self.backend = 'triton'

    def extra_repr(self) -> str:
        s = '{in_channels}, {out_channels}, kernel_size={kernel_size}, stride={stride}'
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.output_padding != (0,) * len(self.output_padding):
            s += ', output_padding={output_padding}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        if self.padding_mode != 'zeros':
            s += ', padding_mode={padding_mode}'
        if self.activation is not None:
            s += ', activation={activation}'
        s += f', backend={self.backend}'
        return s.format(**self.__dict__)

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        cache: torch.Tensor | None = None,
        output_final_state: bool = False,
        cu_seqlens: torch.LongTensor | None = None,
        chunk_indices: torch.LongTensor | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Args:
            x (torch.Tensor):
                Tensor of shape `[B, T, D]`. `B` must be 1 if `cu_seqlens` is provided.
            residual (torch.Tensor, Optional):
                Residual tensor of shape `[B, T, D]`. Default: `None`.
            mask (torch.Tensor, Optional):
                Attention mask for padded positions. Default: `None`.
            cache (torch.Tensor, Optional):
                Previous cache of shape `[N, D, W]`, where `W` is the kernel size. Default: `None`.
                Single-token calls without autograd update it **inplace**.
                Other calls return the updated state when `output_final_state=True`.
            output_final_state (bool, Optional):
                Whether to output the final state of shape `[N, D, W]`. Default: `False`.
            cu_seqlens (torch.Tensor, Optional):
                Cumulative sequence lengths of shape `[N+1]`. Default: `None`.
            chunk_indices (torch.Tensor, Optional):
                Chunk indices for variable-length sequences. Default: `None`.

        Returns:
            y (torch.Tensor):
                Convolution output of shape `[B, T, D]`.
            final_state (torch.Tensor, Optional):
                Cache of shape `[N, D, W]` when requested, otherwise `None`.
        """
        B, T, *_ = x.shape
        N = B if cu_seqlens is None else len(cu_seqlens) - 1
        if mask is not None:
            if cu_seqlens is not None:
                raise ValueError("`mask` and `cu_seqlens` cannot be provided at the same time")
            x = x.mul_(mask.unsqueeze(-1))

        # the in-place decode kernel requires autograd off and exactly one token per sequence, including packed inputs.
        if not torch.is_grad_enabled() and B * T == N and (cu_seqlens is None or bool((cu_seqlens.diff() == 1).all())):
            y, cache = self.step(
                x=x,
                residual=residual,
                cache=cache,
                output_final_state=output_final_state,
                cu_seqlens=cu_seqlens,
            )
            return y, cache

        # use Triton for stateful calls unsupported by the CUDA backend.
        if self.backend == 'cuda' and cache is not None:
            warnings.warn(
                "The CUDA backend does not support both `cu_seqlens` and `cache` being provided, "
                "or both `cu_seqlens` and `output_final_state` being provided. "
                "Switching to the Triton backend instead. ",
                stacklevel=2,
            )
            self.backend = 'triton'

        return causal_conv1d(
            x=x,
            weight=rearrange(self.weight, "d 1 w -> d w"),
            bias=self.bias,
            residual=residual,
            initial_state=cache,
            output_final_state=output_final_state,
            activation=self.activation,
            backend=self.backend,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            **kwargs,
        )

    def step(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None,
        cache: torch.Tensor | None,
        output_final_state: bool = False,
        cu_seqlens: torch.LongTensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        B, _, D, W = *x.shape, self.kernel_size[0]
        N = B if cu_seqlens is None else len(cu_seqlens) - 1
        # the update kernel requires a cache even when the final state is not requested.
        if cache is None:
            cache = x.new_zeros(N, D, W)
        if self.backend == 'triton':
            y, cache = causal_conv1d_update(
                x=x,
                cache=cache,
                residual=residual,
                weight=rearrange(self.weight, "d 1 w -> d w"),
                bias=self.bias,
                activation=self.activation,
            )
            return y, (cache if output_final_state else None)

        shape = x.shape
        x = x.squeeze(0) if cu_seqlens is not None else x.squeeze(1)
        # the CUDA update mutates cache in place.
        y = causal_conv1d_update_cuda(
            x=x,
            conv_state=cache,
            weight=rearrange(self.weight, "d 1 w -> d w"),
            bias=self.bias,
            activation=self.activation,
        )
        y = y.view(shape)
        if residual is not None:
            y.add_(residual)
        return y, (cache if output_final_state else None)

    @property
    def state_size(self) -> int:
        return self.hidden_size * self.kernel_size[0]
