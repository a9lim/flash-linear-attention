# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Main interface for causal 1D convolution operations."""

import torch

from fla.ops.cp import FLACPContext
from fla.utils import input_guard


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
):
    """
    A causal 1D convolution implementation that powers Mamba/Mamba2 and DeltaNet architectures.

    When a residual connection is provided, this implements the Canon operation
    described in the paper at https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5240330.

    Args:
        x (torch.Tensor):
            Input tensor of shape [B, T, D].
        weight (Optional[torch.Tensor]):
            Weight tensor of shape [D, W]. Default: `None`.
        bias (Optional[torch.Tensor]):
            Bias tensor of shape [D]. Default: `None`.
        residual (Optional[torch.Tensor]):
            Residual tensor of shape [B, T, D]. Default: `None`.
        initial_state (Optional[torch.Tensor]):
            Initial state tensor of shape [N, D, W],
            where `N` is the number of sequences in the batch and `W` is the kernel size.
            If provided, the initial state is used to initialize the cache. Default: `None`.
        output_final_state (Optional[bool]):
            Whether to output the final state of shape [N, D, W]. Default: `False`.
        activation (Optional[str]):
            Activations applied to output, only `swish`/`silu` or `None` (i.e., no activation) are supported.
            Default: `None`.
        backend (Optional[str]):
            Specifies the backend to use for the convolution operation. Supported values are `'cuda'` 、 `'triton'` and `'mix'`.
            Default: `'triton'`.
        cu_seqlens (Optional[torch.Tensor]):
            Cumulative sequence lengths (optional)
        chunk_indices (Optional[torch.LongTensor]):
            Chunk indices for variable-length sequences (optional)
        l2norm_head_dim (Optional[int]):
            When set, the Triton backend fuses a per-head L2 normalization (heads of
            this many channels, applied after the activation) into the convolution
            for the first `l2norm_channels` channels (all channels by default).
            Dense fixed-length training only: no cache state, no `cu_seqlens`.
        l2norm_channels (Optional[int]):
            Number of leading channels the fused L2 normalization covers.
        l2norm_eps (float):
            Epsilon inside the fused normalization's square root. Default: `1e-6`.
        split_outputs (Optional[tuple[int, ...]]):
            Channel widths of separate outputs to write, in order. Each width must
            be a positive multiple of `l2norm_head_dim` and they must sum to `D`.
            The convolution then writes one contiguous `[B, T, w_i]` tensor per
            width instead of a single `[B, T, D]` output, and its backward takes
            one gradient per width. Requires the fused conv+L2-norm triton path
            and no residual.

    Returns:
        Tuple of (output, final_state), where the output is a tuple of tensors
        when `split_outputs` is given and a single tensor otherwise.
        If `output_final_state` is `False`, the final state is `None`.
    """
    # Import here to avoid circular dependencies
    from fla.modules.conv.cp import causal_conv1d_cp
    from fla.modules.conv.cuda import causal_conv1d_cuda, fast_causal_conv1d_fn
    from fla.modules.conv.triton import CausalConv1dFunction

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
            x,
            weight,
            bias,
            residual,
            initial_state,
            output_final_state,
            activation,
            cu_seqlens,
            cu_seqlens_cpu=cu_seqlens_cpu,
            chunk_indices=chunk_indices,
            seq_idx=seq_idx,
        )
    elif backend == 'cuda':
        return causal_conv1d_cuda(
            x,
            weight,
            bias,
            residual,
            initial_state,
            output_final_state,
            activation,
            cu_seqlens,
            cu_seqlens_cpu=cu_seqlens_cpu,
            **kwargs,
        )
    else:
        raise ValueError(f"Unsupported backend: {backend}")
