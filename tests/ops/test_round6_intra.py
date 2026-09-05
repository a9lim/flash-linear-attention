# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Frozen-baseline comparisons for bounded and unbounded diagonal tiles."""

import pytest

from benchmarks.round6_intra import compare, load_baseline, make_inputs


@pytest.fixture(scope='module')
def baseline(tmp_path_factory):
    return load_baseline('34848cb2', tmp_path_factory.mktemp('round6_intra'))


@pytest.mark.parametrize('regime', ['mild', 'mixed', 'extreme'])
@pytest.mark.parametrize('length,width,varlen', [(65, 32, False), (128, 64, True), (1024, 128, False)])
def test_round6_intra_baseline(baseline, regime, length, width, varlen):
    values = make_inputs(batch=1, length=length, heads=3, width=width, regime=regime, varlen=varlen)
    compare(baseline, values)
