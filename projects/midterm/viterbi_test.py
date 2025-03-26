# SPDX-License-Identifier: GPL-3.0-or-later
#
# viterbi_test.py -- Viterbi decoder test
# Copyright (C) 2025  Jacob Koziej <jacobkoziej@gmail.com>

import numpy as np
import pytest

from galois import GF2
from numpy.random import Generator

from plcp import ConvolutionalEncoder
from viterbi import (
    Viterbi,
    poly2matrix,
)


@pytest.mark.parametrize(
    "polynomials, k",
    (
        ((0b111, 0b101), 3),
        ((0o133, 0o171), 7),
    ),
)
def test_viterbi(
    rng: Generator,
    random_count: int,
    polynomials: tuple[int, int],
    k: int,
) -> None:
    G = poly2matrix(polynomials, k)

    x = GF2.Random(random_count, seed=rng)
    x = np.concatenate([x, [0] * (k - 1)])

    c = ConvolutionalEncoder(G)

    y = c(x)

    v = Viterbi(G)

    bit_flips = rng.integers(0, random_count, k)

    y[bit_flips] ^= y[bit_flips]

    decoded = v(y)

    assert not np.sum(np.array(x + decoded))
