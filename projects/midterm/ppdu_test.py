# SPDX-License-Identifier: GPL-3.0-or-later
#
# ppdu.py -- Physical layer Protocol Data Unit frame format tests
# Copyright (C) 2025  Jacob Koziej <jacobkoziej@gmail.com>

import numpy as np

from fractions import Fraction

from galois import GF2
from numpy.random import Generator

from ppdu import (
    Interleaver,
    Puncturer,
    Scrambler,
    rate_parameter,
)


def test_interleaver(rate: int) -> None:
    parameter = rate_parameter(rate)

    interleaver = Interleaver(
        bpsc=parameter.bpsc,
        cbps=parameter.cbps,
    )

    x = np.arange(parameter.cbps)

    interleaved = interleaver.forward(x)

    assert np.any(x != interleaved)

    deinterleaved = interleaver.reverse(interleaved)

    assert np.all(x == deinterleaved)


def test_puncturer(rng: Generator, random_count: int, rate: int) -> None:
    parameter = rate_parameter(rate)

    puncturer = Puncturer(parameter.coding_rate)

    data = GF2.Random(random_count + (-random_count % 36), seed=rng)

    punctured = puncturer.forward(data)

    assert len(punctured) <= len(data)

    unpunctured = puncturer.reverse(punctured)

    assert len(unpunctured) == len(data)

    if parameter.coding_rate == Fraction(1, 2):
        return

    assert np.any(unpunctured != data)


def test_scrambler() -> None:
    sequence = GF2(
        [
            # fmt: off
            0, 0, 0, 0, 1, 1, 1, 0,
            1, 1, 1, 1, 0, 0, 1, 0,
            1, 1, 0, 0, 1, 0, 0, 1,
            0, 0, 0, 0, 0, 0, 1, 0,
            0, 0, 1, 0, 0, 1, 1, 0,
            0, 0, 1, 0, 1, 1, 1, 0,
            1, 0, 1, 1, 0, 1, 1, 0,
            0, 0, 0, 0, 1, 1, 0, 0,
            1, 1, 0, 1, 0, 1, 0, 0,
            1, 1, 1, 0, 0, 1, 1, 1,
            1, 0, 1, 1, 0, 1, 0, 0,
            0, 0, 1, 0, 1, 0, 1, 0,
            1, 1, 1, 1, 1, 0, 1, 0,
            0, 1, 0, 1, 0, 0, 0, 1,
            1, 0, 1, 1, 1, 0, 0, 0,
            1, 1, 1, 1, 1, 1, 1,
            # fmt: on
        ],
    )

    scrambler = Scrambler(0o177)

    x = GF2(0)

    for _ in range(2):
        for i, bit in enumerate(sequence):
            assert scrambler(x) == bit
