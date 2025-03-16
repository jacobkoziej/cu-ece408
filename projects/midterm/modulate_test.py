# SPDX-License-Identifier: GPL-3.0-or-later
#
# modulate_test.py -- subcarrier modulation mapping tests
# Copyright (C) 2025  Jacob Koziej <jacobkoziej@gmail.com>

import numpy as np
import pytest

from dataclasses import dataclass

from numpy import ndarray
from numpy.random import Generator

from plcp import rate_parameter
from modulate import (
    demodulate,
    modulate,
)


@dataclass(frozen=True)
class Data:
    rate: int
    bits: ndarray


@pytest.fixture
def data(rng: Generator, random_count: int, rate: int) -> Data:
    bpsc = rate_parameter(rate).bpsc

    bits = rng.integers(0, (1 << bpsc), size=random_count, dtype=np.uint8)

    return Data(rate, bits)


@pytest.mark.parametrize("disturbance", [0, 0.1, 0.2])
def test_modulation(data: Data, disturbance: int) -> None:
    d = modulate(data.bits, data.rate)
    r = d + (disturbance + 1j * disturbance)
    x = demodulate(r, data.rate)

    assert np.all(x == data.bits)
