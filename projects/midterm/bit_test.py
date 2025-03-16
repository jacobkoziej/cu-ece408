# SPDX-License-Identifier: GPL-3.0-or-later
#
# bit.py -- bit manipulation tests
# Copyright (C) 2025  Jacob Koziej <jacobkoziej@gmail.com>

import numpy as np
import pytest

from numpy import ndarray
from numpy.random import Generator
from pytest import FixtureRequest

from bit import (
    packbits,
    unpackbits,
)


@pytest.fixture(params=[2, 3, 5, 7, 8], scope="module")
def count(request: FixtureRequest) -> int:
    return request.param


@pytest.fixture
def values(rng: Generator, random_count: int) -> ndarray:
    return rng.integers(0, 9, random_count, dtype=np.uint8)


@pytest.mark.parametrize("shape", [(), (1,), (1, 1)])
def test_packing(values: ndarray, count: int, shape: tuple[int, ...]) -> None:
    values = values.reshape(shape + values.shape)

    unpacked = unpackbits(values, count=count)

    assert values.shape + (count,) == unpacked.shape

    packed = packbits(unpacked)

    assert np.all(packed == (values & (1 << count) - 1))
