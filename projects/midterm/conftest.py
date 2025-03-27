# SPDX-License-Identifier: GPL-3.0-or-later
#
# conftest.py -- pytest configuration
# Copyright (C) 2025  Jacob Koziej <jacobkoziej@gmail.com>

import numpy as np
import pytest

from dataclasses import dataclass

from numpy import ndarray
from numpy.random import Generator
from pytest import FixtureRequest

from ppdu import rate_parameter


@dataclass(frozen=True)
class Data:
    rate: int
    bits: ndarray


@pytest.fixture
def data(rng: Generator, random_count: int, rate: int) -> Data:
    bpsc = rate_parameter(rate).bpsc

    bits = rng.integers(0, (1 << bpsc), size=random_count, dtype=np.uint8)

    return Data(rate, bits)


@pytest.fixture(scope="session")
def random_count() -> int:
    return 1024


@pytest.fixture(params=[6, 9, 12, 18, 24, 36, 48, 54], scope="session")
def rate(request: FixtureRequest) -> int:
    return request.param


@pytest.fixture
def rng(seed) -> Generator:
    return np.random.default_rng(seed)


@pytest.fixture(scope="session")
def seed() -> int:
    return 0xBB485B7A
