# SPDX-License-Identifier: GPL-3.0-or-later
#
# conftest.py -- pytest configuration
# Copyright (C) 2025  Jacob Koziej <jacobkoziej@gmail.com>

import numpy as np
import pytest

from numpy.random import Generator
from pytest import FixtureRequest


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
