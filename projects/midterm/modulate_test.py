# SPDX-License-Identifier: GPL-3.0-or-later
#
# modulate_test.py -- subcarrier modulation mapping tests
# Copyright (C) 2025  Jacob Koziej <jacobkoziej@gmail.com>

import numpy as np
import pytest

from conftest import Data
from modulate import (
    demodulate,
    modulate,
)


@pytest.mark.parametrize("disturbance", [0, 0.1, 0.2])
def test_modulation(data: Data, disturbance: int) -> None:
    d = modulate(data.bits, data.rate)
    r = d + (disturbance + 1j * disturbance)
    x = demodulate(r, data.rate)

    assert np.all(x == data.bits)
