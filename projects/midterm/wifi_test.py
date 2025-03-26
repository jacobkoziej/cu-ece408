# SPDX-License-Identifier: GPL-3.0-or-later
#
# wifi_test.py -- IEEE Std 802.11a-1999 tx/rx pair test
# Copyright (C) 2025  Jacob Koziej <jacobkoziej@gmail.com>

import numpy as np
import pytest

from typing import Final

from numpy.random import Generator

from wifi import (
    Rx,
    Tx,
)

from conftest import Data


FREQUENCY_OFFSET_ANGLE: Final[float] = 2e-2


@pytest.fixture
def rx() -> Rx:
    return Rx()


@pytest.fixture
def tx(rng: Generator) -> Tx:
    return Tx(rng=rng)


def test_wifi(rx: Rx, tx: Tx, data: Data) -> None:
    bits = data.bits

    signal = tx(bits, data.rate)

    frequency_offset = np.exp(
        1j * FREQUENCY_OFFSET_ANGLE * np.arange(signal.size)
    )

    signal *= frequency_offset

    recieved = rx(signal)

    assert np.all(recieved == bits)
