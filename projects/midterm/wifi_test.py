# SPDX-License-Identifier: GPL-3.0-or-later
#
# wifi_test.py -- IEEE Std 802.11a-1999 tx/rx pair test
# Copyright (C) 2025  Jacob Koziej <jacobkoziej@gmail.com>

import numpy as np
import pytest

from wifi import (
    Rx,
    Tx,
)

from numpy.random import Generator

from conftest import Data


@pytest.fixture
def rx() -> Rx:
    return Rx()


@pytest.fixture
def tx(rng: Generator) -> Tx:
    return Tx(rng=rng)


def test_wifi(rx: Rx, tx: Tx, data: Data) -> None:
    bits = data.bits

    signal = tx(bits, data.rate)
    recieved = rx(signal)

    assert np.all(recieved == bits)
