# SPDX-License-Identifier: GPL-3.0-or-later
#
# ofdm_test.py -- orthogonal frequency-division multiplexing tests
# Copyright (C) 2025  Jacob Koziej <jacobkoziej@gmail.com>

import numpy as np
import pytest

import modulate
import ofdm

from numpy import ndarray

from conftest import Data
from plcp import rate_parameter
from ofdm import (
    carrier_frequency_offset,
)


@pytest.fixture(scope="module")
def short_training_sequence() -> ndarray:
    return ofdm.short_training_sequence()


@pytest.mark.parametrize("phi", [0.0, 0.1, 0.2])
@pytest.mark.parametrize("symbols", [1, 2, 4])
def test_carrier_frequency_offset(
    short_training_sequence: ndarray,
    phi: float,
    symbols: int,
) -> None:
    samples = ofdm.SHORT_TRAINING_SYMBOLS * ofdm.SHORT_TRAINING_SYMBOL_SAMPLES

    cfo = np.exp(
        1j * (phi / ofdm.SHORT_TRAINING_SYMBOL_SAMPLES) * np.arange(samples)
    )

    theta = carrier_frequency_offset(short_training_sequence * cfo)

    assert np.isclose(phi, theta)


def test_modulation(data: Data) -> None:
    rate = data.rate

    parameter = rate_parameter(rate)

    bits = data.bits[
        : ofdm.SUBCARRIERS_DATA * (len(data.bits) // ofdm.SUBCARRIERS_DATA)
    ]
    bits = bits.reshape(-1, ofdm.SUBCARRIERS_DATA)

    d = np.stack(
        [
            bits ^ (1 << 0),
            bits ^ (1 << 1),
            bits ^ (1 << 2),
            bits ^ (1 << 3),
            bits ^ (1 << 4),
        ],
    )
    d &= (1 << parameter.bpsc) - 1

    modulated = modulate.modulate(d, rate)

    s = ofdm.modulate(modulated)
    r = ofdm.demodulate(s)

    demodulated = modulate.demodulate(r, rate)

    assert np.all(demodulated == d)
