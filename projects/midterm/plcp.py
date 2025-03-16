# SPDX-License-Identifier: GPL-3.0-or-later
#
# plcp.py -- OFDM PLCP sublayer
# Copyright (C) 2025  Jacob Koziej <jacobkoziej@gmail.com>

import numpy as np

from collections import deque
from dataclasses import dataclass
from fractions import Fraction
from typing import Final

from numpy import ndarray


@dataclass(frozen=True, kw_only=True)
class RateParameter:
    modulation: str
    coding_rate: Fraction
    bpsc: int
    cbps: int
    dbps: int


_DECODE_RATE: Final[dict[int, int]] = {
    0b1011: 6,
    0b1111: 9,
    0b1010: 12,
    0b1110: 18,
    0b1001: 24,
    0b1101: 36,
    0b1000: 48,
    0b1100: 54,
}
_ENCODE_RATE: Final[dict[int, int]] = {
    6: 0b1011,
    9: 0b1111,
    12: 0b1010,
    18: 0b1110,
    24: 0b1001,
    36: 0b1101,
    48: 0b1000,
    54: 0b1100,
}
_RATE_PARAMETERS: Final[dict[int, RateParameter]] = {
    6: RateParameter(
        modulation="BPSK",
        coding_rate=Fraction(1, 2),
        bpsc=1,
        cbps=48,
        dbps=24,
    ),
    9: RateParameter(
        modulation="BPSK",
        coding_rate=Fraction(3, 4),
        bpsc=1,
        cbps=48,
        dbps=36,
    ),
    12: RateParameter(
        modulation="QPSK",
        coding_rate=Fraction(1, 2),
        bpsc=2,
        cbps=96,
        dbps=48,
    ),
    18: RateParameter(
        modulation="QPSK",
        coding_rate=Fraction(3, 4),
        bpsc=2,
        cbps=96,
        dbps=72,
    ),
    24: RateParameter(
        modulation="16-QAM",
        coding_rate=Fraction(1, 2),
        bpsc=4,
        cbps=192,
        dbps=96,
    ),
    36: RateParameter(
        modulation="16-QAM",
        coding_rate=Fraction(3, 4),
        bpsc=4,
        cbps=192,
        dbps=144,
    ),
    48: RateParameter(
        modulation="64-QAM",
        coding_rate=Fraction(1, 2),
        bpsc=6,
        cbps=288,
        dbps=192,
    ),
    54: RateParameter(
        modulation="64-QAM",
        coding_rate=Fraction(3, 4),
        bpsc=6,
        cbps=288,
        dbps=216,
    ),
}


class ConvolutionalEncoder:
    k: Final[int] = 7

    def __call__(self, x: ndarray) -> ndarray:
        assert x.shape == ()
        assert x.dtype == np.uint8
        assert x <= 1

        state = self._state

        code = np.zeros(2, dtype=np.uint8)

        code[0] = x ^ state[1] ^ state[2] ^ state[4] ^ state[5]
        code[1] = x ^ state[0] ^ state[1] ^ state[2] ^ state[5]

        _ = state.pop()

        state.appendleft(x)

        return code

    def __init__(self) -> None:
        self._state = deque(np.zeros(self.k - 1, dtype=np.uint8))


class Scrambler:
    CONSTRAINT_LENGTH: Final[int] = 7

    def __call__(self, x: ndarray):
        assert x.shape == ()
        assert x.dtype == np.uint8
        assert x <= 1

        state = self._state

        feedback = state[6] ^ state[3]

        _ = state.pop()

        state.appendleft(feedback)

        return x ^ feedback

    def __init__(self, state: ndarray):
        assert len(state) == self.CONSTRAINT_LENGTH
        assert state.dtype == np.uint8
        assert np.all(state <= 1)

        self._state = deque(state)


def decode_rate(rate: int) -> int:
    try:
        return _DECODE_RATE[rate]

    except Exception as _:
        raise KeyError(f"Unsupported rate: {bin(rate)}")


def encode_rate(rate: int) -> int:
    try:
        return _ENCODE_RATE[rate]

    except Exception as _:
        raise KeyError(f"Unsupported rate: {rate}")


def rate_parameter(rate: int) -> RateParameter:
    try:
        return _RATE_PARAMETERS[rate]

    except Exception as _:
        raise KeyError(f"Unsupported rate: {rate}")
