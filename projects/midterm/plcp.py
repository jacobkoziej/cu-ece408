# SPDX-License-Identifier: GPL-3.0-or-later
#
# plcp.py -- OFDM PLCP sublayer
# Copyright (C) 2025  Jacob Koziej <jacobkoziej@gmail.com>

import numpy as np

from collections import deque
from dataclasses import dataclass
from fractions import Fraction
from math import floor
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
_PUNCTURE_MATRIX: Final[dict[Fraction, ndarray]] = {
    Fraction(1, 2): np.array(
        [1, 1],
        dtype=np.bool,
    ),
    Fraction(2, 3): np.array(
        [
            [1, 1],
            [1, 0],
            [1, 1],
            [1, 0],
            [1, 1],
            [1, 0],
        ],
        dtype=np.bool,
    ),
    Fraction(3, 4): np.array(
        [
            [1, 1],
            [1, 0],
            [0, 1],
            [1, 1],
            [1, 0],
            [0, 1],
            [1, 1],
            [1, 0],
            [0, 1],
        ],
        dtype=np.bool,
    ),
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


class Interleaver:
    def __init__(self, *, bpsc: int, cbps: int) -> None:
        self._bpsc = bpsc
        self._cbps = cbps

    def forward(self, x: ndarray) -> ndarray:
        cbps = self._cbps

        assert len(x) == self._cbps

        y = np.zeros(x.shape, dtype=x.dtype)

        for k in range(cbps):
            i = 1

            i *= cbps / 16
            i *= k % 16
            i += floor(k / 16)

            y[int(i)] = x[k]

        z = np.zeros(y.shape, dtype=y.dtype)

        for i in range(cbps):
            s = max(self._bpsc / 2, 1)

            j = 1

            j *= s * floor(i / s)
            j += (i + cbps - floor(16 * i / cbps)) % s

            z[int(j)] = y[i]

        return z

    def reverse(self, x: ndarray) -> ndarray:
        cbps = self._cbps

        assert len(x) == self._cbps

        y = np.zeros(x.shape, dtype=x.dtype)

        for i in range(cbps):
            s = max(self._bpsc / 2, 1)

            j = 1

            j *= s * floor(i / s)
            j += (i + cbps - floor(16 * i / cbps)) % s

            y[i] = x[int(j)]

        z = np.zeros(y.shape, dtype=y.dtype)

        for k in range(cbps):
            i = 1

            i *= cbps / 16
            i *= k % 16
            i += floor(k / 16)

            z[k] = y[int(i)]

        return z


class Puncturer:
    def __init__(self, coding_rate: Fraction) -> None:
        self._puncture_matrix = _PUNCTURE_MATRIX[coding_rate]
        self._punctured_shape = np.sum(self._puncture_matrix)

    def forward(self, x: ndarray) -> ndarray:
        p = self._puncture_matrix

        y = x.reshape((-1,) + p.shape)
        y = y[..., p]

        return y.flatten()

    def reverse(self, x: ndarray) -> ndarray:
        p = self._puncture_matrix
        s = (self._punctured_shape,)

        y = x.reshape((-1,) + s)
        z = np.zeros(y.shape[:-1] + p.shape, dtype=x.dtype)

        z[..., p] = y

        return z.flatten()


class Scrambler:
    k: Final[int] = 8

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
        assert len(state) == self.k - 1
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
