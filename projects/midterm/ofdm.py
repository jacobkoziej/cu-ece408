# SPDX-License-Identifier: GPL-3.0-or-later
#
# ofdm.py -- orthogonal frequency-division multiplexing
# Copyright (C) 2025  Jacob Koziej <jacobkoziej@gmail.com>

import numpy as np

from typing import (
    Final,
    Optional,
)

from numpy import ndarray
from numpy.fft import (
    fft,
    fftshift,
    ifft,
    ifftshift,
)
from scipy.signal import resample

from plcp import Scrambler

SUBCARRIERS_DATA: Final[int] = 48
SUBCARRIERS_PILOT: Final[int] = 4
SUBCARRIERS_TOTAL: Final[int] = SUBCARRIERS_DATA + SUBCARRIERS_PILOT

SHORT_TRAINING_SYMBOL_SAMPLES: Final[int] = 16
SHORT_TRAINING_SYMBOLS: Final[int] = 10
LONG_TRAINING_SYMBOLS: Final[int] = 2

_FFT_SIZE: Final[int] = 64
_FFT_INDEX_SHIFT: Final[int] = ((_FFT_SIZE - SUBCARRIERS_TOTAL) // 2) - 1

_PILOT_INDICES: ndarray = np.zeros(_FFT_SIZE, dtype=np.bool)
_PILOT_INDICES[np.array([5, 19, 33, 47]) + _FFT_INDEX_SHIFT] = True

_DATA_INDICES: ndarray = ~_PILOT_INDICES
_DATA_INDICES[_FFT_SIZE // 2] = False
_DATA_INDICES[:6] = False
_DATA_INDICES[-5:] = False

_PILOTS: ndarray = np.array([1, 1, 1, -1])

CIRCULAR_PREFIX: Final[int] = 16
FRAME_SIZE: Final[int] = _FFT_SIZE + CIRCULAR_PREFIX


def add_circular_prefix(x: ndarray, size: int = CIRCULAR_PREFIX) -> ndarray:
    assert size > 0
    assert size < x.shape[-1]

    prefix = x[..., -size:]

    return np.concatenate((prefix, x), axis=-1)


def apply_window(x: ndarray) -> ndarray:
    y = np.copy(x)

    y[..., +0] *= 0.5
    y[..., -1] *= 0.5

    return y


def carrier_frequency_offset(s: ndarray, symbols: int = 4) -> ndarray:
    assert symbols > 0
    assert symbols < SHORT_TRAINING_SYMBOLS

    s = s.reshape(-1, SHORT_TRAINING_SYMBOL_SAMPLES)

    phi = np.angle(
        s[..., -(symbols + 1) : -1, :].conj() * s[..., -symbols:, :]
    )

    return np.mean(phi, axis=(-2, -1))


def demodulate(s: ndarray, equalizer: Optional[ndarray] = None) -> ndarray:
    if equalizer is None:
        equalizer = np.array(1)

    d = fftshift(fft(s / equalizer), axes=-1)

    return d[..., _DATA_INDICES]


def long_training_sequence() -> ndarray:
    L = np.array(
        [
            # fmt: off
            +0, +0, +0, +0, +0, +0, +1, +1,
            -1, -1, +1, +1, -1, +1, -1, +1,
            +1, +1, +1, +1, +1, -1, -1, +1,
            +1, -1, +1, -1, +1, +1, +1, +1,
            +0, +1, -1, -1, +1, +1, -1, +1,
            -1, +1, -1, -1, -1, -1, -1, +1,
            +1, -1, -1, +1, -1, +1, -1, +1,
            +1, +1, +1, +0, +0, +0, +0, +0,
            # fmt: on
        ],
    )

    l = ifft(ifftshift(L, axes=-1))  # noqa: E741

    return np.tile(l, LONG_TRAINING_SYMBOLS)


def modulate(d: ndarray) -> ndarray:
    shape = d.shape[:-1]

    s = np.zeros(shape + (_FFT_SIZE,), dtype=np.complex128)

    frames = 1 if s.ndim <= 1 else s.shape[-2]

    s[..., _PILOT_INDICES] = pilots(frames)
    s[..., _DATA_INDICES] = d

    return ifft(ifftshift(s, axes=-1))


def pilots(frames: int) -> ndarray:
    assert frames > 0

    scrambler = Scrambler(0o177)
    polarity = np.zeros(frames, dtype=np.int8)

    for i in range(frames):
        polarity[i] = -1 if scrambler(polarity[i].astype(np.uint8)) else 1

    return polarity[:, None] * _PILOTS[None, :]


def short_training_sequence() -> ndarray:
    S = np.zeros(_FFT_SIZE, dtype=np.complex128)

    # fmt: off
    S[2  + _FFT_INDEX_SHIFT] = +1 + 1j
    S[6  + _FFT_INDEX_SHIFT] = -1 - 1j
    S[10 + _FFT_INDEX_SHIFT] = +1 + 1j
    S[14 + _FFT_INDEX_SHIFT] = -1 - 1j
    S[18 + _FFT_INDEX_SHIFT] = -1 - 1j
    S[22 + _FFT_INDEX_SHIFT] = +1 + 1j
    S[30 + _FFT_INDEX_SHIFT] = -1 - 1j
    S[34 + _FFT_INDEX_SHIFT] = -1 - 1j
    S[38 + _FFT_INDEX_SHIFT] = +1 + 1j
    S[42 + _FFT_INDEX_SHIFT] = +1 + 1j
    S[46 + _FFT_INDEX_SHIFT] = +1 + 1j
    S[50 + _FFT_INDEX_SHIFT] = +1 + 1j
    # fmt: on

    s = ifft(ifftshift(np.sqrt(13 / 6) * S, axes=-1))
    s = np.tile(s, SHORT_TRAINING_SYMBOLS)

    return resample(s, SHORT_TRAINING_SYMBOLS * SHORT_TRAINING_SYMBOL_SAMPLES)
