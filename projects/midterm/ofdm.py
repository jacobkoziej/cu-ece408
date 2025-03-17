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
from scipy.signal import resample_poly

from plcp import Scrambler

SUBCARRIERS_DATA: Final[int] = 48
SUBCARRIERS_PILOT: Final[int] = 4
SUBCARRIERS_TOTAL: Final[int] = SUBCARRIERS_DATA + SUBCARRIERS_PILOT

_PILOT_INDICES: ndarray = np.zeros(SUBCARRIERS_TOTAL + 1, dtype=np.bool)
_PILOT_INDICES[[5, 19, 33, 47]] = True

_DATA_INDICES: ndarray = ~_PILOT_INDICES
_DATA_INDICES[(SUBCARRIERS_TOTAL + 1) // 2] = False

_PILOTS: ndarray = np.array([1, 1, 1, -1])


def demodulate(s: ndarray, equalizer: Optional[ndarray] = None) -> ndarray:
    if equalizer is None:
        equalizer = np.array(1)

    d = fftshift(fft(s / equalizer), axes=-1)

    return d[..., _DATA_INDICES]


def modulate(d: ndarray) -> ndarray:
    shape = d.shape[:-1]

    s = np.zeros(shape + (SUBCARRIERS_TOTAL + 1,), dtype=np.complex128)

    frames = 1 if s.ndim <= 1 else s.shape[-2]

    s[..., _PILOT_INDICES] = pilots(frames)
    s[..., _DATA_INDICES] = d

    return ifft(ifftshift(s, axes=-1))


def pilots(frames: int) -> ndarray:
    assert frames > 0

    scrambler = Scrambler(np.ones(Scrambler.k - 1, dtype=np.uint8))
    polarity = np.zeros(frames, dtype=np.int8)

    for i in range(frames):
        polarity[i] = -1 if scrambler(polarity[i].astype(np.uint8)) else 1

    return polarity[:, None] * _PILOTS[None, :]


def short_training_sequence() -> ndarray:
    S = np.zeros(SUBCARRIERS_TOTAL, dtype=np.complex128)

    S[2] = +1 + 1j
    S[6] = -1 - 1j
    S[10] = +1 + 1j
    S[14] = -1 - 1j
    S[18] = -1 - 1j
    S[22] = +1 + 1j
    S[30] = -1 - 1j
    S[34] = -1 - 1j
    S[38] = +1 + 1j
    S[42] = +1 + 1j
    S[46] = +1 + 1j
    S[50] = +1 + 1j

    s = ifft(ifftshift(np.sqrt(13 / 6) * S, axes=-1))
    s = np.repeat(s, 10)

    return resample_poly(s, 160, len(s))
