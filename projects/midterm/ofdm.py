# SPDX-License-Identifier: GPL-3.0-or-later
#
# ofdm.py -- orthogonal frequency-division multiplexing
# Copyright (C) 2025  Jacob Koziej <jacobkoziej@gmail.com>

import numpy as np

from numpy import ndarray

from plcp import Scrambler

_PILOTS: ndarray = np.array([1, 1, 1, -1])


def pilots(frames: int) -> ndarray:
    assert frames > 0

    scrambler = Scrambler(np.ones(Scrambler.k - 1, dtype=np.uint8))
    polarity = np.zeros(frames, dtype=np.int8)

    for i in range(frames):
        polarity[i] = -1 if scrambler(polarity[i].astype(np.uint8)) else 1

    return polarity[:, None] * _PILOTS[None, :]
