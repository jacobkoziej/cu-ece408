# SPDX-License-Identifier: GPL-3.0-or-later
#
# bit.py -- bit manipulation
# Copyright (C) 2025  Jacob Koziej <jacobkoziej@gmail.com>

import numpy as np

from typing import Final

from numpy import ndarray

_BITORDER: Final[str] = "little"


def packbits(x: ndarray) -> ndarray:
    shape = x.shape[:-1]

    x = np.packbits(x, axis=-1, bitorder=_BITORDER)

    return x.reshape(shape)


def unpackbits(x: ndarray, *, count: int = 8) -> ndarray:
    assert count >= 1
    assert count <= 8

    x = x.reshape(x.shape + (1,))

    return np.unpackbits(x, axis=-1, count=count, bitorder=_BITORDER)
