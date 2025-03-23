# SPDX-License-Identifier: GPL-3.0-or-later
#
# bit.py -- bit manipulation
# Copyright (C) 2025  Jacob Koziej <jacobkoziej@gmail.com>

import numpy as np

from typing import Final

from galois import GF2
from numpy import ndarray

_BITORDER: Final[str] = "little"


def packbits(x: GF2) -> ndarray:
    shape = x.shape[:-1]

    x = np.packbits(np.array(x), axis=-1, bitorder=_BITORDER)

    return x.reshape(shape)


def unpackbits(x: ndarray, *, count: int = 8) -> GF2:
    assert count >= 1
    assert count <= 8

    x = x.reshape(x.shape + (1,))

    return GF2(np.unpackbits(x, axis=-1, count=count, bitorder=_BITORDER))
