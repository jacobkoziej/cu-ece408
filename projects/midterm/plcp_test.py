# SPDX-License-Identifier: GPL-3.0-or-later
#
# plcp.py -- OFDM PLCP sublayer tests
# Copyright (C) 2025  Jacob Koziej <jacobkoziej@gmail.com>

import numpy as np

from plcp import (
    Scrambler,
)


def test_scrambler() -> None:
    sequence = np.array(
        [
            [0, 0, 0, 0, 1, 1, 1, 0],
            [1, 1, 1, 1, 0, 0, 1, 0],
            [1, 1, 0, 0, 1, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 1, 1, 0],
            [0, 0, 1, 0, 1, 1, 1, 0],
            [1, 0, 1, 1, 0, 1, 1, 0],
            [0, 0, 0, 0, 1, 1, 0, 0],
            [1, 1, 0, 1, 0, 1, 0, 0],
            [1, 1, 1, 0, 0, 1, 1, 1],
            [1, 0, 1, 1, 0, 1, 0, 0],
            [0, 0, 1, 0, 1, 0, 1, 0],
            [1, 1, 1, 1, 1, 0, 1, 0],
            [0, 1, 0, 1, 0, 0, 0, 1],
            [1, 0, 1, 1, 1, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 1, 1],
        ],
        dtype=np.uint8,
    ).flatten()[
        :-1
    ]  # drop dummy bit since we have only 127 values

    scrambler = Scrambler(np.ones(Scrambler.CONSTRAINT_LENGTH, dtype=np.uint8))

    x = np.array(0, dtype=np.uint8)

    for _ in range(2):
        for i, bit in enumerate(sequence):
            assert scrambler(x) == bit
