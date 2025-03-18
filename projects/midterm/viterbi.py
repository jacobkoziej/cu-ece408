# SPDX-License-Identifier: GPL-3.0-or-later
#
# viterbi.py -- Viterbi decoder
# Copyright (C) 2025  Jacob Koziej <jacobkoziej@gmail.com>

import galois
import numpy as np

from galois import GF2

from bit import packbits


class Viterbi:
    def __init__(self, polynomials: list[int], k: int) -> None:
        assert k > 0
        assert len(polynomials) >= 1

        generator_matrix = GF2(
            [
                galois.Poly.Int(polynomial, field=GF2).coefficients(k, "asc")
                for polynomial in polynomials
            ]
        ).T

        states = GF2.Zeros((1 << (k - 1), k))
        states[:, 1:] = GF2(
            [
                galois.Poly.Int(i, field=GF2).coefficients(k - 1, "asc")
                for i in range(1 << (k - 1))
            ]
        )

        self._zero_branch = packbits(np.array(states[:, :-1]))
        self._zero_expected = np.array(states @ generator_matrix)

        states[:, 0] = 1

        self._one_branch = packbits(np.array(states[:, :-1]))
        self._one_expected = np.array(states @ generator_matrix)
