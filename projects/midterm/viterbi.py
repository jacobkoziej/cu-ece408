# SPDX-License-Identifier: GPL-3.0-or-later
#
# viterbi.py -- Viterbi decoder
# Copyright (C) 2025  Jacob Koziej <jacobkoziej@gmail.com>

import galois
import numpy as np

from galois import GF2

from bit import packbits


def polynomial2generator_matrix(polynomials: list[int], k: int) -> GF2:
    assert k > 0
    assert len(polynomials) >= 1

    return GF2(
        [
            galois.Poly.Int(polynomial, field=GF2).coefficients(k, "asc")
            for polynomial in polynomials
        ]
    ).T


class Viterbi:
    def __init__(
        self,
        generator_matrix: GF2,
        *,
        initial_state: int = 0,
        final_state: int = 0,
    ) -> None:
        self.generator_matrix = generator_matrix
        self.initial_state = initial_state
        self.final_state = final_state

        self.n = generator_matrix.shape[1]

        self.k = k = generator_matrix.shape[0]

        states = 1 << (k - 1)

        states = GF2.Zeros((states, k))
        states[:, 1:] = GF2(
            [
                galois.Poly.Int(i, field=GF2).coefficients(k - 1, "asc")
                for i in range(states.shape[0])
            ]
        )

        zero_branch = packbits(np.array(states[:, :-1]))
        zero_expected = states @ generator_matrix

        states[:, 0] = 1

        one_branch = packbits(np.array(states[:, :-1]))
        one_expected = states @ generator_matrix

        self._branch = np.stack([zero_branch, one_branch])
        self._expected = np.stack([zero_expected, one_expected])
