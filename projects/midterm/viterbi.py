# SPDX-License-Identifier: GPL-3.0-or-later
#
# viterbi.py -- Viterbi decoder
# Copyright (C) 2025  Jacob Koziej <jacobkoziej@gmail.com>

import galois
import numpy as np

from galois import GF2
from numpy import ndarray

from bit import packbits


class Viterbi:
    def __call__(self, x: GF2) -> GF2:
        n = self.n
        states = self.states

        x = x.reshape(-1, n)

        cost = np.full((len(x) + 1, n, states, states), np.inf)
        cost[0, 0, 0, 0] = 0

        for i, x_i in enumerate(x):
            new_cost = self._forward_step(x_i, cost[i])

            cost[i + 1] = new_cost

        y = GF2.Zeros(len(x) + 1)

        path = np.full(len(x) + 1, -1)
        path[-1] = 0

        for i in range(len(y) - 2, -1, -1):
            bit, new_path = self._reverse_step(cost[i], path[i + 1])

            y[i] = bit
            path[i] = new_path

        return y[1:]

    def __init__(self, generator_matrix: GF2, k: int, n: int) -> None:
        assert k > 0
        assert n > 0

        self.generator_matrix = generator_matrix
        self.k = k
        self.n = n

        self.states = states = 1 << (k - 1)

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

    def _forward_step(self, x: GF2, cost: ndarray) -> GF2:
        cost = np.min(cost, axis=-1)

        inf = cost == np.inf

        cost.T[inf.T] = cost.T[inf[::-1].T]

        branch_metric = np.sum(np.array(self._expected ^ x), axis=-1)
        path_metric = branch_metric + cost

        branch = self._branch
        new_cost = np.full((self.n, self.states, self.states), np.inf)

        for s, n in enumerate(np.argmin(path_metric, axis=0)):
            new_cost[n, branch[n, s], s] = path_metric[n, s]

        return new_cost

    def _reverse_step(
        self,
        cost: ndarray,
        state: ndarray,
    ) -> tuple[GF2, ndarray]:
        previous_state = np.argmin(cost[..., state, :], axis=-1)

        bit = np.zeros(self.n, dtype=cost.dtype)

        for i, p in enumerate(previous_state):
            bit[i] = cost[i, state, p]

        bit = GF2(np.argmin(bit))

        return bit, previous_state[bit]


def poly2matrix(polynomials: list[int], k: int) -> GF2:
    assert k > 0
    assert len(polynomials) >= 1

    return GF2(
        [
            galois.Poly.Int(polynomial, field=GF2).coefficients(k, "asc")
            for polynomial in polynomials
        ]
    ).T
