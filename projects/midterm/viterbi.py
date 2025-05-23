# SPDX-License-Identifier: GPL-3.0-or-later
#
# viterbi.py -- Viterbi decoder
# Copyright (C) 2025  Jacob Koziej <jacobkoziej@gmail.com>

import galois
import numpy as np

from typing import Optional

from einops import reduce
from galois import GF2
from numpy import ndarray

from bit import packbits


class Viterbi:
    def __call__(self, x: GF2, valid: Optional[ndarray] = None) -> GF2:
        if valid is None:
            valid = np.full(x.shape, True)

        n = self.n
        states = self.states

        x = x.reshape(-1, n)
        valid = valid.reshape(x.shape)

        cost = np.full((len(x) + 1, n, states, states), np.inf)
        cost[0, 0, 0, 0] = 0

        for i, (x_i, valid_i) in enumerate(zip(x, valid)):
            new_cost = self._forward_step(x_i, valid_i, cost[i])

            cost[i + 1] = new_cost

        y = GF2.Zeros(len(x))

        path = np.full(len(x) + 1, -1)
        path[-1] = 0

        for i in range(len(y), 0, -1):
            bit, new_path = self._reverse_step(cost[i], path[i])

            y[i - 1] = bit
            path[i - 1] = new_path

        return y

    def __init__(self, generator_matrix: GF2) -> None:
        self.generator_matrix = generator_matrix

        self.k = k = generator_matrix.shape[0]
        self.n = n = generator_matrix.shape[1]

        # we only support 2-bit convolutional codes
        assert n == 2

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

    def _forward_step(self, x: GF2, valid: ndarray, cost: ndarray) -> GF2:
        cost = reduce(cost, "input state branch -> state", "min")

        branch_metric = np.array(self._expected ^ x)
        branch_metric = reduce(
            branch_metric * valid,
            "input state bits -> input state",
            "sum",
        )
        path_metric = branch_metric + cost

        branch = self._branch
        new_cost = np.full((self.n, self.states, self.states), np.inf)

        for n in range(self.n):
            for s in range(self.states):
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
