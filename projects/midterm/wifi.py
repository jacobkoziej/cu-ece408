# SPDX-License-Identifier: GPL-3.0-or-later
#
# wifi.py -- IEEE Std 802.11a-1999 tx/rx pair
# Copyright (C) 2025  Jacob Koziej <jacobkoziej@gmail.com>

import numpy as np

import plcp

from math import ceil
from typing import (
    Final,
    Optional,
)

from galois import GF2
from numpy import ndarray
from numpy.random import Generator

from bit import (
    packbits,
    unpackbits,
)
from modulate import (
    demodulate,
    modulate,
)
from plcp import (
    GENERATOR_CONSTRAINT_LENGTH,
    GENERATOR_POLYNOMIALS,
    SCRAMBLER_SERVICE_BITS,
    SERVICE_BITS,
    SIGNAL_BITS,
    ConvolutionalEncoder,
    Interleaver,
    Puncturer,
    Scrambler,
    Signal,
    decode_signal,
    encode_signal,
)
from viterbi import (
    Viterbi,
    poly2matrix,
)


TAIL_BITS: Final[int] = 6


def _calculate_data_bits(length: int, dbps: int) -> int:
    n_sym = ceil((SERVICE_BITS + 8 * length + TAIL_BITS) / dbps)
    n_data = n_sym * dbps

    return n_data


def _calculate_pad_bits(length: int, n_data: int) -> int:
    return n_data - (SERVICE_BITS + 8 * length + TAIL_BITS)


def _chunk_data(x: GF2, cbps: int) -> GF2:
    return x.reshape(-1, cbps)


class Rx:
    def __call__(self, x: ndarray) -> Optional[ndarray]:
        # fmt: off
        signal = x[:SIGNAL_BITS]
        data   = x[SIGNAL_BITS:]
        # fmt: on

        signal = self._demodulate_data(signal, 6)
        signal = decode_signal(signal)

        if signal is None:
            return None

        self._update_state(signal)

        data = self._demodulate_data(data, self._rate)

        data = self._deinterleave_data(data)
        valid = GF2.Ones(data.shape)
        valid = self._depuncture_data(valid)
        valid = np.array(valid).astype(np.bool)

        data = self._depuncture_data(data)
        data = self._apply_viterbi_decoder(data, valid)
        state = self._estimate_scrambler_state(data[:SCRAMBLER_SERVICE_BITS])
        data = self._descramble_data(data, state)

        y = self._decode_data(data)

        return y

    def __init__(self) -> None:
        generator_matrix = poly2matrix(
            GENERATOR_POLYNOMIALS,
            GENERATOR_CONSTRAINT_LENGTH,
        )
        self.decoder = Viterbi(generator_matrix)

        self.scrambler = Scrambler(0)

    def _apply_viterbi_decoder(self, x: GF2, valid: ndarray) -> GF2:
        return self.decoder(x, valid)

    def _decode_data(self, data: GF2) -> ndarray:
        psdu = data[SERVICE_BITS : -(TAIL_BITS + self._n_pad)]

        y = packbits(psdu.reshape(-1, 8))

        return y

    def _deinterleave_data(self, x: GF2) -> GF2:
        cbps = self._cbps

        x = _chunk_data(x, cbps)

        interleaver = Interleaver(bpsc=self._bpsc, cbps=cbps)

        y = interleaver.reverse(x)

        return y.flatten()

    def _demodulate_data(self, x: GF2, rate: int) -> ndarray:
        bpsc = plcp.rate_parameter(rate).bpsc

        x = demodulate(x, rate)

        return unpackbits(x, count=bpsc).flatten()

    def _depuncture_data(self, x: GF2) -> GF2:
        puncturer = Puncturer(self._coding_rate)

        return puncturer.reverse(x)

    def _descramble_data(self, data: GF2, state: int) -> ndarray:
        self.scrambler.seed(state)

        descrambled = self.scrambler(data)

        n_pad = self._n_pad

        descrambled[-(TAIL_BITS + n_pad) : -n_pad] = 0

        return descrambled

    def _estimate_scrambler_state(self, service: GF2) -> int:
        zeros = GF2.Zeros(service.shape)

        for state in range(1, 1 << (Scrambler.k - 1)):
            self.scrambler.seed(state)

            descrambled = self.scrambler(service)

            if np.all(descrambled == zeros):
                break

        return state

    def _update_state(self, signal: Signal) -> None:
        self._rate = signal.rate
        self._length = signal.length

        rate_parameter = plcp.rate_parameter(self._rate)

        self._coding_rate = rate_parameter.coding_rate

        self._bpsc = rate_parameter.bpsc
        self._cbps = rate_parameter.cbps
        self._dbps = rate_parameter.dbps

        self._n_data = _calculate_data_bits(self._length, self._dbps)
        self._n_pad = _calculate_pad_bits(self._length, self._n_data)


class Tx:
    def __call__(self, x: ndarray, rate: int) -> ndarray:
        self._update_state(x, rate)

        signal = Signal(rate, self._length)
        signal = encode_signal(signal)
        signal = self._modulate_data(signal, 6)

        data = self._encode_data(x)
        data = self._scramble_data(data)
        data = self._apply_convolutional_encoder(data)
        data = self._puncture_data(data)
        data = self._interleave_data(data)
        data = self._modulate_data(data, rate)

        return np.concatenate([signal, data])

    def __init__(self, *, rng: Generator = None):
        if rng is None:
            rng = np.random.default_rng()

        self.rng = rng

        generator_matrix = poly2matrix(
            GENERATOR_POLYNOMIALS,
            GENERATOR_CONSTRAINT_LENGTH,
        )
        self.encoder = ConvolutionalEncoder(generator_matrix)

        self.scrambler = Scrambler(0)

    def _apply_convolutional_encoder(self, x: GF2) -> GF2:
        return self.encoder(x).flatten()

    def _encode_data(self, x: ndarray) -> GF2:
        data = GF2.Zeros(self._n_data)

        psdu = unpackbits(x).flatten()

        data[0:SERVICE_BITS] = plcp.service()
        data[SERVICE_BITS : -(TAIL_BITS + self._n_pad)] = psdu

        return data

    def _interleave_data(self, x: GF2) -> GF2:
        cbps = self._cbps

        x = _chunk_data(x, cbps)

        interleaver = Interleaver(bpsc=self._bpsc, cbps=cbps)

        y = interleaver.forward(x)

        return y.flatten()

    def _modulate_data(self, x: GF2, rate: int) -> ndarray:
        bpsc = plcp.rate_parameter(rate).bpsc

        x = x.reshape(-1, bpsc)
        x = packbits(x)

        return modulate(x, rate)

    def _puncture_data(self, x: GF2) -> GF2:
        puncturer = Puncturer(self._coding_rate)

        return puncturer.forward(x)

    def _scramble_data(self, x: GF2) -> GF2:
        seed = int(self.rng.integers(1, 1 << (Scrambler.k - 1)))

        self.scrambler.seed(seed)

        scrambled = self.scrambler(x)

        n_pad = self._n_pad

        scrambled[-(TAIL_BITS + n_pad) : -n_pad] = 0

        return scrambled

    def _update_state(self, x: ndarray, rate: int) -> None:
        assert x.dtype == np.uint8

        self._length = len(x)

        rate_parameter = plcp.rate_parameter(rate)

        self._coding_rate = rate_parameter.coding_rate

        self._bpsc = rate_parameter.bpsc
        self._cbps = rate_parameter.cbps
        self._dbps = rate_parameter.dbps

        self._n_data = _calculate_data_bits(self._length, self._dbps)
        self._n_pad = _calculate_pad_bits(self._length, self._n_data)
