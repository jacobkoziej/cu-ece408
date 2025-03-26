# SPDX-License-Identifier: GPL-3.0-or-later
#
# wifi.py -- IEEE Std 802.11a-1999 tx/rx pair
# Copyright (C) 2025  Jacob Koziej <jacobkoziej@gmail.com>

import numpy as np

import ofdm
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
from ofdm import (
    FRAME_SIZE,
    LONG_TRAINING_SIZE,
    LONG_TRAINING_SYMBOLS,
    LONG_TRAINING_SYMBOL_SAMPLES,
    SHORT_TRAINING_SIZE,
    SHORT_TRAINING_SYMBOLS,
    SHORT_TRAINING_SYMBOL_SAMPLES,
    SUBCARRIERS_DATA,
    add_circular_prefix,
    apply_window,
    carrier_frequency_offset,
    remove_circular_prefix,
    unapply_window,
)
from plcp import (
    ConvolutionalEncoder,
    GENERATOR_CONSTRAINT_LENGTH,
    GENERATOR_POLYNOMIALS,
    Interleaver,
    Puncturer,
    SCRAMBLER_SERVICE_BITS,
    SERVICE_BITS,
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
        short_training_sequence = x[:SHORT_TRAINING_SIZE]

        coarse_offset = carrier_frequency_offset(
            short_training_sequence,
            SHORT_TRAINING_SYMBOL_SAMPLES,
            SHORT_TRAINING_SYMBOLS - 1,
        )

        x *= np.exp(-1j * coarse_offset * np.arange(x.size))
        x = x[SHORT_TRAINING_SIZE:]

        long_training_sequence = remove_circular_prefix(x[:LONG_TRAINING_SIZE])

        fine_offset = carrier_frequency_offset(
            long_training_sequence,
            LONG_TRAINING_SYMBOL_SAMPLES,
            LONG_TRAINING_SYMBOLS - 1,
        )

        x *= np.exp(-1j * fine_offset * np.arange(x.size))
        x = x[LONG_TRAINING_SIZE:]

        # fmt: off
        signal = x[:FRAME_SIZE]
        data   = x[FRAME_SIZE:]
        # fmt: on

        signal = self._ofdm_demodulate(signal)
        signal = self._demodulate(signal, 6)
        signal = self._deinterleave(signal, 6)
        signal = self._apply_viterbi_decoder(signal)
        signal = decode_signal(signal)

        if signal is None:
            return None

        self._update_state(signal)

        data = self._ofdm_demodulate(data)
        data = self._demodulate(data)
        data = self._deinterleave(data)

        valid = GF2.Ones(data.shape)
        valid = self._depuncture(valid)
        valid = np.array(valid).astype(np.bool)

        data = self._depuncture(data)
        data = self._apply_viterbi_decoder(data, valid)
        state = self._estimate_scrambler_state(data[:SCRAMBLER_SERVICE_BITS])
        data = self._descramble(data, state)

        y = self._decode(data)

        return y

    def __init__(self) -> None:
        generator_matrix = poly2matrix(
            GENERATOR_POLYNOMIALS,
            GENERATOR_CONSTRAINT_LENGTH,
        )
        self.decoder = Viterbi(generator_matrix)

        self.scrambler = Scrambler(0)

    def _apply_viterbi_decoder(
        self,
        x: GF2,
        valid: Optional[ndarray] = None,
    ) -> GF2:
        return self.decoder(x, valid)

    def _decode(self, data: GF2) -> ndarray:
        psdu = data[SERVICE_BITS : -(TAIL_BITS + self._n_pad)]

        y = packbits(psdu.reshape(-1, 8))

        return y

    def _deinterleave(self, x: GF2, rate: Optional[int] = None) -> GF2:
        if rate is None:
            rate = self._rate

        rate_parameter = plcp.rate_parameter(rate)

        bpsc = rate_parameter.bpsc
        cbps = rate_parameter.cbps

        x = _chunk_data(x, cbps)

        interleaver = Interleaver(bpsc=bpsc, cbps=cbps)

        y = interleaver.reverse(x)

        return y.flatten()

    def _demodulate(self, x: GF2, rate: Optional[int] = None) -> ndarray:
        if rate is None:
            rate = self._rate

        bpsc = plcp.rate_parameter(rate).bpsc

        x = demodulate(x, rate)

        return unpackbits(x, count=bpsc).flatten()

    def _depuncture(self, x: GF2) -> GF2:
        puncturer = Puncturer(self._coding_rate)

        return puncturer.reverse(x)

    def _descramble(self, data: GF2, state: int) -> ndarray:
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

    def _ofdm_demodulate(self, x: ndarray) -> ndarray:
        x = x.reshape(-1, FRAME_SIZE)
        x = unapply_window(x)
        x = remove_circular_prefix(x)

        return ofdm.demodulate(x).flatten()

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

        short_training_sequence = apply_window(ofdm.short_training_sequence())
        long_training_sequence = apply_window(ofdm.long_training_sequence())

        signal = Signal(rate, self._length)
        signal = encode_signal(signal)
        signal = self._apply_convolutional_encoder(signal)
        signal = self._interleave(signal, 6)
        signal = self._modulate(signal, 6)
        signal = self._ofdm_modulate(signal)

        data = self._encode(x)
        data = self._scramble(data)
        data = self._apply_convolutional_encoder(data)
        data = self._puncture(data)
        data = self._interleave(data)
        data = self._modulate(data)
        data = self._ofdm_modulate(data)

        return np.concatenate(
            [
                short_training_sequence,
                long_training_sequence,
                signal,
                data,
            ],
        )

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

    def _encode(self, x: ndarray) -> GF2:
        data = GF2.Zeros(self._n_data)

        psdu = unpackbits(x).flatten()

        data[0:SERVICE_BITS] = plcp.service()
        data[SERVICE_BITS : -(TAIL_BITS + self._n_pad)] = psdu

        return data

    def _interleave(self, x: GF2, rate: Optional[int] = None) -> GF2:
        if rate is None:
            rate = self._rate

        rate_parameter = plcp.rate_parameter(rate)

        bpsc = rate_parameter.bpsc
        cbps = rate_parameter.cbps

        x = _chunk_data(x, cbps)

        interleaver = Interleaver(bpsc=bpsc, cbps=cbps)

        y = interleaver.forward(x)

        return y.flatten()

    def _modulate(self, x: GF2, rate: Optional[int] = None) -> ndarray:
        if rate is None:
            rate = self._rate

        bpsc = plcp.rate_parameter(rate).bpsc

        x = x.reshape(-1, bpsc)
        x = packbits(x)

        return modulate(x, rate)

    def _ofdm_modulate(self, x: ndarray) -> ndarray:
        x = x.reshape(-1, SUBCARRIERS_DATA)
        x = ofdm.modulate(x)
        x = add_circular_prefix(x)

        return apply_window(x).flatten()

    def _puncture(self, x: GF2) -> GF2:
        puncturer = Puncturer(self._coding_rate)

        return puncturer.forward(x)

    def _scramble(self, x: GF2) -> GF2:
        seed = int(self.rng.integers(1, 1 << (Scrambler.k - 1)))

        self.scrambler.seed(seed)

        scrambled = self.scrambler(x)

        n_pad = self._n_pad

        scrambled[-(TAIL_BITS + n_pad) : -n_pad] = 0

        return scrambled

    def _update_state(self, x: ndarray, rate: int) -> None:
        assert x.dtype == np.uint8

        self._rate = rate
        self._length = len(x)

        rate_parameter = plcp.rate_parameter(rate)

        self._coding_rate = rate_parameter.coding_rate

        self._bpsc = rate_parameter.bpsc
        self._cbps = rate_parameter.cbps
        self._dbps = rate_parameter.dbps

        self._n_data = _calculate_data_bits(self._length, self._dbps)
        self._n_pad = _calculate_pad_bits(self._length, self._n_data)
