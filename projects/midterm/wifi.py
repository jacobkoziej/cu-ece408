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
from plcp import (
    SERVICE_BITS,
    SIGNAL_BITS,
    Signal,
    decode_signal,
    encode_signal,
)


TAIL_BITS: Final[int] = 6


def _calculate_data_bits(length: int, dbps: int) -> int:
    n_sym = ceil((SERVICE_BITS + 8 * length + TAIL_BITS) / dbps)
    n_data = n_sym * dbps

    return n_data


def _calculate_pad_bits(length: int, n_data: int) -> int:
    return n_data - (SERVICE_BITS + 8 * length + TAIL_BITS)


class Rx:
    def __call__(self, x: ndarray) -> Optional[ndarray]:
        x = GF2(x)

        # fmt: off
        signal = x[:SIGNAL_BITS]
        data   = x[SIGNAL_BITS:]
        # fmt: on

        signal = decode_signal(signal)

        if signal is None:
            return None

        rate_parameter = plcp.rate_parameter(signal.rate)

        length = signal.length

        dbps = rate_parameter.dbps

        n_data = _calculate_data_bits(length, dbps)
        n_pad = _calculate_pad_bits(length, n_data)

        psdu = data[SERVICE_BITS : -(TAIL_BITS + n_pad)]

        y = packbits(psdu.reshape(-1, 8))

        return y


class Tx:
    def __call__(self, x: ndarray, rate: int) -> ndarray:
        assert x.dtype == np.uint8

        length = len(x)

        signal = Signal(rate, length)
        signal = encode_signal(signal)

        service = plcp.service()
        rate_parameter = plcp.rate_parameter(rate)

        dbps = rate_parameter.dbps

        n_data = _calculate_data_bits(length, dbps)
        n_pad = _calculate_pad_bits(length, n_data)

        psdu = unpackbits(x).flatten()

        data = GF2.Zeros(n_data)

        data[0:SERVICE_BITS] = service
        data[SERVICE_BITS : -(TAIL_BITS + n_pad)] = psdu

        return np.concatenate([signal, data]).astype(np.uint8)

    def __init__(self, *, rng: Generator = None):
        if rng is None:
            rng = np.random.default_rng()

        self.rng = rng
