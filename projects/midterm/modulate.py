# SPDX-License-Identifier: GPL-3.0-or-later
#
# modulate.py -- subcarrier modulation mapping
# Copyright (C) 2025  Jacob Koziej <jacobkoziej@gmail.com>

import numpy as np

from dataclasses import dataclass
from typing import Final

from numpy import ndarray


@dataclass(frozen=True, kw_only=True)
class _Mapping:
    mask: int
    shift: int
    k_mod: ndarray
    decode: ndarray
    encode: ndarray

    real: bool = False


_MAPPING_BPSK: _Mapping = _Mapping(
    mask=0x1,
    shift=0,
    k_mod=1 / np.sqrt(1),
    decode=np.array(
        [
            0b0,  # -1
            0b1,  # +1
        ],
        dtype=np.uint8,
    ),
    encode=np.array(
        [
            -1,  # 0b0
            +1,  # 0b1
        ],
        dtype=np.int8,
    ),
    real=True,
)
_MAPPING_QPSK: _Mapping = _Mapping(
    mask=0x1,
    shift=1,
    k_mod=1 / np.sqrt(2),
    decode=np.array(
        [
            0b0,  # -1
            0b1,  # +1
        ],
        dtype=np.uint8,
    ),
    encode=np.array(
        [
            -1,  # 0b0
            +1,  # 0b1
        ],
        dtype=np.int8,
    ),
)
_MAPPING_16_QAM: _Mapping = _Mapping(
    mask=0x3,
    shift=2,
    k_mod=1 / np.sqrt(10),
    decode=np.array(
        [
            0b00,  # -3
            0b10,  # -1
            0b11,  # +1
            0b01,  # +3
        ],
        dtype=np.uint8,
    ),
    encode=np.array(
        [
            -3,  # 0b00
            +3,  # 0b01
            -1,  # 0b10
            +1,  # 0b11
        ],
        dtype=np.int8,
    ),
)
_MAPPING_64_QAM: _Mapping = _Mapping(
    mask=0x7,
    shift=3,
    k_mod=1 / np.sqrt(42),
    decode=np.array(
        [
            0b000,  # -7
            0b100,  # -5
            0b110,  # -3
            0b010,  # -1
            0b011,  # +1
            0b111,  # +3
            0b101,  # +5
            0b001,  # +7
        ],
        dtype=np.uint8,
    ),
    encode=np.array(
        [
            -7,  # 0b000
            +7,  # 0b001
            -1,  # 0b010
            +1,  # 0b011
            -5,  # 0b100
            +5,  # 0b101
            -3,  # 0b110
            +3,  # 0b111
        ],
        dtype=np.int8,
    ),
)

_MAPPING: Final[dict[int, _Mapping]] = {
    6: _MAPPING_BPSK,
    9: _MAPPING_BPSK,
    12: _MAPPING_QPSK,
    18: _MAPPING_QPSK,
    24: _MAPPING_16_QAM,
    36: _MAPPING_16_QAM,
    48: _MAPPING_64_QAM,
    54: _MAPPING_64_QAM,
}


def _decode_component(x: ndarray, mapping: _Mapping) -> ndarray:
    mask = mapping.mask

    index = np.clip(np.rint(x), a_min=-mask, a_max=mask)
    index = (index + mask).astype(np.uint8) // 2

    return mapping.decode[index]


def _get_mapping(rate: int) -> _Mapping:
    try:
        mapping = _MAPPING[rate]

    except Exception as _:
        raise KeyError(f"Unsupported rate: {rate}")

    return mapping


def demodulate(d: ndarray, rate: int) -> ndarray:
    mapping = _get_mapping(rate)

    d = d / mapping.k_mod

    if mapping.real:
        d = d.real

    i = _decode_component(d.real, mapping)
    q = _decode_component(d.imag, mapping)

    return (q << mapping.shift) | i


def modulate(x: ndarray, rate: int) -> ndarray:
    mapping = _get_mapping(rate)

    i = x & mapping.mask
    q = (x >> mapping.shift) & mapping.mask

    d = (mapping.encode[i] + 1j * mapping.encode[q]) * mapping.k_mod

    if mapping.real:
        d = d.real

    return d
