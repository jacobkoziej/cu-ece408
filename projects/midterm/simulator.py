#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
#
# simulator.py -- IEEE Std 802.11a-1999 simulator
# Copyright (C) 2025  Jacob Koziej <jacobkoziej@gmail.com>

import sys

import numpy as np

import bit
import ppdu

from argparse import ArgumentParser
from pathlib import Path

from numpy import ndarray
from numpy.random import Generator
from tqdm import trange

from wifi import (
    Rx,
    Tx,
)


def calculate_ber(value: ndarray, expected: ndarray) -> ndarray:
    value = bit.unpackbits(value)
    expected = bit.unpackbits(expected)

    return np.mean(value != expected, axis=(-2, -1))


def main() -> None:
    parser = ArgumentParser()

    parser.add_argument(
        "-b",
        "--bytes",
        default=32,
        type=int,
    )
    parser.add_argument(
        "-i",
        "--iterations",
        default=512,
        type=int,
    )
    parser.add_argument(
        "-p",
        "--points",
        default=41,
        type=int,
    )
    parser.add_argument(
        "-r",
        "--rate",
        required=True,
        type=int,
    )
    parser.add_argument(
        "--seed",
        default=0x48F76461,
        type=int,
    )
    parser.add_argument(
        "--snr-min",
        default=0.0,
        type=float,
    )
    parser.add_argument(
        "--snr-max",
        default=40.0,
        type=float,
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
    )

    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    snr = np.linspace(args.snr_min, args.snr_max, args.points)
    ber = np.zeros((args.points, args.iterations))

    for i in trange(args.points, ncols=80):
        for j in range(args.iterations):
            data, received = sim(rng, args.bytes, args.rate, snr[i])
            ber[i, j] = calculate_ber(received, data)

    ber = np.mean(ber, axis=-1)

    output = Path(f"{args.rate}.csv") if args.output is None else args.output

    np.savetxt(
        output,
        np.array([snr, ber]).T,
        header="snr, ber",
        delimiter=",",
    )


def sim(
    rng: Generator,
    bytes: int,
    rate: int,
    snr_db: np.double,
) -> tuple[ndarray, ndarray]:
    tx = Tx(rng=rng)
    rx = Rx()

    data = rng.integers(0, 256, bytes, dtype=np.uint8)

    signal = tx(data, rate)

    p_signal = np.mean(np.abs(signal) ** 2)
    snr = 10 ** (snr_db / 10)

    noise_scale = np.sqrt((p_signal / snr) / 2)

    noise = 0
    noise += rng.normal(0, noise_scale, signal.shape) + 0j
    noise += 1j * rng.normal(0, noise_scale, signal.shape)

    received = rx(signal + noise, ppdu.Signal(rate, data.size))

    return data, received


if __name__ == "__main__":
    sys.exit(main())
