# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
# SPDX-License-Identifier: GPL-3.0-or-later
#
# report.py -- IEEE Std 802.11a-1999
# Copyright (C) 2025  Jacob Koziej <jacobkoziej@gmail.com>

# %% [markdown]
# For this project, I decided to write a TX/RX pair that implements most
# of IEEE Std 802.11a-1999's physical (PHY) layer, specifically the
# Physical layer Protocol Data Unit (PPDU) frame format. On top of this,
# I've taken the liberty of writing a Viterbi hard-decision decoder for
# the receiver along with various unit tests to ensure that each of the
# sub-components of the TX/RX pair behave as expected.

# %% [markdown]
# # PPDU Frame Format
#
# ```
# |<------------------- PLCP Header ------------------>|
# +------+----------+--------+--------+------+---------+------+------+-----+
# | RATE | RESERVED | LENGTH | PARITY | TAIL | SERVICE | PSDU | TAIL | PAD |
# | 4b   | 1b       | 12b    | 1b     | 6b   | 16b     |      | 6b   |     |
# +------+----------+--------+--------+------+---------+------+------+-----+
# |                                          |                             |
# +------------------------+ Coded / OFDM    | Coded / OFDM                |
#                          | (BPSK, r=1/2)   | (RATE indicated in SIGNAL)  |
#                          |<--------------->|<--------------------------->|
#          +---------------+-----------------+-----------------------------+
#          | PLCP Preamble | SIGNAL          | DATA                        |
#          | 12 symbols    | One OFDM Symbol | Variable OFDM Symbols       |
#          +---------------+-----------------+-----------------------------+
# ```

# %% [markdown]
# ## PLCP Header
#
# We first turn our attention to the Physical Layer Convergence Protocol
# (PLCP) header as it serves as a great entry point to understanding how
# to decode a PPDU frame.
#
# - The `RATE` field specifies the encoded rate of the `DATA` field of
#   our PPDU frame. Importantly, this tells us the modulation, coding
#   rate ($R$), coded bits per suhcarrier ($N_{BPSC}$), coded bits per
#   OFDM symbol ($N_{CBPS}$), and data bits per OFDM symbol
#   ($N_{DBPS}$).
# - The `RESERVED` field is reserved and always encoded as zero.
# - The `LENGTH` field encodes the number of bytes encoded in the
#   current PPDU frame.
# - The `PARITY` field ensures even parity with the preceding 15 bits.
# - The `TAIL` field ensures that our convolutional coder returns back
#   to the zero state.
# - The `SERVICE` field consists of 16 bits, of which the last 10 are
#   reserved (and are always encoded as zero). The first six bits of
#   this field are always zero and allow the receiver to determine the
#   initial state of the PLCP `DATA` scrambler.
