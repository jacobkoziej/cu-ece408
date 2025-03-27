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
