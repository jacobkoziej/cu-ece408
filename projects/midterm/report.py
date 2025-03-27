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

# %%
import matplotlib.pyplot as plt
import numpy as np

import modulate
import ofdm
import plcp

from fractions import Fraction

from galois import GF2

# %%
rng: np.random.Generator = np.random.default_rng(0x509355EF)

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

# %% [markdown]
# ## PLCP `DATA` Scrambler
#
# We define a frame synchronous scrambler over $\mathbb{F}_2$ with the
# polynomial $S(x) = x^7 + x^4 + 1$. This polynomial can be realized as
# a simple shift register of seven elements with feedback, clocked for
# every input bit. By its definition, this polynomial produces a
# pseudo-random 127-bit sequence.
#
# Adding a scrambler may seem like a strange thing, but it introduces
# entropy into our signal (for example, it lets us work around a user
# that insists on sending vast quantities of zeros) to *hopefully* aid
# in our data successfully finding its way to the receiver. To scramble
# our data, we initialize the shift register with a non-zero random
# state and clock and XOR our data bitstream with the least significant
# bit of the shift register. To descramble, we reset the initial state
# and repeat the process.

# %% [markdown]
# ## Convolutional Encoder
#
# To encode the scrambled bitstream, we utilize a convolutional coder
# defined over $\mathbb{F}_2$ with the industry standard generator
# polynomials, $g_0 = 133_8$ and $g_1 = 171_8$, of rate $R = 1/2$,
# meaning that for every one bit of data, we encode two bits.
#
# The benefit of a convolutional code over a block code is that they
# handle burst errors better since they rely on an internal state to
# "remember" previously encoded bits to aid in error correction using a
# reverse process like the Viterbi algorithm.
#
# In our case, the two polynomials have a constraint length of $K = 7$,
# meaning it has an internal "memory" of six, or $2^6$ states. Since we
# want a deterministic way of decoding our encoded bitstream, we, by
# convention, start and end our bitstream in the zero state. We achieve
# this by sending a sequence of six zeros which perfectly coincides with
# the `TAIL` field.

# %% [markdown]
# ## Punctured Coding
#
# Since we're using a convolutional encoder and decoding our bitstream
# with error correction, we can take advantage of this receiver-side
# process and artificially introduce bit errors by "puncturing" our
# encoded bitstream, increasing bandwidth. On the receiver-side, we
# simply re-insert "dummy" bits in place of the real bits and decode the
# bitstream with error correction.
#
# For example, a $R = 2/3$ code will encode two bits of input data with
# three bits (where `-1` corresponds to a punctured bit):

# %% tags=["hide-input"]
puncturer = plcp.Puncturer(Fraction(2, 3))

punctured = puncturer.forward(GF2.Ones(12))
depunctured = puncturer.reverse(punctured)

x = np.arange(depunctured.size)
x[~np.array(depunctured, dtype=np.bool)] = -1

x.reshape(-1, 2).T

# %% [markdown]
# ## Interleaving
#
# In an effort to combat burst errors, we apply an interleaver to our
# data. We apply two permutations to the bitstream:
#
# - The first ensures that adjacent coded bits are mapped onto
#   non-adjacent subcarriers.
# - The second ensures that adjacent coded bits are mapped alternately
#   onto less and more significant bits of the constellations to combat
#   long runs of low reliability.
#
# By applying this strategy, we can treat bit errors as independent
# events on the side of the receiver following deinterleaving. As an
# example, let us look at the output of interleaving a block of
# $N_{CBPS} = 48$ bits:

# %% tags=["hide-input"]
bpsc = 1
cbps = 48
interleaver = plcp.Interleaver(bpsc=bpsc, cbps=cbps)

interleaved = interleaver.forward(np.arange(cbps))

interleaved.reshape(-1, 8)

# %% [markdown]
# ## Subcarrier Modulation Mapping
#
# We modulate data with either BPSK, QPSK, 16-QAM, or 64-QAM gray-coded
# constellations depending of the value of the `RATE` field. Since each
# constellation encodes a different amount of bits, each must also be
# properly scaled such that power stays uniform during transmission.
# We can demodulate by rescaling the constellation and the performing
# hard decisions on each symbol.
#
# As an example, let's look at the 16-QAM constellation which is
# associated with a `RATE` of 24:

# %% tags=["hide-input"]
x = np.arange(1 << 4, dtype=np.uint8)
s = modulate.modulate(x, 24)

plt.figure(figsize=(6, 6))
plt.scatter(s.real, s.imag, color="red")
for x, s in zip(x, s):
    plt.annotate(
        f"0b{int(x):04b}",
        (s.real, s.imag),
        xytext=(-2, 1),
        textcoords="offset fontsize",
    )
plt.xlabel("In-phase")
plt.ylabel("Quadrature")
plt.title("16-QAM")
plt.axhline(0, color="black", linewidth=0.5)
plt.axvline(0, color="black", linewidth=0.5)
plt.xlim(-1.5, 1.5)
plt.ylim(-1.5, 1.5)
plt.show()

# %% [markdown]
# ## OFDM
#
# Orthogonal Frequency-Division Multiplexing (OFDM) is a modulation
# technique that allows us to transmit data by splitting what would be a
# high-speed channel into multiple, low-speed channels with orthogonal
# subcarriers, improving the robustness of data transmission.
#
# In our case, we start with a DFT of size 64, and of these bin
# frequencies, we designate four to be pilot signals, and 48 for data.
# The remaining bins are left unused. The reason for a DFT of size 64 is
# twofold: we assume a digital frequency normalized to 20 MHz and FFTs
# of size 64 are common. We enumerate the bins starting at negative
# frequencies using indices -31 through 32, where index 0 refers to DC.

# %% [markdown]
# ### Pilot Subcarriers
#
# The pilots serve a crucial role in decoding as they allow a receiver
# to estimate the effects of a channel and invert it. Since OFDM
# operates in the frequency domain, it is sufficient to just divide
# through each of the bins with the channel estimate to invert the
# channel.
#
# One important detail is that the pilots themselves are scrambled
# (inverted depending on the state of the aforementioned scrambler when
# seeded with $177_8$). When the output of the scrambler is one, the
# pilots are inverted. The reason for doing so is to prevent spectral
# lines from showing up.

# %% tags=["hide-input"]
d = np.zeros(48)
s = np.fft.ifft(ofdm.modulate(d), axis=-1)

dft_bin = np.arange(s.size) - (s.size // 2) + 1

plt.figure()
plt.stem(dft_bin, np.abs(s))
plt.xlabel("DFT Bin")
plt.ylabel("Magnitude")
plt.title("Pilot Subcarriers")
plt.show()

# %% [markdown]
# ### Cyclic Prefix
#
# To mitigate the effect of inter-symbol interference (ISI), once the
# frequency domain representation of the encoded data is brought into
# the time domain, a cyclic prefix gets inserted. This is nothing more
# than taking the last $N$ output samples and prepending them to the
# front of the signal. This segment of the signal is called the "guard
# interval."
#
# In adding this prefix, we essentially change the linear convolution
# performed by our channel into a cyclic one (given that the channel
# group delay does not exceed the length of the guard interval). When
# decoding the signal, this prefix gets discarded as its sole purpose is
# for ISI protection in transit.

# %% tags=["hide-input"]
b = rng.integers(0, 1 << 4, 48, dtype=np.uint8)
d = modulate.modulate(b, 24)
s = ofdm.modulate(d)

plt.figure()
plt.stem(s.real)
plt.stem(s.imag, linefmt="r-", markerfmt="ro")
plt.legend(["In-phase", "Quadrature"])
plt.xlabel("Sample")
plt.ylabel("Value")
plt.title("OFDM without Cyclic Prefix")
plt.show()

# %% tags=["hide-input"]
s = ofdm.add_circular_prefix(s, ofdm.CIRCULAR_PREFIX)

plt.figure()
plt.stem(s.real)
plt.stem(s.imag, linefmt="r-", markerfmt="ro")
plt.legend(["In-phase", "Quadrature"])
plt.fill_between(
    np.arange(ofdm.CIRCULAR_PREFIX),
    -0.15,
    0.15,
    facecolor="green",
    alpha=0.25,
)
plt.fill_between(
    s.size - np.arange(ofdm.CIRCULAR_PREFIX) - 1,
    -0.15,
    0.15,
    facecolor="green",
    alpha=0.25,
)
plt.xlabel("Sample")
plt.ylabel("Value")
plt.title("OFDM with Cyclic Prefix")
plt.show()

# %% [markdown]
# ### Training Sequences
#
# Since our local and foreign oscillators are not perfect, when taken
# from baseband to bandpass and then back to baseband, our signal will
# incur a carrier frequency offset, which translates to a constant
# rotation in the complex plane between samples. This renders decoding a
# useless exercise as symbols will map incorrectly inside of
# constellations.
#
# To work around this, we transmit training sequences of predefined
# symbols that are mutually agreed upon. We first transmit a sequence of
# ten "short" symbols which we can utilize to determine a coarse
# frequency offset. Then we transmit a two "long" symbols over the same
# time period for calculating a fine frequency offset.
#
# When decoding, we can simply average the phase offset between
# associated samples between symbols with the following:
#
# $$
# \varphi
# =
# \frac{1}{N}\arg\left(\Sigma^{N - 1}_{i = 0} S_i^* S_{i + N}\right),
# $$
#
# where $N$ is equal to the number of samples per symbol and $S_i$ is
# the $i^\text{th}$ sample of the received signal in the time domain.
# This can be further improved by averaging $\varphi$ between symbols.
# After we calculate $\varphi$, it is sufficient to multiply samples by
# $e^{-1j\varphi n}$, where $n$ is the sample index, to remove the
# carrier offset.
