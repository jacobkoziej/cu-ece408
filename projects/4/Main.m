% SPDX-License-Identifier: GPL-3.0-or-later
%
% Main.m -- project 4
% Copyright (C) 2025  Jacob Koziej <jacobkoziej@gmail.com>

clc;
clear;
close all;

rng(0x93970dda);

% simulation parameters
TRAINING_SYMBOLS = 128;
MESSAGE_SYMBOLS  = 1024;
BITS_PER_SYMBOL  = 4;
MOD_FUNC         = @qammod;
DEMOD_FUNC       = @qamdemod;
TX_CHANNELS      = 2;
RX_CHANNELS      = 2;
FADING_VARIANCE  = 1.6;
SNR              = 8;
OFDM_FFT_BINS    = 64;
OFDM_CP_LENGTH   = 16;
MIMO_OFDM_TAPS   = 3;

%%% Simulation Data
% For this simulation, I've decided to generate a short training
% sequence and a longer sequence of message symbols. Since we are not
% generating bit error rate (BER) plots, I've opted to have a single
% run to gauge the effectiveness of the following equalization
% techniques.

%%
M = 2^BITS_PER_SYMBOL;

training_bits = randi(M - 1, TX_CHANNELS, TRAINING_SYMBOLS);
message_bits  = randi(M - 1, TX_CHANNELS, MESSAGE_SYMBOLS);

X_train   = MOD_FUNC(training_bits, M);
X_message = MOD_FUNC(message_bits, M);

%%% Flat Fading Channel
% To simulate a multi-input multi-output (MIMO) flat fading channel, we
% can create an $M$ by $N$ matrix where $M$ represents the number of
% outputs and $N$ the number of inputs of the MIMO system. Each of the
% entries in this matrix are sampled from a complex, zero-mean, Gaussian
% random variable. Since there is just a single filter coefficient to
% represent our channel, we can say that our channel fits the flat
% fading criteria as it merely varies the amplitude and phase of each of
% our samples, introducing no inter-symbol interference (ISI).

%%
CN = @(variance, M, N) sqrt(variance / 2) .* (randn(M, N) + 1j * randn(M, N));

H = CN(FADING_VARIANCE, RX_CHANNELS, TX_CHANNELS);

%%% Additive White Gaussian Noise
P_signal = mean(abs([X_train, X_message]).^2, 'all');
P_noise  = P_signal / (10^(SNR / 10));

N_train   = CN(P_noise, RX_CHANNELS, TRAINING_SYMBOLS);
N_message = CN(P_noise, RX_CHANNELS, MESSAGE_SYMBOLS);

%%% Received Data
Y_train   = (H * X_train) + N_train;
Y_message = (H * X_message) + N_message;

%%% Pre-coding
% When pre-coding, the we have channel state information at the
% transceiver (CSIT). This allows us to invert the channel before
% transmitting so that by the time our signal reaches the receiver,
% there should be no fading, only additive white Gaussian noise (AWGN).
% We can achieve this mathematically with the Moore-Penrose inverse, and
% perform a matrix multiply with our signal before transmission.
W_precode = @(H) pinv(H);

X_precode_message = (H * W_precode(H) * X_message) + N_message;
[~, ber_precode] = biterr(message_bits, DEMOD_FUNC(X_precode_message, M));

%%% Zero-forcing
% Zero-forcing pushes off the responsibility of modeling the channel
% onto the receiver, conveniently called channel state information at
% the receiver (CSIR). The process of zero-forcing tries to find the
% channel inverse which minimizes the spectral norm: $||y - Hx||$, where
% $y$ is the received signal, $H$ is the channel, and $x$ is the
% original signal. The solution to this problem involves the
% Moore-Penrose inverses of $H$. Although great on initial impression,
% the drawback of this approach is that it can amplify noise present at
% the receiver while inverting the channel. Since the noise is
% independent of the channel, it gets caught in the crossfire and gets
% in the way of us successfully decoding our signal.
%
% Since we usually have no knowledge of the channel, we can sound it
% using a training sequence. By solving for the Moore-Penrose inverse of
% the symbols sent at the transmitter, we can retrieve an estimate for
% the channel with the assumption that the noise found in $y$ is not
% significant.

%%
H_zf = Y_train * pinv(X_train);

X_zf_message = pinv(H_zf) * Y_message;
[~, ber_zf] = biterr(message_bits, DEMOD_FUNC(X_zf_message, M));

%%% Minimum Mean Square Error
% To mitigate amplifying the noise term in our received signal, we can
% throw in a normalization term that takes into account the relative
% power of the noise in our signal so that we don't entirely drown out
% our barely detectable signal. Linear minimum mean square error (MMSE)
% estimation does just that by increasing the lower bound of our
% original minimization problem.
%
% We can achieve this mathematically by injecting an identity matrix
% scaled by the variance of the noise into the Moore-Penrose inverse.

%%
H_mmse = Y_train * X_train' * ...
    inv(X_train * X_train' + P_noise * eye(TX_CHANNELS));

X_mmse_message = pinv(H_mmse) * Y_message;
[~, ber_mmse] = biterr(message_bits, DEMOD_FUNC(X_mmse_message, M));

%%% Results
% From all of these techniques, pre-coding works best since it entirely
% mitigates the effect of the transmission channel, leaving us with
% nothing but AWGN. Although it is the most effective of all the three
% strategies, it is also the most impractical as it assumes knowledge of
% the channel instead of an estimate. Zero-forcing and MMSE get around
% this by estimating the channel at the receiver, however, this has the
% limitation of including AWGN in the estimates, meaning they will only
% ever approach the effectiveness of pre-coding.

%%
display(ber_precode);
display(ber_zf);
display(ber_mmse);

%%% OFDM
% Orthogonal frequency-division multiplexing (OFDM) allows for us to
% transmit data simultaneous using multiple sub-carriers unlike
% traditional single carrier systems. Since sub-carriers are designed to
% be orthogonal, they do not interfere with each other during
% transmission. Another benefit of this approach is that large amounts
% of data is sent over numerous low-rate streams, making the system
% resilient to burst errors in a channel.
%
% The process of encoding a signal with OFDM involves populating
% discrete Fourier transform (DFT) bins and then taking the inverse DFT.
% If this signal were sent in this form, our symbols would all
% experience ISI if our channel has frequency selective fading. To
% mitigate this, we prepend a cyclic prefix to our signal to make it
% periodic by prepending a block of the end of our signal. So long as
% the cyclic prefix is longer than a channel, the linear convolution
% performed in the time domain turns into a cyclic convolution in the
% frequency domain, preventing ISI. This then allows us to perform
% OFDM's famous "single tap" equalization after removing the cyclic
% prefix and before performing a DFT. We can further improve this by
% adding pilot tones in the frequency domain which stay constant and
% allow us to better estimate changes in the channel.

%%
% h = [1, 0.2, 0.4];
% h = [0.888, 0.233, 0.902, 0.123, 0.334];
h = [0.227, 0.460, 0.688, 0.460, 0.227];

H = fftshift(freqz(h, 1, OFDM_FFT_BINS, 'whole'));

X_ofdm = ofdmmod(X_message.', OFDM_FFT_BINS, OFDM_CP_LENGTH);

P_ofdm       = mean(abs(X_ofdm).^2, 'all');
P_ofdm_noise = P_ofdm / (10^(SNR / 10));

N_ofdm = CN(P_ofdm_noise, size(X_ofdm, 1), size(X_ofdm, 2));

Y_ofdm = filter(h, 1, X_ofdm) + N_ofdm;

C_zf   = H;
C_mmse = conj(H) ./ (abs(H).^2 + (P_ofdm_noise / P_ofdm));

X_ofdm_zf = ofdmdemod(Y_ofdm, OFDM_FFT_BINS, OFDM_CP_LENGTH, C_zf).';
[~, ber_ofdm_zf] = biterr(message_bits, DEMOD_FUNC(X_ofdm_zf, M));

X_ofdm_mmse = ofdmdemod(Y_ofdm, OFDM_FFT_BINS, OFDM_CP_LENGTH, C_mmse).';
[~, ber_ofdm_mmse] = biterr(message_bits, DEMOD_FUNC(X_ofdm_mmse, M));

%%% Results
% Strangely MMSE performs worse than zero-forcing.

%%
display(ber_ofdm_zf);
display(ber_ofdm_mmse);

%%% MIMO OFDM
H = zeros(MIMO_OFDM_TAPS, RX_CHANNELS, TX_CHANNELS);
for i = 1:MIMO_OFDM_TAPS
    H(i, :, :) = CN(FADING_VARIANCE, RX_CHANNELS, TX_CHANNELS);
end

H_freq = zeros(RX_CHANNELS, TX_CHANNELS, OFDM_FFT_BINS);

for i = 1:RX_CHANNELS
    for j = 1:TX_CHANNELS
        h = squeeze(H(:, i, j));
        H_freq(i, j, :) = fftshift(fft(h, OFDM_FFT_BINS));
    end
end

X_ofdm = ofdmmod(X_message.', OFDM_FFT_BINS, OFDM_CP_LENGTH).';

H_merge = eye(size(H, 2, 3));

for i = 1:MIMO_OFDM_TAPS
    H_merge = squeeze(H(i, :, :)) * H_merge;
end

Y_ofdm = H_merge * X_ofdm + N_ofdm.';

Y_ofdm_eq = Y_ofdm;
for i = 1:OFDM_FFT_BINS
    h = squeeze(H_freq(:, :, i));
    y = squeeze(Y_ofdm(:, OFDM_CP_LENGTH + i));
    Y_ofdm_eq(:, i) = pinv(h) * y;
end

X_ofdm_message = ofdmdemod(Y_ofdm_eq.', OFDM_FFT_BINS, OFDM_CP_LENGTH, 1).';
[~, ber_ofdm_message] = biterr(message_bits, DEMOD_FUNC(X_ofdm_message, M));

display(ber_ofdm_message);
