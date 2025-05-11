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
FADING_VARIANCE  = 1;
SNR              = 40;

%%% Simulation Data
M = 2^BITS_PER_SYMBOL;

training_bits = randi(M - 1, TX_CHANNELS, TRAINING_SYMBOLS);
message_bits  = randi(M - 1, TX_CHANNELS, MESSAGE_SYMBOLS);

X_train   = MOD_FUNC(training_bits, M);
X_message = MOD_FUNC(message_bits, M);

%%% Flat Fading Channel
CN = @(variance, M, N) sqrt(variance / 2) .* (randn(M, N) + 1j * randn(M, N));

H = CN(FADING_VARIANCE, RX_CHANNELS, TX_CHANNELS);

%%% Additive White Gaussian Noise
P_signal = mean(abs([X_train, X_message]), 'all');
P_noise  = P_signal / (10^(SNR / 10));

N_train   = CN(P_noise, RX_CHANNELS, TRAINING_SYMBOLS);
N_message = CN(P_noise, RX_CHANNELS, MESSAGE_SYMBOLS);

%%% Received Data
Y_train   = (H * X_train) + N_train;
Y_message = (H * X_message) + N_message;

%%% Pre-coding
W_precode = @(H) pinv(H);

X_precode_message = (H * W_precode(H) * X_message) + N_message;
[~, ber_precode] = biterr(message_bits, DEMOD_FUNC(X_precode_message, M));

%%% Zero-forcing
H_zf = Y_train * pinv(X_train);

X_zf_message = pinv(H_zf) * Y_message;
[~, ber_zf] = biterr(message_bits, DEMOD_FUNC(X_zf_message, M));

%%% Minimum Mean Square Error
H_mmse = Y_train * X_train' * ...
    inv(X_train * X_train' + P_noise * eye(TX_CHANNELS));

X_mmse_message = pinv(H_mmse) * Y_message;
[~, ber_mmse] = biterr(message_bits, DEMOD_FUNC(X_mmse_message, M));
