% SPDX-License-Identifier: GPL-3.0-or-later
%
% Main.m -- project 2
% Copyright (C) 2025  Jacob Koziej <jacobkoziej@gmail.com>

clc;
clear;
close all;

rng(0xf7fc753);

global N
global PATHS
global CHANNEL_GAIN
global F_M
global F_M_VARIATION

%% Parameters
ITERATIONS    = 50;
N             = 2^12;
M             = 2;
SIGNAL_GAIN   = db2mag(20);
PATHS         = 8;
CHANNEL_GAIN  = db2mag(0);
F_M           = 0.1;
F_M_VARIATION = 0.8;
SNR           = linspace(0, 50, 10);

%% Simulation
NO_DIVERSITY = 1;
MRRC_2RX     = 2;
MRRC_4RX     = 3;
ALAMOUTI_1RX = 4;
ALAMOUTI_2RX = 5;

TECHNIQUES   = 5;

BER = zeros(ITERATIONS, TECHNIQUES, numel(SNR));

for i = 1:ITERATIONS
    fprintf('progress: %d/%d\n', i, ITERATIONS);

    for j = 1:numel(SNR)
        snr       = SNR(j);
        bitstream = randi(M, 1, N) - 1;
        s         = SIGNAL_GAIN * pskmod(bitstream, M);

        % no diversity
        h        = generate_channel(1);
        s_hat    = mrrc(s, h, snr);
        [~, ber] = biterr(bitstream, pskdemod(s_hat, M));

        BER(i, NO_DIVERSITY, j) = ber;

        % mrrc (2 rx)
        h        = generate_channel(2);
        s_hat    = mrrc(s, h, snr);
        [~, ber] = biterr(bitstream, pskdemod(s_hat, M));

        BER(i, MRRC_2RX, j) = ber;

        % mrrc (4 rx)
        h        = generate_channel(4);
        s_hat    = mrrc(s, h, snr);
        [~, ber] = biterr(bitstream, pskdemod(s_hat, M));

        BER(i, MRRC_4RX, j) = ber;

        % alamouti (1 rx)
        h        = generate_channel(2);
        h        = h(:, 1:end / 2);
        s_hat    = alamouti(s, h, snr);
        [~, ber] = biterr(bitstream, pskdemod(s_hat, M));

        BER(i, ALAMOUTI_1RX, j) = ber;

        % alamouti (2 rx)
        h_0      = generate_channel(2);
        h_1      = generate_channel(2);
        h_0      = h_0(:, 1:end / 2);
        h_1      = h_1(:, 1:end / 2);
        s_hat    = alamouti(s, h_0, snr) + alamouti(s, h_1, snr);
        [~, ber] = biterr(bitstream, pskdemod(s_hat, M));

        BER(i, ALAMOUTI_2RX, j) = ber;
    end
end

BER = reshape(mean(BER), TECHNIQUES, numel(SNR));

%% Results
semilogy(SNR, BER);
title('Figure 4');
xlabel('SNR [dB]');
ylabel('Bit Error Rate');
legend({
        'No Diversity (1 Tx, 1 Rx)'
        'MRRC (1 Tx, 2 Rx)'
        'MRRC (1 Tx, 4 Rx)'
        'Alamouti (2 Tx, 1 Rx)'
        'Alamouti (2 Tx, 2 Rx)'
       });

%% Helpers
function h = generate_channel(channels)
    global N
    global PATHS
    global CHANNEL_GAIN
    global F_M
    global F_M_VARIATION

    r = zeros(channels, PATHS, N);

    for i = 1:channels
        for j = 1:PATHS
            f_m = F_M * unifrnd(1 - F_M_VARIATION, 1);
            r(i, j, :) = CHANNEL_GAIN * rayleigh_channel(f_m, N);
        end

        % introduce random path delays
        theta = unifrnd(0, 2 * pi, 1, PATHS, 1);
        r(i, :, :) = r(i, :, :) .* exp(1j * theta);
    end

    h = reshape(sum(r, 2), channels, N);
end
