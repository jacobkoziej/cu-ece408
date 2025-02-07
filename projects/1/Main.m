% SPDX-License-Identifier: GPL-3.0-or-later
%
% Main.m -- project 1
% Copyright (C) 2025  Jacob Koziej <jacobkoziej@gmail.com>

clc;
clear;
close all;

rng(0x2b708595);

%% Parameters
ITERATIONS = 1024;
SNRS       = 64;
SYMBOLS    = 1000;
SPS        = 1;

%% Part 1.a (Theoretical 16-QAM)
snrs = linspace(0, 16, SNRS);

bers = zeros(ITERATIONS, SNRS);
sers = zeros(ITERATIONS, SNRS);

bers_theoretical = zeros(1, SNRS);
sers_theoretical = zeros(1, SNRS);

config.channel = 1;
config.m       = 16;
config.symbols = SYMBOLS;

k = nextpow2(config.m);

%%% Contents of |part1a.m|
% <include>part1a.m</include>

%%% Simulation Loop
for i = 1:SNRS
    config.snr = snrs(i);

    ebno = config.snr + 10 * log10(SPS / k);
    [ber_theoretical, ser_theoretical] = berawgn(ebno, 'qam', config.m);

    bers_theoretical(i) = ber_theoretical;
    sers_theoretical(i) = ser_theoretical;

    for j = 1:ITERATIONS
        [ber, ser] = part1a(config);

        bers(j, i) = ber;
        sers(j, i) = ser;
    end
end

bers = mean(bers);
sers = mean(sers);

%%% Results
figure;
sgtitle('Part 1.a');

subplot(1, 2, 1);
hold on;
plot(snrs, bers_theoretical);
plot(snrs, bers);
ylabel('BER');
xlabel('SNR');
legend('Theoretical', 'Simulated');

subplot(1, 2, 2);
hold on;
plot(snrs, sers_theoretical);
plot(snrs, sers);
ylabel('SER');
xlabel('SNR');
legend('Theoretical', 'Simulated');

%% Part 1.b (Equalized BPSK Over Moderate ISI Channel)
config.channel          = [1.0, 0.2, 0.4];
config.m                = 2;
config.reference_tap    = 1;
config.snr              = 12;
config.tap_weights      = 3;
config.training_symbols = 50;

k = nextpow2(config.m);
ebno = config.snr + 10 * log10(SPS / k);
[ber_theoretical, ser_theoretical] = berawgn(ebno, 'psk', config.m, 'nondiff');

bers = zeros(1, ITERATIONS);
sers = zeros(1, ITERATIONS);
bers_rls = zeros(1, ITERATIONS);
sers_rls = zeros(1, ITERATIONS);
errs = zeros(ITERATIONS, config.symbols);
tap_weights = zeros(ITERATIONS, config.tap_weights);

%%% Equalization Approach
% To equalize the Inter-Symbol Interference (ISI) channel, I've opted to
% apply the Recursive Least Squares (RLS). Initially, I went with Least
% Mean Squares (LMS) algorithm without much success. I started with a
% learning rate of 1e-2, 5 tap weights, a center reference tap, and 64
% training symbols. Unsurprisingly, this approach did not work. To
% resolve this, I changed the number of taps to match that of the
% channel and set the reference tap to be the first element. This
% yielded better results, but looking at the learning curve and
% generated tap weights, it was obvious that the amount of training
% symbols was not sufficient. After cranking up the training symbols to
% 256, I finally observed a BER below 1e-4. To further improve my
% equalization, I took advantage of RLS's much faster convergence since
% it approximates the signal's auto-correlation matrix as opposed to
% relying on gradient descent. It turns out that using a conservative
% forgetting factor of |0.99| yielded great performance with as few as
% 50 training symbols!

%%% Contents of |part1b.m|
% <include>part1b.m</include>

%%% Simulation Loop
for i = 1:ITERATIONS
    [ber, ser] = part1b(config, false);

    bers(i) = ber;
    sers(i) = ser;

    [ber, ser, err, h] = part1b(config, true);

    bers_rls(i) = ber;
    sers_rls(i) = ser;
    errs(i, :) = err;
    tap_weights(i, :) = h;
end

snr = config.snr;
channel = config.channel;

ber = mean(bers);
ser = mean(sers);

ber_rls = mean(bers_rls);
ser_rls = mean(sers_rls);
errs = mean(errs);
tap_weights = mean(tap_weights);

%%% Results
display('Part 1b');
display(snr);
display(ber_theoretical);
display(ser_theoretical);
display(ber);
display(ser);
display(channel);
display(tap_weights);
display(ber_rls);
display(ser_rls);

figure;
plot(abs(err));
title('Part 1.b');
xlabel('Symbol');
ylabel('|Reference Tap Error|');

%% Part 2 (Error-Corrected Equalized BPSK Over Moderate ISI Channel)
config.training_symbols = 16;
config.tap_weights = 3;
config.reference_tap = 1;
config.snr = 12;

bers = zeros(1, ITERATIONS);
sers = zeros(1, ITERATIONS);
errs = zeros(ITERATIONS, config.symbols);
tap_weights = zeros(ITERATIONS, config.tap_weights);

%%% Contents of |part2.m|
% <include>part2.m</include>

%%% Error-Correction Approach
% To error correct the BPSK equalized channel, I've opted to apply
% Reed-Solomon codes to my symbol stream. I went with this approach
% mainly because I wanted to learn more about error correction codes.
% Reed-Solomon takes advantage of modulo-arithmetic over GF(N), making
% it especially fast to implement in hardware, and great for real-time
% applications like communication links. To implement this, I first
% calculated the number of bits I could transmit over a packet when
% taking account of the training sequence along with additional
% error-correction data. Once calculated, I generated a bitstream and
% encoded this stream with Reed-Solomon over GF(5). Then I padded the
% packet's training sequence with additional symbols that would not fit
% into the packet with the encoding scheme. On the receiver end I
% performed the inverse of the original process to retrieve the
% transmitted bits, achieving a near zero BER.

%%% Simulation Loop
for i = 1:ITERATIONS
    [ber, ser, err, h] = part2(config);

    bers(i) = ber;
    sers(i) = ser;
    errs(i, :) = err;
    tap_weights(i, :) = h;
end

snr = config.snr;
channel = config.channel;

ber = mean(bers);
ser = mean(sers);
err = mean(errs);
tap_weights = mean(tap_weights);

%%% Results
display('Part 2');
display(snr);
display(ber);
display(ser);
display(channel);
display(tap_weights);
figure;
plot(abs(err));
title('Part 2');
xlabel('Symbol');
ylabel('|Reference Tap Error|');
