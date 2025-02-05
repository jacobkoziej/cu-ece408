% SPDX-License-Identifier: GPL-3.0-or-later
%
% Main.m -- project 1
% Copyright (C) 2025  Jacob Koziej <jacobkoziej@gmail.com>

clc;
clear;
close all;

rng(0x2b708595);

%% parameters
ITERATIONS = 16;
SNRS       = 128;
SYMBOLS    = 1024;
SPS        = 1;

snrs = linspace(0, 16, SNRS);

%% part 1a
bers = zeros(ITERATIONS, SNRS);
sers = zeros(ITERATIONS, SNRS);

bers_theoretical = zeros(1, SNRS);
sers_theoretical = zeros(1, SNRS);

config.channel = 1;
config.m       = 16;
config.symbols = SYMBOLS;

k = nextpow2(config.m);

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

%% part 1b
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

display('part 1b');
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
