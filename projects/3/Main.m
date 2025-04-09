% SPDX-License-Identifier: GPL-3.0-or-later
%
% Main.m -- project 3
% Copyright (C) 2025  Jacob Koziej <jacobkoziej@gmail.com>

clc;
clear;
close all;

%% Signal Parameters
OVERSAMPLE = 4;
CHIP_RATE = 1e6;
B_RCOS = [
          +0.0038
          +0.0052
          -0.0044
          -0.0121
          -0.0023
          +0.0143
          +0.0044
          -0.0385
          -0.0563
          +0.0363
          +0.2554
          +0.4968
          +0.6025
          +0.4968
          +0.2554
          +0.0363
          -0.0563
          -0.0385
          +0.0044
          +0.0143
          -0.0023
          -0.0121
          -0.0044
          +0.0052
          +0.0038
         ]';
WALSH_CHANNELS = 8;
FRAME_CHIPS = 255;
PN_TAPS = [8, 7, 6, 1, 0];

%% Apply Root Raised Cosine Filter at Receiver
load Rcvd_Koziej.mat;

r = downsample(filter(B_RCOS, 1, Rcvd), OVERSAMPLE);
r = reshape(r, FRAME_CHIPS, []);

%% Invert PN Sequence
pn_seqs = pskmod(pn_sequences(PN_TAPS), 2);

pilot = r(:, 1);
pn_correlations = pn_seqs .* pilot;
[~, initial_condition] = max(abs(sum(pn_correlations)));

pn_sequence = pn_seqs(:, initial_condition);

r = r .* pn_sequence;
