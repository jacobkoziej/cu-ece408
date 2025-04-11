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
PN_TAPS = [8, 7, 6, 1, 0];

FRAME_CHIPS = 255;
DATA_CHIPS  = 1:(64 * 3);

DATA_FRAME_START = 2;
DATA_FRAME_END   = 1;

WALSH_CHANNELS = 8;
PILOT_CHANNEL  = 1;
DATA_CHANNEL   = 6;

CFO_SAMPLES = 128;

%% Apply Root Raised Cosine Filter at Receiver
load Rcvd_Koziej.mat;

r = upfirdn(Rcvd, B_RCOS, 1, OVERSAMPLE);

% remove filter ramp-up
r = r(ceil(numel(B_RCOS) / OVERSAMPLE):end);

r = reshape(r, FRAME_CHIPS, []);

%% Invert PN Sequence
pn_seqs = pskmod(pn_sequences(PN_TAPS), 2);

pilot = sub2ind(size(r), 1:size(r, 1), 1);

pn_correlations = pn_seqs .* r(pilot).';
[~, initial_condition] = max(abs(sum(pn_correlations)));

pn_sequence = pn_seqs(:, initial_condition);

r = r .* pn_sequence;

%% Invert Walsh Channel Orthogonal Spreading
W = hadamard(WALSH_CHANNELS);

data = r(DATA_CHIPS, DATA_FRAME_START:end - DATA_FRAME_END);
data = reshape(data, WALSH_CHANNELS, []);

decoded = W * data;

%% Equalize Channels with Pilot
decoded = decoded ./ decoded(PILOT_CHANNEL, :);

%% Decode Little Endian Message
msg = decoded(DATA_CHANNEL, :);
msg = reshape(msg, 8, []);
msg = pskdemod(msg, 2);
msg = bit2int(msg, 8, false);
msg = char(msg);

display(msg);

%% Determine Carrier Frequency Offset
get_cfo_samples = @(x) x(end - (CFO_SAMPLES - 1):end);

fine_phi = carrier_frequency_offset(get_cfo_samples(r(pilot)));

sample_index = reshape(1:numel(r), size(r));

fine_cfo = exp(-1j .* fine_phi .* sample_index);

r = r .* fine_cfo;

coarse_phi = mean(angle(get_cfo_samples(r(pilot))));

carrier_frequency_offset = coarse_phi + fine_phi;

display(carrier_frequency_offset);
