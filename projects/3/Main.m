% SPDX-License-Identifier: GPL-3.0-or-later
%
% Main.m -- project 3
% Copyright (C) 2025  Jacob Koziej <jacobkoziej@gmail.com>

clc;
clear;
close all;

% signal parameters
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

%%% Apply Root-Raised-Cosine Filter at Receiver
% The transmitter applies a root-raised-cosine (RRC) filter to minimise
% inter-symbol interference (ISI) since it satisfies the Nyquist ISI
% criterion. Since this filter is root-raised, we need to apply the same
% filter on the receiver to complete the matched filter.
%
% Since the received filter is oversampled by a factor of 4x, we need to
% downsample our signal to operate at the correct chip rate. Another
% thing we must consider is the filter ramp-up time. We need to remove
% this to avoid misaligning our frames and decoding garbage. We can
% determine this by applying the floor function to the length of our
% RRC filter by the downsample factor and dropping that amount of
% samples from the start of our receiver buffer following decimation.

%%
load Rcvd_Koziej.mat;

r = upfirdn(Rcvd, B_RCOS, 1, OVERSAMPLE);

ramp_up_cutoff = ceil(numel(B_RCOS) / OVERSAMPLE);

%%
figure;
stem(0:15, abs(r(1:16)));
xline(ramp_up_cutoff - 1, '--', {'Filter Ramp-up', 'Cutoff'});
title('RRC Filtered Signal');
xlabel('Sample [n]');
ylabel('Magnitude');
ylim([0, max(abs(r(1:16))) + 0.25]);

%%

% remove filter ramp-up
r = r(ramp_up_cutoff:end);

% construct frames
r = reshape(r, FRAME_CHIPS, []);

%%
figure;
stem(0:FRAME_CHIPS - 1, abs(r));
xline(length(DATA_CHIPS), '--', 'Data Chips Cutoff');
title('Superimposed Frames');
xlabel('Sample [n]');
ylabel('Magnitude');

%%%
% After constructing each of our frames, we can superimpose them to make
% sure that we haven't made a mistake. In our superimposed plot, we can
% clearly see the CDMA spread spectrum due to our 8-ary Hadamard
% spreading codes along with a clear distinction of where the chip data
% ends. If our frames were misaligned, we wouldn't see the clear cutoff
% of where the data in each frame ends.

%%% Invert PN Sequence
% Next order of business is inverting the pseudo-noise (PN) sequence
% applied to our signal. A PN sequence gets applied to our signal to
% make more secure as the generator polynomial needs to be known at both
% the transmitter and receiver, but it also prevents spectral smearing
% from a continuous stream of one symbol. In our case, our generator
% polynomial is $g = 123_8$, generating an M-sequence of 255. This works
% out perfectly as our frame size is 255 chips, allowing us to
% correlated with all 255 different PN sequences to determine the
% initial state of the linear-feedback shift register at the
% transmitter. Once the initial state is found, every frame needs to be
% multiplied by the PN sequence to invert it.

%%
pn_seqs = pskmod(pn_sequences(PN_TAPS), 2);

%%
figure;
imagesc(pskdemod(pn_seqs, 2));
title('PN Sequences');
xlabel('Output [n]');
ylabel('Initial State [s]');

%%
pilot = sub2ind(size(r), 1:size(r, 1), 1);

pn_correlations = pn_seqs .* r(pilot).';
pn_correlations = abs(sum(pn_correlations));
[~, initial_state] = max(pn_correlations);

%%
figure;
stem(pn_correlations);
text(initial_state + 2, pn_correlations(initial_state) + 2, 'Best State');
xlim([1, length(pn_correlations)]);
title('Absolute Sum of PN Correlations');
xlabel('Initial State [s]');

%%
pn_sequence = pn_seqs(:, initial_state);

r = r .* pn_sequence;

%%% Invert Walsh Channel Orthogonal Spreading
% CDMA operates over a spread spectrum constructed with a Hadamard code,
% also known as a Walsh code. We can construct an 8-th order Hadamard
% matrix to create an orthogonal basis of codes (channels) which we can
% utilize bi-directionally to encode and decode data. Once we decode the
% channels, it is easy to see which channels actually contain signals as
% their magnitude will be significantly higher than channels without
% data.

%%
W = hadamard(WALSH_CHANNELS);

%%
figure;
imagesc(W);
title('8th Order Hadamard Matrix');
xlabel('Sample [n]');
ylabel('Channel [c]');

%%
% select only data chips
data = r(DATA_CHIPS, DATA_FRAME_START:end - DATA_FRAME_END);
data = reshape(data, WALSH_CHANNELS, []);

% decode walsh channels
decoded = W * data;

%%
figure;
t = tiledlayout(1, 2);

nexttile;
imagesc(abs(data));
title('Encoded Walsh Channels');

nexttile;
imagesc(abs(decoded));
title('Decoded Walsh Channels');

xlabel(t, 'Symbol Magnitude [n]');
ylabel(t, 'Channel [c]');

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
cfo_samples = r(pilot);
cfo_samples = cfo_samples(end - (CFO_SAMPLES - 1):end);

carrier_frequency_offset = conj(cfo_samples(1:end - 1)) .* cfo_samples(2:end);
carrier_frequency_offset = mean(angle(carrier_frequency_offset));

display(carrier_frequency_offset);
