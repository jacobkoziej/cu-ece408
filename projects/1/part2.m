% SPDX-License-Identifier: GPL-3.0-or-later
%
% part2.m -- part 2
% Copyright (C) 2025  Jacob Koziej <jacobkoziej@gmail.com>

function [ber, ser, err, h, bitrate] = part2(config)
    M = 2;
    K = nextpow2(M);

    m = 5;
    n = 2^m - 1;
    k = n - 2;

    % calculate transmission bits, accounting
    % for training symbols and error-correction
    training_symbols = config.training_symbols;
    bits = (config.symbols - training_symbols) * K;

    symbols = floor(bits / (m * n)) * k;

    s = randi(2, 1, symbols * m) - 1;

    % generate symbol stream from bitstream
    tx = reshape(s, m, []);
    tx = bit2int(tx, m);

    % encode symbols with Reed-Solomon
    tx = reshape(tx, [], k);
    tx = gf(tx, m);
    tx = rsenc(tx, n, k);
    tx = double(tx.x);
    tx = tx(1:end);

    % encode Reed-Solomon as symbol stream
    tx = int2bit(tx, m);
    tx = reshape(tx, K, []);
    tx = bit2int(tx, K);

    training_symbols = training_symbols + ...
        (config.symbols - training_symbols - numel(tx));

    tx = [randi(M, 1, training_symbols) - 1, tx];

    tx = pskmod(tx, M);
    rx = filter(config.channel, 1, tx);
    rx = awgn(rx, config.snr, 'measured');

    eqrls = comm.LinearEqualizer( ...
                                 'Algorithm', 'RLS', ...
                                 'Constellation', pskmod(0:(M - 1), M), ...
                                 'ForgettingFactor', 0.99, ...
                                 'NumTaps', config.tap_weights, ...
                                 'ReferenceTap', config.reference_tap ...
                                );

    [r, err, h] = eqrls(rx.', tx(1:training_symbols).');

    r = pskdemod(r.', M);

    signal_symbols = true(size(r));
    signal_symbols(1:training_symbols) = false;

    % serialize symbols and decode Reed-Solomon stream
    r = r(signal_symbols);
    r = int2bit(r, K);

    r = reshape(r, m, []);
    r = bit2int(r, m);
    r = reshape(r, [], n);
    r = gf(r, m);

    [r, ~] = rsdec(r, n, k);

    r = double(r.x);
    r = r(1:end);
    r = int2bit(r, m);
    r = r(1:end);

    [~, ber] = biterr(s, r);
    [~, ser] = symerr(s, r);

    bitrate = (m * symbols) / config.symbols;
end
