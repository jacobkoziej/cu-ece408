% SPDX-License-Identifier: GPL-3.0-or-later
%
% part1b.m -- part 1b
% Copyright (C) 2025  Jacob Koziej <jacobkoziej@gmail.com>

function [ber, ser] = part1b(config, enable_lms)
    M = config.m;

    eqrls = comm.LinearEqualizer( ...
                                 'Algorithm', 'RLS', ...
                                 'Constellation', pskmod(0:(M - 1), M), ...
                                 'NumTaps', config.tap_weights, ...
                                 'ReferenceTap', config.reference_tap ...
                                );

    s = randi(M, 1, config.symbols) - 1;

    tx = pskmod(s, M);
    rx = filter(config.channel, 1, tx);
    rx = awgn(rx, config.snr, 'measured');

    if enable_rls
        [r, err, h] = eqrls(rx.', tx(1:config.training_symbols).');
    else
        r = rx.';
    end

    r = pskdemod(r.', M);

    signal_symbols = true(size(s));
    signal_symbols(1:config.training_symbols) = false;

    s = s(signal_symbols);
    r = r(signal_symbols);

    [~, ber] = biterr(s, r);
    [~, ser] = symerr(s, r);
end
