% SPDX-License-Identifier: GPL-3.0-or-later
%
% part1b.m -- part 1b
% Copyright (C) 2025  Jacob Koziej <jacobkoziej@gmail.com>

function [ber, ser] = part1b(config, enable_lms)
    M = config.m;

    eqlms = comm.LinearEqualizer( ...
                                 'AdaptAfterTraining', false, ...
                                 'Algorithm', 'LMS', ...
                                 'Constellation', pskmod(0:(M - 1), M), ...
                                 'NumTaps', config.tap_weights, ...
                                 'ReferenceTap', config.reference_tap, ...
                                 'StepSize', config.learning_rate ...
                                );

    s = randi(M, 1, config.symbols) - 1;

    train_flag = false(size(s));
    train_flag(2:config.training_symbols) = true;

    tx = pskmod(s, M);
    rx = filter(config.channel, 1, tx);
    rx = awgn(rx, config.snr, 'measured');

    if enable_lms
        [r, err, h] = eqlms(rx.', tx(1:config.training_symbols).');
    else
        r = rx.';
    end

    r = pskdemod(r.', M);

    s = s(~train_flag);
    r = r(~train_flag);

    [~, ber] = biterr(s, r);
    [~, ser] = symerr(s, r);
end
