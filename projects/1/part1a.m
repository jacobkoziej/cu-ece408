% SPDX-License-Identifier: GPL-3.0-or-later
%
% part1a.m -- part 1a
% Copyright (C) 2025  Jacob Koziej <jacobkoziej@gmail.com>

function [ber, ser] = part1a(config)
    M = config.m;

    s = randi(M, 1, config.symbols) - 1;

    tx = qammod(s, M);
    rx = filter(config.channel, 1, tx);
    rx = awgn(rx, config.snr, 'measured');

    r = qamdemod(rx, M);

    [~, ber] = biterr(s, r);
    [~, ser] = symerr(s, r);
end
