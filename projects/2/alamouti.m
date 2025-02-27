% SPDX-License-Identifier: GPL-3.0-or-later
%
% alamouti.m -- alamouti diversity technique
% Copyright (C) 2025  Jacob Koziej <jacobkoziej@gmail.com>

function s_hat = alamouti(s, h, snr)
    % we know that
    %
    % r_0 = h_1 * s_1       + h_0 * s_0       + n_0
    % r_1 = h_1 * conj(s_1) - h_0 * conj(s_1) + n_1
    %
    % s_0 = conj(h_0) * r_0 + h_1 * conj(r_1)
    % s_1 = conj(h_1) * r_0 - h_0 * conj(r_1)
    %
    % taking note that the channel is constant for two symbol lengths,
    % we modify our s, h, and r matrices to take advantage of the
    % alternating patterns

    s_0       = s;
    s_0       = reshape(s_0, [], numel(s_0) / 2);
    s_0(2, :) = -conj(s_0(2, :));

    s_1       = s;
    s_1       = reshape(s_1, [], numel(s_1) / 2);
    s_1       = flipud(s_1);
    s_1(2, :) = conj(s_1(2, :));

    s_0 = s_0(1:end);
    s_1 = s_1(1:end);

    % s = [s_0, -conj(s_1), s_2, -conj(s_3), ...]
    %     [s_1, +conj(s_0), s_3, +conj(s_2), ...]
    s = [s_0; s_1];

    % h = [h_0, h_0, h_2, h_2, ...]
    %     [h_1, h_1, h_3, h_3, ...]
    h             = upsample(h.', 2).';
    h(:, 2:2:end) = h(:, 1:2:end - 1);

    r = awgn(sum(s .* h), snr);

    % r = [r_0,       r_0,       r_2,       r_2,       ...]
    %     [conj(r_1), conj(r_1), conj(r_3), conj(r_3), ...]
    r             = reshape(r, [], numel(r) / 2);
    r             = upsample(r.', 2).';
    r(:, 2:2:end) = r(:, 1:2:end - 1);
    r(2, :)       = conj(r(2, :));

    % h = [conj(h_0), +conj(h_1), conj(h_2), +conj(h_3), ...]
    % h = [h_1,       -h_0,       h_3,       -h_2),      ...]
    h(:, 2:2:end) = flipud(h(:, 2:2:end));
    h(1, :)       = conj(h(1, :));
    h(4:4:end)    = -h(4:4:end);

    s_hat = sum(h .* r);
end
