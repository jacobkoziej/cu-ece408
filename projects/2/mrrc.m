% SPDX-License-Identifier: GPL-3.0-or-later
%
% mrrc.m -- maximal-ratio receive combining scheme
% Copyright (C) 2025  Jacob Koziej <jacobkoziej@gmail.com>

function s_hat = mrrc(s, h, n)
    r = (s .* h) + n;
    s_hat = sum(r .* conj(h));
end
