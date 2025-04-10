% SPDX-License-Identifier: GPL-3.0-or-later
%
% carrier_frequency_offset.m -- calculate carrier frequency offset
% Copyright (C) 2025  Jacob Koziej <jacobkoziej@gmail.com>

function phi = carrier_frequency_offset(s)
    phi = angle(conj(s(1:end - 1)) .* s(2:end));
    phi = mean(phi);
end
