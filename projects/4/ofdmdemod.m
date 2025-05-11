% SPDX-License-Identifier: GPL-3.0-or-later
%
% ofdmmod.m -- demodulate with OFDM
% Copyright (C) 2025  Jacob Koziej <jacobkoziej@gmail.com>

function Y = ofdmdemod(X, N, CP, H)
    channels = size(X, 2);

    X = reshape(X, N + CP, []);
    X = X(CP + 1:end, :);

    frames = size(X, 2);

    Y = zeros(N, frames);

    for i = 1:frames
        Y(:, i) = fftshift(fft(X(:, i))) ./ H;
    end

    Y = reshape(Y, [], channels);
end
