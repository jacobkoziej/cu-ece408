% SPDX-License-Identifier: GPL-3.0-or-later
%
% ofdmmod.m -- modulate with OFDM
% Copyright (C) 2025  Jacob Koziej <jacobkoziej@gmail.com>

function Y = ofdmmod(X, N, CP)
    symbols  = size(X, 1);
    channels = size(X, 2);

    channel_frames = ceil(log(symbols) / log(N));

    frames = channels * channel_frames;

    Y = zeros(N, frames);

    X = [X; zeros((channel_frames * N) - symbols, channels)];
    X = reshape(X, N, frames);

    for i = 1:frames
        Y(:, i) = ifft(ifftshift(X(:, i)));
    end

    Y = [Y(end - (CP - 1):end, :); Y];
    Y = reshape(Y, (N + CP) * channel_frames, channels);
end
