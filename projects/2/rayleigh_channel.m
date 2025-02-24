% SPDX-License-Identifier: GPL-3.0-or-later
%
% rayleigh_channel.m -- baseband rayleigh channel simulator (Smith's algorithm)
% Copyright (C) 2025  Jacob Koziej <jacobkoziej@gmail.com>

function r = rayleigh_channel(f_m, n)
    points = floor(n / 2);

    g_i = gen_noise(points);
    g_q = gen_noise(points);

    sqrt_spectrum = sqrt(fading_spectrum(f_m, n));

    r_i = ifft(fftshift(g_i .* sqrt_spectrum), 'symmetric').^2;
    r_q = ifft(fftshift(g_q .* sqrt_spectrum), 'symmetric').^2;

    r = sqrt(r_i.^2 + r_q.^2);
end

function g = gen_noise(n)
    g = rand(1, n) + 1j * rand(1, n);
    g = [fliplr(conj(g)), g];
end

function spectrum = fading_spectrum(f_m, n)
    f_m_edge = f_m - (f_m / n);

    f = linspace(-f_m_edge, f_m_edge, n);

    spectrum = 1.5 ./ (pi .* f_m .* sqrt(1 - (f ./ f_m).^2));
end
