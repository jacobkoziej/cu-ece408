% SPDX-License-Identifier: GPL-3.0-or-later
%
% pn_sequences.m -- generate pseudorandom noise sequences
% Copyright (C) 2025  Jacob Koziej <jacobkoziej@gmail.com>

function seqs = pn_sequences(polynomial)
    n = max(polynomial);

    states = 2^n - 1;

    seqs = zeros(states);

    for i = 1:states
        seq = comm.PNSequence( ...
                              'Polynomial', polynomial, ...
                              'InitialConditions', int2bit(i, n), ...
                              'SamplesPerFrame', states);

        seqs(:, i) = seq();
    end
end
