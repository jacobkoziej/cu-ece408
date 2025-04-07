% SPDX-License-Identifier: GPL-3.0-or-later
%
% Main.m -- project 3
% Copyright (C) 2025  Jacob Koziej <jacobkoziej@gmail.com>

clc;
clear;
close all;

%% Signal Parameters
OVERSAMPLE = 4;
CHIP_RATE = 1e6;
B_RCOS = [
          +0.0038
          +0.0052
          -0.0044
          -0.0121
          -0.0023
          +0.0143
          +0.0044
          -0.0385
          -0.0563
          +0.0363
          +0.2554
          +0.4968
          +0.6025
          +0.4968
          +0.2554
          +0.0363
          -0.0563
          -0.0385
          +0.0044
          +0.0143
          -0.0023
          -0.0121
          -0.0044
          +0.0052
          +0.0038
         ]';
