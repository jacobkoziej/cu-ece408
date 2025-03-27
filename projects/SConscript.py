# SPDX-License-Identifier: GPL-3.0-or-later
#
# Copyright (C) 2024  Jacob Koziej <jacobkoziej@gmail.com>

# ruff: noqa: F821

from SCons.Script import (
    Import,
    Return,
    SConscript,
)

Import("env")

projects = []

projects += SConscript(
    "midterm/SConscript.py",
    exports=[
        "env",
    ],
)

Return("projects")
