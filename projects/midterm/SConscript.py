# SPDX-License-Identifier: GPL-3.0-or-later
#
# Copyright (C) 2024  Jacob Koziej <jacobkoziej@gmail.com>

# ruff: noqa: F821

from pathlib import Path

from SCons.Script import (
    Import,
    Return,
)

Import("env")

notebook_raw = env.Jupytext("report-raw.ipynb", "report.py")[0]

notebook = env.Papermill("report.ipynb", notebook_raw)[0]

report = env.NbConvert(str(Path(str(notebook)).with_suffix(".html")), notebook)

Return("report")
