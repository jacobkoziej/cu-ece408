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

report_raw = env.Jupytext("report-raw.ipynb", "report.py")[0]

report = env.Papermill("report.ipynb", report_raw)[0]
env.Depends(report, env.Glob("*.py"))

report = env.NbConvert(str(Path(str(report)).with_suffix(".html")), report)

Return("report")
