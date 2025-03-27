# SPDX-License-Identifier: GPL-3.0-or-later
#
# Copyright (C) 2025  Jacob Koziej <jacobkoziej@gmail.com>

import os

from SCons.Environment import Environment
from SCons.Script import (
    EnsurePythonVersion,
    EnsureSConsVersion,
    SConscript,
)


EnsureSConsVersion(4, 7, 0)
EnsurePythonVersion(3, 12)


env = Environment(
    ENV={
        "PATH": os.environ["PATH"],
        "TERM": os.environ.get("TERM"),
    },
    tools=[
        "default",
        "github.jacobkoziej.scons-tools.Jupyter.NbConvert",
        "github.jacobkoziej.scons-tools.Jupytext",
        "github.jacobkoziej.scons-tools.Papermill",
    ],
)
env.AppendUnique(
    NBCONVERTFLAGS=[
        "--TagRemovePreprocessor.enabled=True",
        "--TagRemovePreprocessor.remove_cell_tags=\"{'parameters'}\"",
        "--to=html",
    ],
)

build = "build"


projects = SConscript(
    "projects/SConscript.py",
    exports=[
        "env",
    ],
    variant_dir=f"{build}/projects",
)
