#!/usr/bin/env bash
# Add conda prefix lib to runtime library path (for libnixl.so etc.)
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
