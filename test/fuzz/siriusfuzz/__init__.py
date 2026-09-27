# Copyright 2026, Sirius Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# See the LICENSE file at the repo root for the full text.
"""Generative differential fuzzer for Sirius.

Generates random schemas and typed SQL queries over the GPU-supported surface,
runs each query on the GPU and on DuckDB CPU in the same process, and reports
result mismatches, GPU errors, plan-time coverage gaps, hangs and crashes.
See test/fuzz/README.md.
"""

__version__ = "0.1.0"
