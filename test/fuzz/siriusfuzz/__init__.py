# Copyright 2026, Sirius Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# See the LICENSE file at the repo root for the full text.
"""Generative differential fuzzer for Sirius.

Generates random schemas and typed SQL queries, runs each query on DuckDB CPU
and on Sirius through the DuckDB shell built with the extension, and reports
result mismatches, GPU errors, plan-time and runtime fallbacks, hangs and
crashes. See test/fuzz/README.md.
"""

__version__ = "0.1.0"
