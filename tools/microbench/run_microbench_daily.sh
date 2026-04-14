#!/usr/bin/env bash
# Copyright 2025, Sirius Contributors.
# SPDX-License-Identifier: Apache-2.0
#
# Convenience wrapper: run the "daily" microbench profile.
# See run_microbench_sweep.sh for options and environment variables.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$SCRIPT_DIR/run_microbench_sweep.sh" daily
