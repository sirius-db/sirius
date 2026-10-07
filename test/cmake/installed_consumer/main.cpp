// Copyright 2026, Sirius Contributors. SPDX-License-Identifier: Apache-2.0
#include <sirius/duckdb.hpp>
#include <sirius/ffi.hpp>

// Keep symbol references without initializing a GPU during the link smoke test.
auto* volatile context_factory = &sirius::ffi::make_context;
auto* volatile registration    = &sirius::register_duckdb_extension;

int main() { return context_factory == nullptr || registration == nullptr; }
