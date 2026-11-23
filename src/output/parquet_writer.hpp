// Copyright 2026, Sirius Contributors. SPDX-License-Identifier: Apache-2.0
#pragma once
#include "output/parquet_spec.hpp"

#include <cudf/table/table_view.hpp>

#include <cuda/stream_ref>

namespace sirius::output {
bool supported_parquet_compression(parquet_compression codec);
void write_parquet_part(parquet_export_spec const& spec,
                        cudf::table_view table,
                        std::filesystem::path const& path,
                        ::cuda::stream_ref stream);
// The caller selects a configured execution device; the writer restores the
// caller's device after all schema-only allocations and encoding have retired.
void write_empty_parquet(parquet_export_spec const& spec,
                         std::filesystem::path const& path,
                         int device_id);
}  // namespace sirius::output
