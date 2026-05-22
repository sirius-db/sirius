/*
 * Copyright 2026, Sirius Contributors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include "scan_manager/duckdb_native_split_provider.hpp"

#include <cudf/table/table.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <cucascade/memory/memory_space.hpp>

#include <memory>

namespace sirius::op::scan {

struct duckdb_native_scan_info;

/// Resolve a GPU memory space to allocate the decoded table into. Looks up
/// the sirius_state on the scan_info's ClientContext and picks the first
/// GPU-tier space from the memory manager.
cucascade::memory::memory_space* pick_gpu_memory_space_for_duckdb_native_scan(
  duckdb_native_scan_info const& scan_info);

/// Decode a single split (a contiguous run of row-group metadata produced by
/// the walker) into a cudf::table.
///
/// V1 spike scope: fixed-width data codecs UNCOMPRESSED / RLE / BITPACKING,
/// rowid synthesis, UNCOMPRESSED / EMPTY / CONSTANT validity. Throws
/// std::runtime_error on VARCHAR columns, CONSTANT data segments, ROARING
/// validity, or any other codec the walker accepted but the spike does not
/// yet handle. Those are deliberate V2 follow-ups.
std::unique_ptr<cudf::table> decode_duckdb_native_split(
  scan_manager::duckdb_native_split_provider::split_payload const& split,
  cucascade::memory::memory_space& mem_space,
  rmm::cuda_stream_view stream);

}  // namespace sirius::op::scan
