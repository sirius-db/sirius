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
#include "ducklake/unit_descriptor.hpp"
#include "io/io_context.hpp"
#include "op/scan/puffin_reader.hpp"
#include "scan_manager/preparation_stop_token.hpp"

namespace sirius::ducklake {
// Per-unit, local, no-snapshot DV profile. These bounds do not come from catalog counts.
inline constexpr uint64_t dv_max_file_bytes = 64ULL * 1024 * 1024;
inline constexpr size_t dv_max_positions    = 1024 * 1024;
inline constexpr size_t dv_max_footer_bytes = 1024 * 1024;
inline constexpr size_t dv_max_blobs        = 64;
bool parquet_unit_dependencies_supported(unit_descriptor const&);
void validate_parquet_unit_dependencies(unit_descriptor const&);
scan_manager::unit_envelope parquet_unit_envelope(unit_descriptor const&);
struct dv_visibility final : op::scan::prepared_visibility {
  std::string data_identity, delete_identity;
  size_t validated_blobs = 0;
  bool parquet_validated = false;
  std::shared_ptr<dv_visibility const> predecessor;
  uint64_t inline_count = 0;
  std::span<int64_t const> positions;
  std::shared_ptr<op::scan::decoded_delete_positions const> decoded;
  scan_manager::charged_block merged;
};
void validate_delete_position(int64_t position, int64_t& previous);
std::shared_ptr<dv_visibility const> bind_parquet_visibility(std::shared_ptr<dv_visibility const>,
                                                             std::string data_identity,
                                                             int64_t file_rows);
std::shared_ptr<dv_visibility const> combine_inline_visibility(std::shared_ptr<dv_visibility const>,
                                                               std::span<uint64_t const>,
                                                               int64_t file_rows,
                                                               scan_manager::charging_allocator&);
// Structural/source seam for adapter tests; it cannot publish a scan certificate.
std::shared_ptr<dv_visibility const> prepare_dv_visibility(unit_descriptor const&,
                                                           int64_t file_rows,
                                                           std::string data_identity,
                                                           op::scan::puffin_byte_source&,
                                                           scan_manager::charging_allocator&,
                                                           scan_manager::preparation_stop_token,
                                                           std::string delete_identity = {});
std::shared_ptr<dv_visibility const> read_dv_visibility(unit_descriptor const&,
                                                        int64_t file_rows,
                                                        std::string data_identity,
                                                        io::ioctx_resolver const&,
                                                        scan_manager::charging_allocator&,
                                                        scan_manager::preparation_stop_token);
}  // namespace sirius::ducklake
