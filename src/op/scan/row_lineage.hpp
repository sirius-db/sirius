/* Copyright 2026, Sirius Contributors. SPDX-License-Identifier: Apache-2.0 */
#pragma once

#include "op/scan/gpu_row_positions.hpp"
#include "op/scan/owning_table_view.hpp"
#include "op/scan/scan_execution_record.hpp"

#include <cudf/column/column.hpp>

#include <rmm/resource_ref.hpp>

#include <span>
#include <utility>

namespace sirius::op::scan {
struct scan_plan;
// File ordinals belong to this immutable namespace. Intervals describe the
// input before selection, including the original offsets of pruned row groups.
struct lineage_domain {
  struct file {
    std::string identity;
    int64_t rows;
    bool operator==(file const&) const = default;
  };
  struct file_run {
    std::size_t file;
    int64_t first;
    int64_t count;
    bool operator==(file_run const&) const = default;
  };
  std::vector<file> files;
  std::vector<file_run> file_runs;
  bool operator==(lineage_domain const&) const = default;
};

struct row_lineage {
  row_position_state state = row_position_state::none;
  std::shared_ptr<lineage_domain const> domain;
  // Selection runs are (input-domain ordinal, count), never file positions.
  std::vector<std::pair<int64_t, int64_t>> runs;
  std::shared_ptr<cudf::column const> per_row;  // INT64 input-domain ordinals

  [[nodiscard]] int64_t input_domain_length() const;
  [[nodiscard]] int64_t output_length() const;
  // Structural validation cannot inspect device values. validate_on also
  // checks bounds and duplicates, but cannot prove a decoder omitted no rows.
  void validate(int64_t output_rows) const;
  void validate_on(int64_t output_rows,
                   ::cuda::stream_ref,
                   rmm::device_async_resource_ref = cudf::get_current_device_resource_ref()) const;
  [[nodiscard]] std::pair<std::size_t, int64_t> resolve(int64_t ordinal) const;
  [[nodiscard]] std::vector<int64_t> ordinals_on(::cuda::stream_ref) const;
  [[nodiscard]] row_lineage select(std::span<int64_t const> delivered_ordinals,
                                   ::cuda::stream_ref,
                                   rmm::device_async_resource_ref) const;
  [[nodiscard]] row_lineage select(cudf::column_view delivered_ordinals,
                                   ::cuda::stream_ref,
                                   rmm::device_async_resource_ref) const;
  [[nodiscard]] std::unique_ptr<cudf::column> ordinal_column(::cuda::stream_ref,
                                                             rmm::device_async_resource_ref) const;
  // The step's input domain must describe exactly before's delivered sequence.
  static row_lineage compose(row_lineage const& before,
                             row_lineage const& step,
                             ::cuda::stream_ref,
                             rmm::device_async_resource_ref);
  // File namespaces are explicitly remapped by identity; selection offsets
  // advance by input-domain length, including rows removed by earlier filters.
  static row_lineage concatenate(std::span<row_lineage const>,
                                 ::cuda::stream_ref,
                                 rmm::device_async_resource_ref);
};
// Consume the reader's row buffer in place; source indices are released after
// conversion. Only a consumer of canonical ordinals should call this adapter.
row_lineage consume_reader_positions(std::unique_ptr<gpu_row_positions>,
                                     ::cuda::stream_ref,
                                     rmm::device_async_resource_ref);
// Additional temporary bytes above the existing reader row/source buffers.
std::size_t lineage_conversion_workspace_bytes(std::size_t rows, std::size_t intervals);
void append_lineage_virtual_columns(owning_table_view&,
                                    row_lineage const&,
                                    scan_plan const&,
                                    std::span<uint64_t const> file_indices,
                                    ::cuda::stream_ref,
                                    rmm::device_async_resource_ref);
void apply_lineage_visibility(owning_table_view&,
                              row_lineage&,
                              std::function<std::span<int64_t const>(std::string const&)> const&,
                              ::cuda::stream_ref,
                              rmm::device_async_resource_ref);
}  // namespace sirius::op::scan
