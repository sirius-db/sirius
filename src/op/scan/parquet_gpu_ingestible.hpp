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
#include "op/scan/parquet_scan_core.hpp"

#include <typeinfo>
namespace sirius::op::scan {
class parquet_gpu_ingestible : public gpu_ingestible {
 public:
  explicit parquet_gpu_ingestible(
    std::unique_ptr<parquet_ingestible_table_info> info,
    parquet_source_kind source_kind = parquet_source_kind::PLAIN_PARQUET)
    : _core(std::move(info), source_kind)
  {
  }
  [[nodiscard]] std::size_t reader_calls() const noexcept { return _core.reader_calls(); }
  void set_execution_completion(std::shared_ptr<pipeline::completion_handler> c) override
  {
    gpu_ingestible::set_execution_completion(c);
    _core.set_execution_completion(std::move(c));
  }
  std::unique_ptr<batch_coalescer> create_batch_coalescer() const override
  {
    return _core.create_batch_coalescer();
  }
  bool has_processed_all_metadata() const override { return _core.has_processed_all_metadata(); }
  [[nodiscard]] bool source_selection_empty() const noexcept
  {
    return _core.source_selection_empty();
  }
  const ingestible_table_info& table_info() const noexcept override { return _core.table_info(); }
  std::vector<std::size_t> materialized_column_order() const override
  {
    return _core.materialized_column_order();
  }
  bool output_assembly_is_leading_identity() const noexcept override
  {
    return _core.output_assembly_is_leading_identity();
  }
  std::optional<std::size_t> output_prefix_width() const noexcept override
  {
    return _core.output_prefix_width();
  }
  [[nodiscard]] bool can_serve_cached_view() const noexcept override
  {
    // Other adapters need their own proof for visibility and source positions.
    return typeid(*this) == typeid(parquet_gpu_ingestible) && _core.can_serve_cached_view();
  }
  [[nodiscard]] std::optional<cached_view_layout> view_layout(
    filtered_table const& input, std::span<std::size_t const> elided) const override
  {
    if (!can_serve_cached_view()) return std::nullopt;
    return _core.view_layout(input, elided);
  }
  bool has_row_filter() const noexcept override { return _core.has_row_filter(); }
  bool can_report_survivors() const noexcept override { return true; }
  scan_filter_analysis const& filter_analysis() const override { return _core.filter_analysis(); }
  std::unique_ptr<cudf::table> post_filter_and_project(
    filtered_table&& input,
    const cucascade::memory::memory_space& space,
    ::cuda::stream_ref stream,
    bool like                                                    = false,
    std::shared_ptr<const sirius::like_multiliteral_cache> cache = nullptr,
    std::unique_ptr<cudf::column>* survivors                     = nullptr,
    std::span<std::size_t const> elided                          = {},
    expression_query_policy policy                               = {}) override
  {
    return _core.post_filter_and_project(
      std::move(input), space, stream, like, std::move(cache), survivors, elided, policy);
  }
  metadata_scan_task_t next_split_provider(io::ioctx_resolver resolve) override
  {
    return _core.next_split_provider(std::move(resolve));
  }
  stoppable_metadata_scan_task_t next_split_provider_with_stop(io::ioctx_resolver resolve) override
  {
    return _core.next_split_provider_with_stop(std::move(resolve));
  }
  filtered_table materialize_metadata_to_table(
    scan_info const& info,
    const cucascade::memory::memory_space& space,
    ::cuda::stream_ref stream,
    bool like,
    std::shared_ptr<const sirius::like_multiliteral_cache> cache,
    expression_query_policy policy = {}) override
  {
    return _core.materialize_metadata_to_table(info, space, stream, like, std::move(cache), policy);
  }

 private:
  parquet_scan_core _core;
};
std::shared_ptr<parquet_gpu_ingestible> make_ingestible(
  std::unique_ptr<parquet_ingestible_table_info> info);
}  // namespace sirius::op::scan
