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

/**
 * @file resident_scan_test_utils.hpp
 * @brief Fixtures shared by the scan normalization tests in `test_scan_schema_normalization.cpp`
 * and the hidden resident-scan benchmark in `bench_fused_membership.cpp`: one process-wide memory
 * manager and stream, resident GPU source columns, a Parquet table description with an optional
 * residual filter, and the resident admission a pinned split carries.
 */

#include "operator/operator_test_utils.hpp"

#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/table/table.hpp>
#include <cudf/types.hpp>

#include <rmm/cuda_stream.hpp>

#include <cuda/stream>
#include <cuda_runtime.h>

#include <catch.hpp>
#include <cucascade/data/data_batch.hpp>
#include <cucascade/memory/memory_space.hpp>
#include <data/data_batch_utils.hpp>
#include <duckdb/planner/filter/constant_filter.hpp>
#include <op/scan/parquet_gpu_ingestible.hpp>
#include <op/scan/sirius_gpu_scan_operator.hpp>
#include <op/scan/sirius_gpu_scan_operator_data.hpp>
#include <utils/table_filter_test_utils.hpp>

#include <cstddef>
#include <memory>
#include <optional>
#include <vector>

namespace sirius::test::resident_scan {

struct test_env {
  std::unique_ptr<sirius::memory::sirius_memory_reservation_manager> mgr;
  cucascade::memory::memory_space* gpu_space;
  cucascade::memory::memory_space* host_space;
  rmm::cuda_stream conv_stream;

  test_env()
    : mgr(sirius::test::operator_utils::initialize_memory_manager()),
      gpu_space(mgr->get_memory_space(cucascade::memory::Tier::GPU, 0)),
      host_space(mgr->get_memory_space(cucascade::memory::Tier::HOST, 0)),
      conv_stream()
  {
  }

  ::cuda::stream_ref stream() { return conv_stream; }
};

inline test_env& env()
{
  sirius::test::operator_utils::ensure_converter_registry();
  static test_env e;
  return e;
}

inline void admit_fixture_resident_input(sirius::op::scan::scan_operator_input& input)
{
  sirius::op::scan::pin_validation validation;
  validation.identity = validation.layout = validation.structure = {true, true};
  input.resident_contract_id                                     = 1;
  input.resident_validation                                      = validation;
}

/// Three BIGINT columns (`key`, `payload`, `keep`) read through @p columns; @p positive_filter
/// names the projected column a `> 0` residual filter tests.
inline std::unique_ptr<sirius::op::scan::parquet_ingestible_table_info> resident_parquet_info(
  std::vector<std::size_t> const& columns,
  std::size_t output_arity,
  std::optional<std::size_t> positive_filter = std::nullopt)
{
  auto info   = std::make_unique<sirius::op::scan::parquet_ingestible_table_info>();
  info->names = {"key", "payload", "keep"};
  for (std::size_t i = 0; i < info->names.size(); ++i) {
    info->returned_types.push_back(sirius::logical_type::make(sirius::type_id::BIGINT));
  }
  for (auto column : columns) {
    info->column_ids.emplace_back(column);
    info->projection_ids.push_back(info->projection_ids.size());
  }
  info->projected_output_count = output_arity;
  if (positive_filter) {
    info->table_filters = duckdb::make_uniq<duckdb::TableFilterSet>();
    info->table_filters->PushFilter(
      duckdb::ProjectionIndex(*positive_filter),
      sirius::test::constant_filter(duckdb::ExpressionType::COMPARE_GREATERTHAN,
                                    duckdb::Value::BIGINT(0)));
  }
  return info;
}

/// A GPU column of @p values; @p null_row, when given, is the one null row.
template <typename T>
std::unique_ptr<cudf::column> resident_column(
  std::vector<T> const& values,
  cudf::type_id type,
  std::optional<cudf::size_type> null_row = std::nullopt)
{
  auto& e     = env();
  auto column = cudf::make_numeric_column(
    cudf::data_type{type},
    static_cast<cudf::size_type>(values.size()),
    null_row ? cudf::mask_state::ALL_VALID : cudf::mask_state::UNALLOCATED,
    e.stream(),
    e.gpu_space->get_default_allocator());
  if (!values.empty()) {
    REQUIRE(cudaMemcpyAsync(column->mutable_view().data<T>(),
                            values.data(),
                            values.size() * sizeof(T),
                            cudaMemcpyHostToDevice,
                            e.stream().get()) == cudaSuccess);
  }
  if (null_row) {
    cudf::set_null_mask(
      column->mutable_view().null_mask(), *null_row, *null_row + 1, false, e.stream());
    column->set_null_count(1);
  }
  e.stream().sync();
  return column;
}

inline std::shared_ptr<cucascade::data_batch> resident_columns(
  std::vector<std::unique_ptr<cudf::column>> columns)
{
  auto& e = env();
  return sirius::make_data_batch(
    std::make_unique<cudf::table>(std::move(columns)), *e.gpu_space, e.stream(), {});
}

}  // namespace sirius::test::resident_scan
