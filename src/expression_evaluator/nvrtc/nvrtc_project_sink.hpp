/*
 * Copyright 2025, Sirius Contributors.
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

#include <cudf/column/column.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>

#include <rmm/resource_ref.hpp>

#include <cuda/stream>

#include <expression/ast/node.hpp>

#include <cstdint>
#include <memory>
#include <span>
#include <unordered_map>
#include <vector>

namespace sirius::nvrtc {

using boundary_column_map = std::unordered_map<sirius::ast::node const*, std::uint32_t>;

/**
 * @brief Returns whether all expressions in @p expressions can be compiled and fused by the
 * `NVRTC_JIT` `ProjectSink` compiler for the given @p input table.
 */
[[nodiscard]] bool can_compile_project_sink(
  std::span<sirius::ast::node const* const> expressions,
  cudf::table_view const& input,
  boundary_column_map const* boundary_map = nullptr) noexcept;

/**
 * @brief Compiles (or fetches from `codegen::jit::KernelCache`) and executes a single fused
 * multi-output `ProjectSink` CUDA `__global__` kernel for @p expressions over @p input.
 *
 * @param expressions The non-null expression trees to evaluate in projection order.
 * @param input The input table view.
 * @param stream The CUDA stream on which to allocate outputs and launch the kernel.
 * @param mr The device memory resource used for output column allocations.
 * @return A new `cudf::table` containing one column per expression in @p expressions.
 */
[[nodiscard]] std::unique_ptr<cudf::table> evaluate_project_sink(
  std::span<sirius::ast::node const* const> expressions,
  cudf::table_view const& input,
  ::cuda::stream_ref stream,
  rmm::device_async_resource_ref mr,
  std::size_t* narrow_domain_comparison_count = nullptr,
  boundary_column_map const* boundary_map     = nullptr);

/**
 * @brief Collects maximal in-tree `Materialization Boundary` sub-trees inside @p expressions that
 * produce register-supported types (`STRING` or fixed-width) so they can be pre-evaluated into
 * synthetic input columns before compiling @p expressions.
 */
void collect_materialization_boundaries(std::span<sirius::ast::node const* const> expressions,
                                        cudf::table_view const& input,
                                        std::vector<sirius::ast::node const*>& out_boundaries);

/**
 * @brief Returns whether @p predicate can be compiled and executed by the `NVRTC_JIT` `FilterSink`
 * compiler for the given @p input table.
 */
[[nodiscard]] bool can_compile_filter_sink(
  sirius::ast::node const& predicate,
  cudf::table_view const& input,
  boundary_column_map const* boundary_map = nullptr) noexcept;

/**
 * @brief Compiles (or fetches from `codegen::jit::KernelCache`) and executes a `FilterSink` CUDA
 * `__global__` kernel for @p predicate over @p input using `Sparse-to-Dense Compaction`.
 *
 * @param predicate The boolean filter expression tree to evaluate.
 * @param input The input table view (possibly augmented with synthetic boundary columns).
 * @param projected_input The table view of output columns to materialize for surviving rows.
 * @param stream The CUDA stream on which to allocate outputs and launch the kernel.
 * @param mr The device memory resource used for output column allocations.
 * @param out_survivors Optional output parameter receiving the compacted ascending `INT32` row
 *                      indices of surviving rows (used by `select_with_survivors`).
 * @param narrow_domain_comparison_count Optional counter incremented by the number of narrow-domain
 *                                       comparisons emitted during translation.
 * @param boundary_map Optional map of pre-evaluated `Materialization Boundary` nodes to synthetic
 *                     column indices in @p input.
 * @return A new `cudf::table` containing the surviving rows of @p projected_input.
 */
[[nodiscard]] std::unique_ptr<cudf::table> evaluate_filter_sink(
  sirius::ast::node const& predicate,
  cudf::table_view const& input,
  cudf::table_view const& projected_input,
  ::cuda::stream_ref stream,
  rmm::device_async_resource_ref mr,
  std::unique_ptr<cudf::column>* out_survivors = nullptr,
  std::size_t* narrow_domain_comparison_count  = nullptr,
  boundary_column_map const* boundary_map      = nullptr);

}  // namespace sirius::nvrtc
