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

#include <memory>
#include <span>
#include <vector>

namespace sirius::nvrtc {

/**
 * @brief Returns whether all expressions in @p expressions can be compiled and fused by the
 * `NVRTC_JIT` `ProjectSink` compiler for the given @p input table.
 */
[[nodiscard]] bool can_compile_project_sink(std::span<sirius::ast::node const* const> expressions,
                                            cudf::table_view const& input) noexcept;

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
  rmm::device_async_resource_ref mr);

}  // namespace sirius::nvrtc
