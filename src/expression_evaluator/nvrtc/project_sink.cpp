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

#include "codegen/util/cuda_check.hpp"
#include "helper/utils.hpp"

#include <cudf/column/column_factories.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/utilities/traits.hpp>

#include <rmm/device_buffer.hpp>
#include <rmm/device_uvector.hpp>

#include <cuda_runtime_api.h>

#include <expression_evaluator/nvrtc/kernel_abi.hpp>
#include <expression_evaluator/nvrtc/nvrtc_project_sink.hpp>
#include <expression_evaluator/nvrtc/sink_launch.hpp>
#include <expression_evaluator/nvrtc/supportability.hpp>
#include <expression_evaluator/nvrtc/translation_context.hpp>
#include <sirius/exception.hpp>

#include <array>
#include <format>
#include <string>
#include <vector>

namespace sirius::nvrtc {

namespace {

constexpr tile_config project_tile{256, 4};
constexpr char const* project_entry = "sirius_nvrtc_project_kernel";

[[nodiscard]] std::unique_ptr<cudf::column> make_unallocated_fixed_width_output_column(
  cudf::data_type type,
  cudf::size_type num_rows,
  ::cuda::stream_ref stream,
  rmm::device_async_resource_ref mr)
{
  if (cudf::is_numeric(type) || cudf::is_boolean(type)) {
    return cudf::make_numeric_column(type, num_rows, cudf::mask_state::UNALLOCATED, stream, mr);
  }
  if (cudf::is_timestamp(type)) {
    return cudf::make_timestamp_column(type, num_rows, cudf::mask_state::UNALLOCATED, stream, mr);
  }
  if (cudf::is_fixed_point(type)) {
    return cudf::make_fixed_point_column(type, num_rows, cudf::mask_state::UNALLOCATED, stream, mr);
  }
  throw not_implemented_exception("[nvrtc] Cannot allocate output column for type: {}",
                                  cudf::type_to_name(type));
}

[[nodiscard]] void* output_data_pointer(cudf::mutable_column_view& col)
{
  switch (col.type().id()) {
    case cudf::type_id::INT8: return col.data<std::int8_t>();
    case cudf::type_id::INT16: return col.data<std::int16_t>();
    case cudf::type_id::INT32: return col.data<std::int32_t>();
    case cudf::type_id::INT64: return col.data<std::int64_t>();
    case cudf::type_id::UINT8: return col.data<std::uint8_t>();
    case cudf::type_id::UINT16: return col.data<std::uint16_t>();
    case cudf::type_id::UINT32: return col.data<std::uint32_t>();
    case cudf::type_id::UINT64: return col.data<std::uint64_t>();
    case cudf::type_id::FLOAT32: return col.data<float>();
    case cudf::type_id::FLOAT64: return col.data<double>();
    case cudf::type_id::BOOL8: return col.data<bool>();
    case cudf::type_id::TIMESTAMP_DAYS: return col.data<std::int32_t>();
    case cudf::type_id::TIMESTAMP_SECONDS:
    case cudf::type_id::TIMESTAMP_MILLISECONDS:
    case cudf::type_id::TIMESTAMP_MICROSECONDS:
    case cudf::type_id::TIMESTAMP_NANOSECONDS: return col.data<std::int64_t>();
    case cudf::type_id::DECIMAL32: return col.data<std::int32_t>();
    case cudf::type_id::DECIMAL64: return col.data<std::int64_t>();
    case cudf::type_id::DECIMAL128: return col.data<__int128_t>();
    default:
      throw not_implemented_exception("[nvrtc] Unsupported column type: {}",
                                      cudf::type_to_name(col.type()));
  }
}

/**
 * @brief Kernel body for one projection list: one output column per register.
 *
 * Every thread runs the same unchecked row statements for each of its items. An item past the end
 * of the batch evaluates the batch's last row again and is marked not `live`. This is sound because
 * the row statements depend only on `row` and store nothing (the contract of
 * `translation_context::body_code()`) and each output store overwrites one fixed-width slot: the
 * repeated evaluation stores the bytes the row's own thread stores, and raises only the Overflow
 * Poison that row raises.
 *
 * A nullable output stores its payload for every row and takes its validity word from one ballot of
 * `valid && live` per warp and item, so an item past the end of the batch is never counted as a
 * valid row. The kernel counts the valid rows per warp, and each block adds its rows that are not
 * valid to the output's null count, issuing no atomic when it has none.
 */
[[nodiscard]] std::string project_kernel_body(translation_context const& ctx,
                                              std::vector<register_value> const& out_regs,
                                              kernel_abi const& abi,
                                              bool any_nullable_output,
                                              bool any_can_overflow)
{
  std::string src;
  emit_kernel_head(src, project_entry, abi);
  src += std::format(
    "  int32_t const tile_row0 = static_cast<int32_t>(blockIdx.x) * {};\n"
    "  int32_t const lane = static_cast<int32_t>(threadIdx.x) & 31;\n"
    "  int32_t const warp_in_block = static_cast<int32_t>(threadIdx.x) >> 5;\n",
    project_tile.tile_rows());

  for (std::size_t k = 0; k < out_regs.size(); ++k) {
    if (!out_regs[k].is_statically_non_null) {
      src += std::format("  uint32_t warp_valid_{} = 0u;\n", k);
    }
  }
  if (any_can_overflow) { src += "  bool row_overflow = false;\n"; }
  if (ctx.has_subquery_error()) { src += "  bool row_subquery_error = false;\n"; }

  // The loop carries no unroll pragma, so a large row body is not forced through the compiler once
  // per item.
  src += std::format(
    "  for (int item = 0; item < {}; ++item) {{\n"
    "    int32_t const tile_row = tile_row0 + item * {} + static_cast<int32_t>(threadIdx.x);\n"
    "    bool const live = tile_row < num_rows;\n"
    "    int32_t const row = live ? tile_row : num_rows - 1;\n",
    project_tile.items_per_thread,
    project_tile.block_threads);
  src += ctx.body_code();
  for (std::size_t k = 0; k < out_regs.size(); ++k) {
    auto const& reg = out_regs[k];
    src += std::format("    out_{}_data[row] = {}.val;\n", k, reg.var_name);
    if (reg.can_overflow) {
      // A NULL value's Overflow Poison counts too: see the invariant on `reg_val` in
      // `device_runtime/reg_val.cuh`.
      src += std::format("    row_overflow = row_overflow || {}.overflow;\n", reg.var_name);
    }
  }
  for (std::size_t k = 0; k < out_regs.size(); ++k) {
    if (out_regs[k].is_statically_non_null) { continue; }
    // A warp's 32 rows of one item are one validity word, which lane 0 owns.
    src += std::format(
      "    uint32_t const ballot_{0} = __ballot_sync(0xFFFFFFFFu, {1}.valid && live);\n"
      "    if (lane == 0 && live) {{ out_{0}_mask[tile_row >> 5] = ballot_{0}; }}\n"
      "    warp_valid_{0} += static_cast<uint32_t>(__popc(ballot_{0}));\n",
      k,
      out_regs[k].var_name);
  }
  src += "  }\n";

  if (any_can_overflow) {
    emit_status_flag_or(
      src, "row_overflow", "status_and_null_counts", "sirius::nvrtc::abi::status_overflow_bit");
  }
  if (ctx.has_subquery_error()) {
    emit_status_flag_or(src,
                        "row_subquery_error",
                        "status_and_null_counts",
                        "sirius::nvrtc::abi::status_subquery_error_bit");
  }

  if (any_nullable_output) {
    for (std::size_t k = 0; k < out_regs.size(); ++k) {
      if (out_regs[k].is_statically_non_null) { continue; }
      src += std::format(
        "  __shared__ uint32_t s_valid_{0}[{1}];\n"
        "  if (lane == 0) {{ s_valid_{0}[warp_in_block] = warp_valid_{0}; }}\n",
        k,
        project_tile.num_warps());
    }
    // `rows_left` is positive: `grid_dim_x` in `evaluate_project_sink` launches no block past the
    // end of the batch.
    src += std::format(
      "  __syncthreads();\n"
      "  if (threadIdx.x == 0) {{\n"
      "    int32_t const rows_left = num_rows - tile_row0;\n"
      "    uint32_t const block_rows = static_cast<uint32_t>(rows_left < {0} ? rows_left : {0});\n",
      project_tile.tile_rows());
    for (std::size_t k = 0; k < out_regs.size(); ++k) {
      if (out_regs[k].is_statically_non_null) { continue; }
      src += std::format(
        "    uint32_t block_valid_{0} = 0u;\n"
        "    #pragma unroll\n"
        "    for (int w = 0; w < {1}; ++w) {{\n"
        "      block_valid_{0} += s_valid_{0}[w];\n"
        "    }}\n"
        "    if (block_valid_{0} < block_rows) {{\n"
        "      atomicAdd(&status_and_null_counts[{2}], block_rows - block_valid_{0});\n"
        "    }}\n",
        k,
        project_tile.num_warps(),
        1 + k);
    }
    src += "  }\n";
  }
  src += "}\n";
  return src;
}

}  // namespace

project_sink_result evaluate_project_sink(std::span<sirius::ast::node const* const> expressions,
                                          cudf::table_view const& input,
                                          ::cuda::stream_ref stream,
                                          rmm::device_async_resource_ref mr,
                                          sink_options const& options)
{
  auto const num_rows = input.num_rows();
  if (num_rows == 0) {
    std::vector<std::unique_ptr<cudf::column>> empty_cols;
    empty_cols.reserve(expressions.size());
    for (auto const* expr : expressions) {
      empty_cols.push_back(make_unallocated_fixed_width_output_column(
        resolve_output_cudf_type(*expr, input), 0, stream, mr));
    }
    return project_sink_result{std::make_unique<cudf::table>(std::move(empty_cols)), 0};
  }

  translation_context ctx(
    input, options.boundary_map, like_policy{options.like_swar_fastpath, options.classify_like});
  std::vector<register_value> out_regs;
  out_regs.reserve(expressions.size());
  for (auto const* expr : expressions) {
    out_regs.push_back(ctx.produce(*expr));
  }
  bool any_nullable_output = false;
  bool any_can_overflow    = false;
  for (auto const& reg : out_regs) {
    any_nullable_output = any_nullable_output || !reg.is_statically_non_null;
    any_can_overflow    = any_can_overflow || reg.can_overflow;
  }

  kernel_abi abi(ctx);
  for (std::size_t k = 0; k < out_regs.size(); ++k) {
    abi.add_sink_parameter(cuda_type_name(out_regs[k].device_type) + "* __restrict__",
                           std::format("out_{}_data", k));
    abi.add_sink_parameter("uint32_t* __restrict__", std::format("out_{}_mask", k));
  }
  abi.add_sink_parameter("int32_t", "num_rows");
  abi.add_sink_parameter("uint32_t* __restrict__", "status_and_null_counts");

  auto const compiled = compile_sink_kernel(
    project_entry,
    project_kernel_body(ctx, out_regs, abi, any_nullable_output, any_can_overflow),
    options.fuse_multiply_add);
  CUfunction const fn = compiled->func_for_current_device();

  struct output_column_staging {
    rmm::device_buffer null_mask;
    std::unique_ptr<cudf::column> col;
    void* data_ptr{nullptr};
    std::uint32_t* mask_ptr{nullptr};
  };
  std::vector<output_column_staging> outputs;
  outputs.reserve(out_regs.size());
  for (auto const& reg : out_regs) {
    output_column_staging stg{rmm::device_buffer{0, stream, mr}, nullptr};
    if (!reg.is_statically_non_null) {
      stg.null_mask = cudf::create_null_mask(num_rows, cudf::mask_state::UNINITIALIZED, stream, mr);
      stg.mask_ptr  = static_cast<std::uint32_t*>(stg.null_mask.data());
    }
    stg.col    = make_unallocated_fixed_width_output_column(reg.device_type, num_rows, stream, mr);
    auto mview = stg.col->mutable_view();
    stg.data_ptr = output_data_pointer(mview);
    outputs.push_back(std::move(stg));
  }

  // Status words: [0] = status bits, [1..K] = null counts.
  bool const need_status_buffer =
    any_nullable_output || any_can_overflow || ctx.has_subquery_error();
  std::size_t const status_words = 1 + out_regs.size();
  rmm::device_uvector<std::uint32_t> d_status(need_status_buffer ? status_words : 0, stream, mr);
  std::uint32_t* status_ptr = need_status_buffer ? d_status.data() : nullptr;
  if (need_status_buffer) {
    simpatico::throw_if_cuda_error(
      cudaMemsetAsync(status_ptr, 0, status_words * sizeof(std::uint32_t), stream.get()),
      "cudaMemsetAsync(status_and_null_counts)");
  }

  kernel_arguments args(abi, ctx, input, options.string_constants, stream);
  for (auto& out : outputs) {
    args.add_sink_argument(&out.data_ptr);
    args.add_sink_argument(&out.mask_ptr);
  }
  auto num_rows_arg = static_cast<std::int32_t>(num_rows);
  args.add_sink_argument(&num_rows_arg);
  args.add_sink_argument(&status_ptr);

  auto const grid_dim_x = utils::ceil_div(static_cast<std::uint32_t>(num_rows),
                                          static_cast<std::uint32_t>(project_tile.tile_rows()));
  launch_sink_kernel(fn,
                     grid_dim_x,
                     static_cast<unsigned>(project_tile.block_threads),
                     stream,
                     args.data(),
                     "cuLaunchKernel(sirius_nvrtc_project_kernel)");

  std::vector<std::uint32_t> h_status;
  if (need_status_buffer) {
    // The readback waits for the kernel, so the handle outlives the launch.
    h_status.resize(status_words);
    read_status_words(h_status, status_ptr, stream, ctx.overflow_sites());
  } else {
    retain_until_stream_completes(compiled, stream);
  }

  std::vector<std::unique_ptr<cudf::column>> final_columns;
  final_columns.reserve(outputs.size());
  for (std::size_t k = 0; k < outputs.size(); ++k) {
    auto& stg = outputs[k];
    if (!out_regs[k].is_statically_non_null) {
      auto const null_count = static_cast<cudf::size_type>(h_status[1 + k]);
      if (null_count > 0) { stg.col->set_null_mask(std::move(stg.null_mask), null_count); }
    }
    final_columns.push_back(std::move(stg.col));
  }
  return project_sink_result{std::make_unique<cudf::table>(std::move(final_columns)),
                             ctx.narrow_domain_comparison_count()};
}

}  // namespace sirius::nvrtc
