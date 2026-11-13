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

#include "codegen/util/cuda_check.hpp"
#include "helper/utils.hpp"

#include <cudf/column/column_factories.hpp>
#include <cudf/copying.hpp>

#include <rmm/device_uvector.hpp>

#include <cuda_runtime_api.h>

#include <expression_evaluator/nvrtc/kernel_abi.hpp>
#include <expression_evaluator/nvrtc/nvrtc_project_sink.hpp>
#include <expression_evaluator/nvrtc/sink_launch.hpp>
#include <expression_evaluator/nvrtc/translation_context.hpp>
#include <sirius/exception.hpp>

#include <array>
#include <format>
#include <string>
#include <utility>

namespace sirius::nvrtc {

namespace {

constexpr tile_config filter_tile{256, 16};
constexpr int filter_scan_threads   = 1024;
constexpr char const* filter_entry  = "sirius_nvrtc_filter_kernel";
constexpr char const* scan_entry    = "sirius_nvrtc_filter_scan_kernel";
constexpr char const* scatter_entry = "sirius_nvrtc_filter_scatter_kernel";
constexpr std::size_t status_words  = 2;  // [0] = status bits, [1] = total survivors
static_assert(filter_tile.block_threads == 256 && filter_tile.items_per_thread == 16,
              "the scatter kernel source hard-codes the filter tile geometry");

/**
 * @brief Fixed kernels completing the compaction for multi-tile batches: an exclusive scan over the
 *        per-tile survivor counts, and a warp-cooperative scatter of the surviving row indices from
 *       the per-warp ballot words.
 *
 * @note Compiled once per process.
 */
constexpr char const* filter_compaction_source = R"(
#include "codegen/stdint_shim.hpp"

// One block: exclusive prefix sum of tile_counts in place; status_and_counts[1] = total.
extern "C" __global__ void sirius_nvrtc_filter_scan_kernel(
    uint32_t* __restrict__ tile_counts,
    int32_t num_tiles,
    uint32_t* __restrict__ status_and_counts)
{
  constexpr int kThreads = 1024;
  __shared__ uint32_t s_warp_sums[kThreads / 32];
  int32_t const lane = static_cast<int32_t>(threadIdx.x) & 31;
  int32_t const warp = static_cast<int32_t>(threadIdx.x) >> 5;
  uint32_t carry = 0u;
  for (int32_t base = 0; base < num_tiles; base += kThreads) {
    int32_t const idx = base + static_cast<int32_t>(threadIdx.x);
    uint32_t const v = (idx < num_tiles) ? tile_counts[idx] : 0u;
    uint32_t incl = v;
    #pragma unroll
    for (int off = 1; off < 32; off <<= 1) {
      uint32_t const n = __shfl_up_sync(0xFFFFFFFFu, incl, off);
      if (lane >= off) { incl += n; }
    }
    if (lane == 31) { s_warp_sums[warp] = incl; }
    __syncthreads();
    uint32_t warp_prefix = 0u;
    #pragma unroll
    for (int w = 0; w < kThreads / 32; ++w) {
      if (w < warp) { warp_prefix += s_warp_sums[w]; }
    }
    uint32_t chunk_total = 0u;
    #pragma unroll
    for (int w = 0; w < kThreads / 32; ++w) {
      chunk_total += s_warp_sums[w];
    }
    if (idx < num_tiles) { tile_counts[idx] = carry + warp_prefix + incl - v; }
    carry += chunk_total;
    __syncthreads();
  }
  if (threadIdx.x == 0) { status_and_counts[1] = carry; }
}

// One block per tile: each warp walks its share of the tile's 32-row ballot words and writes the
// surviving row indices contiguously (lane rank within the word, word rank within the tile,
// tile prefix from the scan), so the output is ascending and every store instruction is coalesced.
extern "C" __global__ void sirius_nvrtc_filter_scatter_kernel(
    uint32_t const* __restrict__ ballots,
    uint32_t const* __restrict__ tile_prefix,
    int32_t num_rows,
    int32_t* __restrict__ survivors)
{
  constexpr int kThreads = 256;
  constexpr int kWarps = kThreads / 32;
  constexpr int kWordsPerTile = kThreads * 16 / 32;
  constexpr int kWordsPerWarp = kWordsPerTile / kWarps;
  __shared__ uint32_t s_word_prefix[kWordsPerTile];
  __shared__ uint32_t s_warp_sums[kWarps];
  int32_t const tile = static_cast<int32_t>(blockIdx.x);
  int32_t const lane = static_cast<int32_t>(threadIdx.x) & 31;
  int32_t const warp = static_cast<int32_t>(threadIdx.x) >> 5;
  int32_t const num_words = (num_rows >> 5) + ((num_rows & 31) != 0 ? 1 : 0);
  int32_t const word0 = tile * kWordsPerTile;

  // Popcount of every ballot word of the tile (kWordsPerTile <= kThreads: one word per thread).
  static_assert(kWordsPerTile <= kThreads, "one ballot word per thread");
  if (threadIdx.x < kWordsPerTile) {
    int32_t const gw = word0 + static_cast<int32_t>(threadIdx.x);
    s_word_prefix[threadIdx.x] = (gw < num_words) ? static_cast<uint32_t>(__popc(ballots[gw])) : 0u;
  }
  __syncthreads();
  // Each warp turns its contiguous run of kWordsPerWarp counts into exclusive offsets. Every lane
  // takes part in each shuffle round (idle lanes contribute zero) so the warp never diverges.
  uint32_t run_total = 0u;
  for (int k = 0; k < kWordsPerWarp; k += 32) {
    bool const in_run = (k + lane) < kWordsPerWarp;
    int32_t const wi = warp * kWordsPerWarp + k + lane;
    uint32_t const c = in_run ? s_word_prefix[wi] : 0u;
    uint32_t incl = c;
    #pragma unroll
    for (int off = 1; off < 32; off <<= 1) {
      uint32_t const n = __shfl_up_sync(0xFFFFFFFFu, incl, off);
      if (lane >= off) { incl += n; }
    }
    if (in_run) { s_word_prefix[wi] = run_total + incl - c; }
    run_total += __shfl_sync(0xFFFFFFFFu, incl, 31);
  }
  if (lane == 0) { s_warp_sums[warp] = run_total; }
  __syncthreads();
  uint32_t warp_base = tile_prefix[tile];
  for (int w = 0; w < warp; ++w) {
    warp_base += s_warp_sums[w];
  }
  for (int32_t wi = warp * kWordsPerWarp; wi < (warp + 1) * kWordsPerWarp; ++wi) {
    int32_t const gw = word0 + wi;
    if (gw >= num_words) { break; }
    uint32_t const ballot = ballots[gw];
    if (ballot == 0u) { continue; }
    uint32_t const base = warp_base + s_word_prefix[wi];
    if ((ballot >> lane) & 1u) {
      uint32_t const rank = static_cast<uint32_t>(__popc(ballot & ((1u << lane) - 1u)));
      survivors[base + rank] = (gw << 5) + lane;
    }
  }
}
)";

[[nodiscard]] std::unique_ptr<cudf::table> make_empty_projected_table(
  cudf::table_view const& projected_input,
  ::cuda::stream_ref stream,
  rmm::device_async_resource_ref mr)
{
  if (projected_input.num_columns() == 0) {
    return std::make_unique<cudf::table>(std::vector<std::unique_ptr<cudf::column>>{});
  }
  auto sliced = cudf::slice(projected_input, {0, 0}, stream);
  return std::make_unique<cudf::table>(sliced.front(), stream, mr);
}

[[nodiscard]] std::unique_ptr<cudf::column> make_empty_survivors(::cuda::stream_ref stream,
                                                                 rmm::device_async_resource_ref mr)
{
  return cudf::make_numeric_column(
    cudf::data_type{cudf::type_id::INT32}, 0, cudf::mask_state::UNALLOCATED, stream, mr);
}

/**
 * @brief Kernel body for one predicate.
 *
 *  - Phase 1 evaluates the predicate for every row of the tile (one survive bit per item per
 *    thread, one count per (item, warp) in shared memory);
 *  - Phase 2 scans the counts in warp 0 and, for a single-tile batch, scatters the survivors
 *    directly. A multi-tile batch publishes its tile count and ballot words for the scan and
 *    scatter kernels instead, which avoids serializing tiles on a prefix chain.
 */
[[nodiscard]] std::string filter_kernel_body(translation_context const& ctx,
                                             register_value const& pred_reg,
                                             kernel_abi const& abi)
{
  constexpr int scan_entries = filter_tile.items_per_thread * filter_tile.num_warps();
  static_assert(scan_entries % 32 == 0, "the warp scan below consumes whole 32-entry chunks");

  std::string src;
  emit_kernel_head(src, filter_entry, abi);
  src += std::format(
    "  int32_t const tile = static_cast<int32_t>(blockIdx.x);\n"
    "  bool const single_tile = (gridDim.x == 1u);\n"
    "  int32_t const tile_row0 = tile * {};\n"
    "  int32_t const lane = static_cast<int32_t>(threadIdx.x) & 31;\n"
    "  int32_t const warp_in_block = static_cast<int32_t>(threadIdx.x) >> 5;\n"
    "  __shared__ uint32_t s_counts[{}];\n"
    "  __shared__ uint32_t s_block_count;\n"
    "  uint32_t survive_bits = 0u;\n",
    filter_tile.tile_rows(),
    scan_entries);
  if (pred_reg.can_overflow) { src += "  bool row_overflow = false;\n"; }
  if (ctx.has_subquery_error()) { src += "  bool row_subquery_error = false;\n"; }

  auto const body_code = ctx.body_code();
  emit_full_and_partial_tiles(src, filter_tile, [&](bool guarded) {
    std::string const indent = guarded ? "      " : "    ";
    emit_item_loop_open(src, filter_tile);
    src += "    bool survives = false;\n";
    if (guarded) { src += "    if (row < num_rows) {\n"; }
    src += body_code;
    auto const& v = pred_reg.var_name;
    if (pred_reg.can_overflow) {
      // A NULL value's Overflow Poison counts too: see the invariant on `reg_val` in
      // `device_runtime/reg_val.cuh`.
      src += std::format(
        "{0}row_overflow = row_overflow || {1}.overflow;\n"
        "{0}survives = ({1}.valid && !{1}.overflow && {1}.val);\n",
        indent,
        v);
    } else if (pred_reg.is_statically_non_null) {
      src += std::format("{}survives = {}.val;\n", indent, v);
    } else {
      src += std::format("{0}survives = ({1}.valid && {1}.val);\n", indent, v);
    }
    if (guarded) { src += "    }\n"; }
    src += std::format(
      "    uint32_t const warp_ballot = __ballot_sync(0xFFFFFFFFu, survives);\n"
      "    survive_bits |= (survives ? 1u : 0u) << item;\n"
      "    if (lane == 0) {{\n"
      "      s_counts[item * {0} + warp_in_block] = static_cast<uint32_t>(__popc(warp_ballot));\n"
      "      if (!single_tile{1}) {{ ballots[(row - lane) >> 5] = warp_ballot; }}\n"
      "    }}\n"
      "  }}\n",
      filter_tile.num_warps(),
      guarded ? " && row - lane < num_rows" : "");
  });

  if (pred_reg.can_overflow) {
    emit_status_flag_or(
      src, "row_overflow", "status_and_counts", "sirius::nvrtc::abi::status_overflow_bit");
  }
  if (ctx.has_subquery_error()) {
    emit_status_flag_or(src,
                        "row_subquery_error",
                        "status_and_counts",
                        "sirius::nvrtc::abi::status_subquery_error_bit");
  }

  src += std::format(
    "  __syncthreads();\n"
    "  if (warp_in_block == 0) {{\n"
    "    uint32_t carry = 0u;\n"
    "    #pragma unroll\n"
    "    for (int chunk = 0; chunk < {0}; ++chunk) {{\n"
    "      uint32_t const c = s_counts[chunk * 32 + lane];\n"
    "      uint32_t incl = c;\n"
    "      #pragma unroll\n"
    "      for (int off = 1; off < 32; off <<= 1) {{\n"
    "        uint32_t const n = __shfl_up_sync(0xFFFFFFFFu, incl, off);\n"
    "        if (lane >= off) {{ incl += n; }}\n"
    "      }}\n"
    "      s_counts[chunk * 32 + lane] = carry + incl - c;\n"
    "      carry += __shfl_sync(0xFFFFFFFFu, incl, 31);\n"
    "    }}\n"
    "    if (lane == 0) {{\n"
    "      s_block_count = carry;\n"
    "      if (single_tile) {{ status_and_counts[1] = carry; }} else {{ tile_counts[tile] = carry; "
    "}}\n"
    "    }}\n"
    "  }}\n"
    "  if (!single_tile) {{ return; }}\n"
    "  __syncthreads();\n"
    "  #pragma unroll\n"
    "  for (int item = 0; item < {1}; ++item) {{\n"
    "    bool const survives = ((survive_bits >> item) & 1u) != 0u;\n"
    "    uint32_t const warp_ballot = __ballot_sync(0xFFFFFFFFu, survives);\n"
    "    if (survives) {{\n"
    "      uint32_t const lane_rank = static_cast<uint32_t>(__popc(warp_ballot & ((1u << lane) - "
    "1u)));\n"
    "      survivors[s_counts[item * {2} + warp_in_block] + lane_rank] = tile_row0 + item * {3} + "
    "static_cast<int32_t>(threadIdx.x);\n"
    "    }}\n"
    "  }}\n"
    "}}\n",
    scan_entries / 32,
    filter_tile.items_per_thread,
    filter_tile.num_warps(),
    filter_tile.block_threads);
  return src;
}

}  // namespace

filter_sink_result evaluate_filter_sink(sirius::ast::node const& predicate,
                                        cudf::table_view const& input,
                                        cudf::table_view const& projected_input,
                                        bool want_survivors,
                                        ::cuda::stream_ref stream,
                                        rmm::device_async_resource_ref mr,
                                        sink_options const& options)
{
  auto const num_rows = input.num_rows();
  if (num_rows == 0) {
    return filter_sink_result{make_empty_projected_table(projected_input, stream, mr),
                              want_survivors ? make_empty_survivors(stream, mr) : nullptr,
                              0};
  }

  translation_context ctx(
    input, options.boundary_map, like_policy{options.like_swar_fastpath, options.classify_like});
  auto const pred_reg = ctx.produce_condition(predicate);

  // status_and_counts: [0] = status flags,
  //                    [1] = total survivors (written by the predicate kernel for a single tile, by
  //                                           the scan otherwise).
  kernel_abi abi(ctx);
  abi.add_sink_parameter("int32_t", "num_rows");
  abi.add_sink_parameter("uint32_t* __restrict__", "status_and_counts");
  abi.add_sink_parameter("uint32_t* __restrict__", "tile_counts");
  abi.add_sink_parameter("uint32_t* __restrict__", "ballots");
  abi.add_sink_parameter("int32_t* __restrict__", "survivors");

  auto const compiled = compile_sink_kernel(filter_entry, filter_kernel_body(ctx, pred_reg, abi));
  CUfunction const fn = compiled->func_for_current_device();

  auto const grid_dim_x  = utils::ceil_div(static_cast<std::uint32_t>(num_rows),
                                          static_cast<std::uint32_t>(filter_tile.tile_rows()));
  bool const single_tile = std::cmp_equal(grid_dim_x, 1);

  // Workspace: the status words, then (multi-tile only) one count per tile and one ballot word per
  //            32 rows
  auto const num_words = utils::ceil_div(static_cast<std::size_t>(num_rows), std::size_t{32});
  auto const workspace_words =
    status_words +
    (single_tile ? std::size_t{0} : (static_cast<std::size_t>(grid_dim_x) + num_words));
  rmm::device_uvector<std::uint32_t> d_workspace(workspace_words, stream, mr);
  rmm::device_uvector<std::int32_t> d_survivors(static_cast<std::size_t>(num_rows), stream, mr);
  auto* status_ptr = d_workspace.data();
  simpatico::throw_if_cuda_error(
    cudaMemsetAsync(status_ptr, 0, status_words * sizeof(std::uint32_t), stream.get()),
    "cudaMemsetAsync(filter_status)");
  std::uint32_t* tile_counts_ptr  = single_tile ? nullptr : (d_workspace.data() + status_words);
  std::uint32_t* ballots_ptr      = single_tile ? nullptr : (tile_counts_ptr + grid_dim_x);
  std::int32_t* survivors_scratch = d_survivors.data();
  auto num_rows_arg               = static_cast<std::int32_t>(num_rows);

  kernel_arguments args(abi, ctx, input, options.string_constants, stream);
  args.add_sink_argument(&num_rows_arg);
  args.add_sink_argument(&status_ptr);
  args.add_sink_argument(&tile_counts_ptr);
  args.add_sink_argument(&ballots_ptr);
  args.add_sink_argument(&survivors_scratch);
  launch_sink_kernel(fn,
                     grid_dim_x,
                     static_cast<unsigned>(filter_tile.block_threads),
                     stream,
                     args.data(),
                     "cuLaunchKernel(sirius_nvrtc_filter_kernel)");

  if (!single_tile) {
    auto const scan =
      compile_sink_kernel(scan_entry, filter_compaction_source, /*prepend_device_runtime=*/false);
    auto const scatter = compile_sink_kernel(
      scatter_entry, filter_compaction_source, /*prepend_device_runtime=*/false);
    auto num_tiles_arg = static_cast<std::int32_t>(grid_dim_x);
    std::array<void*, 3> scan_args{&tile_counts_ptr, &num_tiles_arg, &status_ptr};
    launch_sink_kernel(scan->func_for_current_device(),
                       1,
                       filter_scan_threads,
                       stream,
                       scan_args.data(),
                       "cuLaunchKernel(sirius_nvrtc_filter_scan_kernel)");
    std::array<void*, 4> scatter_args{
      &ballots_ptr, &tile_counts_ptr, &num_rows_arg, &survivors_scratch};
    launch_sink_kernel(scatter->func_for_current_device(),
                       grid_dim_x,
                       static_cast<unsigned>(filter_tile.block_threads),
                       stream,
                       scatter_args.data(),
                       "cuLaunchKernel(sirius_nvrtc_filter_scatter_kernel)");
  }

  // The readback waits for every launch above, so the kernel handles outlive their launches.
  std::array<std::uint32_t, status_words> h_status{};
  read_status_words(h_status, status_ptr, stream, ctx.overflow_sites());
  auto const survivor_count = static_cast<cudf::size_type>(h_status[1]);
  auto const narrow_count   = ctx.narrow_domain_comparison_count();
  if (survivor_count == 0) {
    return filter_sink_result{make_empty_projected_table(projected_input, stream, mr),
                              want_survivors ? make_empty_survivors(stream, mr) : nullptr,
                              narrow_count};
  }

  // 100% selectivity without a survivors request: no gather needed.
  if (survivor_count == num_rows && !want_survivors) {
    return filter_sink_result{
      std::make_unique<cudf::table>(projected_input, stream, mr), nullptr, narrow_count};
  }

  auto survivors_col   = cudf::make_numeric_column(cudf::data_type{cudf::type_id::INT32},
                                                 survivor_count,
                                                 cudf::mask_state::UNALLOCATED,
                                                 stream,
                                                 mr);
  auto* survivors_data = survivors_col->mutable_view().data<std::int32_t>();
  simpatico::throw_if_cuda_error(
    cudaMemcpyAsync(survivors_data,
                    survivors_scratch,
                    static_cast<std::size_t>(survivor_count) * sizeof(std::int32_t),
                    cudaMemcpyDeviceToDevice,
                    stream.get()),
    "cudaMemcpyAsync(survivors)");

  std::unique_ptr<cudf::table> result_table;
  if (projected_input.num_columns() == 0) {
    result_table = std::make_unique<cudf::table>(std::vector<std::unique_ptr<cudf::column>>{});
  } else if (survivor_count == num_rows) {
    result_table = std::make_unique<cudf::table>(projected_input, stream, mr);
  } else {
    result_table = cudf::gather(
      projected_input, survivors_col->view(), cudf::out_of_bounds_policy::DONT_CHECK, stream, mr);
  }
  return filter_sink_result{
    std::move(result_table), want_survivors ? std::move(survivors_col) : nullptr, narrow_count};
}

}  // namespace sirius::nvrtc
