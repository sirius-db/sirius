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
#include <cudf/copying.hpp>

#include <rmm/device_uvector.hpp>

#include <cuda_runtime_api.h>

#include <expression_evaluator/nvrtc/kernel_abi.hpp>
#include <expression_evaluator/nvrtc/nvrtc_project_sink.hpp>
#include <expression_evaluator/nvrtc/sink_launch.hpp>
#include <expression_evaluator/nvrtc/translation_context.hpp>
#include <sirius/exception.hpp>

#include <array>
#include <concepts>
#include <format>
#include <string>
#include <string_view>
#include <utility>

namespace sirius::nvrtc {

namespace {

constexpr tile_config filter_tile{256, 16};
constexpr char const* filter_entry  = "sirius_nvrtc_filter_kernel";
constexpr char const* scatter_entry = "sirius_nvrtc_filter_scatter_kernel";
constexpr std::size_t status_words  = 2;  // [0] = status bits, [1] = total survivors

// A warp of the predicate kernel and of the scatter kernel owns `words_per_warp` consecutive
// *ballot words*  (32-bit masks), each covering 32 consecutive rows: lane `i` holds word `i` of its
// warp in a register (thus, `words_per_warp <= 32`).
constexpr int words_per_warp = filter_tile.items_per_thread;
constexpr int rows_per_warp  = words_per_warp * 32;
constexpr int words_per_tile = words_per_warp * filter_tile.num_warps();
static_assert(words_per_warp <= 32, "a lane holds at most one ballot word of its warp");
static_assert(filter_tile.block_threads % 32 == 0, "a tile is made of whole warps");

// The predicate kernel adds every tile's survivor count to the total of the tile's group: a group
// owns as many tiles as a scatter block has threads, so that each thread reads the count of at
// most one tile of the block's group.
constexpr int tiles_per_group = filter_tile.block_threads;

/**
 * @brief Appends the statements that replace @p var in every lane by its sum over the warp.
 */
void emit_warp_sum(std::string& out, std::string_view var)
{
  out += std::format(
    "  #pragma unroll\n"
    "  for (int off = 16; off > 0; off >>= 1) {{\n"
    "    {0} += __shfl_xor_sync(0xFFFFFFFFu, {0}, off);\n"
    "  }}\n",
    var);
}

/**
 * @brief Appends the statements that define `warp_total`, the number of survivors in a warp's
 *        ballot words. Expects each lane's word in @p word_var (zero in a lane that owns none).
 */
void emit_warp_total(std::string& out, std::string_view word_var)
{
  out += std::format("  uint32_t warp_total = static_cast<uint32_t>(__popc({}));\n", word_var);
  emit_warp_sum(out, "warp_total");
}

/** @brief Whether the compiler must unroll a generated loop in full or must keep it a loop. */
enum class loop_unroll : bool { full, never };

/**
 * @brief Appends the loop in which a warp writes the row index of every survivor in its ballot
 *        words, in ascending order from `survivors[warp_base]`. Expects each lane's word in
 *        @p word_var and the warp's first row in @p warp_row0.
 *
 * Each word is handed to the whole warp by one shuffle, so the loop reads no memory, and the lanes
 * whose bit is set store to consecutive slots.
 */
void emit_warp_survivor_writes(std::string& out,
                               std::string_view word_var,
                               std::string_view warp_row0,
                               loop_unroll unroll)
{
  out += std::format(
    "  uint32_t const lanes_below = (1u << lane) - 1u;\n"
    "  uint32_t out = warp_base;\n"
    "  {3}\n"
    "  for (int i = 0; i < {0}; ++i) {{\n"
    "    uint32_t const word = __shfl_sync(0xFFFFFFFFu, {1}, i);\n"
    "    if ((word >> lane) & 1u) {{\n"
    "      survivors[out + static_cast<uint32_t>(__popc(word & lanes_below))] = {2} + i * 32 + "
    "lane;\n"
    "    }}\n"
    "    out += static_cast<uint32_t>(__popc(word));\n"
    "  }}\n",
    words_per_warp,
    word_var,
    warp_row0,
    unroll == loop_unroll::full ? "#pragma unroll" : "#pragma unroll 1");
}

/**
 * @brief Returns the source of the fixed kernel that completes the compaction of a multi-tile
 *        batch: it writes the surviving row indices from the ballot words.
 *
 * @note Compiled once per process.
 */
[[nodiscard]] std::string const& filter_scatter_source()
{
  static std::string const source = [] {
    std::string src = std::format(R"(
// One block per tile: lane i of a warp loads ballot word i of the warp once, and the warp writes
// its survivors after those of the tiles and of the warps before it. The survivors before the tile
// are the totals of the groups before its group plus the counts of the tiles before it in its
// group: every thread loads at most one such count and every {2}th group total, and the block sums
// them. The block of the last tile stores the batch total in status_and_counts[1].
extern "C" __global__ void sirius_nvrtc_filter_scatter_kernel(
    uint32_t const* __restrict__ ballots,
    uint32_t const* __restrict__ tile_counts,
    uint32_t const* __restrict__ group_counts,
    int32_t num_rows,
    int32_t* __restrict__ survivors,
    uint32_t* __restrict__ status_and_counts)
{{
  __shared__ uint32_t s_warp_totals[{0}];
  __shared__ uint32_t s_warp_preceding[{0}];
  int32_t const tile = static_cast<int32_t>(blockIdx.x);
  int32_t const thread = static_cast<int32_t>(threadIdx.x);
  int32_t const lane = thread & 31;
  int32_t const warp = thread >> 5;
  int32_t const num_words = (num_rows >> 5) + ((num_rows & 31) != 0 ? 1 : 0);
  int32_t const warp_word0 = tile * {1} + warp * {3};
  int32_t const my_index = warp_word0 + lane;
  int32_t const group = tile / {2};
  int32_t const my_tile = group * {2} + thread;
  uint32_t const my_word = (lane < {3} && my_index < num_words) ? ballots[my_index] : 0u;
  uint32_t preceding = (my_tile < tile) ? tile_counts[my_tile] : 0u;
  #pragma unroll 1
  for (int32_t g = thread; g < group; g += {2}) {{ preceding += group_counts[g]; }}
)",
                                  filter_tile.num_warps(),
                                  words_per_tile,
                                  tiles_per_group,
                                  words_per_warp);
    emit_warp_total(src, "my_word");
    emit_warp_sum(src, "preceding");
    src += std::format(
      "  if (lane == 0) {{\n"
      "    s_warp_totals[warp] = warp_total;\n"
      "    s_warp_preceding[warp] = preceding;\n"
      "  }}\n"
      "  __syncthreads();\n"
      "  uint32_t warp_base = 0u;\n"
      "  uint32_t tile_end = 0u;\n"
      "  #pragma unroll\n"
      "  for (int w = 0; w < {}; ++w) {{\n"
      "    uint32_t const p = s_warp_preceding[w];\n"
      "    uint32_t const c = s_warp_totals[w];\n"
      "    warp_base += p;\n"
      "    tile_end += p + c;\n"
      "    if (w < warp) {{ warp_base += c; }}\n"
      "  }}\n"
      "  if (thread == 0 && tile == static_cast<int32_t>(gridDim.x) - 1) {{\n"
      "    status_and_counts[1] = tile_end;\n"
      "  }}\n"
      "  if (warp_total == 0u) {{ return; }}\n",
      filter_tile.num_warps());
    emit_warp_survivor_writes(src, "my_word", "(warp_word0 << 5)", loop_unroll::full);
    src += "}\n";
    return src;
  }();
  return source;
}

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

/** @brief Whether a tile holds all of its rows or is cut short by the end of the batch. */
enum class tile_fill : bool { full, partial };

/**
 * @brief Appends the full-tile / partial-tile dispatch to @p out: the block's tile is processed by
 *        @c emit_items(tile_fill::full) when it is complete and by
 *        @c emit_items(tile_fill::partial) (which must guard each row by the number of rows)
 *        otherwise. Expects `tile_row0` and `num_rows` in scope.
 */
void emit_full_and_partial_tiles(std::string& out, std::invocable<tile_fill> auto const& emit_items)
{
  out += std::format("  if (num_rows - tile_row0 >= {}) {{\n", filter_tile.tile_rows());
  emit_items(tile_fill::full);
  out += "  } else {\n";
  emit_items(tile_fill::partial);
  out += "  }\n";
}

/**
 * @brief Appends the head of the per-thread item loop to @p out. The caller closes the loop body.
 *
 * Item `i` of a warp covers the 32 rows of the warp's ballot word `i`, one row per lane.
 *
 * Neither loop forces the row body to be unrolled. The loop over a full tile carries no unroll
 * pragma, so unrolling is the compiler's choice and a large row body is not forced through it once
 * per item. The loop over a partial tile runs in at most one block per batch (the last), so it is
 * marked never to be unrolled.
 */
void emit_item_loop_open(std::string& out, tile_fill fill)
{
  out += std::format(
    "{}"
    "  for (int item = 0; item < {}; ++item) {{\n"
    "    int32_t const row = warp_row0 + item * 32 + lane;\n",
    fill == tile_fill::partial ? "  #pragma unroll 1\n" : "",
    filter_tile.items_per_thread);
}

/**
 * @brief Kernel body for one predicate.
 *
 *  - Phase 1 evaluates the predicate for every row of the tile. A warp votes once per item, and
 *    lane `i` keeps the ballot word of item `i` in a register, so the loop writes no memory.
 *  - Phase 2 counts the survivors: each warp sums its words in registers and the block reduces the
 *    warp totals through shared memory. A single-tile batch then writes its survivors directly. A
 *    multi-tile batch publishes its ballot words (one coalesced store per warp) and its tile count
 *    for the scatter kernel instead, and adds the count to the total of its group of tiles, so no
 *    tile waits for the counts of the tiles before it.
 */
[[nodiscard]] std::string filter_kernel_body(translation_context const& ctx,
                                             register_value const& pred_reg,
                                             kernel_abi const& abi)
{
  std::string src;
  emit_kernel_head(src, filter_entry, abi);
  src += std::format(
    "  int32_t const tile = static_cast<int32_t>(blockIdx.x);\n"
    "  bool const single_tile = (gridDim.x == 1u);\n"
    "  int32_t const tile_row0 = tile * {};\n"
    "  int32_t const lane = static_cast<int32_t>(threadIdx.x) & 31;\n"
    "  int32_t const warp_in_block = static_cast<int32_t>(threadIdx.x) >> 5;\n"
    "  int32_t const warp_row0 = tile_row0 + warp_in_block * {};\n"
    "  __shared__ uint32_t s_warp_totals[{}];\n"
    "  uint32_t my_ballot = 0u;\n",
    filter_tile.tile_rows(),
    rows_per_warp,
    filter_tile.num_warps());
  if (pred_reg.can_overflow) { src += "  bool row_overflow = false;\n"; }
  if (ctx.has_subquery_error()) { src += "  bool row_subquery_error = false;\n"; }

  auto const body_code = ctx.body_code();
  emit_full_and_partial_tiles(src, [&](tile_fill fill) {
    bool const guarded       = fill == tile_fill::partial;
    std::string const indent = guarded ? "      " : "    ";
    emit_item_loop_open(src, fill);
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
    src +=
      "    uint32_t const warp_ballot = __ballot_sync(0xFFFFFFFFu, survives);\n"
      "    my_ballot = (lane == item) ? warp_ballot : my_ballot;\n"
      "  }\n";
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

  emit_warp_total(src, "my_ballot");
  // A row past the end of the batch never survives, so a ballot word that starts there is zero and
  // is not stored: the workspace holds no word for it.
  src += std::format(
    "  if (lane == 0) {{ s_warp_totals[warp_in_block] = warp_total; }}\n"
    "  if (!single_tile && lane < {0}) {{\n"
    "    int32_t const word = (warp_row0 >> 5) + lane;\n"
    "    if ((word << 5) < num_rows) {{ ballots[word] = my_ballot; }}\n"
    "  }}\n"
    "  __syncthreads();\n"
    "  uint32_t warp_base = 0u;\n"
    "  uint32_t tile_total = 0u;\n"
    "  #pragma unroll\n"
    "  for (int w = 0; w < {1}; ++w) {{\n"
    "    uint32_t const c = s_warp_totals[w];\n"
    "    if (w < warp_in_block) {{ warp_base += c; }}\n"
    "    tile_total += c;\n"
    "  }}\n"
    "  if (!single_tile) {{\n"
    "    if (threadIdx.x == 0) {{\n"
    "      tile_counts[tile] = tile_total;\n"
    "      atomicAdd(&group_counts[tile / {2}], tile_total);\n"
    "    }}\n"
    "    return;\n"
    "  }}\n"
    "  if (threadIdx.x == 0) {{ status_and_counts[1] = tile_total; }}\n",
    words_per_warp,
    filter_tile.num_warps(),
    tiles_per_group);
  // Every predicate kernel compiles this loop and only the one block of a single-tile batch runs
  // it, so it is never unrolled.
  emit_warp_survivor_writes(src, "my_ballot", "warp_row0", loop_unroll::never);
  src += "}\n";
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
  //                                           the scatter kernel otherwise).
  kernel_abi abi(ctx);
  abi.add_sink_parameter("int32_t", "num_rows");
  abi.add_sink_parameter("uint32_t* __restrict__", "status_and_counts");
  abi.add_sink_parameter("uint32_t* __restrict__", "group_counts");
  abi.add_sink_parameter("uint32_t* __restrict__", "tile_counts");
  abi.add_sink_parameter("uint32_t* __restrict__", "ballots");
  abi.add_sink_parameter("int32_t* __restrict__", "survivors");

  auto const grid_dim_x  = utils::ceil_div(static_cast<std::uint32_t>(num_rows),
                                          static_cast<std::uint32_t>(filter_tile.tile_rows()));
  bool const single_tile = std::cmp_equal(grid_dim_x, 1);

  // Both kernels are compiled before the first launch and their handles live until this function
  // returns, after the status readback has waited for every launch.
  auto const compiled = compile_sink_kernel(
    filter_entry, filter_kernel_body(ctx, pred_reg, abi), options.fuse_multiply_add);
  auto const scatter =
    single_tile
      ? nullptr
      : compile_sink_kernel(scatter_entry, filter_scatter_source(), options.fuse_multiply_add);

  // Workspace: the status words, then (multi-tile only) one total per group of tiles, one count per
  //            tile and one ballot word per 32 rows. The predicate kernel accumulates into the
  //            status words and the group totals, so both start at zero.
  auto const num_words  = utils::ceil_div(static_cast<std::size_t>(num_rows), std::size_t{32});
  auto const num_groups = single_tile ? std::size_t{0}
                                      : utils::ceil_div(static_cast<std::size_t>(grid_dim_x),
                                                        static_cast<std::size_t>(tiles_per_group));
  auto const workspace_words =
    status_words + num_groups +
    (single_tile ? std::size_t{0} : (static_cast<std::size_t>(grid_dim_x) + num_words));
  rmm::device_uvector<std::uint32_t> d_workspace(workspace_words, stream, mr);
  rmm::device_uvector<std::int32_t> d_survivors(static_cast<std::size_t>(num_rows), stream, mr);
  auto* status_ptr = d_workspace.data();
  simpatico::throw_if_cuda_error(
    cudaMemsetAsync(
      status_ptr, 0, (status_words + num_groups) * sizeof(std::uint32_t), stream.get()),
    "cudaMemsetAsync(filter_status)");
  std::uint32_t* group_counts_ptr = single_tile ? nullptr : (d_workspace.data() + status_words);
  std::uint32_t* tile_counts_ptr  = single_tile ? nullptr : (group_counts_ptr + num_groups);
  std::uint32_t* ballots_ptr      = single_tile ? nullptr : (tile_counts_ptr + grid_dim_x);
  std::int32_t* survivors_scratch = d_survivors.data();
  auto num_rows_arg               = static_cast<std::int32_t>(num_rows);

  kernel_arguments args(abi, ctx, input, options.string_constants, stream);
  args.add_sink_argument(&num_rows_arg);
  args.add_sink_argument(&status_ptr);
  args.add_sink_argument(&group_counts_ptr);
  args.add_sink_argument(&tile_counts_ptr);
  args.add_sink_argument(&ballots_ptr);
  args.add_sink_argument(&survivors_scratch);
  launch_sink_kernel(compiled->func_for_current_device(),
                     grid_dim_x,
                     static_cast<unsigned>(filter_tile.block_threads),
                     stream,
                     args.data(),
                     "cuLaunchKernel(sirius_nvrtc_filter_kernel)");

  if (!single_tile) {
    std::array<void*, 6> scatter_args{&ballots_ptr,
                                      &tile_counts_ptr,
                                      &group_counts_ptr,
                                      &num_rows_arg,
                                      &survivors_scratch,
                                      &status_ptr};
    launch_sink_kernel(scatter->func_for_current_device(),
                       grid_dim_x,
                       static_cast<unsigned>(filter_tile.block_threads),
                       stream,
                       scatter_args.data(),
                       "cuLaunchKernel(sirius_nvrtc_filter_scatter_kernel)");
  }

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
