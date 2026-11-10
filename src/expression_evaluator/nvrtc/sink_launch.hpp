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

#include "codegen/jit/nvrtc_compiler.hpp"

#include <cuda.h>
#include <cuda/stream>

#include <expression_evaluator/nvrtc/kernel_abi.hpp>

#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <string_view>

/**
 * @file
 * @brief What every `NVRTC_JIT` sink shares around its generated kernel: the tile shape and the
 *        emitters for the full/partial tile dispatch and the status flags, compiling or fetching
 *        the kernel behind the device runtime preamble, launching it, keeping its handle alive, and
 *        reading the status words back.
 *
 * All functions here are stream-ordered on the stream they are given and throw `sirius` exceptions
 * on failure.
 */

namespace sirius::nvrtc {

/**
 * @brief Rows a block processes per launch: @p block_threads threads each handling
 *        @p items_per_thread rows strided by the block width.
 */
struct tile_config {
  int block_threads;
  int items_per_thread;
  [[nodiscard]] constexpr int tile_rows() const noexcept
  {
    return block_threads * items_per_thread;
  }
  [[nodiscard]] constexpr int num_warps() const noexcept { return block_threads / 32; }
};

/**
 * @brief Appends the kernel head `extern "C" __global__ void <entry>(<parameters>)` and the opening
 *        brace with the ABI's per-thread prelude to @p out.
 */
void emit_kernel_head(std::string& out, std::string_view entry, kernel_abi const& abi);

/**
 * @brief Appends the full-tile / partial-tile dispatch to @p out: the block's tile is processed by
 *        @c emit_items(false) when it is complete and by @c emit_items(true) (which must guard each
 *        row by the number of rows) otherwise. Expects `tile_row0` and `num_rows` in scope.
 */
template <typename EmitItems>
void emit_full_and_partial_tiles(std::string& out, tile_config tile, EmitItems&& emit_items)
{
  out += "  if (num_rows - tile_row0 >= " + std::to_string(tile.tile_rows()) + ") {\n";
  emit_items(false);
  out += "  } else {\n";
  emit_items(true);
  out += "  }\n";
}

/**
 * @brief Appends the head of the per-thread item loop to @p out: an unrolled loop over
 *        @p items_per_thread. The caller closes the loop body.
 */
void emit_item_loop_open(std::string& out, tile_config tile);

/**
 * @brief Appends the end-of-kernel reduction of a per-thread flag: one warp ballot of @p flag_var,
 *        then lane 0 ORs @p status_bit into `<status_array>[0]`.
 */
void emit_status_flag_or(std::string& out,
                         std::string_view flag_var,
                         std::string_view status_array,
                         std::string_view status_bit);

/**
 * @brief Returns the compiled kernel @p entry whose source is the device runtime preamble followed
 *        by @p kernel_body (or @p kernel_body alone when @p prepend_device_runtime is false),
 *        compiling it with NVRTC when neither this process nor the disk cache has it.
 *        `SIRIUS_NVRTC_DUMP_DIR` receives the full source of every compile.
 *
 * The in-process lookup is keyed by (entry, body, device architecture), so it is proportional to
 * the generated body rather than the preamble and never hands a kernel built for one SM to another.
 *
 * @throws internal_exception with the NVRTC log and the source when compilation fails.
 */
[[nodiscard]] std::shared_ptr<codegen::jit::CompiledKernel const> compile_sink_kernel(
  std::string_view entry, std::string const& kernel_body, bool prepend_device_runtime = true);

/**
 * @brief Returns the compiled kernel @p entry of the complete, self-contained @p source (no device
 *        runtime preamble is added), compiling it with NVRTC when no cache has it.
 *
 * @throws internal_exception with the NVRTC log and the source when compilation fails.
 */
[[nodiscard]] std::shared_ptr<codegen::jit::CompiledKernel const> compile_kernel_source(
  std::string_view entry, std::string const& source);

/**
 * @brief Launches @p fn with a 1-D grid and block on @p stream; @p what names the kernel in
 *        error messages.
 */
void launch_sink_kernel(CUfunction fn,
                        unsigned grid_dim_x,
                        unsigned block_dim_x,
                        ::cuda::stream_ref stream,
                        void** args,
                        char const* what);

/**
 * @brief Keeps @p kernel loaded until the work queued on @p stream so far has completed, for a
 *        launch the caller does not wait for before dropping its handle (@c KernelCache::clear()
 *        would otherwise be free to unload a running kernel's library).
 */
void retain_until_stream_completes(std::shared_ptr<codegen::jit::CompiledKernel const> kernel,
                                   ::cuda::stream_ref stream);

/**
 * @brief Copies the status words at @p device into @p host, waiting for @p stream, and raises the
 *        error the status bits of word 0 report.
 *
 * @throws invalid_input_exception when a poisoned value reached the sink (@c status_overflow_bit)
 * or a scalar subquery produced more than one row (@c status_subquery_error_bit).
 */
void read_status_words(std::span<std::uint32_t> host,
                       std::uint32_t const* device,
                       ::cuda::stream_ref stream);

}  // namespace sirius::nvrtc
