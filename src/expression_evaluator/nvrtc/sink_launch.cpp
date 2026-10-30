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

#include "codegen/jit/kernel_cache.hpp"
#include "codegen/util/cuda_check.hpp"
#include "util/host_observation.hpp"

#include <cuda_runtime_api.h>

#include <expression_evaluator/nvrtc/device_runtime.hpp>
#include <expression_evaluator/nvrtc/device_runtime/abi_constants.cuh>
#include <expression_evaluator/nvrtc/sink_launch.hpp>
#include <sirius/exception.hpp>

#include <atomic>
#include <cstdlib>
#include <format>
#include <fstream>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace sirius::nvrtc {

namespace {

int cached_arch_cc_for_current_device()
{
  int device = 0;
  simpatico::throw_if_cuda_error(cudaGetDevice(&device), "cudaGetDevice(arch_cc)");
  static std::mutex mutex;
  static std::unordered_map<int, int> arch_by_device;
  std::lock_guard<std::mutex> const lock(mutex);
  auto const it = arch_by_device.find(device);
  if (it != arch_by_device.end()) { return it->second; }
  int const arch = codegen::jit::arch_cc_for_current_device();
  arch_by_device.emplace(device, arch);
  return arch;
}

/**
 * @brief Debug aid: when `SIRIUS_NVRTC_DUMP_DIR` names a directory, every generated kernel source
 *        is written there as `<seq>_<entry>.cu` so it can be inspected or compiled offline (e.g.
 *        with @c nvcc --resource-usage).
 *
 * @note Resolved once per process; unset disables the dump.
 */
void maybe_dump_kernel_source(std::string_view entry, std::string const& source)
{
  static std::string const dir = []() -> std::string {
    char const* env = std::getenv("SIRIUS_NVRTC_DUMP_DIR");
    return (env != nullptr && *env != '\0') ? std::string(env) : std::string{};
  }();
  if (dir.empty()) { return; }
  static std::atomic<unsigned> seq{0};
  auto const path = std::format("{}/{}_{}.cu", dir, seq.fetch_add(1), entry);
  std::ofstream out(path, std::ios::binary);
  out << source;
}

/**
 * @brief Kernels already resolved by this process, keyed by (entry, kernel body) so the
 *        evaluation-time cost is proportional to the generated body rather than to the ~50 KB
 *        preamble.
 *
 * @note The table is subordinate to @c KernelCache: it is dropped whenever @c KernelCache::clear()
 *       advances the generation, so a cleared cache compiles again exactly as before.
 */
class compiled_sink_table {
 public:
  std::shared_ptr<codegen::jit::CompiledKernel const> find(std::string_view entry,
                                                           std::string const& body,
                                                           int arch_cc,
                                                           std::uint64_t generation)
  {
    std::lock_guard<std::mutex> const lock(_mutex);
    if (!adopt_generation(generation)) { return nullptr; }
    auto const it = _entries.find(key(entry, body, arch_cc));
    return it == _entries.end() ? nullptr : it->second;
  }

  void insert(std::string_view entry,
              std::string const& body,
              int arch_cc,
              std::uint64_t generation,
              std::shared_ptr<codegen::jit::CompiledKernel const> kernel)
  {
    std::lock_guard<std::mutex> const lock(_mutex);
    if (!adopt_generation(generation)) { return; }
    _entries.emplace(key(entry, body, arch_cc), std::move(kernel));
  }

 private:
  // Generations only move forward: a caller that observed an older generation than the table
  // raced with a clear() and must neither read nor publish through the table.
  bool adopt_generation(std::uint64_t generation)
  {
    if (generation < _generation) { return false; }
    if (generation > _generation) {
      _entries.clear();
      _generation = generation;
    }
    return true;
  }

  static std::string key(std::string_view entry, std::string const& body, int arch_cc)
  {
    std::string k = std::to_string(arch_cc);
    k.reserve(k.size() + entry.size() + 2 + body.size());
    k.push_back('\0');
    k.append(entry).push_back('\0');
    return k.append(body);
  }

  std::mutex _mutex;
  std::uint64_t _generation{0};
  std::unordered_map<std::string, std::shared_ptr<codegen::jit::CompiledKernel const>> _entries;
};

compiled_sink_table& sink_table()
{
  static compiled_sink_table table;
  return table;
}

/**
 * @brief Kernel handles whose launches may still be in flight, each paired with an event recorded
 *        after the launch.
 *
 * Completed entries are released on the next retain, so the list stays as short as the number of
 * concurrently executing sink kernels. Events are pooled; process-exit destruction may run after
 * the CUDA context is gone, so it ignores the CUDA results.
 */
class kernel_retire_list {
 public:
  ~kernel_retire_list()
  {
    for (auto& e : _pending) {
      (void)cudaEventDestroy(e.event);
    }
    for (auto& [device, events] : _free_events) {
      for (auto ev : events) {
        (void)cudaEventDestroy(ev);
      }
    }
  }

  void retain(std::shared_ptr<codegen::jit::CompiledKernel const> kernel, ::cuda::stream_ref stream)
  {
    // An event may only be recorded on streams of the device it was created on, so the free
    // lists are per device; the calling thread's device is the stream's device.
    int device = 0;
    simpatico::throw_if_cuda_error(cudaGetDevice(&device), "cudaGetDevice(kernel retire)");
    std::lock_guard<std::mutex> const lock(_mutex);
    std::erase_if(_pending, [&](entry& e) {
      if (cudaEventQuery(e.event) != cudaSuccess) { return false; }
      _free_events[e.device].push_back(e.event);
      return true;
    });
    auto& free_events = _free_events[device];
    cudaEvent_t event = nullptr;
    if (!free_events.empty()) {
      event = free_events.back();
      free_events.pop_back();
    } else {
      simpatico::throw_if_cuda_error(cudaEventCreateWithFlags(&event, cudaEventDisableTiming),
                                     "cudaEventCreate(kernel retire)");
    }
    if (auto const rc = cudaEventRecord(event, stream.get()); rc != cudaSuccess) {
      free_events.push_back(event);
      simpatico::throw_if_cuda_error(rc, "cudaEventRecord(kernel retire)");
    }
    _pending.push_back(entry{event, device, std::move(kernel)});
  }

 private:
  struct entry {
    cudaEvent_t event;
    int device;
    std::shared_ptr<codegen::jit::CompiledKernel const> kernel;
  };
  std::mutex _mutex;
  std::vector<entry> _pending;
  std::unordered_map<int, std::vector<cudaEvent_t>> _free_events;
};

kernel_retire_list& retire_list()
{
  static kernel_retire_list list;
  return list;
}

}  // namespace

void emit_kernel_head(std::string& out, std::string_view entry, kernel_abi const& abi)
{
  out += std::format("extern \"C\" __global__ void {}", entry);
  abi.emit_parameter_list(out);
  out += "{\n";
  abi.emit_kernel_prelude(out);
}

void emit_item_loop_open(std::string& out, tile_config tile)
{
  out += std::format(
    "  #pragma unroll\n"
    "  for (int item = 0; item < {}; ++item) {{\n"
    "    int32_t const row = tile_row0 + item * {} + static_cast<int32_t>(threadIdx.x);\n",
    tile.items_per_thread,
    tile.block_threads);
}

void emit_status_flag_or(std::string& out,
                         std::string_view flag_var,
                         std::string_view status_array,
                         std::string_view status_bit)
{
  out += std::format(
    "  if (__ballot_sync(0xFFFFFFFFu, {0}) != 0u && lane == 0) {{\n"
    "    atomicOr(&{1}[0], {2});\n"
    "  }}\n",
    flag_var,
    status_array,
    status_bit);
}

std::shared_ptr<codegen::jit::CompiledKernel const> compile_sink_kernel(
  std::string_view entry, std::string const& kernel_body, bool prepend_device_runtime)
{
  auto const arch_cc    = cached_arch_cc_for_current_device();
  auto const generation = codegen::jit::KernelCache::instance().generation();
  if (auto found = sink_table().find(entry, kernel_body, arch_cc, generation)) { return found; }

  std::shared_ptr<codegen::jit::CompiledKernel const> kernel;
  if (prepend_device_runtime) {
    auto const runtime = nvrtc_device_runtime_source();
    std::string source;
    source.reserve(runtime.size() + kernel_body.size());
    source.append(runtime).append(kernel_body);
    kernel = compile_kernel_source(entry, source);
  } else {
    kernel = compile_kernel_source(entry, kernel_body);
  }
  sink_table().insert(entry, kernel_body, arch_cc, generation, kernel);
  return kernel;
}

std::shared_ptr<codegen::jit::CompiledKernel const> compile_kernel_source(std::string_view entry,
                                                                          std::string const& source)
{
  codegen::jit::CompileOptions opts;
  opts.arch_cc       = cached_arch_cc_for_current_device();
  opts.device_int128 = true;
  maybe_dump_kernel_source(entry, source);
  try {
    return codegen::jit::KernelCache::instance().get_or_compile_plain(
      source, std::string(entry), opts);
  } catch (codegen::jit::CompileError const& err) {
    throw internal_exception("[nvrtc] {}\nLog:\n{}\nSource:\n{}", err.what(), err.log, err.source);
  }
}

void launch_sink_kernel(CUfunction fn,
                        unsigned grid_dim_x,
                        unsigned block_dim_x,
                        ::cuda::stream_ref stream,
                        void** args,
                        char const* what)
{
  simpatico::throw_if_cu_error(cuLaunchKernel(fn,
                                              grid_dim_x,
                                              1,
                                              1,
                                              block_dim_x,
                                              1,
                                              1,
                                              0,
                                              reinterpret_cast<CUstream>(stream.get()),
                                              args,
                                              nullptr),
                               what);
}

void retain_until_stream_completes(std::shared_ptr<codegen::jit::CompiledKernel const> kernel,
                                   ::cuda::stream_ref stream)
{
  retire_list().retain(std::move(kernel), stream);
}

void read_status_words(std::span<std::uint32_t> host,
                       std::uint32_t const* device,
                       ::cuda::stream_ref stream)
{
  simpatico::read_device_bytes_completed(host.data(), device, host.size_bytes(), stream);
  if ((host[0] & abi::status_subquery_error_bit) != 0U) {
    throw invalid_input_exception(
      "[expression_evaluator:case]: More than one row returned by a subquery used as an "
      "expression.");
  }
  if ((host[0] & abi::status_overflow_bit) != 0U) {
    throw out_of_range_exception("[expression_evaluator]: Overflow in NVRTC evaluation");
  }
}

}  // namespace sirius::nvrtc
