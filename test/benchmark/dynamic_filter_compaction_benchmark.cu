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
 * See the License for the specific language governing permissions and limitations under
 * the License.
 */

// Compares three ways to apply two independent dynamic membership filters:
//   A: compact the whole table after each mask (current/origin-dev behavior)
//   B: compact only key B (+ row ids) after mask A, then gather non-B columns once
//   C: probe both original key columns, AND the masks, compact the whole table once

#include <cudf/binaryop.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/copying.hpp>
#include <cudf/filling.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/table/table.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/cuda_stream.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/mr/cuda_async_memory_resource.hpp>
#include <rmm/mr/per_device_resource.hpp>

#include <cuda/memory_resource>
#include <cuda_profiler_api.h>
#include <cuda_runtime.h>

#include <cudf/cudf_utils.hpp>
#include <op/dynamic_filter/sirius_dynamic_filter.hpp>
#include <telemetry/nvtx.hpp>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

using sirius::op::sirius_dynamic_bloom_filter;
using sirius::op::sirius_dynamic_in_list_filter;
using sirius::op::sirius_mask_applicable;

enum class filter_kind { in_list, bloom };

struct options {
  cudf::size_type rows         = 32 * 1024 * 1024;
  cudf::size_type build_keys   = 1024 * 1024;
  cudf::size_type domain       = 4 * 1024 * 1024;
  int payload_columns          = 8;
  int string_bytes             = 0;  // 0: all INT64 payloads; >0: append one STRING payload
  int iterations               = 8;
  int warmup                   = 2;
  std::string strategy         = "all";
  filter_kind kind             = filter_kind::in_list;
  bool profile                 = false;
};

struct strategy_result {
  std::unique_ptr<cudf::table> output;
  cudf::size_type after_a = 0;
  cudf::size_type final   = 0;
};

__device__ std::uint64_t mix64(std::uint64_t x)
{
  x ^= x >> 30;
  x *= 0xbf58476d1ce4e5b9ULL;
  x ^= x >> 27;
  x *= 0x94d049bb133111ebULL;
  return x ^ (x >> 31);
}

__global__ void fill_probe_columns(std::int64_t* a,
                                   std::int64_t* b,
                                   std::int64_t** payloads,
                                   int payload_columns,
                                   cudf::size_type rows,
                                   cudf::size_type domain)
{
  auto const tid    = static_cast<cudf::size_type>(blockIdx.x * blockDim.x + threadIdx.x);
  auto const stride = static_cast<cudf::size_type>(blockDim.x * gridDim.x);
  for (auto i = tid; i < rows; i += stride) {
    auto const x = static_cast<std::uint64_t>(i);
    a[i]         = static_cast<std::int64_t>(mix64(x + 0x243f6a8885a308d3ULL) % domain);
    b[i]         = static_cast<std::int64_t>(mix64(x + 0x13198a2e03707344ULL) % domain);
    for (int col = 0; col < payload_columns; ++col) {
      payloads[col][i] =
        static_cast<std::int64_t>(mix64(x + static_cast<std::uint64_t>(col + 1) *
                                             0x9e3779b97f4a7c15ULL));
    }
  }
}

// Fixed-length printable payload, one row = `nchars` bytes. Cheap stand-in for a comment
// column: gather copies chars+offsets, not 8-byte INT64s.
__global__ void fill_string_chars(char* chars, int nchars, cudf::size_type rows)
{
  auto const tid    = static_cast<cudf::size_type>(blockIdx.x * blockDim.x + threadIdx.x);
  auto const stride = static_cast<cudf::size_type>(blockDim.x * gridDim.x);
  for (auto i = tid; i < rows; i += stride) {
    auto const base = static_cast<std::uint64_t>(i) * static_cast<std::uint64_t>(nchars);
    auto word       = mix64(static_cast<std::uint64_t>(i) + 0xa5a5a5a5a5a5a5a5ULL);
    for (int c = 0; c < nchars; ++c) {
      if ((c & 7) == 0 && c != 0) {
        word = mix64(word + static_cast<std::uint64_t>(c));
      }
      chars[base + static_cast<std::uint64_t>(c)] =
        static_cast<char>('a' + ((word >> ((c & 7) * 8)) % 26));
    }
  }
}

std::unique_ptr<cudf::table> make_probe_table(options const& opts,
                                              rmm::cuda_stream_view stream,
                                              rmm::device_async_resource_ref mr)
{
  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.reserve(static_cast<std::size_t>(opts.payload_columns + 2));
  for (int i = 0; i < opts.payload_columns + 2; ++i) {
    columns.push_back(cudf::make_numeric_column(cudf::data_type{cudf::type_id::INT64},
                                                opts.rows,
                                                cudf::mask_state::UNALLOCATED,
                                                stream,
                                                mr));
  }

  std::vector<std::int64_t*> payload_ptrs;
  payload_ptrs.reserve(static_cast<std::size_t>(opts.payload_columns));
  for (int i = 0; i < opts.payload_columns; ++i) {
    payload_ptrs.push_back(columns[static_cast<std::size_t>(i + 2)]
                             ->mutable_view()
                             .data<std::int64_t>());
  }
  rmm::device_uvector<std::int64_t*> device_payload_ptrs(payload_ptrs.size(), stream);
  cudaMemcpyAsync(device_payload_ptrs.data(),
                  payload_ptrs.data(),
                  payload_ptrs.size() * sizeof(std::int64_t*),
                  cudaMemcpyHostToDevice,
                  stream.value());

  constexpr int block_size = 256;
  auto const blocks =
    std::min<int>(65535, (static_cast<int64_t>(opts.rows) + block_size - 1) / block_size);
  fill_probe_columns<<<blocks, block_size, 0, stream.value()>>>(
    columns[0]->mutable_view().data<std::int64_t>(),
    columns[1]->mutable_view().data<std::int64_t>(),
    device_payload_ptrs.data(),
    opts.payload_columns,
    opts.rows,
    opts.domain);
  auto const status = cudaGetLastError();
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string{"fill_probe_columns failed: "} +
                             cudaGetErrorString(status));
  }

  if (opts.string_bytes > 0) {
    auto const nchars = static_cast<std::int64_t>(opts.rows) * opts.string_bytes;
    if (nchars > std::numeric_limits<cudf::size_type>::max()) {
      throw std::invalid_argument("rows * --string-bytes exceeds INT32 string offsets");
    }
    auto offsets = cudf::sequence(
      opts.rows + 1,
      cudf::numeric_scalar<cudf::size_type>{0, true, stream, mr},
      cudf::numeric_scalar<cudf::size_type>{
        static_cast<cudf::size_type>(opts.string_bytes), true, stream, mr},
      stream,
      mr);
    rmm::device_buffer chars(static_cast<std::size_t>(nchars), stream, mr);
    fill_string_chars<<<blocks, block_size, 0, stream.value()>>>(
      static_cast<char*>(chars.data()), opts.string_bytes, opts.rows);
    auto const str_status = cudaGetLastError();
    if (str_status != cudaSuccess) {
      throw std::runtime_error(std::string{"fill_string_chars failed: "} +
                               cudaGetErrorString(str_status));
    }
    columns.push_back(cudf::make_strings_column(
      opts.rows, std::move(offsets), std::move(chars), 0, rmm::device_buffer{0, stream, mr}));
  }
  stream.synchronize();
  return std::make_unique<cudf::table>(std::move(columns));
}

std::unique_ptr<cudf::column> make_build_keys(options const& opts,
                                              rmm::cuda_stream_view stream,
                                              rmm::device_async_resource_ref mr)
{
  return cudf::sequence(opts.build_keys,
                        cudf::numeric_scalar<std::int64_t>{0, true, stream, mr},
                        cudf::numeric_scalar<std::int64_t>{1, true, stream, mr},
                        stream,
                        mr);
}

strategy_result strategy_a(cudf::table_view input,
                           sirius_mask_applicable const& filter_a,
                           sirius_mask_applicable const& filter_b,
                           rmm::cuda_stream_view stream,
                           rmm::device_async_resource_ref mr)
{
  sirius::nvtx_scoped_range range{"compaction_bench::A_full_cascade"};
  auto mask_a = filter_a.compute_mask(input.column(0), -1, stream, mr);
  auto after_a = sirius::ApplyRetentionMask(input, mask_a->view(), stream, mr);
  auto mask_b  = filter_b.compute_mask(after_a->view().column(1), -1, stream, mr);
  auto output  = sirius::ApplyRetentionMask(after_a->view(), mask_b->view(), stream, mr);
  auto const final_rows = output->num_rows();
  return {std::move(output), after_a->num_rows(), final_rows};
}

strategy_result strategy_b(cudf::table_view input,
                           sirius_mask_applicable const& filter_a,
                           sirius_mask_applicable const& filter_b,
                           rmm::cuda_stream_view stream,
                           rmm::device_async_resource_ref mr)
{
  sirius::nvtx_scoped_range range{"compaction_bench::B_compact_key_then_stitch"};
  auto mask_a = filter_a.compute_mask(input.column(0), -1, stream, mr);
  auto row_ids =
    cudf::sequence(input.num_rows(),
                   cudf::numeric_scalar<cudf::size_type>{0, true, stream, mr},
                   cudf::numeric_scalar<cudf::size_type>{1, true, stream, mr},
                   stream,
                   mr);

  auto compact_b_and_ids = sirius::ApplyRetentionMask(
    cudf::table_view{{input.column(1), row_ids->view()}}, mask_a->view(), stream, mr);
  auto mask_b =
    filter_b.compute_mask(compact_b_and_ids->view().column(0), -1, stream, mr);
  auto final_b_and_ids =
    sirius::ApplyRetentionMask(compact_b_and_ids->view(), mask_b->view(), stream, mr);

  std::vector<cudf::size_type> non_b_indices;
  non_b_indices.reserve(static_cast<std::size_t>(input.num_columns() - 1));
  non_b_indices.push_back(0);
  for (cudf::size_type i = 2; i < input.num_columns(); ++i) {
    non_b_indices.push_back(i);
  }
  auto non_b = cudf::gather(input.select(non_b_indices),
                            final_b_and_ids->view().column(1),
                            cudf::out_of_bounds_policy::DONT_CHECK,
                            stream,
                            mr);

  auto b_and_ids_columns = final_b_and_ids->release();
  auto non_b_columns      = non_b->release();
  std::vector<std::unique_ptr<cudf::column>> output_columns;
  output_columns.reserve(static_cast<std::size_t>(input.num_columns()));
  output_columns.push_back(std::move(non_b_columns[0]));
  output_columns.push_back(std::move(b_and_ids_columns[0]));
  for (std::size_t i = 1; i < non_b_columns.size(); ++i) {
    output_columns.push_back(std::move(non_b_columns[i]));
  }
  auto const after_a = compact_b_and_ids->num_rows();
  auto output        = std::make_unique<cudf::table>(std::move(output_columns));
  auto const final_rows = output->num_rows();
  return {std::move(output), after_a, final_rows};
}

strategy_result strategy_c(cudf::table_view input,
                           sirius_mask_applicable const& filter_a,
                           sirius_mask_applicable const& filter_b,
                           rmm::cuda_stream_view stream,
                           rmm::device_async_resource_ref mr)
{
  sirius::nvtx_scoped_range range{"compaction_bench::C_probe_both_gather_once"};
  auto mask_a = filter_a.compute_mask(input.column(0), -1, stream, mr);
  auto mask_b = filter_b.compute_mask(input.column(1), -1, stream, mr);
  auto final_mask =
    cudf::binary_operation(mask_a->view(),
                           mask_b->view(),
                           cudf::binary_operator::LOGICAL_AND,
                           cudf::data_type{cudf::type_id::BOOL8},
                           stream,
                           mr);
  auto output = sirius::ApplyRetentionMask(input, final_mask->view(), stream, mr);
  auto const final_rows = output->num_rows();
  return {std::move(output), -1, final_rows};
}

strategy_result run_strategy(char strategy,
                             cudf::table_view input,
                             sirius_mask_applicable const& filter_a,
                             sirius_mask_applicable const& filter_b,
                             rmm::cuda_stream_view stream,
                             rmm::device_async_resource_ref mr)
{
  switch (strategy) {
    case 'a': return strategy_a(input, filter_a, filter_b, stream, mr);
    case 'b': return strategy_b(input, filter_a, filter_b, stream, mr);
    case 'c': return strategy_c(input, filter_a, filter_b, stream, mr);
    default: throw std::invalid_argument("strategy must be a, b, or c");
  }
}

std::vector<char> selected_strategies(std::string const& strategy)
{
  if (strategy == "all") return {'a', 'b', 'c'};
  if (strategy.size() == 1 && strategy[0] >= 'a' && strategy[0] <= 'c') return {strategy[0]};
  throw std::invalid_argument("--strategy must be all, a, b, or c");
}

long long parse_integer(char const* value, char const* option)
{
  char* end = nullptr;
  auto parsed = std::strtoll(value, &end, 10);
  if (end == value || *end != '\0' || parsed <= 0) {
    throw std::invalid_argument(std::string{option} + " requires a positive integer");
  }
  return parsed;
}

options parse_options(int argc, char** argv)
{
  options result;
  for (int i = 1; i < argc; ++i) {
    auto next = [&](char const* option) -> char const* {
      if (++i >= argc) throw std::invalid_argument(std::string{option} + " requires a value");
      return argv[i];
    };
    std::string_view arg{argv[i]};
    if (arg == "--rows") {
      result.rows = static_cast<cudf::size_type>(parse_integer(next("--rows"), "--rows"));
    } else if (arg == "--build-keys") {
      result.build_keys =
        static_cast<cudf::size_type>(parse_integer(next("--build-keys"), "--build-keys"));
    } else if (arg == "--domain") {
      result.domain = static_cast<cudf::size_type>(parse_integer(next("--domain"), "--domain"));
    } else if (arg == "--payload-columns") {
      result.payload_columns =
        static_cast<int>(parse_integer(next("--payload-columns"), "--payload-columns"));
    } else if (arg == "--string-bytes") {
      result.string_bytes =
        static_cast<int>(parse_integer(next("--string-bytes"), "--string-bytes"));
    } else if (arg == "--iterations") {
      result.iterations = static_cast<int>(parse_integer(next("--iterations"), "--iterations"));
    } else if (arg == "--warmup") {
      result.warmup = static_cast<int>(parse_integer(next("--warmup"), "--warmup"));
    } else if (arg == "--strategy") {
      result.strategy = next("--strategy");
    } else if (arg == "--filter-kind") {
      auto const kind = std::string{next("--filter-kind")};
      if (kind == "in-list" || kind == "in_list") {
        result.kind = filter_kind::in_list;
      } else if (kind == "bloom") {
        result.kind = filter_kind::bloom;
      } else {
        throw std::invalid_argument("--filter-kind must be in-list or bloom");
      }
    } else if (arg == "--profile") {
      result.profile = true;
    } else {
      throw std::invalid_argument("unknown option: " + std::string{arg});
    }
  }
  if (result.build_keys > result.domain) {
    throw std::invalid_argument("--build-keys must not exceed --domain");
  }
  return result;
}

void print_header(options const& opts)
{
  auto const selectivity = static_cast<double>(opts.build_keys) / opts.domain;
  auto const row_bytes =
    static_cast<std::size_t>(opts.payload_columns + 2) * sizeof(int64_t) +
    (opts.string_bytes > 0
       ? static_cast<std::size_t>(opts.string_bytes) + sizeof(cudf::size_type)
       : 0);
  std::cerr << "rows=" << opts.rows << " build_keys=" << opts.build_keys
            << " domain=" << opts.domain << " expected_selectivity=" << selectivity
            << " payload_columns=" << opts.payload_columns << " string_bytes=" << opts.string_bytes
            << " row_bytes=" << row_bytes
            << " filter_kind=" << (opts.kind == filter_kind::bloom ? "bloom" : "in-list") << '\n';
  std::cout << "strategy,iteration,gpu_ms,wall_ms,after_a_rows,final_rows\n";
}

}  // namespace

int main(int argc, char** argv)
{
  try {
    auto const opts = parse_options(argc, argv);
    cudaFree(nullptr);

    rmm::mr::cuda_async_memory_resource async_mr;
    rmm::mr::set_current_device_resource(
      cuda::mr::any_resource<cuda::mr::device_accessible>{
        rmm::device_async_resource_ref{async_mr}});
    auto mr = rmm::device_async_resource_ref{async_mr};
    rmm::cuda_stream stream{rmm::cuda_stream::flags::non_blocking};

    auto probe      = make_probe_table(opts, stream.view(), mr);
    auto build_keys = make_build_keys(opts, stream.view(), mr);
    std::unique_ptr<sirius_mask_applicable> filter_a;
    std::unique_ptr<sirius_mask_applicable> filter_b;
    if (opts.kind == filter_kind::bloom) {
      filter_a = std::make_unique<sirius_dynamic_bloom_filter>(build_keys->view(), stream.view(), mr);
      filter_b = std::make_unique<sirius_dynamic_bloom_filter>(build_keys->view(), stream.view(), mr);
    } else {
      filter_a =
        std::make_unique<sirius_dynamic_in_list_filter>(build_keys->view(), stream.view(), mr);
      filter_b =
        std::make_unique<sirius_dynamic_in_list_filter>(build_keys->view(), stream.view(), mr);
    }
    stream.synchronize();

    print_header(opts);
    auto const strategies = selected_strategies(opts.strategy);
    for (auto const strategy : strategies) {
      for (int i = 0; i < opts.warmup; ++i) {
        auto result =
          run_strategy(strategy, probe->view(), *filter_a, *filter_b, stream.view(), mr);
        stream.synchronize();
        if (!result.output) throw std::runtime_error("strategy produced no output");
      }
    }

    if (opts.profile) cudaProfilerStart();
    for (auto const strategy : strategies) {
      auto const iterations = opts.profile ? 1 : opts.iterations;
      for (int iteration = 0; iteration < iterations; ++iteration) {
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        auto const wall_start = std::chrono::steady_clock::now();
        cudaEventRecord(start, stream.value());
        auto result =
          run_strategy(strategy, probe->view(), *filter_a, *filter_b, stream.view(), mr);
        cudaEventRecord(stop, stream.value());
        cudaEventSynchronize(stop);
        auto const wall_stop = std::chrono::steady_clock::now();
        float gpu_ms         = 0.0F;
        cudaEventElapsedTime(&gpu_ms, start, stop);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        auto const wall_ms =
          std::chrono::duration<double, std::milli>(wall_stop - wall_start).count();
        std::cout << strategy << ',' << iteration << ',' << std::fixed << std::setprecision(3)
                  << gpu_ms << ',' << wall_ms << ',' << result.after_a << ',' << result.final
                  << '\n';
      }
    }
    if (opts.profile) cudaProfilerStop();
    stream.synchronize();
    return 0;
  } catch (std::exception const& e) {
    std::cerr << "ERROR: " << e.what() << '\n';
    return 1;
  }
}
