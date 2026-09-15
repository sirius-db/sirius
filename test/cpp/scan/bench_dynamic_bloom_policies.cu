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

#include <cuco/bloom_filter.cuh>
#include <cuco/bloom_filter_policies.cuh>
#include <cuco/hash_functions.cuh>
#include <cuda/std/array>
#include <cuda/std/cstdint>
#include <cuda/std/limits>
#include <cuda/std/utility>
#include <cuda_runtime.h>
#include <thrust/count.h>
#include <thrust/device_vector.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/system/cuda/execution_policy.h>

#include <catch.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <vector>

namespace {

template <class KeyT>
class legacy_sirius_bloom_policy {
 public:
  using hasher    = cuco::xxhash_64<KeyT>;
  using word_type = std::uint32_t;
  using hash_type = std::uint64_t;

  static constexpr std::uint32_t words_per_block            = 8;
  static constexpr std::uint32_t pattern_bits               = 8;
  static constexpr std::uint32_t add_horizontal_layout      = 1;
  static constexpr std::uint32_t add_vertical_layout        = 8;
  static constexpr std::uint32_t contains_horizontal_layout = 1;
  static constexpr std::uint32_t contains_vertical_layout   = 8;
  static constexpr bool conditional_add                     = false;
  static constexpr bool early_exit_contains                 = false;
  static constexpr std::size_t max_filter_blocks = cuda::std::numeric_limits<std::uint32_t>::max();

  __host__ __device__ constexpr legacy_sirius_bloom_policy(hasher hash = {}) : hash_{hash} {}

  template <class Key>
  __device__ constexpr cuda::std::pair<hash_type, hash_type> split_hash(Key const& key) const
  {
    auto const hash = hash_(key);
    return {hash, hash};
  }

  template <class Extent>
  __device__ constexpr std::uint32_t block_index(hash_type hash, Extent num_blocks) const
  {
    auto const wide =
      static_cast<__uint128_t>(hash) *
      static_cast<__uint128_t>(static_cast<typename Extent::value_type>(num_blocks));
    return static_cast<std::uint32_t>(static_cast<std::uint64_t>(wide >> 64));
  }

  template <std::uint32_t LoopIndex, std::uint32_t VerticalLayout>
  __device__ constexpr cuda::std::array<word_type, VerticalLayout> array_pattern(
    hash_type hash) const
  {
    static_assert(LoopIndex == 0);
    static_assert(VerticalLayout == words_per_block);
    cuda::std::array<word_type, VerticalLayout> pattern{};
#pragma unroll
    for (std::uint32_t word = 0; word < words_per_block; ++word) {
      pattern[word] = word_type{1} << ((hash >> (word * 5)) & 31);
    }
    return pattern;
  }

 private:
  hasher hash_{};
};

using key_type       = std::int64_t;
using extent_type    = cuco::extent<std::size_t>;
using legacy_filter  = cuco::bloom_filter<key_type,
                                          extent_type,
                                          cuda::thread_scope_device,
                                          legacy_sirius_bloom_policy<key_type>>;
using default_filter = cuco::bloom_filter<key_type,
                                          extent_type,
                                          cuda::thread_scope_device,
                                          cuco::default_filter_policy<key_type>>;

class cuda_event {
 public:
  cuda_event() { REQUIRE(cudaEventCreate(&_event) == cudaSuccess); }
  ~cuda_event() { cudaEventDestroy(_event); }

  cuda_event(cuda_event const&)            = delete;
  cuda_event& operator=(cuda_event const&) = delete;

  operator cudaEvent_t() const { return _event; }

 private:
  cudaEvent_t _event{};
};

template <class Function>
float time_ms(cudaStream_t stream, Function&& function)
{
  cuda_event start;
  cuda_event stop;
  REQUIRE(cudaEventRecord(start, stream) == cudaSuccess);
  function();
  REQUIRE(cudaEventRecord(stop, stream) == cudaSuccess);
  REQUIRE(cudaEventSynchronize(stop) == cudaSuccess);
  float elapsed = 0.0F;
  REQUIRE(cudaEventElapsedTime(&elapsed, start, stop) == cudaSuccess);
  return elapsed;
}

float median(std::vector<float> values)
{
  auto const middle = values.begin() + static_cast<std::ptrdiff_t>(values.size() / 2);
  std::nth_element(values.begin(), middle, values.end());
  return *middle;
}

template <class Filter>
float sample_add(Filter& filter,
                 thrust::counting_iterator<key_type> first,
                 std::size_t num_keys,
                 cuda::stream_ref stream)
{
  filter.clear(stream);
  return time_ms(stream.get(), [&] { filter.add_async(first, first + num_keys, stream); });
}

template <class Filter>
float sample_contains(Filter const& filter,
                      thrust::counting_iterator<key_type> first,
                      std::size_t num_keys,
                      thrust::device_vector<bool>& output,
                      cuda::stream_ref stream)
{
  return time_ms(stream.get(),
                 [&] { filter.contains_async(first, first + num_keys, output.begin(), stream); });
}

template <class Filter>
double false_positive_rate(Filter const& filter,
                           thrust::counting_iterator<key_type> first,
                           std::size_t num_keys,
                           thrust::device_vector<bool>& output,
                           cuda::stream_ref stream)
{
  filter.contains(first, first + num_keys, output.begin(), stream);
  auto const false_positives =
    thrust::count(thrust::cuda::par.on(stream.get()), output.begin(), output.end(), true);
  REQUIRE(cudaStreamSynchronize(stream.get()) == cudaSuccess);
  return static_cast<double>(false_positives) / static_cast<double>(num_keys);
}

double throughput(std::size_t num_keys, float milliseconds)
{ return static_cast<double>(num_keys) / (static_cast<double>(milliseconds) * 1.0e6); }

}  // namespace

TEST_CASE("benchmark legacy and default dynamic Bloom policies", "[!benchmark][bloom_policy_bench]")
{
  constexpr int warmup_iterations  = 3;
  constexpr int samples            = 21;
  constexpr std::size_t probe_keys = 16U << 20;

  cudaStream_t raw_stream{};
  REQUIRE(cudaStreamCreateWithFlags(&raw_stream, cudaStreamNonBlocking) == cudaSuccess);
  cuda::stream_ref const stream{raw_stream};
  thrust::counting_iterator<key_type> const build_begin{0};
  thrust::device_vector<bool> output(probe_keys);

  std::printf("policy,num_keys,filter_mib,add_gelem_s,contains_gelem_s,false_positive_rate\n");

  for (auto const num_keys :
       {std::size_t{1} << 10, std::size_t{1} << 16, std::size_t{1} << 20, std::size_t{1} << 23}) {
    auto const num_blocks = (num_keys + 15) / 16;
    auto const filter_mib = static_cast<double>(num_blocks * 32) / static_cast<double>(1U << 20);

    legacy_filter legacy{extent_type{num_blocks}, {}, {}, {}, stream};
    default_filter current{extent_type{num_blocks}, {}, {}, {}, stream};
    auto const probe_begin = build_begin + num_keys;

    for (int i = 0; i < warmup_iterations; ++i) {
      (void)sample_add(legacy, build_begin, num_keys, stream);
      (void)sample_add(current, build_begin, num_keys, stream);
      (void)sample_contains(legacy, probe_begin, probe_keys, output, stream);
      (void)sample_contains(current, probe_begin, probe_keys, output, stream);
    }

    std::vector<float> legacy_add;
    std::vector<float> current_add;
    std::vector<float> legacy_contains;
    std::vector<float> current_contains;
    legacy_add.reserve(samples);
    current_add.reserve(samples);
    legacy_contains.reserve(samples);
    current_contains.reserve(samples);

    for (int i = 0; i < samples; ++i) {
      if (i % 2 == 0) {
        legacy_add.push_back(sample_add(legacy, build_begin, num_keys, stream));
        current_add.push_back(sample_add(current, build_begin, num_keys, stream));
        legacy_contains.push_back(sample_contains(legacy, probe_begin, probe_keys, output, stream));
        current_contains.push_back(
          sample_contains(current, probe_begin, probe_keys, output, stream));
      } else {
        current_add.push_back(sample_add(current, build_begin, num_keys, stream));
        legacy_add.push_back(sample_add(legacy, build_begin, num_keys, stream));
        current_contains.push_back(
          sample_contains(current, probe_begin, probe_keys, output, stream));
        legacy_contains.push_back(sample_contains(legacy, probe_begin, probe_keys, output, stream));
      }
    }

    auto const legacy_fpr  = false_positive_rate(legacy, probe_begin, probe_keys, output, stream);
    auto const current_fpr = false_positive_rate(current, probe_begin, probe_keys, output, stream);
    auto const legacy_add_ms       = median(legacy_add);
    auto const current_add_ms      = median(current_add);
    auto const legacy_contains_ms  = median(legacy_contains);
    auto const current_contains_ms = median(current_contains);

    std::printf("legacy,%zu,%.6f,%.6f,%.6f,%.9f\n",
                num_keys,
                filter_mib,
                throughput(num_keys, legacy_add_ms),
                throughput(probe_keys, legacy_contains_ms),
                legacy_fpr);
    std::printf("default,%zu,%.6f,%.6f,%.6f,%.9f\n",
                num_keys,
                filter_mib,
                throughput(num_keys, current_add_ms),
                throughput(probe_keys, current_contains_ms),
                current_fpr);

    auto build_output = thrust::device_vector<bool>(num_keys);
    legacy.contains(build_begin, build_begin + num_keys, build_output.begin(), stream);
    REQUIRE(thrust::count(
              thrust::cuda::par.on(stream.get()), build_output.begin(), build_output.end(), true) ==
            num_keys);
    current.contains(build_begin, build_begin + num_keys, build_output.begin(), stream);
    REQUIRE(thrust::count(
              thrust::cuda::par.on(stream.get()), build_output.begin(), build_output.end(), true) ==
            num_keys);
  }

  REQUIRE(cudaStreamDestroy(raw_stream) == cudaSuccess);
}
