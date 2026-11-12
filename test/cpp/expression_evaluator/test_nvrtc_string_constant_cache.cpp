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

#include "ast_test_support.hpp"
#include "codegen/jit/kernel_cache.hpp"

#include <cudf/column/column_factories.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/aligned.hpp>
#include <rmm/cuda_device.hpp>
#include <rmm/cuda_stream.hpp>
#include <rmm/device_buffer.hpp>

#include <cuda_runtime_api.h>

#include <catch.hpp>
#include <cucascade/memory/memory_space.hpp>
#include <cucascade/memory/reservation_aware_resource_adaptor.hpp>
#include <cucascade/memory/reservation_manager_configurator.hpp>
#include <expression_evaluator/expression_evaluator.hpp>
#include <expression_evaluator/nvrtc/string_constant_cache.hpp>
#include <expression_evaluator/query_caches.hpp>
#include <memory/sirius_memory_reservation_manager.hpp>
#include <sirius/exception.hpp>

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <format>
#include <latch>
#include <optional>
#include <set>
#include <string>
#include <thread>
#include <vector>

using sirius::nvrtc::string_constant_cache;
using namespace sirius::expr_test;

namespace {

/** Copies @p size device bytes at @p device_bytes to the host in the order of @p stream. */
std::string read_back(char const* device_bytes, std::size_t size, rmm::cuda_stream const& stream)
{
  std::string host(size, '\0');
  REQUIRE(
    cudaMemcpyAsync(host.data(), device_bytes, size, cudaMemcpyDeviceToHost, stream.value()) ==
    cudaSuccess);
  stream.synchronize();
  return host;
}

/** A pool with an embedded NUL, a UTF-8 sequence and a payload longer than a few words. */
std::string const mixed_pool =
  std::string("a\0b", 3) + "\xC3\xA9t\xC3\xA9" + std::string(4096, 'x');

/**
 * A memory manager configured like the engine's, with per-thread reservation tracking and a
 * stream-ordered allocator on GPU 0.
 */
std::unique_ptr<sirius::memory::sirius_memory_reservation_manager> make_engine_like_manager()
{
  cucascade::memory::reservation_manager_configurator builder;
  builder.set_number_of_gpus(1)
    .set_gpu_usage_limit(std::size_t{256} << 20)
    .set_reservation_fraction_per_gpu(0.75)
    .set_per_numa_region_capacity(std::size_t{256} << 20)
    .use_gpu_id_as_host_id()
    .set_reservation_fraction_per_numa_region(0.75)
    .track_reservation_per_stream(false);
  return std::make_unique<sirius::memory::sirius_memory_reservation_manager>(builder.build());
}

/** Host work that holds a stream until @c open is called or ten seconds pass. */
class stream_gate {
 public:
  explicit stream_gate(rmm::cuda_stream const& stream)
  {
    REQUIRE(cudaLaunchHostFunc(stream.value(), &stream_gate::wait, this) == cudaSuccess);
  }
  void open() { _open.store(true); }
  [[nodiscard]] bool passed() const { return _passed.load(); }

 private:
  static void wait(void* self)
  {
    auto* const gate      = static_cast<stream_gate*>(self);
    auto const give_up_at = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    while (!gate->_open.load() && std::chrono::steady_clock::now() < give_up_at) {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    gate->_passed.store(true);
  }

  std::atomic<bool> _open{false};
  std::atomic<bool> _passed{false};
};

/** Builds a STRING column with INT32 offsets from @p rows, all valid. */
std::unique_ptr<cudf::column> make_strings(std::vector<std::string> const& rows)
{
  auto const stream = cudf::get_default_stream();
  auto const mr     = cudf::get_current_device_resource_ref();
  std::string chars;
  std::vector<std::int32_t> offsets{0};
  for (auto const& row : rows) {
    chars += row;
    offsets.push_back(static_cast<std::int32_t>(chars.size()));
  }
  auto offsets_col = cudf::make_numeric_column(cudf::data_type{cudf::type_id::INT32},
                                               static_cast<cudf::size_type>(offsets.size()),
                                               cudf::mask_state::UNALLOCATED,
                                               stream,
                                               mr);
  REQUIRE(cudaMemcpy(offsets_col->mutable_view().data<std::int32_t>(),
                     offsets.data(),
                     offsets.size() * sizeof(std::int32_t),
                     cudaMemcpyHostToDevice) == cudaSuccess);
  rmm::device_buffer chars_buf(chars.data(), chars.size(), stream, mr);
  stream.sync();
  return cudf::make_strings_column(
    static_cast<cudf::size_type>(rows.size()), std::move(offsets_col), std::move(chars_buf), 0, {});
}

auto const varchar = sirius::logical_type::make(sirius::type_id::VARCHAR);
auto const boolean = sirius::logical_type::make(sirius::type_id::BOOLEAN);

std::unique_ptr<ast_node> equals(std::string literal)
{
  return make_cmp(
    sirius::comparison_type::equal, make_ref_typed(0, varchar), make_str_const(std::move(literal)));
}

std::unique_ptr<ast_node> contains(std::string literal)
{
  std::vector<std::unique_ptr<ast_node>> args;
  args.push_back(make_ref_typed(0, varchar));
  args.push_back(make_str_const(std::move(literal)));
  return make_func(sirius::function_id::contains, std::move(args), boolean);
}

/** An `NVRTC_JIT` evaluator of @p expressions over the query caches @p caches. */
sirius::expression_evaluator nvrtc_evaluator(
  duckdb::vector<std::unique_ptr<ast_node>> const& expressions,
  std::shared_ptr<sirius::expression_query_caches const> caches)
{
  return sirius::expression_evaluator(
    expressions,
    cudf::get_current_device_resource_ref(),
    cudf::get_default_stream(),
    sirius::expression_query_policy{sirius::expression_evaluator_strategy::NVRTC_JIT},
    sirius::expression_evaluator::default_min_ast_size,
    /*like_swar_fastpath=*/false,
    std::move(caches));
}

}  // namespace

TEST_CASE("string_constant_cache - one upload per pool serves every stream of a device",
          "[nvrtc_jit][string_constant_cache]")
{
  rmm::cuda_stream upload_stream;
  rmm::cuda_stream consumer_stream;
  string_constant_cache const cache;

  auto const uploaded = cache.acquire(mixed_pool, upload_stream);
  auto const reused   = cache.acquire(mixed_pool, consumer_stream);

  CHECK(cache.upload_count_for_testing() == 1);
  CHECK(reused.data() == uploaded.data());
  // The consumer stream reads the bytes without any host wait on the upload.
  CHECK(read_back(reused.data(), mixed_pool.size(), consumer_stream) == mixed_pool);
}

TEST_CASE("string_constant_cache - the upload does not wait for work queued on the caller's stream",
          "[nvrtc_jit][string_constant_cache]")
{
  rmm::cuda_stream blocked_stream;
  rmm::cuda_stream other_stream;
  string_constant_cache const cache;
  stream_gate gate(blocked_stream);

  std::optional<string_constant_cache::lease> blocked;
  {
    auto const before = std::chrono::steady_clock::now();
    blocked.emplace(cache.acquire(mixed_pool, blocked_stream));
    CHECK(std::chrono::steady_clock::now() - before < std::chrono::seconds(5));
  }
  auto const other = cache.acquire(mixed_pool, other_stream);
  CHECK(read_back(other.data(), mixed_pool.size(), other_stream) == mixed_pool);
  CHECK_FALSE(gate.passed());

  gate.open();
  CHECK(read_back(blocked->data(), mixed_pool.size(), blocked_stream) == mixed_pool);
}

TEST_CASE("string_constant_cache - the source bytes may change once acquire returns",
          "[nvrtc_jit][string_constant_cache]")
{
  rmm::cuda_stream stream;
  string_constant_cache const cache;
  std::string source = mixed_pool;

  auto const copy = cache.acquire(source, stream);
  source.assign(source.size(), '?');

  CHECK(read_back(copy.data(), mixed_pool.size(), stream) == mixed_pool);
}

TEST_CASE("string_constant_cache - distinct pools are uploaded separately",
          "[nvrtc_jit][string_constant_cache]")
{
  rmm::cuda_stream stream;
  string_constant_cache const cache;

  auto const first  = cache.acquire("alpha", stream);
  auto const second = cache.acquire("beta", stream);
  auto const again  = cache.acquire("alpha", stream);

  CHECK(cache.upload_count_for_testing() == 2);
  CHECK(again.data() == first.data());
  CHECK(read_back(first.data(), 5, stream) == "alpha");
  CHECK(read_back(second.data(), 4, stream) == "beta");
}

TEST_CASE(
  "string_constant_cache - destruction returns before the work queued under a lease completes, "
  "and that work still reads the copy",
  "[nvrtc_jit][string_constant_cache]")
{
  rmm::cuda_stream stream;
  // Pinned, so that the read queues behind the gate instead of waiting for it on the host.
  char* pinned = nullptr;
  REQUIRE(cudaMallocHost(&pinned, mixed_pool.size()) == cudaSuccess);
  std::unique_ptr<char, decltype(&cudaFreeHost)> const host{pinned, &cudaFreeHost};
  auto const manager = make_engine_like_manager();
  std::optional<string_constant_cache> cache{
    std::in_place,
    std::vector<cucascade::memory::memory_space const*>{
      manager->get_memory_space(cucascade::memory::Tier::GPU, 0)}};
  std::optional<stream_gate> gate;
  {
    auto const copy = cache->acquire(mixed_pool, stream);
    gate.emplace(stream);
    REQUIRE(cudaMemcpyAsync(
              host.get(), copy.data(), mixed_pool.size(), cudaMemcpyDeviceToHost, stream.value()) ==
            cudaSuccess);
  }

  auto const before = std::chrono::steady_clock::now();
  cache.reset();
  CHECK(std::chrono::steady_clock::now() - before < std::chrono::seconds(5));
  CHECK_FALSE(gate->passed());

  gate->open();
  stream.synchronize();
  CHECK(std::string(host.get(), mixed_pool.size()) == mixed_pool);
}

TEST_CASE("string_constant_cache - concurrent acquisitions of one pool share one upload",
          "[nvrtc_jit][string_constant_cache]")
{
  constexpr int thread_count = 8;
  string_constant_cache const cache;
  std::latch start{thread_count};
  std::vector<char const*> copies(thread_count);
  std::vector<std::string> read(thread_count);
  std::vector<std::thread> threads;
  for (int t = 0; t < thread_count; ++t) {
    threads.emplace_back([&, t] {
      rmm::cuda_stream stream;
      start.arrive_and_wait();
      auto const copy = cache.acquire(mixed_pool, stream);
      auto const own  = cache.acquire(std::format("pool of thread {}", t), stream);
      copies[t]       = copy.data();
      read[t]         = read_back(copy.data(), mixed_pool.size(), stream);
    });
  }
  for (auto& thread : threads) {
    thread.join();
  }

  CHECK(cache.upload_count_for_testing() == 1 + thread_count);
  CHECK(std::set<char const*>(copies.begin(), copies.end()).size() == 1);
  for (auto const& bytes : read) {
    CHECK(bytes == mixed_pool);
  }
}

TEST_CASE(
  "string_constant_cache - a cache over a GPU memory space charges its copies to the space for "
  "its own lifetime, never to the calling thread's reservation",
  "[nvrtc_jit][string_constant_cache]")
{
  // Per-thread tracking charges every allocation a thread makes through the space's allocator to
  // the reservation attached to that thread.
  auto const manager  = make_engine_like_manager();
  auto* const space   = manager->get_memory_space(cucascade::memory::Tier::GPU, 0);
  auto* const charged = space->get_memory_resource_of<cucascade::memory::Tier::GPU>();
  REQUIRE(charged != nullptr);

  rmm::cuda_stream stream;
  auto task_reservation = space->make_reservation(std::size_t{1} << 20);
  REQUIRE(task_reservation != nullptr);
  REQUIRE(charged->attach_reservation_to_tracker(stream, std::move(task_reservation)));
  auto const unrelated = charged->get_total_allocated_bytes();

  std::optional<string_constant_cache> cache{
    std::in_place, std::vector<cucascade::memory::memory_space const*>{space}};
  {
    auto const copy = cache->acquire(mixed_pool, stream);
    CHECK(read_back(copy.data(), mixed_pool.size(), stream) == mixed_pool);
  }
  CHECK(charged->get_allocated_bytes(stream) == 0);
  CHECK(charged->get_total_allocated_bytes() ==
        unrelated + rmm::align_up(mixed_pool.size(), rmm::CUDA_ALLOCATION_ALIGNMENT));

  cache.reset();
  CHECK(charged->get_total_allocated_bytes() == unrelated);

  charged->reset_stream_reservation(stream);
  stream.synchronize();
}

TEST_CASE("string_constant_cache - an empty pool is rejected", "[nvrtc_jit][string_constant_cache]")
{
  rmm::cuda_stream stream;
  string_constant_cache const cache;
  CHECK_THROWS_AS(cache.acquire({}, stream), sirius::internal_exception);
  CHECK(cache.upload_count_for_testing() == 0);
}

TEST_CASE("string_constant_cache - every device gets its own copy",
          "[nvrtc_jit][string_constant_cache][multi_gpu]")
{
  int device_count = 0;
  REQUIRE(cudaGetDeviceCount(&device_count) == cudaSuccess);
  string_constant_cache const cache;
  std::set<char const*> copies;
  for (int device = 0; device < device_count; ++device) {
    rmm::cuda_set_device_raii const on_device{rmm::cuda_device_id{device}};
    rmm::cuda_stream upload_stream;
    rmm::cuda_stream consumer_stream;
    auto const uploaded = cache.acquire(mixed_pool, upload_stream);
    auto const reused   = cache.acquire(mixed_pool, consumer_stream);
    CHECK(reused.data() == uploaded.data());
    CHECK(read_back(reused.data(), mixed_pool.size(), consumer_stream) == mixed_pool);
    copies.insert(uploaded.data());
  }
  CHECK(copies.size() == static_cast<std::size_t>(device_count));
  CHECK(cache.upload_count_for_testing() == static_cast<std::size_t>(device_count));
}

TEST_CASE(
  "NVRTC_JIT - evaluators of one query upload a string-constant pool once over many batches",
  "[nvrtc_jit][string_constant_cache]")
{
  std::string const long_literal(300, 'L');
  std::string const with_nul("a\0b", 3);
  std::string const utf8 = "\xC3\xA9t\xC3\xA9";
  std::vector<std::string> keys;
  for (int k = 0; k < 40; ++k) {
    keys.push_back(std::format("k{:02}", k));
  }
  std::vector<std::string> const rows{
    "", long_literal, with_nul, "a", "x" + utf8 + "y", "k07", "k39", std::string(299, 'L')};
  auto const column = make_strings(rows);
  cudf::table_view const input{{column->view()}};

  duckdb::vector<std::unique_ptr<ast_node>> expressions;
  expressions.push_back(equals(""));
  expressions.push_back(equals(long_literal));
  expressions.push_back(equals(with_nul));
  expressions.push_back(contains(utf8));
  for (auto const& key : keys) {
    expressions.push_back(equals(key));
  }
  std::vector<std::vector<std::uint8_t>> expected(expressions.size());
  for (auto const& row : rows) {
    expected[0].push_back(row.empty());
    expected[1].push_back(row == long_literal);
    expected[2].push_back(row == with_nul);
    expected[3].push_back(row.find(utf8) != std::string::npos);
    for (std::size_t k = 0; k < keys.size(); ++k) {
      expected[4 + k].push_back(row == keys[k]);
    }
  }

  auto const caches = std::make_shared<sirius::expression_query_caches>();
  for (int task = 0; task < 2; ++task) {
    auto evaluator = nvrtc_evaluator(expressions, caches);
    for (int batch = 0; batch < 8; ++batch) {
      auto const output = evaluator.evaluate(input);
      REQUIRE(output->num_columns() == static_cast<cudf::size_type>(expressions.size()));
      for (cudf::size_type c = 0; c < output->num_columns(); ++c) {
        CAPTURE(task, batch, c);
        CHECK(copy_bool_column_to_host(output->view().column(c)) ==
              expected[static_cast<std::size_t>(c)]);
      }
    }
  }
  CHECK(caches->string_constants.upload_count_for_testing() == 1);
}

TEST_CASE("NVRTC_JIT - different string literals share one kernel",
          "[nvrtc_jit][string_constant_cache]")
{
  auto const column = make_strings({"alpha", "a much longer literal", "beta"});
  cudf::table_view const input{{column->view()}};
  auto const caches = std::make_shared<sirius::expression_query_caches>();
  auto const run    = [&](std::string literal) {
    duckdb::vector<std::unique_ptr<ast_node>> expressions;
    expressions.push_back(equals(std::move(literal)));
    auto evaluator = nvrtc_evaluator(expressions, caches);
    return copy_bool_column_to_host(evaluator.evaluate(input)->view().column(0));
  };

  CHECK(run("alpha") == std::vector<std::uint8_t>{1, 0, 0});
  auto const kernels = codegen::jit::KernelCache::instance().size();
  CHECK(run("a much longer literal") == std::vector<std::uint8_t>{0, 1, 0});
  CHECK(codegen::jit::KernelCache::instance().size() == kernels);
  CHECK(caches->string_constants.upload_count_for_testing() == 2);
}
