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

// The readahead's device tier: the per-GPU speculative budget, the issuer that reads a split's hint
// ranges straight into device buffers, and the claim/shed protocol the GPU executor runs before it
// reserves memory for the demand read.
//
// Most tests drive the issuer through a scripted ioctx whose reads the test completes, fails or
// holds by hand (latches and promises, no sleeps), so every interleaving is deterministic.  The
// end-to-end tests read a real local file: through kvikIO with no host cache, and through the
// engine's own io_uring backend with the prefetching cache on.

#include "catch.hpp"
#include "event/query_event_publisher.hpp"
#include "exec/query_lifecycle_registry.hpp"
#include "exec/semi_future.hpp"
#include "io/io_context.hpp"
#include "io/kvikio/kvikio_context.hpp"
#include "io/sirius_datasource.hpp"
#include "io/types.hpp"
#include "memory/sirius_memory_reservation_manager.hpp"
#include "memory/topology_index.hpp"
#include "op/scan/gpu_ingestible_types.hpp"
#include "op/scan/prefetched_ranges.hpp"
#include "op/scan/sirius_gpu_scan_operator_data.hpp"
#include "op/sirius_physical_operator.hpp"
#include "pipeline/gpu_pipeline_task.hpp"
#include "pipeline/pipeline_build_context.hpp"
#include "pipeline/sirius_pipeline.hpp"
#include "planner/query.hpp"
#include "scan_manager/config.hpp"
#include "scan_manager/device_prefetch.hpp"
#include "scan_manager/readahead_scan_manager.hpp"
#include "scan_manager/sirius_scan_manager.hpp"
#include "utils/sirius_test_env.hpp"
#include "utils/telemetry_utils.hpp"

#include <cudf/io/text/byte_range_info.hpp>

#include <rmm/cuda_device.hpp>
#include <rmm/cuda_stream.hpp>

#include <cuda_runtime_api.h>

#include <cucascade/memory/config.hpp>
#include <cucascade/memory/memory_space.hpp>
#include <cucascade/memory/reservation_manager_configurator.hpp>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <barrier>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <fstream>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <span>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace {

namespace scan = sirius::op::scan;
namespace sm   = sirius::scan_manager;

using byte_range        = cudf::io::text::byte_range_info;
using read_future       = sirius::exec::semi_future<std::size_t>;
using prefetched_ranges = scan::prefetched_ranges;
using prefetch_outcome  = sm::prefetch_outcome_kind;
using readahead_manager = sm::readahead_scan_manager;
using device_issue      = sm::device_readahead_issue;
using device_issue_done = sm::device_readahead_result;
using prepared_slice    = sirius::io::prepared_io_slice;
using std::chrono_literals::operator""s;

constexpr std::size_t MiB        = std::size_t{1} << 20;
constexpr std::size_t k_file     = 8 * MiB;
constexpr std::size_t k_capacity = 512 * MiB;

/// Polls @p condition until it holds or @p timeout passes.  Used only for effects another thread
/// produces with no hook to wait on (the readahead worker's outcome counters, a hook that runs when
/// the last reference drops).
template <class Condition>
bool eventually(Condition condition, std::chrono::seconds timeout = 20s)
{
  auto const deadline = std::chrono::steady_clock::now() + timeout;
  while (!condition()) {
    if (std::chrono::steady_clock::now() >= deadline) { return false; }
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  return true;
}

//===----------------------------------------------------------------------===//
// Memory and registry
//===----------------------------------------------------------------------===//

/// One or two GPU spaces (plus their host regions), keeping the configs the registry is sized from.
struct gpu_env {
  std::vector<cucascade::memory::memory_space_config> configs;
  std::unique_ptr<sirius::memory::sirius_memory_reservation_manager> manager;

  explicit gpu_env(std::size_t n_gpus                 = 1,
                   double reservation_fraction        = 0.75,
                   bool per_stream_tracking           = true,
                   std::size_t capacity_per_gpu_bytes = k_capacity)
  {
    cucascade::memory::reservation_manager_configurator builder;
    builder.set_number_of_gpus(n_gpus)
      .set_gpu_usage_limit(capacity_per_gpu_bytes)
      .set_reservation_fraction_per_gpu(reservation_fraction)
      .set_per_numa_region_capacity(std::size_t{1} << 30)
      .use_gpu_id_as_host_id()
      .set_reservation_fraction_per_numa_region(0.75)
      .track_reservation_per_stream(per_stream_tracking);
    configs = builder.build();
    manager = std::make_unique<sirius::memory::sirius_memory_reservation_manager>(configs);
  }

  [[nodiscard]] cucascade::memory::memory_space* gpu_space(int device = 0) const
  {
    auto* space = manager->get_memory_space(cucascade::memory::Tier::GPU, device);
    REQUIRE(space != nullptr);
    return space;
  }

  [[nodiscard]] cucascade::memory::gpu_memory_space_config const& gpu_config(int device = 0) const
  {
    for (auto const& config : configs) {
      if (auto const* gpu = std::get_if<cucascade::memory::gpu_memory_space_config>(&config);
          gpu != nullptr && gpu->device_id == device) {
        return *gpu;
      }
    }
    FAIL("no GPU config for device " << device);
    throw std::logic_error("unreachable");
  }

  [[nodiscard]] std::shared_ptr<sm::device_readahead_registry> registry(
    double budget_fraction, double threshold_fraction = 0.9) const
  {
    return std::make_shared<sm::device_readahead_registry>(
      *manager, budget_fraction, threshold_fraction, configs);
  }
};

//===----------------------------------------------------------------------===//
// Scripted storage
//===----------------------------------------------------------------------===//

class scripted_object final : public sirius::io::io_object {
 public:
  explicit scripted_object(std::string path) : _path(std::move(path)) {}

  [[nodiscard]] std::string const& raw_file_cache_id() const noexcept override { return _path; }
  [[nodiscard]] std::string const& object_path() const noexcept override { return _path; }
  [[nodiscard]] std::size_t size() const noexcept override { return k_file; }
  [[nodiscard]] std::string_view validation_tag() const noexcept override { return {}; }

 private:
  std::string _path;
};

/// The byte a scripted read leaves in a destination that starts at file offset @p offset.
std::uint8_t fill_value(std::size_t offset)
{
  return static_cast<std::uint8_t>(0x10 + offset / MiB);
}

/// A device destination one scripted read was asked to fill.
struct read_target {
  std::uint8_t* dst;
  std::size_t size;
  std::size_t offset;
  cudaStream_t stream;
  int device;
};

std::vector<read_target> targets_of(std::vector<prepared_slice> const& slices)
{
  int device = 0;
  REQUIRE(cudaGetDevice(&device) == cudaSuccess);
  std::vector<read_target> targets;
  for (auto const& slice : slices) {
    targets.push_back(
      {slice.d_buffer.data, slice.size(), slice.offset(), slice.d_buffer.stream.get(), device});
  }
  return targets;
}

std::size_t total_of(std::vector<read_target> const& targets)
{
  std::size_t total = 0;
  for (auto const& target : targets) {
    total += target.size;
  }
  return total;
}

/// Writes the scripted pattern into every target, stream-ordered on the stream the read named.
void fill_targets(std::vector<read_target> const& targets)
{
  for (auto const& target : targets) {
    rmm::cuda_set_device_raii bind{rmm::cuda_device_id{target.device}};
    REQUIRE(cudaMemsetAsync(target.dst, fill_value(target.offset), target.size, target.stream) ==
            cudaSuccess);
  }
}

/// An ioctx with no storage behind it: every device read goes to @p respond, which returns the
/// future the test wants (settled now, held, failed or absent).  Whether the read went through the
/// raw backend or a prefetching cache is observable from the context: a scripted context has no
/// cache, so a read that arrives here is the backend's vectored device read.
class scripted_ioctx final : public sirius::io::ioctx {
 public:
  using responder = std::function<read_future(std::vector<prepared_slice>&)>;

  explicit scripted_ioctx(responder respond) : _respond(std::move(respond)) {}
  ~scripted_ioctx() override { pre_destroy(); }

  [[nodiscard]] sirius::io::io_context_type type() const noexcept override
  {
    return sirius::io::io_context_type::restful;
  }
  void shutdown() noexcept override {}
  [[nodiscard]] bool supports(std::string_view) const noexcept override { return true; }
  [[nodiscard]] bool supports_device_read() const noexcept override { return true; }
  [[nodiscard]] bool supports_host_to_device_read() const noexcept override { return false; }
  [[nodiscard]] bool supports_vector_host_read() const noexcept override { return false; }
  [[nodiscard]] bool supports_device_range_read() const noexcept override { return true; }

  [[nodiscard]] std::vector<cudf::io::text::byte_range_info> align_and_coalesce(
    std::span<cudf::io::text::byte_range_info const> ranges,
    std::optional<std::size_t>) const noexcept override
  {
    return {ranges.begin(), ranges.end()};
  }

  std::size_t host_read_io(sirius::io::io_object const&,
                           std::size_t,
                           std::size_t,
                           std::uint8_t*) override
  {
    throw std::runtime_error("the scripted backend serves only device reads");
  }

  read_future mixed_readv_async_io(sirius::io::io_object const&,
                                   std::vector<prepared_slice>&& slices) noexcept override
  {
    try {
      return _respond(slices);
    } catch (...) {
      return sirius::exec::make_semi_future<std::size_t>(std::current_exception());
    }
  }

 protected:
  std::shared_ptr<sirius::io::io_object> create_io_object(std::string path) override
  {
    return std::make_shared<scripted_object>(std::move(path));
  }

 private:
  responder _respond;
};

std::shared_ptr<sirius::io::sirius_datasource> scripted_datasource(
  std::shared_ptr<scripted_ioctx> const& context, std::string path = "scripted://object")
{
  return std::make_shared<sirius::io::sirius_datasource>(
    context, std::make_shared<scripted_object>(std::move(path)));
}

/// Every read fills its destinations and settles at once with the bytes it asked for.
std::shared_ptr<scripted_ioctx> instant_context()
{
  return std::make_shared<scripted_ioctx>([](std::vector<prepared_slice>& slices) {
    auto const targets = targets_of(slices);
    fill_targets(targets);
    return sirius::exec::make_semi_future<std::size_t>(total_of(targets));
  });
}

/// Every read fails with @p message, settled before the read call returns.
std::shared_ptr<scripted_ioctx> failing_context(
  std::string message = "injected device read failure")
{
  return std::make_shared<scripted_ioctx>([message](std::vector<prepared_slice>&) {
    return sirius::exec::make_semi_future<std::size_t>(
      std::make_exception_ptr(std::runtime_error(message)));
  });
}

/// Every read settles successfully but one byte short, the way a backend clamping at end of file
/// does.
std::shared_ptr<scripted_ioctx> short_context()
{
  return std::make_shared<scripted_ioctx>([](std::vector<prepared_slice>& slices) {
    return sirius::exec::make_semi_future<std::size_t>(total_of(targets_of(slices)) - 1);
  });
}

/// Every read hands back an invalid future: there is nothing to install a callback on.
std::shared_ptr<scripted_ioctx> no_future_context()
{
  return std::make_shared<scripted_ioctx>(
    [](std::vector<prepared_slice>&) { return read_future{}; });
}

/// One read that stays in flight until the test opens (or fails) the gate.  Opening writes the
/// scripted pattern into the destinations the read named -- which the fan-in must still keep alive
/// -- and then settles the read.
class read_gate {
 public:
  [[nodiscard]] std::shared_ptr<scripted_ioctx> context()
  {
    return std::make_shared<scripted_ioctx>([this](std::vector<prepared_slice>& slices) {
      auto targets = targets_of(slices);
      std::lock_guard lock(_mutex);
      if (_reads++ > 0) {
        return sirius::exec::make_semi_future<std::size_t>(
          std::make_exception_ptr(std::logic_error("a gate serves one read")));
      }
      _targets = std::move(targets);
      for (auto const& target : _targets) {
        _streams.insert(target.stream);
      }
      _cv.notify_all();
      return _promise.get_semi_future();
    });
  }

  /// Blocks until the read arrived (a latch, not a sleep).
  [[nodiscard]] bool wait_for_read(std::chrono::seconds timeout = 30s)
  {
    std::unique_lock lock(_mutex);
    return _cv.wait_for(lock, timeout, [&] { return _reads > 0; });
  }

  [[nodiscard]] int reads() const
  {
    std::lock_guard lock(_mutex);
    return _reads;
  }

  [[nodiscard]] std::set<cudaStream_t> streams() const
  {
    std::lock_guard lock(_mutex);
    return _streams;
  }

  void open()
  {
    std::vector<read_target> targets;
    {
      std::lock_guard lock(_mutex);
      targets = _targets;
    }
    fill_targets(targets);
    _promise.set_value(total_of(targets));
  }

  void fail()
  {
    _promise.set_exception(
      std::make_exception_ptr(std::runtime_error("injected device read failure")));
  }

 private:
  mutable std::mutex _mutex;
  std::condition_variable _cv;
  int _reads{0};
  std::vector<read_target> _targets;
  std::set<cudaStream_t> _streams;
  sirius::exec::promise<std::size_t> _promise;
};

//===----------------------------------------------------------------------===//
// Splits
//===----------------------------------------------------------------------===//

/// A split that can decode from device copies of its hint ranges.
struct device_split : scan::scan_info {
  using scan::scan_info::scan_info;
  [[nodiscard]] bool supports_device_prefetch() const noexcept override { return true; }
};

/// Two ranges of one object, 3 MiB in all: one hint, two buffers.
std::vector<scan::scan_info::fadvise_entry> two_ranges(
  std::shared_ptr<sirius::io::sirius_datasource> const& ds)
{
  return {{ds, {byte_range(0, 1 * MiB), byte_range(3 * MiB, 2 * MiB)}}};
}

constexpr std::size_t k_split_bytes = 3 * MiB;

std::shared_ptr<device_split> split_on(std::shared_ptr<scripted_ioctx> const& context,
                                       std::optional<int> device = 0,
                                       std::string path          = "scripted://object")
{
  auto split =
    std::make_shared<device_split>(two_ranges(scripted_datasource(context, std::move(path))));
  split->set_preferred_device(device);
  return split;
}

/// Copies a device span back to the host.
std::vector<std::uint8_t> to_host(cudf::device_span<std::uint8_t const> span, cudaStream_t stream)
{
  std::vector<std::uint8_t> host(span.size());
  REQUIRE(cudaMemcpyAsync(host.data(), span.data(), span.size(), cudaMemcpyDeviceToHost, stream) ==
          cudaSuccess);
  REQUIRE(cudaStreamSynchronize(stream) == cudaSuccess);
  return host;
}

bool all_equal(std::vector<std::uint8_t> const& bytes, std::uint8_t value)
{
  return std::all_of(bytes.begin(), bytes.end(), [value](std::uint8_t b) { return b == value; });
}

/// What on_done reported, and how many times it ran.
struct done_probe {
  std::promise<device_issue_done> promise;
  std::future<device_issue_done> future   = promise.get_future();
  std::shared_ptr<std::atomic<int>> calls = std::make_shared<std::atomic<int>>(0);

  sirius::exec::invocable<void(device_issue_done) noexcept> callback()
  {
    return [p = std::move(promise), calls = calls](device_issue_done result) mutable noexcept {
      calls->fetch_add(1);
      p.set_value(std::move(result));
    };
  }

  device_issue_done wait()
  {
    REQUIRE(future.wait_for(30s) == std::future_status::ready);
    return future.get();
  }
};

std::string message_of(std::exception_ptr const& error)
{
  try {
    std::rethrow_exception(error);
  } catch (std::exception const& e) {
    return e.what();
  } catch (...) {
    return "unknown";
  }
}

//===----------------------------------------------------------------------===//
// Issuer fixture
//===----------------------------------------------------------------------===//

struct issuer_fixture {
  gpu_env env{1};
  cucascade::memory::memory_space* space = env.gpu_space();
  /// Declared after the env: the registry's GPU state references the space.
  std::shared_ptr<sm::device_readahead_registry> registry = env.registry(0.5);
  sm::device_readahead_gpu* gpu                           = registry->gpu(0);
  std::shared_ptr<scan::device_stream_pool> pool          = scan::device_stream_pool::create(0, 2);
  std::size_t baseline                                    = space->get_available_memory();

  issuer_fixture() { REQUIRE(gpu != nullptr); }

  [[nodiscard]] device_issue issue(std::shared_ptr<scan::scan_info> const& split, done_probe& done)
  {
    return sm::issue_device_readahead(*gpu, pool->acquire(), split, done.callback());
  }

  /// Everything the prefetch held is given back: budget token, tracked device memory.
  void require_released()
  {
    REQUIRE(eventually([&] { return gpu->reserved() == 0; }));
    for (std::size_t i = 0; i < pool->size(); ++i) {
      REQUIRE(cudaStreamSynchronize(pool->acquire().view.get()) == cudaSuccess);
    }
    REQUIRE(eventually([&] { return space->get_available_memory() == baseline; }));
  }
};

/// The rounded reservation of the standard split (both ranges are already 256-byte multiples).
constexpr std::size_t k_split_reserved = k_split_bytes;

}  // namespace

//===----------------------------------------------------------------------===//
// Registry and budget
//===----------------------------------------------------------------------===//

TEST_CASE("the registry budgets by configured capacity, not by the reservation limit",
          "[scan_manager][device_prefetch][budget]")
{
  if (!sirius::test::has_gpus(1)) { return; }
  // Reservations are admitted up to half the capacity; the budget is a fraction of the capacity.
  gpu_env env{1, /*reservation_fraction=*/0.5};
  auto const& config = env.gpu_config();
  REQUIRE(env.gpu_space()->get_max_memory() == config.reservation_limit());
  REQUIRE(config.reservation_limit() < config.memory_capacity);

  SECTION("budget and ceiling")
  {
    auto registry = env.registry(/*budget=*/0.75, /*threshold=*/0.4);
    REQUIRE(registry->enabled());
    auto* gpu = registry->gpu(0);
    REQUIRE(gpu != nullptr);
    CHECK(gpu->budget_bytes() == static_cast<std::size_t>(0.75 * static_cast<double>(k_capacity)));
    CHECK(gpu->budget_bytes() > config.reservation_limit());
    // The ceiling is the lesser of the threshold and the downgrade stop.
    CHECK(gpu->ceiling_bytes() == static_cast<std::size_t>(0.4 * static_cast<double>(k_capacity)));
    CHECK(env.registry(0.75, 1.0)->gpu(0)->ceiling_bytes() == config.downgrade_stop_threshold());
    CHECK(gpu->device_id() == 0);
    CHECK(&gpu->space() == env.gpu_space());
  }

  SECTION("a reservation above the reservation limit is still within the budget")
  {
    auto registry    = env.registry(0.75, 1.0);
    auto* gpu        = registry->gpu(0);
    auto const bytes = config.reservation_limit() + 1 * MiB;
    REQUIRE(bytes <= gpu->budget_bytes());
    REQUIRE(bytes <= gpu->ceiling_bytes());
    CHECK(gpu->try_reserve(bytes));
    CHECK(gpu->reserved() == bytes);
    gpu->release(bytes);
    CHECK(gpu->reserved() == 0);
  }

  SECTION("disabled")
  {
    CHECK_FALSE(env.registry(0.0)->enabled());
    CHECK(env.registry(0.0)->gpu(0) == nullptr);
    sm::device_readahead_registry no_configs{*env.manager, 0.5, 0.5, {}};
    CHECK_FALSE(no_configs.enabled());
    CHECK(no_configs.gpu(0) == nullptr);
    CHECK(env.registry(0.5)->gpu(1) == nullptr);
  }
}

TEST_CASE("try_reserve rations the budget, the usage ceiling and the demand guard",
          "[scan_manager][device_prefetch][budget]")
{
  if (!sirius::test::has_gpus(1)) { return; }
  gpu_env env;
  auto* space = env.gpu_space();
  sm::device_readahead_gpu gpu{*space, k_capacity, /*budget=*/100 * MiB, /*ceiling=*/200 * MiB};

  SECTION("the budget")
  {
    CHECK(gpu.try_reserve(60 * MiB));
    CHECK(gpu.reserved() == 60 * MiB);
    CHECK_FALSE(gpu.try_reserve(60 * MiB));
    CHECK(gpu.reserved() == 60 * MiB);
    CHECK(gpu.try_reserve(40 * MiB));
    gpu.release(60 * MiB);
    CHECK(gpu.try_reserve(60 * MiB));
    gpu.release(100 * MiB);
    CHECK(gpu.reserved() == 0);
  }

  SECTION("the usage ceiling counts what anyone else holds")
  {
    auto hog = space->make_reservation_or_null(150 * MiB);
    REQUIRE(hog != nullptr);
    CHECK_FALSE(gpu.try_reserve(60 * MiB));  // 150 + 60 > 200
    CHECK(gpu.try_reserve(40 * MiB));        // 150 + 40 <= 200
    gpu.release(40 * MiB);
    hog.reset();
    CHECK(gpu.try_reserve(60 * MiB));
    gpu.release(60 * MiB);
  }

  SECTION("the demand guard")
  {
    CHECK_FALSE(gpu.demand_active());
    gpu.enter_demand_guard();
    gpu.enter_demand_guard();
    CHECK(gpu.demand_active());
    CHECK_FALSE(gpu.try_reserve(1 * MiB));
    gpu.exit_demand_guard();
    CHECK(gpu.demand_active());
    gpu.exit_demand_guard();
    CHECK_FALSE(gpu.demand_active());
    CHECK(gpu.try_reserve(1 * MiB));
    gpu.release(1 * MiB);
  }
}

//===----------------------------------------------------------------------===//
// The issuer
//===----------------------------------------------------------------------===//

TEST_CASE_METHOD(issuer_fixture,
                 "issue reads the hint ranges into device buffers and lands them ready",
                 "[scan_manager][device_prefetch][issuer]")
{
  if (!sirius::test::has_gpus(1)) { return; }
  auto split = split_on(instant_context());
  done_probe done;
  REQUIRE(issue(split, done) == device_issue::issued);

  auto const result = done.wait();
  CHECK(result.ok);
  CHECK(result.bytes == k_split_bytes);
  CHECK(result.error == nullptr);
  CHECK(done.calls->load() == 1);

  auto ranges = split->prefetched();
  REQUIRE(ranges != nullptr);
  CHECK(ranges->current() == prefetched_ranges::state::ready);
  CHECK(ranges->device_id() == 0);
  CHECK(ranges->total_bytes() == k_split_bytes);
  CHECK(ranges->reserved_bytes() == k_split_reserved);
  // The budget token is held until the claim, and the allocation is tracked on the space.
  CHECK(gpu->reserved() == k_split_reserved);
  CHECK(space->get_available_memory() + k_split_bytes <= baseline);

  rmm::cuda_stream consumer;
  auto taken = ranges->take(::cuda::stream_ref{consumer.value()});
  REQUIRE(taken.has_value());
  REQUIRE(taken->size() == 1);
  auto const& entry = taken->front();
  REQUIRE(entry.spans.size() == 2);
  CHECK(entry.spans[0].size() == 1 * MiB);
  CHECK(entry.spans[1].size() == 2 * MiB);
  CHECK(all_equal(to_host(entry.spans[0], consumer.value()), fill_value(0)));
  CHECK(all_equal(to_host(entry.spans[1], consumer.value()), fill_value(3 * MiB)));
  // take() claims: the token is back although the buffers are still in use.
  CHECK(gpu->reserved() == 0);

  taken.reset();
  ranges.reset();
  split.reset();
  require_released();
}

TEST_CASE("issue works under per-stream and per-thread reservation tracking",
          "[scan_manager][device_prefetch][issuer]")
{
  if (!sirius::test::has_gpus(1)) { return; }
  // The engine runs per-thread tracking by default.
  bool const per_stream = GENERATE(true, false);
  INFO("per_stream_tracking=" << per_stream);
  gpu_env env{1, 0.75, per_stream};
  auto registry       = env.registry(0.5);
  auto pool           = scan::device_stream_pool::create(0, 1);
  auto const baseline = env.gpu_space()->get_available_memory();
  {
    auto split = split_on(instant_context());
    done_probe done;
    REQUIRE(sm::issue_device_readahead(
              *registry->gpu(0), pool->acquire(), split, done.callback()) == device_issue::issued);
    REQUIRE(done.wait().ok);
    CHECK(split->prefetched()->current() == prefetched_ranges::state::ready);
    split->claim_device_readahead();
    CHECK(registry->gpu(0)->reserved() == 0);
  }
  REQUIRE(cudaStreamSynchronize(pool->acquire().view.get()) == cudaSuccess);
  CHECK(env.gpu_space()->get_available_memory() == baseline);
}

TEST_CASE_METHOD(issuer_fixture,
                 "work the issuer cannot or should not do is left to the host tier",
                 "[scan_manager][device_prefetch][issuer]")
{
  if (!sirius::test::has_gpus(1)) { return; }
  done_probe done;

  SECTION("a split that cannot decode from device copies")
  {
    auto split =
      std::make_shared<scan::scan_info>(two_ranges(scripted_datasource(instant_context())));
    CHECK(issue(split, done) == device_issue::unsupported);
    CHECK(split->prefetched() == nullptr);
  }
  SECTION("no hints")
  {
    auto split = std::make_shared<device_split>(std::vector<scan::scan_info::fadvise_entry>{});
    CHECK(issue(split, done) == device_issue::no_ranges);
  }
  SECTION("a split larger than the whole speculative budget")
  {
    auto const budget_bytes = gpu->budget_bytes();
    auto split = std::make_shared<device_split>(std::vector<scan::scan_info::fadvise_entry>{
      {scripted_datasource(instant_context()),
       {byte_range(0, static_cast<std::int64_t>(budget_bytes + 1))}}});
    CHECK(issue(split, done) == device_issue::too_large);
  }
  SECTION("a split the consumer already reached")
  {
    auto split = split_on(instant_context());
    split->set_scan_stage(sirius::io::cache::scan_stage::preparing);
    CHECK(issue(split, done) == device_issue::fell_behind);
  }
  SECTION("a split with an occupied slot")
  {
    auto split    = split_on(instant_context());
    auto occupant = std::make_shared<prefetched_ranges>(
      0,
      pool->acquire(),
      std::vector<prefetched_ranges::device_ranges>{},
      0,
      0,
      []() noexcept {},
      [](prefetched_ranges::release_event) noexcept {});
    REQUIRE(split->set_prefetched_ranges(occupant));
    CHECK(issue(split, done) == device_issue::error);
    CHECK(split->prefetched() == occupant);
  }

  // None of them installs anything, reads anything or reports completion.
  CHECK(done.calls->load() == 0);
  CHECK(gpu->reserved() == 0);
  require_released();
}

TEST_CASE_METHOD(issuer_fixture,
                 "memory pressure refuses without installing and without leaking the token",
                 "[scan_manager][device_prefetch][issuer]")
{
  if (!sirius::test::has_gpus(1)) { return; }
  done_probe done;
  auto split = split_on(instant_context());

  SECTION("the budget is exhausted")
  {
    REQUIRE(gpu->try_reserve(gpu->budget_bytes() - 1 * MiB));
    CHECK(issue(split, done) == device_issue::pressure);
    gpu->release(gpu->budget_bytes() - 1 * MiB);
  }
  SECTION("a demand reservation is admitting")
  {
    gpu->enter_demand_guard();
    CHECK(issue(split, done) == device_issue::pressure);
    gpu->exit_demand_guard();
  }
  SECTION("the real reservation is refused")
  {
    // Admission is against the reservation limit, which the token budget does not know about.
    auto hog = space->make_reservation_or_null(space->get_max_memory() - 1 * MiB);
    REQUIRE(hog != nullptr);
    auto const consumed_before = space->get_available_memory();
    // The ceiling would refuse first, so lift it by building a GPU state that allows it.
    sm::device_readahead_gpu permissive{*space, k_capacity, k_capacity, k_capacity};
    CHECK(sm::issue_device_readahead(permissive, pool->acquire(), split, done.callback()) ==
          device_issue::memory);
    CHECK(permissive.reserved() == 0);
    CHECK(space->get_available_memory() == consumed_before);
    hog.reset();
  }

  CHECK(split->prefetched() == nullptr);
  CHECK(done.calls->load() == 0);
  CHECK(gpu->reserved() == 0);
}

TEST_CASE_METHOD(issuer_fixture,
                 "a short read is a failure and an errored read never throws",
                 "[scan_manager][device_prefetch][issuer]")
{
  if (!sirius::test::has_gpus(1)) { return; }
  done_probe done;
  std::shared_ptr<scripted_ioctx> context;
  std::string expected;
  SECTION("short")
  {
    context  = short_context();
    expected = "short device read";
  }
  SECTION("errored")
  {
    context  = failing_context();
    expected = "injected device read failure";
  }
  SECTION("no future")
  {
    context  = no_future_context();
    expected = "no future";
  }

  auto split = split_on(context);
  device_issue outcome{};
  REQUIRE_NOTHROW(outcome = issue(split, done));
  REQUIRE(outcome == device_issue::issued);
  auto const result = done.wait();
  CHECK_FALSE(result.ok);
  CHECK(result.bytes == k_split_bytes);
  REQUIRE(result.error != nullptr);
  CHECK(message_of(result.error).find(expected) != std::string::npos);
  CHECK(done.calls->load() == 1);

  auto ranges = split->prefetched();
  REQUIRE(ranges != nullptr);
  CHECK(ranges->current() == prefetched_ranges::state::failed);
  CHECK_FALSE(ranges->abandoned());
  CHECK(ranges->peek().empty());

  // The failed buffers are freed, and the token returned, when the last reference drops.
  ranges.reset();
  split.reset();
  require_released();
}

TEST_CASE_METHOD(issuer_fixture,
                 "one failed hint fails the prefetch only once every hint settled",
                 "[scan_manager][device_prefetch][issuer]")
{
  if (!sirius::test::has_gpus(1)) { return; }
  read_gate gate;
  auto split = std::make_shared<device_split>(std::vector<scan::scan_info::fadvise_entry>{
    {scripted_datasource(failing_context(), "scripted://a"), {byte_range(0, 1 * MiB)}},
    {scripted_datasource(gate.context(), "scripted://b"), {byte_range(3 * MiB, 2 * MiB)}}});
  split->set_preferred_device(0);

  done_probe done;
  REQUIRE(issue(split, done) == device_issue::issued);
  // The failed hint settled inline; the other is still in flight, so nothing is reported yet.
  REQUIRE(gate.wait_for_read());
  CHECK(done.calls->load() == 0);
  auto ranges = split->prefetched();
  REQUIRE(ranges != nullptr);
  CHECK(ranges->current() == prefetched_ranges::state::pending);

  gate.open();
  auto const result = done.wait();
  CHECK_FALSE(result.ok);
  CHECK(result.bytes == k_split_bytes);
  CHECK(message_of(result.error) == "injected device read failure");
  CHECK(ranges->current() == prefetched_ranges::state::failed);
  CHECK(done.calls->load() == 1);

  ranges.reset();
  split.reset();
  require_released();
}

TEST_CASE_METHOD(issuer_fixture,
                 "the consumer is ordered after the producer through the ready event",
                 "[scan_manager][device_prefetch][issuer][ordering]")
{
  if (!sirius::test::has_gpus(1)) { return; }
  // The read keeps the producer stream busy before it writes its destination.  A consumer that did
  // not wait on the ready event would read the buffer before the write.
  constexpr std::size_t big = 64 * MiB;
  auto busy  = std::make_shared<scripted_ioctx>([](std::vector<prepared_slice>& slices) {
    auto const targets = targets_of(slices);
    rmm::cuda_stream_view const producer{targets.front().stream};
    rmm::device_buffer scratch(big, producer);
    for (int i = 0; i < 16; ++i) {
      REQUIRE(cudaMemsetAsync(scratch.data(), i, big, producer.value()) == cudaSuccess);
    }
    fill_targets(targets);
    // The scratch buffer is freed stream-ordered, after the fill.
    return sirius::exec::make_semi_future<std::size_t>(total_of(targets));
  });
  auto split = std::make_shared<device_split>(std::vector<scan::scan_info::fadvise_entry>{
    {scripted_datasource(busy), {byte_range(0, static_cast<std::int64_t>(big))}}});
  split->set_preferred_device(0);

  done_probe done;
  REQUIRE(issue(split, done) == device_issue::issued);
  REQUIRE(done.wait().ok);
  auto ranges = split->prefetched();
  REQUIRE(ranges != nullptr);

  rmm::cuda_stream consumer;  // a different stream from the pool's
  auto taken = ranges->take(::cuda::stream_ref{consumer.value()});
  REQUIRE(taken.has_value());
  CHECK(all_equal(to_host(taken->front().spans.front(), consumer.value()), fill_value(0)));
  taken.reset();
  ranges.reset();
  split.reset();
  require_released();
}

//===----------------------------------------------------------------------===//
// Claim, shed, lifetime
//===----------------------------------------------------------------------===//

namespace {

/// Reports a fall-behind only after the slot was installed (the issuer's second check), and runs
/// @p between_checks at that point -- the instant the executor's claim can land between the
/// install and the submission of the reads.
struct overtaken_split final : device_split {
  using device_split::device_split;
  std::function<void()> between_checks;
  mutable std::atomic<int> checks{0};

  [[nodiscard]] bool has_fallen_behind() const noexcept override
  {
    if (checks.fetch_add(1) == 0) { return false; }
    if (between_checks) { between_checks(); }
    return false;  // the claim alone must stop the issue
  }
};

}  // namespace

TEST_CASE_METHOD(issuer_fixture,
                 "a claim before the install keeps the prefetch out",
                 "[scan_manager][device_prefetch][claim]")
{
  if (!sirius::test::has_gpus(1)) { return; }
  read_gate gate;
  auto split = split_on(gate.context());
  split->claim_device_readahead();

  done_probe done;
  CHECK(issue(split, done) != device_issue::issued);
  CHECK(split->prefetched() == nullptr);
  CHECK(gate.reads() == 0);
  CHECK(done.calls->load() == 0);
  CHECK(split->device_readahead_claimed());
  // Nothing was published, so the speculative allocation made on the way is given back.
  split.reset();
  require_released();
}

TEST_CASE_METHOD(issuer_fixture,
                 "a claim between the install and the reads stops the issue and frees everything",
                 "[scan_manager][device_prefetch][claim]")
{
  if (!sirius::test::has_gpus(1)) { return; }
  read_gate gate;
  auto split = std::make_shared<overtaken_split>(two_ranges(scripted_datasource(gate.context())));
  split->set_preferred_device(0);
  std::shared_ptr<prefetched_ranges> installed;
  split->between_checks = [&] {
    installed = split->prefetched();  // already visible to the executor
    split->claim_device_readahead();  // the claim lands between install and issue
  };

  done_probe done;
  CHECK(issue(split, done) == device_issue::fell_behind);
  REQUIRE(installed != nullptr);
  // The issuer abandoned it (a consumer already waiting wakes and reads for itself), dropped the
  // slot and submitted nothing.
  CHECK(installed->abandoned());
  CHECK(installed->current() == prefetched_ranges::state::failed);
  CHECK(split->prefetched() == nullptr);
  CHECK(gate.reads() == 0);
  CHECK(done.calls->load() == 0);
  // The claim released the token; the buffers die with the last reference.
  CHECK(gpu->reserved() == 0);
  installed.reset();
  split.reset();
  require_released();
}

TEST_CASE_METHOD(issuer_fixture,
                 "claim and install racing each other never leak the budget or the memory",
                 "[scan_manager][device_prefetch][claim]")
{
  if (!sirius::test::has_gpus(1)) { return; }
  constexpr int rounds = 64;
  auto context         = instant_context();
  for (int round = 0; round < rounds; ++round) {
    auto split = split_on(context);
    done_probe done;
    std::barrier start{2};
    device_issue outcome{};
    std::thread issuer([&] {
      start.arrive_and_wait();
      outcome = issue(split, done);
    });
    start.arrive_and_wait();
    split->claim_device_readahead();
    issuer.join();

    // Whichever side won, the claim made the prefetch unpublishable and its token is back.
    CHECK(split->device_readahead_claimed());
    if (outcome == device_issue::issued) {
      REQUIRE(done.wait().ok);
    } else {
      CHECK(done.calls->load() == 0);
    }
    CHECK(gpu->reserved() == 0);
    CHECK_FALSE(split->set_prefetched_ranges(std::make_shared<prefetched_ranges>(
      0,
      pool->acquire(),
      std::vector<prefetched_ranges::device_ranges>{},
      0,
      0,
      []() noexcept {},
      [](prefetched_ranges::release_event) noexcept {})));
  }
  require_released();
}

namespace {

/// A real executor claim: the production call chain from the task the executor holds down to the
/// split's slot, with nothing but the memory space of the GPU it runs on.
struct claim_harness {
  std::shared_ptr<sirius::pipeline::sirius_pipeline_task_global_state> global =
    std::make_shared<sirius::pipeline::sirius_pipeline_task_global_state>(
      nullptr, sirius::test::make_test_telemetry_context());

  /// A task whose input is @p split, as the scheduler builds it for a scan.
  [[nodiscard]] std::unique_ptr<sirius::pipeline::gpu_pipeline_task> task_for(
    std::shared_ptr<scan::scan_info> const& split, int id = 1) const
  {
    auto input = std::make_unique<scan::scan_operator_input>(split);
    return std::make_unique<sirius::pipeline::gpu_pipeline_task>(
      id,
      std::vector<cucascade::shared_data_repository*>{},
      std::make_unique<sirius::pipeline::gpu_pipeline_task_local_state>(std::move(input)),
      global);
  }
};

}  // namespace

TEST_CASE_METHOD(issuer_fixture,
                 "the executor's claim releases the budget before it reserves, buffers intact",
                 "[scan_manager][device_prefetch][claim]")
{
  if (!sirius::test::has_gpus(1)) { return; }
  read_gate gate;
  auto split = split_on(gate.context());
  claim_harness harness;
  auto task = harness.task_for(split);

  done_probe done;
  REQUIRE(issue(split, done) == device_issue::issued);
  REQUIRE(gate.wait_for_read());
  auto ranges = split->prefetched();
  REQUIRE(ranges != nullptr);
  REQUIRE(gpu->reserved() == k_split_reserved);

  // The executor's first step for the task, before any reservation or memory wait.  The prefetch is
  // still in flight.
  auto const held = task->claim_device_readahead(*space);
  CHECK(held == k_split_reserved);  // what the demand reservation is reduced by
  CHECK(gpu->reserved() == 0);      // the speculative budget is free again
  CHECK(split->device_readahead_claimed());
  CHECK(ranges->claimed());
  CHECK(split->prefetched() == ranges);  // still installed
  CHECK(ranges->entry_count() == 1);     // buffers untouched
  CHECK(ranges->current() == prefetched_ranges::state::pending);
  CHECK(space->get_available_memory() + k_split_bytes <= baseline);  // memory is still held

  // Idempotent: a second claim neither re-releases nor changes what is held.
  CHECK(task->claim_device_readahead(*space) == k_split_reserved);
  CHECK(gpu->reserved() == 0);

  // The in-flight read lands later; the decode takes the very buffers the claim left alone.
  gate.open();
  REQUIRE(done.wait().ok);
  REQUIRE(ranges->current() == prefetched_ranges::state::ready);
  rmm::cuda_stream consumer;
  auto taken = ranges->take(::cuda::stream_ref{consumer.value()});
  REQUIRE(taken.has_value());
  CHECK(all_equal(to_host(taken->front().spans[1], consumer.value()), fill_value(3 * MiB)));
  CHECK(gpu->reserved() == 0);

  taken.reset();
  ranges.reset();
  task.reset();
  split.reset();
  require_released();
}

TEST_CASE_METHOD(
  issuer_fixture,
  "while a task secures its reservation the guard blocks refills and shed spares claims",
  "[scan_manager][device_prefetch][claim]")
{
  if (!sirius::test::has_gpus(1)) { return; }
  read_gate claimed_gate;
  read_gate shed_gate;
  auto claimed = split_on(claimed_gate.context(), 0, "scripted://claimed");
  auto shed    = split_on(shed_gate.context(), 0, "scripted://shed");
  claim_harness harness;
  auto task = harness.task_for(claimed);

  done_probe claimed_done;
  done_probe shed_done;
  REQUIRE(issue(claimed, claimed_done) == device_issue::issued);
  REQUIRE(issue(shed, shed_done) == device_issue::issued);
  REQUIRE(claimed_gate.wait_for_read());
  REQUIRE(shed_gate.wait_for_read());
  auto claimed_ranges = claimed->prefetched();
  auto shed_ranges    = shed->prefetched();
  REQUIRE(gpu->reserved() == 2 * k_split_reserved);

  // The executor's sequence: claim, then raise the guard under the allocation mutex.
  CHECK(task->claim_device_readahead(*space) == k_split_reserved);
  {
    std::lock_guard lock(gpu->allocation_mutex());
    gpu->enter_demand_guard();
  }
  // The issuer cannot refill what the demand read is waiting for.
  auto refused = split_on(instant_context(), 0, "scripted://late");
  done_probe refused_done;
  CHECK(issue(refused, refused_done) == device_issue::pressure);

  // First refusal of the reservation: shed the speculation nobody claimed.
  CHECK(gpu->shed_unclaimed() == 1);
  CHECK(shed_ranges->abandoned());
  CHECK(shed->prefetched() == nullptr);
  CHECK_FALSE(claimed_ranges->abandoned());
  CHECK(claimed->prefetched() == claimed_ranges);
  // The shed buffer stays alive while its IO is in flight, and its token is held with it: both
  // are returned when the read settles (checked below), never while a read may still write.
  CHECK(shed_ranges->entry_count() == 1);
  CHECK(gpu->reserved() == k_split_reserved);
  gpu->exit_demand_guard();

  // Late completion of the shed read writes into buffers that are still there, then frees them.
  shed_gate.open();
  REQUIRE(shed_done.wait().ok);
  shed_ranges.reset();
  shed.reset();
  CHECK(gpu->reserved() == 0);

  claimed_gate.open();
  REQUIRE(claimed_done.wait().ok);
  CHECK(claimed_ranges->current() == prefetched_ranges::state::ready);
  claimed_ranges.reset();
  claimed.reset();
  task.reset();
  require_released();
}

TEST_CASE_METHOD(issuer_fixture,
                 "a refused demand reservation succeeds once the claim and the shed made room",
                 "[scan_manager][device_prefetch][claim]")
{
  if (!sirius::test::has_gpus(1)) { return; }
  auto claimed = split_on(instant_context(), 0, "scripted://claimed");
  auto spare   = split_on(instant_context(), 0, "scripted://spare");
  claim_harness harness;
  auto task = harness.task_for(claimed);
  done_probe claimed_done;
  done_probe spare_done;
  REQUIRE(issue(claimed, claimed_done) == device_issue::issued);
  REQUIRE(issue(spare, spare_done) == device_issue::issued);
  REQUIRE(claimed_done.wait().ok);
  REQUIRE(spare_done.wait().ok);

  // Another consumer fills the rest of the reservation limit, so nothing is left for the demand
  // read of the claimed task.
  auto const limit = space->get_max_memory();
  auto hog         = space->make_reservation_or_null(limit - 2 * k_split_bytes);
  REQUIRE(hog != nullptr);
  REQUIRE(space->make_reservation_or_null(1 * MiB) == nullptr);

  // The executor's order: claim (the buffers it holds are already allocated, so the request shrinks
  // by their size), raise the guard, ask once, shed, ask again -- all before any wait.
  auto const held = task->claim_device_readahead(*space);
  REQUIRE(held == k_split_reserved);
  {
    std::lock_guard lock(gpu->allocation_mutex());
    gpu->enter_demand_guard();
  }
  CHECK(gpu->reserved() == k_split_reserved);  // only the unclaimed spare still holds budget
  auto const needed = k_split_bytes;
  CHECK(space->make_reservation_or_null(needed) == nullptr);
  CHECK(gpu->shed_unclaimed() == 1);
  auto reservation = space->make_reservation_or_null(needed);
  CHECK(reservation != nullptr);
  CHECK(gpu->reserved() == 0);
  gpu->exit_demand_guard();

  // The claimed prefetch is untouched and still decodes.
  auto ranges = claimed->prefetched();
  REQUIRE(ranges != nullptr);
  rmm::cuda_stream consumer;
  auto taken = ranges->take(::cuda::stream_ref{consumer.value()});
  REQUIRE(taken.has_value());
  CHECK(all_equal(to_host(taken->front().spans[0], consumer.value()), fill_value(0)));

  taken.reset();
  ranges.reset();
  reservation.reset();
  hog.reset();
  task.reset();
  claimed.reset();
  spare.reset();
  require_released();
}

TEST_CASE_METHOD(issuer_fixture,
                 "shedding a landed prefetch returns its memory and token at once",
                 "[scan_manager][device_prefetch][claim]")
{
  if (!sirius::test::has_gpus(1)) { return; }
  auto split = split_on(instant_context());
  done_probe done;
  REQUIRE(issue(split, done) == device_issue::issued);
  REQUIRE(done.wait().ok);
  REQUIRE(gpu->reserved() == k_split_reserved);

  // Nothing is in flight, so the shed drops the last reference: the demand read that asked for
  // room finds it straight away.
  CHECK(gpu->shed_unclaimed() == 1);
  CHECK(split->prefetched() == nullptr);
  CHECK(gpu->reserved() == 0);
  require_released();
  // Shedding again finds nothing.
  CHECK(gpu->shed_unclaimed() == 0);
}

TEST_CASE_METHOD(issuer_fixture,
                 "shedding wakes a waiting consumer so it reads for itself",
                 "[scan_manager][device_prefetch][claim]")
{
  if (!sirius::test::has_gpus(1)) { return; }
  read_gate gate;
  auto split = split_on(gate.context());
  done_probe done;
  REQUIRE(issue(split, done) == device_issue::issued);
  REQUIRE(gate.wait_for_read());
  auto ranges = split->prefetched();
  REQUIRE(ranges != nullptr);

  std::promise<prefetched_ranges::state> seen;
  std::thread consumer([&] { seen.set_value(ranges->wait_for(30s)); });
  CHECK(gpu->shed_unclaimed() == 1);
  auto future = seen.get_future();
  REQUIRE(future.wait_for(30s) == std::future_status::ready);
  consumer.join();
  CHECK(future.get() == prefetched_ranges::state::failed);
  CHECK(ranges->abandoned());
  CHECK(split->prefetched() == nullptr);

  // The abandoned read still lands, and the buffers go once it has.
  gate.open();
  REQUIRE(done.wait().ok);
  ranges.reset();
  split.reset();
  require_released();
}

TEST_CASE_METHOD(issuer_fixture,
                 "a split destroyed before its read lands frees its memory when the read does",
                 "[scan_manager][device_prefetch][lifetime]")
{
  if (!sirius::test::has_gpus(1)) { return; }
  read_gate gate;
  auto split = split_on(gate.context());
  done_probe done;
  REQUIRE(issue(split, done) == device_issue::issued);
  REQUIRE(gate.wait_for_read());

  // Cancellation: the owner drops the split while the IO is in flight.  The fan-in keeps the
  // buffers alive, so the late write goes somewhere valid and the memory stays counted until then.
  split.reset();
  CHECK(done.calls->load() == 0);
  CHECK(space->get_available_memory() + k_split_bytes <= baseline);
  gate.open();
  REQUIRE(done.wait().ok);
  require_released();
}

TEST_CASE_METHOD(issuer_fixture,
                 "a read that fails after its split is gone still returns everything once",
                 "[scan_manager][device_prefetch][lifetime]")
{
  if (!sirius::test::has_gpus(1)) { return; }
  read_gate gate;
  auto split = split_on(gate.context());
  done_probe done;
  REQUIRE(issue(split, done) == device_issue::issued);
  REQUIRE(gate.wait_for_read());
  split.reset();
  gate.fail();
  auto const result = done.wait();
  CHECK_FALSE(result.ok);
  CHECK(done.calls->load() == 1);
  require_released();
}

TEST_CASE("the per-GPU state outlives the registry for an in-flight prefetch",
          "[scan_manager][device_prefetch][lifetime]")
{
  if (!sirius::test::has_gpus(1)) { return; }
  gpu_env env;
  auto pool                                    = scan::device_stream_pool::create(0, 1);
  auto registry                                = env.registry(0.5);
  std::weak_ptr<sm::device_readahead_gpu> weak = registry->gpu(0)->weak_from_this();
  REQUIRE_FALSE(weak.expired());

  read_gate gate;
  auto split = split_on(gate.context());
  done_probe done;
  REQUIRE(sm::issue_device_readahead(*registry->gpu(0), pool->acquire(), split, done.callback()) ==
          device_issue::issued);
  REQUIRE(gate.wait_for_read());

  // The runtime lets go of the registry while the read is still in flight.
  registry.reset();
  CHECK_FALSE(weak.expired());
  gate.open();
  REQUIRE(done.wait().ok);
  // The unconsumed prefetch dies later and still gives its token back to the live state.
  auto state = weak.lock();
  REQUIRE(state != nullptr);
  CHECK(state->reserved() == k_split_reserved);
  split.reset();
  CHECK(state->reserved() == 0);
  state.reset();
  CHECK(weak.expired());
  REQUIRE(cudaStreamSynchronize(pool->acquire().view.get()) == cudaSuccess);
}

//===----------------------------------------------------------------------===//
// Through the readahead worker
//===----------------------------------------------------------------------===//

namespace {

struct scan_operator : sirius::op::sirius_physical_operator {
  scan_operator()
    : sirius::op::sirius_physical_operator(sirius::op::SiriusPhysicalOperatorType::GPU_SCAN, {}, 0)
  {
    operator_id = 1;
  }
};

/// A query with one GPU_SCAN operator (id 1), which is what the readahead keys its work on.
class scan_query {
 public:
  scan_query()
  {
    auto pipeline = std::make_shared<sirius::pipeline::sirius_pipeline>(_ctx);
    _build.set_pipeline_source(*pipeline, *_scan);
    auto const id = sirius::make_query_id(1);
    _query        = std::make_unique<sirius::planner::query>(
      std::vector<std::shared_ptr<sirius::pipeline::sirius_pipeline>>{pipeline},
      _telemetry->context(),
      id,
      sirius::telemetry::query_telemetry_info{quent::now_v7(), _telemetry->worker_id().raw(), id});
  }

  [[nodiscard]] sirius::planner::query const& get() const { return *_query; }

 private:
  std::shared_ptr<const sirius::telemetry::telemetry_context> _telemetry =
    sirius::test::make_test_telemetry_context();
  sirius::pipeline::pipeline_build_context _ctx{nullptr, true};
  sirius::pipeline::sirius_pipeline_build_state _build;
  std::unique_ptr<scan_operator> _scan = std::make_unique<scan_operator>();
  std::unique_ptr<sirius::planner::query> _query;
};

/// One query's readahead over the shared @p registry.  Each owns its stream pools.
struct readahead_query {
  std::shared_ptr<sirius::event::query_event_publisher> publisher;
  scan_query query;
  std::shared_ptr<readahead_manager> manager;

  readahead_query(std::shared_ptr<sirius::event::query_event_publisher> shared_publisher,
                  std::shared_ptr<sm::device_readahead_registry> registry)
    : publisher(std::move(shared_publisher)),
      manager(std::make_shared<readahead_manager>(
        *publisher, 2, nullptr, std::nullopt, nullptr, std::move(registry)))
  {
    manager->prepare_for_query(query.get());
  }

  ~readahead_query() { manager->stop(); }

  [[nodiscard]] std::uint64_t outcomes(prefetch_outcome kind) const
  {
    return manager->counters().outcome(kind);
  }

  /// Publishes @p split (a closed operator emits nothing more) and starts the worker.  An operator
  /// must be closed only after its splits were registered: a closed queue refuses further pushes.
  void run(std::shared_ptr<scan::scan_info> const& split, bool close = false)
  {
    manager->register_scan_task(split, 1);
    if (close) { manager->mark_operator_closed(1); }
    manager->start();
  }
};

struct readahead_fixture {
  gpu_env env{1};
  cucascade::memory::memory_space* space = env.gpu_space();
  std::shared_ptr<sirius::event::query_event_publisher> publisher =
    std::make_shared<sirius::event::query_event_publisher>();
  std::size_t baseline = space->get_available_memory();

  [[nodiscard]] std::shared_ptr<sm::device_readahead_registry> registry(
    double budget_bytes, double threshold = 0.9) const
  {
    return env.registry(budget_bytes / static_cast<double>(k_capacity), threshold);
  }
};

}  // namespace

TEST_CASE_METHOD(
  readahead_fixture,
  "the worker lands a split on the device when enabled and leaves it to the host tier when not",
  "[scan_manager][device_prefetch][readahead]")
{
  if (!sirius::test::has_gpus(1)) { return; }

  std::shared_ptr<sm::device_readahead_registry> reg;
  bool expect_device = false;
  std::shared_ptr<device_split> split;
  read_gate gate;
  auto context = gate.context();

  SECTION("enabled")
  {
    reg           = registry(64 * MiB);
    expect_device = true;
    split         = split_on(context, 0);
  }
  SECTION("no registry at all")
  {
    reg   = nullptr;
    split = split_on(context, 0);
  }
  SECTION("a zero budget")
  {
    reg   = env.registry(0.0);
    split = split_on(context, 0);
  }
  SECTION("the split has no preferred device")
  {
    reg   = registry(64 * MiB);
    split = split_on(context, std::nullopt);
  }
  SECTION("the preferred GPU does not speculate")
  {
    reg   = registry(64 * MiB);
    split = split_on(context, 1);
  }
  SECTION("the format cannot decode from device copies")
  {
    reg        = registry(64 * MiB);
    auto plain = std::make_shared<scan::scan_info>(two_ranges(scripted_datasource(context)));
    plain->set_preferred_device(0);
    readahead_query query{publisher, reg};
    query.run(plain);
    REQUIRE(eventually([&] { return query.outcomes(prefetch_outcome::nothing_to_issue) == 1; }));
    CHECK(plain->prefetched() == nullptr);
    CHECK(gate.reads() == 0);
    return;
  }

  readahead_query query{publisher, reg};
  query.run(split, /*close=*/true);

  if (expect_device) {
    REQUIRE(gate.wait_for_read());
    gate.open();
    REQUIRE(eventually([&] { return query.outcomes(prefetch_outcome::prefetched) == 1; }));
    auto ranges = split->prefetched();
    REQUIRE(ranges != nullptr);
    CHECK(ranges->current() == prefetched_ranges::state::ready);
    CHECK(ranges->device_id() == 0);
    CHECK(reg->gpu(0)->reserved() == k_split_reserved);
  } else {
    // No scripted cache either: the host tier has nothing to stage into and settles the attempt.
    REQUIRE(eventually([&] { return query.outcomes(prefetch_outcome::nothing_to_issue) == 1; }));
    CHECK(split->prefetched() == nullptr);
    CHECK(gate.reads() == 0);
    CHECK(query.manager->counters().memory_retries.load() == 0);
  }
}

TEST_CASE_METHOD(readahead_fixture,
                 "the speculative budget is one pool per GPU shared by every query",
                 "[scan_manager][device_prefetch][readahead][budget]")
{
  if (!sirius::test::has_gpus(1)) { return; }
  // Exactly one standard split fits.
  auto reg = registry(static_cast<double>(k_split_bytes));
  REQUIRE(reg->gpu(0)->budget_bytes() == k_split_bytes);

  read_gate first_gate;
  read_gate third_gate;
  auto first  = split_on(first_gate.context(), 0, "scripted://first");
  auto second = split_on(instant_context(), 0, "scripted://second");
  auto third  = split_on(third_gate.context(), 0, "scripted://third");

  readahead_query query_a{publisher, reg};
  readahead_query query_b{publisher, reg};

  // Query A's read is in flight and holds the whole budget.
  query_a.run(first);
  REQUIRE(first_gate.wait_for_read());
  CHECK(reg->gpu(0)->reserved() == k_split_bytes);

  // Query B finds none: its split falls to the host tier (no cache, so nothing to issue) and
  // nothing is read for it on the device.
  query_b.run(second);
  REQUIRE(eventually([&] { return query_b.outcomes(prefetch_outcome::nothing_to_issue) == 1; }));
  CHECK(second->prefetched() == nullptr);
  CHECK(reg->gpu(0)->reserved() == k_split_bytes);

  // A's executor claims its split: the budget is free before the read has even finished.
  first->claim_device_readahead();
  CHECK(reg->gpu(0)->reserved() == 0);
  query_b.manager->register_scan_task(third, 1);
  REQUIRE(third_gate.wait_for_read());
  CHECK(reg->gpu(0)->reserved() == k_split_bytes);

  // The two queries' reads ran on different streams: each readahead owns its own pool.
  CHECK(first_gate.streams() != third_gate.streams());

  first_gate.open();
  third_gate.open();
  REQUIRE(eventually([&] { return query_a.outcomes(prefetch_outcome::prefetched) == 1; }));
  REQUIRE(eventually([&] { return query_b.outcomes(prefetch_outcome::prefetched) == 1; }));
}

TEST_CASE_METHOD(
  readahead_fixture,
  "past the usage ceiling the split falls back to the host tier without a device read",
  "[scan_manager][device_prefetch][readahead][budget]")
{
  if (!sirius::test::has_gpus(1)) { return; }
  read_gate gate;
  std::shared_ptr<sm::device_readahead_registry> reg;

  SECTION("the consumption threshold")
  {
    // Ceiling 20 MiB, 18 MiB already held by someone else: 18 + 3 would cross it.
    reg = registry(256 * MiB, 20.0 / static_cast<double>(k_capacity / MiB));
    REQUIRE(reg->gpu(0)->ceiling_bytes() == 20 * MiB);
  }
  SECTION("the downgrade stop, which caps a threshold above it") { reg = registry(256 * MiB, 1.0); }

  auto const ceiling = reg->gpu(0)->ceiling_bytes();
  auto held          = space->make_reservation_or_null(ceiling - 2 * MiB);
  REQUIRE(held != nullptr);

  auto split = split_on(gate.context());
  readahead_query query{publisher, reg};
  query.run(split, /*close=*/true);
  REQUIRE(eventually([&] { return query.outcomes(prefetch_outcome::nothing_to_issue) == 1; }));
  CHECK(split->prefetched() == nullptr);
  CHECK(gate.reads() == 0);
  CHECK(reg->gpu(0)->reserved() == 0);
  query.manager->stop();

  // Once the other consumer lets go, the same GPU speculates again.
  held.reset();
  auto retry = split_on(instant_context());
  done_probe done;
  auto pool = scan::device_stream_pool::create(0, 1);
  REQUIRE(sm::issue_device_readahead(*reg->gpu(0), pool->acquire(), retry, done.callback()) ==
          device_issue::issued);
  REQUIRE(done.wait().ok);
  retry.reset();
  REQUIRE(cudaStreamSynchronize(pool->acquire().view.get()) == cudaSuccess);
  CHECK(space->get_available_memory() == baseline);
}

TEST_CASE_METHOD(readahead_fixture,
                 "a prefetch outlives its readahead manager, its stream pool and its registry",
                 "[scan_manager][device_prefetch][readahead][lifetime]")
{
  if (!sirius::test::has_gpus(1)) { return; }
  read_gate gate;
  auto split = split_on(gate.context());
  std::weak_ptr<sm::device_readahead_gpu> gpu_state;
  {
    auto reg  = registry(64 * MiB);
    gpu_state = reg->gpu(0)->weak_from_this();
    readahead_query query{publisher, reg};
    query.run(split, /*close=*/true);
    REQUIRE(gate.wait_for_read());
    // The query ends (stop joins the worker; the manager and its stream pool are dropped) with the
    // read still in flight.
  }
  REQUIRE_FALSE(gpu_state.expired());
  auto ranges = split->prefetched();
  REQUIRE(ranges != nullptr);
  CHECK(ranges->current() == prefetched_ranges::state::pending);

  // The read lands after the manager, the pool handle and the registry are gone.
  gate.open();
  REQUIRE(ranges->wait_for(30s) == prefetched_ranges::state::ready);
  rmm::cuda_stream consumer;
  auto taken = ranges->take(::cuda::stream_ref{consumer.value()});
  REQUIRE(taken.has_value());
  CHECK(all_equal(to_host(taken->front().spans[0], consumer.value()), fill_value(0)));
  taken.reset();
  ranges.reset();
  split.reset();
  CHECK(gpu_state.expired());
  CHECK(space->get_available_memory() == baseline);
}

//===----------------------------------------------------------------------===//
// Real local IO, end to end
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Retained stream-pool groups
//===----------------------------------------------------------------------===//

TEST_CASE("the registry lends private stream pools and reuses them once their borrowers are gone",
          "[scan_manager][device_prefetch][pool]")
{
  if (!sirius::test::has_gpus(1)) { return; }
  gpu_env env;
  auto registry = env.registry(0.5);

  // Concurrent borrowers never share a stream.
  auto a = registry->acquire_pool(0);
  auto b = registry->acquire_pool(0);
  REQUIRE((a != nullptr && b != nullptr));
  CHECK(a != b);
  std::set<cudaStream_t> streams;
  for (auto const& pool : {a, b}) {
    for (std::size_t i = 0; i < pool->size(); ++i) {
      streams.insert(pool->acquire().view.get());
    }
  }
  CHECK(streams.size() == a->size() + b->size());

  // A group whose borrower is gone is the next one lent: the same stream handles serve again.
  auto* const first = a.get();
  a.reset();
  auto again = registry->acquire_pool(0);
  CHECK(again.get() == first);

  // A prefetch buffer still leasing a group keeps it busy, manager or no manager.
  auto lease = again->acquire();
  again.reset();
  auto other = registry->acquire_pool(0);
  CHECK(other.get() != first);
  other.reset();
  lease         = scan::device_stream_pool::lease{};
  auto released = registry->acquire_pool(0);
  CHECK(released.get() == first);
}

TEST_CASE("the registry creates a bounded number of stream groups and then refuses",
          "[scan_manager][device_prefetch][pool]")
{
  if (!sirius::test::has_gpus(1)) { return; }
  gpu_env env;
  auto registry = env.registry(0.5);

  std::vector<std::shared_ptr<scan::device_stream_pool>> held;
  for (std::size_t i = 0; i < sm::device_readahead_registry::max_pool_groups; ++i) {
    auto pool = registry->acquire_pool(0);
    REQUIRE(pool != nullptr);
    held.push_back(std::move(pool));
  }
  CHECK(registry->acquire_pool(0) == nullptr);

  // Letting one go makes exactly that group available again.
  auto* const freed = held.back().get();
  held.pop_back();
  auto reused = registry->acquire_pool(0);
  CHECK(reused.get() == freed);
  CHECK(registry->acquire_pool(0) == nullptr);
}

TEST_CASE_METHOD(readahead_fixture,
                 "sequential queries reuse one retained group instead of creating one per query",
                 "[scan_manager][device_prefetch][readahead][pool]")
{
  if (!sirius::test::has_gpus(1)) { return; }
  auto reg = registry(64 * MiB);

  // Remember which group an idle registry lends first.
  auto* first = [&] {
    auto probe = reg->acquire_pool(0);
    REQUIRE(probe != nullptr);
    return probe.get();
  }();

  // More queries than the registry would ever create groups for: a pool per query would be
  // refused (host fallback) after the cap, so every query landing on the device proves reuse.
  for (std::size_t i = 0; i < sm::device_readahead_registry::max_pool_groups + 8; ++i) {
    {
      readahead_query query{publisher, reg};
      auto split = split_on(instant_context(), 0);
      query.run(split, /*close=*/true);
      REQUIRE(eventually([&] { return query.outcomes(prefetch_outcome::prefetched) == 1; }));
      REQUIRE(split->prefetched() != nullptr);
    }
    // The manager and its split are gone; the group is the registry's again.
    REQUIRE(eventually([&] {
      auto pool = reg->acquire_pool(0);
      return pool != nullptr && pool.get() == first;
    }));
  }
}

TEST_CASE_METHOD(readahead_fixture,
                 "a group stays busy while a prefetch buffer outlives its manager",
                 "[scan_manager][device_prefetch][readahead][pool]")
{
  if (!sirius::test::has_gpus(1)) { return; }
  auto reg    = registry(64 * MiB);
  auto* first = [&] {
    auto probe = reg->acquire_pool(0);
    REQUIRE(probe != nullptr);
    return probe.get();
  }();

  std::shared_ptr<prefetched_ranges> landed;
  {
    readahead_query query{publisher, reg};
    auto split = split_on(instant_context(), 0);
    query.run(split, /*close=*/true);
    REQUIRE(eventually([&] { return query.outcomes(prefetch_outcome::prefetched) == 1; }));
    landed = split->prefetched();
    REQUIRE(landed != nullptr);
  }
  // The manager is destroyed but its buffer is alive: a new query must not be handed that group.
  {
    auto other = reg->acquire_pool(0);
    REQUIRE(other != nullptr);
    CHECK(other.get() != first);
  }
  landed.reset();
  REQUIRE(eventually([&] {
    auto pool = reg->acquire_pool(0);
    return pool != nullptr && pool.get() == first;
  }));
}

TEST_CASE_METHOD(readahead_fixture,
                 "with every group busy the worker falls back to the host tier",
                 "[scan_manager][device_prefetch][readahead][pool]")
{
  if (!sirius::test::has_gpus(1)) { return; }
  auto reg = registry(64 * MiB);
  std::vector<std::shared_ptr<scan::device_stream_pool>> held;
  for (std::size_t i = 0; i < sm::device_readahead_registry::max_pool_groups; ++i) {
    held.push_back(reg->acquire_pool(0));
    REQUIRE(held.back() != nullptr);
  }

  read_gate gate;
  auto split = split_on(gate.context(), 0);
  readahead_query query{publisher, reg};
  query.run(split, /*close=*/true);
  // No device copy and no device read; with no scripted cache the host tier settles it.
  REQUIRE(eventually([&] { return query.outcomes(prefetch_outcome::nothing_to_issue) == 1; }));
  CHECK(split->prefetched() == nullptr);
  CHECK(gate.reads() == 0);
  CHECK(reg->gpu(0)->reserved() == 0);
}

namespace {

/// An 8 MiB file whose bytes are a position-dependent pattern, so a misplaced range is caught.
struct pattern_file {
  std::filesystem::path path;

  static std::uint8_t byte_at(std::size_t offset)
  {
    return static_cast<std::uint8_t>((offset * 31 + (offset >> 12)) & 0xff);
  }

  pattern_file()
  {
    path = std::filesystem::temp_directory_path() /
           ("sirius_device_prefetch_" + std::to_string(::getpid()) + "_" +
            std::to_string(reinterpret_cast<std::uintptr_t>(this)));
    std::ofstream out(path, std::ios::binary);
    std::vector<char> block(1 * MiB);
    for (std::size_t base = 0; base < k_file; base += block.size()) {
      for (std::size_t i = 0; i < block.size(); ++i) {
        block[i] = static_cast<char>(byte_at(base + i));
      }
      out.write(block.data(), static_cast<std::streamsize>(block.size()));
    }
  }

  ~pattern_file()
  {
    std::error_code ec;
    std::filesystem::remove(path, ec);
  }
};

void require_file_bytes(cudf::device_span<std::uint8_t const> span,
                        std::size_t offset,
                        cudaStream_t stream)
{
  auto const host = to_host(span, stream);
  for (std::size_t i = 0; i < host.size(); ++i) {
    if (host[i] != pattern_file::byte_at(offset + i)) {
      FAIL("byte " << i << " of the range at " << offset << " differs from the file");
    }
  }
}

std::shared_ptr<const sirius::memory::topology_index> single_gpu_topology()
{
  cucascade::memory::system_topology_info topology;
  topology.num_gpus = 1;
  cucascade::memory::gpu_topology_info gpu;
  gpu.id        = 0;
  gpu.numa_node = 0;
  topology.gpus.push_back(std::move(gpu));
  return std::make_shared<sirius::memory::topology_index>(topology, std::vector<int>{0});
}

/// Publishes @p split the way the engine does -- stamping the device, hinting and registering it --
/// and returns the input that owns it.
std::unique_ptr<scan::scan_operator_input> publish(
  std::shared_ptr<scan::scan_info> const& split, std::shared_ptr<readahead_manager> const& manager)
{
  return std::make_unique<scan::scan_operator_input>(split, manager, 1, 0);
}

}  // namespace

TEST_CASE("a real local file reaches the device through kvikIO with the cache off",
          "[scan_manager][device_prefetch][e2e]")
{
  if (!sirius::test::has_gpus(1)) { return; }
  pattern_file file;
  gpu_env env;
  auto* space         = env.gpu_space();
  auto const baseline = space->get_available_memory();
  auto registry       = env.registry(0.5);
  auto publisher      = std::make_shared<sirius::event::query_event_publisher>();

  auto context = std::make_shared<sirius::io::kvikio_context>();
  std::shared_ptr<sirius::io::sirius_datasource> ds = context->open_datasource(file.path.string());
  REQUIRE_FALSE(context->uses_prefetching_cache());

  {
    readahead_query query{publisher, registry};
    auto split = std::make_shared<device_split>(two_ranges(ds));
    auto input = publish(split, query.manager);
    query.manager->mark_operator_closed(1);
    query.manager->start();
    REQUIRE(eventually([&] { return query.outcomes(prefetch_outcome::prefetched) == 1; }));

    auto ranges = split->prefetched();
    REQUIRE(ranges != nullptr);
    REQUIRE(ranges->wait_for(30s) == prefetched_ranges::state::ready);
    CHECK(ranges->total_bytes() == k_split_bytes);

    // The executor's claim, then the decode takes the buffers.
    CHECK(input->claim_device_readahead(*space) == k_split_reserved);
    CHECK(registry->gpu(0)->reserved() == 0);
    rmm::cuda_stream consumer;
    auto taken = ranges->take(::cuda::stream_ref{consumer.value()});
    REQUIRE(taken.has_value());
    REQUIRE(taken->front().spans.size() == 2);
    require_file_bytes(taken->front().spans[0], 0, consumer.value());
    require_file_bytes(taken->front().spans[1], 3 * MiB, consumer.value());
    taken.reset();
    ranges.reset();
    consumer.synchronize();
    input.reset();
  }
  CHECK(space->get_available_memory() == baseline);
}

TEST_CASE("a real local file reaches the device through io_uring and the prefetching cache",
          "[scan_manager][device_prefetch][e2e]")
{
  if (!sirius::test::has_gpus(1)) { return; }
  pattern_file file;
  gpu_env env;
  auto* space         = env.gpu_space();
  auto const baseline = space->get_available_memory();

  sm::scan_manager_config cfg;
  cfg.thread_pool.num_threads      = 3;
  cfg.uring_n_reactors             = 1;
  cfg.cache.mode                   = sirius::io::cache::cache_mode::sirius;
  cfg.cache.eviction               = sirius::io::cache::eviction_policy::idle;
  cfg.device_prefetching_budget    = 0.5;
  cfg.device_prefetching_threshold = 0.9;
  cfg.apply_cache_mode();

  // Declared before the datasource: the manager owns the cached ioctx it reads through.
  sirius::exec::query_lifecycle_registry lifecycle;
  sm::sirius_scan_manager manager{
    lifecycle, cfg, *env.manager, single_gpu_topology(), nullptr, env.configs};
  auto* cache = manager.io_ctx()->cache();
  REQUIRE(cache != nullptr);
  // The runtime built the registry from the same configs.
  auto registry = manager.device_readahead();
  REQUIRE(registry != nullptr);
  REQUIRE(registry->gpu(0) != nullptr);
  CHECK(registry->gpu(0)->budget_bytes() ==
        static_cast<std::size_t>(0.5 * static_cast<double>(k_capacity)));

  auto ds = manager.create_datasource(file.path.string());
  REQUIRE(ds->io_ctx()->uses_prefetching_cache());
  auto publisher = std::make_shared<sirius::event::query_event_publisher>();

  SECTION("the device copy lands, byte for byte")
  {
    {
      readahead_query query{publisher, registry};
      auto split = std::make_shared<device_split>(two_ranges(ds));
      auto input = publish(split, query.manager);
      query.manager->mark_operator_closed(1);
      query.manager->start();
      REQUIRE(eventually([&] { return query.outcomes(prefetch_outcome::prefetched) == 1; }));

      auto ranges = split->prefetched();
      REQUIRE(ranges != nullptr);
      REQUIRE(ranges->wait_for(30s) == prefetched_ranges::state::ready);
      rmm::cuda_stream consumer;
      auto taken = ranges->take(::cuda::stream_ref{consumer.value()});
      REQUIRE(taken.has_value());
      require_file_bytes(taken->front().spans[0], 0, consumer.value());
      require_file_bytes(taken->front().spans[1], 3 * MiB, consumer.value());
      taken.reset();
      ranges.reset();
      consumer.synchronize();
      input.reset();
    }
    CHECK(registry->gpu(0)->reserved() == 0);
  }

  SECTION("past the ceiling the host tier stages the split through the cache instead")
  {
    // The registry's ceiling is the downgrade stop at 0.65 of capacity: stand just under it.
    auto const ceiling = registry->gpu(0)->ceiling_bytes();
    auto held          = space->make_reservation_or_null(ceiling - 1 * MiB);
    REQUIRE(held != nullptr);

    readahead_query query{publisher, registry};
    auto split = std::make_shared<device_split>(two_ranges(ds));
    auto input = publish(split, query.manager);
    query.manager->mark_operator_closed(1);
    query.manager->start();
    REQUIRE(eventually([&] { return query.outcomes(prefetch_outcome::prefetched) == 1; }));

    CHECK(split->prefetched() == nullptr);  // no device copy
    CHECK(registry->gpu(0)->reserved() == 0);
    // The host tier did the work: the split's cache request owns staging, and a demand read of
    // its range is served with the file's bytes.
    CHECK(cache->claimed_bytes() > 0);
    std::vector<std::uint8_t> bytes(4096);
    REQUIRE(ds->host_read(3 * MiB, bytes.size(), bytes.data()) == bytes.size());
    for (std::size_t i = 0; i < bytes.size(); ++i) {
      REQUIRE(bytes[i] == pattern_file::byte_at(3 * MiB + i));
    }
    input.reset();
  }
  ds.reset();
  CHECK(space->get_available_memory() == baseline);
}

TEST_CASE("260 sequential cached device prefetches reuse one retained stream group",
          "[scan_manager][device_prefetch][pool][e2e]")
{
  if (!sirius::test::has_gpus(1)) { return; }
  pattern_file file;
  gpu_env env;
  auto* space         = env.gpu_space();
  auto const baseline = space->get_available_memory();

  sm::scan_manager_config cfg;
  cfg.thread_pool.num_threads      = 3;
  cfg.uring_n_reactors             = 1;
  cfg.cache.mode                   = sirius::io::cache::cache_mode::sirius;
  cfg.cache.eviction               = sirius::io::cache::eviction_policy::idle;
  cfg.device_prefetching_budget    = 0.5;
  cfg.device_prefetching_threshold = 0.9;
  cfg.apply_cache_mode();

  // One backend, one cache and one registry for every query below.  The manager owns the registry
  // and shuts every cache down BEFORE its members are destroyed, so no cache completion poll can
  // see a stream the retained groups release afterwards.
  sirius::exec::query_lifecycle_registry lifecycle;
  sm::sirius_scan_manager manager{
    lifecycle, cfg, *env.manager, single_gpu_topology(), nullptr, env.configs};
  auto registry = manager.device_readahead();
  REQUIRE(registry != nullptr);
  auto ds = manager.create_datasource(file.path.string());
  REQUIRE(ds->io_ctx()->uses_prefetching_cache());
  auto publisher = std::make_shared<sirius::event::query_event_publisher>();

  // More queries than the cache's completion poll has lanes for stream handles (256), and more
  // than the registry has groups (32): a stream, or a group, per query would exhaust one of them.
  constexpr int k_queries               = 260;
  scan::device_stream_pool const* group = nullptr;
  std::set<cudaStream_t> streams;
  rmm::cuda_stream consumer;
  for (int i = 0; i < k_queries; ++i) {
    {
      readahead_query query{publisher, registry};
      auto split = std::make_shared<device_split>(two_ranges(ds));
      auto input = publish(split, query.manager);
      query.manager->mark_operator_closed(1);
      query.manager->start();
      REQUIRE(eventually([&] { return query.outcomes(prefetch_outcome::prefetched) == 1; }));

      auto ranges = split->prefetched();
      REQUIRE(ranges != nullptr);
      REQUIRE(ranges->wait_for(30s) == prefetched_ranges::state::ready);
      auto taken = ranges->take(::cuda::stream_ref{consumer.value()});
      REQUIRE(taken.has_value());
      require_file_bytes(taken->front().spans[0], 0, consumer.value());
      require_file_bytes(taken->front().spans[1], 3 * MiB, consumer.value());
      taken.reset();
      ranges.reset();
      consumer.synchronize();
      input.reset();
    }  // the manager, its split and its buffers are gone

    // The retained group is the registry's again, and it is the very same object every time, so
    // its fixed stream set is what the cache saw on query one.
    std::shared_ptr<scan::device_stream_pool> pool;
    REQUIRE(eventually([&] {
      pool = registry->acquire_pool(0);
      return pool != nullptr;
    }));
    if (group == nullptr) {
      group = pool.get();
      for (std::size_t k = 0; k < pool->size(); ++k) {
        streams.insert(pool->acquire().view.get());
      }
    }
    REQUIRE(pool.get() == group);
    for (std::size_t k = 0; k < pool->size(); ++k) {
      REQUIRE(streams.contains(pool->acquire().view.get()));
    }
    CHECK(registry->gpu(0)->reserved() == 0);
  }
  ds.reset();
}

//===----------------------------------------------------------------------===//
// Multiple GPUs
//===----------------------------------------------------------------------===//

TEST_CASE("each GPU rations its own budget and a prefetch on the wrong GPU is discarded",
          "[scan_manager][device_prefetch][multi_gpu]")
{
  if (!sirius::test::has_gpus(2)) { return; }
  gpu_env env{2};
  auto registry = env.registry(0.5);
  REQUIRE(registry->gpu(0) != nullptr);
  REQUIRE(registry->gpu(1) != nullptr);
  REQUIRE(registry->gpu(0) != registry->gpu(1));
  auto* space0 = env.gpu_space(0);
  auto* space1 = env.gpu_space(1);

  auto pool0 = scan::device_stream_pool::create(0, 1);
  auto pool1 = scan::device_stream_pool::create(1, 1);
  CHECK(pool0->device() == 0);
  CHECK(pool1->device() == 1);

  auto split = split_on(instant_context(), 1);
  done_probe done;
  REQUIRE(sm::issue_device_readahead(*registry->gpu(1), pool1->acquire(), split, done.callback()) ==
          device_issue::issued);
  REQUIRE(done.wait().ok);

  auto ranges = split->prefetched();
  REQUIRE(ranges != nullptr);
  CHECK(ranges->device_id() == 1);
  // The budget and the memory are GPU 1's; GPU 0 is untouched.
  CHECK(registry->gpu(1)->reserved() == k_split_reserved);
  CHECK(registry->gpu(0)->reserved() == 0);
  CHECK(space1->get_available_memory() + k_split_bytes <= k_capacity);
  CHECK(space0->get_available_memory() == k_capacity);
  {
    cudaPointerAttributes attributes{};
    REQUIRE(cudaPointerGetAttributes(&attributes, ranges->peek().front().spans.front().data()) ==
            cudaSuccess);
    CHECK(attributes.device == 1);
  }

  // The task was re-pinned to GPU 0: the executor's claim there finds a prefetch on the wrong GPU.
  scan::scan_operator_input input{split};
  CHECK(input.claim_device_readahead(*space0) == 0);
  CHECK(ranges->claimed());
  CHECK(registry->gpu(1)->reserved() == 0);
  CHECK(split->prefetched() == nullptr);  // discarded; the decode reads for itself
  rmm::cuda_stream consumer;
  CHECK_FALSE(ranges->take(::cuda::stream_ref{consumer.value()})
                .has_value());  // flagged wrong-device: refused

  // On its own GPU the same claim keeps it.
  auto second = split_on(instant_context(), 1, "scripted://again");
  done_probe second_done;
  REQUIRE(sm::issue_device_readahead(
            *registry->gpu(1), pool1->acquire(), second, second_done.callback()) ==
          device_issue::issued);
  REQUIRE(second_done.wait().ok);
  scan::scan_operator_input second_input{second};
  CHECK(second_input.claim_device_readahead(*space1) == k_split_reserved);
  CHECK(second->prefetched() != nullptr);

  ranges.reset();
  second_input.discard_device_readahead();
  second.reset();
  split.reset();
  {
    rmm::cuda_set_device_raii bind{rmm::cuda_device_id{1}};
    REQUIRE(cudaStreamSynchronize(pool1->acquire().view.get()) == cudaSuccess);
  }
  CHECK(space1->get_available_memory() == k_capacity);
}

TEST_CASE("the worker places a split on its preferred GPU and a GPU without budget falls back",
          "[scan_manager][device_prefetch][readahead][multi_gpu]")
{
  if (!sirius::test::has_gpus(2)) { return; }
  gpu_env env{2};
  // Only GPU 1 speculates: build the registry from its config alone.
  std::vector<cucascade::memory::memory_space_config> only_gpu1;
  for (auto const& config : env.configs) {
    if (auto const* gpu = std::get_if<cucascade::memory::gpu_memory_space_config>(&config);
        gpu != nullptr && gpu->device_id == 1) {
      only_gpu1.push_back(config);
    }
  }
  auto registry =
    std::make_shared<sm::device_readahead_registry>(*env.manager, 0.5, 0.9, only_gpu1);
  REQUIRE(registry->gpu(0) == nullptr);
  REQUIRE(registry->gpu(1) != nullptr);

  auto publisher = std::make_shared<sirius::event::query_event_publisher>();
  readahead_query query{publisher, registry};
  auto on_one  = split_on(instant_context(), 1, "scripted://one");
  auto on_zero = split_on(instant_context(), 0, "scripted://zero");
  query.manager->register_scan_task(on_one, 1);
  query.manager->register_scan_task(on_zero, 1);
  query.manager->mark_operator_closed(1);
  query.manager->start();
  REQUIRE(eventually([&] {
    return query.outcomes(prefetch_outcome::prefetched) == 1 &&
           query.outcomes(prefetch_outcome::nothing_to_issue) == 1;
  }));
  REQUIRE(on_one->prefetched() != nullptr);
  CHECK(on_one->prefetched()->device_id() == 1);
  CHECK(on_zero->prefetched() == nullptr);
}
