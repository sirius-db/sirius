// SPDX-License-Identifier: Apache-2.0
#include "api/simpatico_codegen.hpp"
#include "codegen/decode/jit/renderer.hpp"
#include "codegen/jit/kernel_cache.hpp"
#include "decode/decode_session.hpp"
#include "decode_session_test_access.hpp"
#include "test_utils.hpp"

#include <cudf/copying.hpp>
#include <cudf/dictionary/dictionary_factories.hpp>
#include <cudf/utilities/pinned_memory.hpp>

#include <rmm/cuda_stream.hpp>
#include <rmm/error.hpp>
#include <rmm/mr/cuda_async_memory_resource.hpp>
#include <rmm/mr/per_device_resource.hpp>

#include <cuda/memory_resource>

#include <array>
#include <atomic>
#include <bit>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <map>
#include <mutex>
#include <numeric>
#include <optional>
#include <thread>
#include <utility>

// Linker wrappers affect only explicit calls from this executable. Injection is scoped to the
// submitting thread and one raw stream; event markers and independent gate controllers stay real.
namespace stream_fault {
enum class operation { none, query, synchronize };
struct calls {
  cudaStream_t stream          = nullptr;
  std::size_t queries          = 0;
  std::size_t synchronizations = 0;
  bool occupied                = false;
};
thread_local operation next      = operation::none;
thread_local cudaStream_t target = nullptr;
thread_local bool observing      = false;
thread_local std::array<calls, 4> counts{};

void observe(operation op, cudaStream_t stream) noexcept
{
  if (!observing) return;
  for (auto& count : counts) {
    if (!count.occupied || count.stream == stream) {
      count.occupied = true;
      count.stream   = stream;
      if (op == operation::query)
        ++count.queries;
      else
        ++count.synchronizations;
      return;
    }
  }
}
bool consume(operation expected, cudaStream_t stream) noexcept
{
  if (next != expected || stream != target) return false;
  next = operation::none;
  return true;
}
}  // namespace stream_fault

extern "C" cudaError_t __real_cudaStreamQuery(cudaStream_t);
extern "C" cudaError_t __real_cudaStreamSynchronize(cudaStream_t);
extern "C" cudaError_t __wrap_cudaStreamQuery(cudaStream_t stream)
{
  stream_fault::observe(stream_fault::operation::query, stream);
  return stream_fault::consume(stream_fault::operation::query, stream)
           ? cudaErrorInvalidValue
           : __real_cudaStreamQuery(stream);
}
extern "C" cudaError_t __wrap_cudaStreamSynchronize(cudaStream_t stream)
{
  stream_fault::observe(stream_fault::operation::synchronize, stream);
  // Complete the real work before returning the injected error, leaving the context reusable.
  auto const status = __real_cudaStreamSynchronize(stream);
  return stream_fault::consume(stream_fault::operation::synchronize, stream) ? cudaErrorInvalidValue
                                                                             : status;
}

namespace {

struct stream_failure_scope {
  stream_failure_scope(stream_fault::operation operation, cudaStream_t stream)
  {
    stream_fault::target = stream;
    stream_fault::next   = operation;
  }
  ~stream_failure_scope() { stream_fault::next = stream_fault::operation::none; }
};

struct stream_observation_scope {
  stream_observation_scope()
  {
    stream_fault::counts    = {};
    stream_fault::observing = true;
  }
  ~stream_observation_scope() { stream_fault::observing = false; }
  stream_fault::calls count(cudaStream_t stream) const
  {
    for (auto const& calls : stream_fault::counts)
      if (calls.occupied && calls.stream == stream) return calls;
    return {};
  }
};

void cuda_check(cudaError_t status)
{
  if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}

template <typename F>
void expect_failure(F&& action, char const* message)
{
  bool failed = false;
  try {
    action();
  } catch (std::exception const&) {
    failed = true;
  }
  expect(failed, message);
}

class event_markers {
 public:
  explicit event_markers(simpatico::stream_pool const& pool) : streams_(pool.streams)
  {
    events_.resize(streams_.size(), nullptr);
    try {
      for (auto& event : events_)
        cuda_check(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
    } catch (...) {
      destroy();
      throw;
    }
  }
  ~event_markers() { destroy(); }
  event_markers(event_markers const&)            = delete;
  event_markers& operator=(event_markers const&) = delete;

  void record()
  {
    for (std::size_t i = 0; i < events_.size(); ++i)
      cuda_check(cudaEventRecord(events_[i], streams_[i]));
  }
  bool complete() const
  {
    for (auto event : events_) {
      if (cudaEventQuery(event) != cudaSuccess) return false;
    }
    return true;
  }

 private:
  void destroy() noexcept
  {
    for (auto event : events_) {
      if (event) (void)cudaEventDestroy(event);
    }
  }
  std::vector<cudaStream_t> streams_;
  std::vector<cudaEvent_t> events_;
};

class injected_out_of_memory final : public rmm::out_of_memory {
 public:
  explicit injected_out_of_memory(std::size_t bytes)
    : rmm::out_of_memory("async decode injected allocation failure"), requested_bytes(bytes)
  {
  }
  std::size_t requested_bytes;
};

// Shared state keeps resource copies observable. Deallocation is stream ordered and never waits.
class checked_resource {
 public:
  struct allocation {
    std::size_t bytes;
    cudaStream_t stream;
  };
  struct state {
    std::mutex mutex;
    std::thread::id owner = std::this_thread::get_id();
    std::map<void*, allocation> live;
    std::vector<cudaStream_t> attempts;
    std::optional<std::size_t> fail_at;
    std::size_t failed_request_bytes           = 0;
    event_markers* failure_markers             = nullptr;
    std::atomic<bool>* release_required        = nullptr;
    std::atomic<bool>* second_release_required = nullptr;
    bool wrong_thread                          = false;
    bool wrong_stream                          = false;
    bool early_release                         = false;
    bool synchronous_access                    = false;
  };

  explicit checked_resource(rmm::device_async_resource_ref upstream)
    : observations(std::make_shared<state>()), upstream_(upstream)
  {
  }

  void* allocate(cuda::stream_ref stream, std::size_t bytes, std::size_t alignment)
  {
    std::lock_guard lock(observations->mutex);
    observations->wrong_thread |= observations->owner != std::this_thread::get_id();
    auto const attempt = observations->attempts.size();
    observations->attempts.push_back(stream.get());
    if (observations->fail_at == attempt) {
      if (observations->failure_markers) observations->failure_markers->record();
      observations->failed_request_bytes = bytes;
      throw injected_out_of_memory(bytes);
    }
    auto* ptr = upstream_.allocate(stream, bytes, alignment);
    try {
      if (ptr) observations->live.emplace(ptr, allocation{bytes, stream.get()});
    } catch (...) {
      upstream_.deallocate(stream, ptr, bytes, alignment);
      throw;
    }
    return ptr;
  }

  void deallocate(cuda::stream_ref stream,
                  void* ptr,
                  std::size_t bytes,
                  std::size_t alignment) noexcept
  {
    std::lock_guard lock(observations->mutex);
    observations->wrong_thread |= observations->owner != std::this_thread::get_id();
    if ((observations->release_required && !observations->release_required->load()) ||
        (observations->second_release_required && !observations->second_release_required->load())) {
      observations->early_release = true;
    }
    if (ptr) {
      auto const found = observations->live.find(ptr);
      observations->wrong_stream |= found == observations->live.end() ||
                                    found->second.stream != stream.get() ||
                                    found->second.bytes != bytes;
      if (found != observations->live.end()) observations->live.erase(found);
    }
    upstream_.deallocate(stream, ptr, bytes, alignment);
  }

  void* allocate_sync(std::size_t bytes, std::size_t alignment)
  {
    observations->synchronous_access = true;
    auto* result = allocate(cuda::stream_ref{cudaStream_t{nullptr}}, bytes, alignment);
    cuda_check(cudaStreamSynchronize(nullptr));
    return result;
  }

  void deallocate_sync(void* ptr, std::size_t bytes, std::size_t alignment) noexcept
  {
    observations->synchronous_access = true;
    deallocate(cuda::stream_ref{cudaStream_t{nullptr}}, ptr, bytes, alignment);
    (void)cudaStreamSynchronize(nullptr);
  }

  bool operator==(checked_resource const& other) const noexcept
  {
    return observations == other.observations;
  }
  friend void get_property(checked_resource const&, cuda::mr::device_accessible) noexcept {}

  void reset()
  {
    expect(observations->live.empty(), "resource reset with live allocations");
    observations->attempts.clear();
    observations->fail_at.reset();
    observations->failed_request_bytes    = 0;
    observations->failure_markers         = nullptr;
    observations->release_required        = nullptr;
    observations->second_release_required = nullptr;
    observations->wrong_thread            = false;
    observations->wrong_stream            = false;
    observations->early_release           = false;
    observations->synchronous_access      = false;
  }
  void check() const
  {
    expect(observations->live.empty(), "decode leaked allocations");
    expect(!observations->wrong_thread, "decode allocated or freed on another CPU thread");
    expect(!observations->wrong_stream,
           "decode freed on a different stream or with incorrect size");
    expect(!observations->early_release,
           "pending owner released storage before its stream completed");
    expect(!observations->synchronous_access, "decode used synchronous resource access");
  }

  std::shared_ptr<state> observations;

 private:
  rmm::device_async_resource_ref upstream_;
};

static_assert(cuda::mr::resource_with<checked_resource, cuda::mr::device_accessible>);

class current_resource_guard {
 public:
  explicit current_resource_guard(rmm::device_async_resource_ref resource)
    : previous_(rmm::mr::set_current_device_resource(
        cuda::mr::any_resource<cuda::mr::device_accessible>{resource}))
  {
  }
  ~current_resource_guard() { rmm::mr::set_current_device_resource(std::move(previous_)); }
  current_resource_guard(current_resource_guard const&)            = delete;
  current_resource_guard& operator=(current_resource_guard const&) = delete;

 private:
  cuda::mr::any_resource<cuda::mr::device_accessible> previous_;
};

class pinned_resource_guard {
 public:
  explicit pinned_resource_guard(rmm::host_device_async_resource_ref resource)
    : previous_(cudf::set_pinned_memory_resource(resource))
  {
  }
  ~pinned_resource_guard() { cudf::set_pinned_memory_resource(previous_); }
  pinned_resource_guard(pinned_resource_guard const&)            = delete;
  pinned_resource_guard& operator=(pinned_resource_guard const&) = delete;

 private:
  rmm::host_device_async_resource_ref previous_;
};

// Reuse one pinned address without adding waits: publication must finish before returning it.
class pinned_scalar_resource {
 public:
  struct state {
    explicit state(rmm::cuda_stream_view stream)
      : storage(sizeof(int64_t), stream, cudf::get_pinned_memory_resource())
    {
    }
    rmm::device_buffer storage;
    cudaStream_t allocated_stream = nullptr;
    std::size_t attempts          = 0;
    bool live                     = false;
    bool fail_next                = false;
    bool invalid_release          = false;
  };

  explicit pinned_scalar_resource(rmm::cuda_stream_view stream)
    : observations(std::make_shared<state>(stream))
  {
  }

  void* allocate(cuda::stream_ref stream, std::size_t bytes, std::size_t alignment)
  {
    ++observations->attempts;
    expect(!observations->live && bytes == sizeof(int64_t) &&
             alignment <= rmm::CUDA_ALLOCATION_ALIGNMENT,
           "pinned scalar fixture received an invalid allocation");
    if (observations->fail_next) {
      observations->fail_next = false;
      throw injected_out_of_memory(bytes);
    }
    observations->live             = true;
    observations->allocated_stream = stream.get();
    // Poison each recycled slot so reading it before the GPU result cannot look like valid metadata.
    *static_cast<int64_t*>(observations->storage.data()) = std::numeric_limits<int64_t>::min();
    return observations->storage.data();
  }

  void deallocate(cuda::stream_ref stream,
                  void* ptr,
                  std::size_t bytes,
                  std::size_t alignment) noexcept
  {
    observations->invalid_release |=
      !observations->live || ptr != observations->storage.data() || bytes != sizeof(int64_t) ||
      alignment > rmm::CUDA_ALLOCATION_ALIGNMENT || stream.get() != observations->allocated_stream;
    observations->live = false;
  }

  void* allocate_sync(std::size_t, std::size_t)
  {
    throw std::logic_error("pinned observation requested synchronous allocation");
  }
  void deallocate_sync(void*, std::size_t, std::size_t) noexcept
  {
    observations->invalid_release = true;
  }
  bool operator==(pinned_scalar_resource const& other) const noexcept
  {
    return observations == other.observations;
  }
  friend void get_property(pinned_scalar_resource const&, cuda::mr::host_accessible) noexcept {}
  friend void get_property(pinned_scalar_resource const&, cuda::mr::device_accessible) noexcept {}

  std::shared_ptr<state> observations;
};

static_assert(cuda::mr::resource_with<pinned_scalar_resource,
                                      cuda::mr::host_accessible,
                                      cuda::mr::device_accessible>);

class release_observation {
 public:
  release_observation(checked_resource& resource,
                      std::atomic<bool>& released,
                      std::atomic<bool>* second_released = nullptr)
    : resource_(resource)
  {
    resource_.observations->release_required        = &released;
    resource_.observations->second_release_required = second_released;
  }
  ~release_observation()
  {
    resource_.observations->release_required        = nullptr;
    resource_.observations->second_release_required = nullptr;
  }
  release_observation(release_observation const&)            = delete;
  release_observation& operator=(release_observation const&) = delete;

 private:
  checked_resource& resource_;
};

// The callback only touches atomics. The controller releases on request or after a deadlock
// watchdog.
class stream_gate {
 public:
  explicit stream_gate(rmm::cuda_stream_view stream) : stream_(stream)
  {
    controller_ = std::thread([this] {
      auto const deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
      while (!released.load()) {
        if (release_after_delay.load()) {
          std::this_thread::sleep_for(std::chrono::milliseconds(50));
          released.store(true);
          break;
        }
        if (std::chrono::steady_clock::now() >= deadline) {
          timed_out.store(true);
          released.store(true);
          break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
      }
    });
    try {
      cuda_check(cudaLaunchHostFunc(
        stream.value(),
        [](void* data) {
          auto& gate = *static_cast<stream_gate*>(data);
          gate.entered.store(true);
          while (!gate.released.load())
            std::this_thread::yield();
        },
        this));
    } catch (...) {
      released.store(true);
      controller_.join();
      throw;
    }
  }
  ~stream_gate()
  {
    released.store(true);
    controller_.join();
    stream_.synchronize_no_throw();
  }
  stream_gate(stream_gate const&)            = delete;
  stream_gate& operator=(stream_gate const&) = delete;

  std::atomic<bool> released{false};
  std::atomic<bool> entered{false};
  std::atomic<bool> timed_out{false};
  std::atomic<bool> release_after_delay{false};

 private:
  rmm::cuda_stream_view stream_;
  std::thread controller_;
};

std::string repeated_plan(std::string const& plan, int columns)
{
  std::string result;
  for (int i = 0; i < columns; ++i) {
    if (i) result += "\n---\n";
    result += plan;
  }
  return result;
}

std::vector<rmm::cuda_stream_view> stream_views(simpatico::stream_pool const& pool)
{
  return {pool.streams.begin(), pool.streams.end()};
}

simpatico::column_decode_request value_request(simpatico::compressed_column const& column)
{
  return {.source = std::cref(*column.plan_tree), .result = simpatico::value_result{column.dtype}};
}

bool columns_equal_completed(cudf::column_view expected, cudf::column_view actual)
{
  auto const equal = columns_equal(expected, actual);
  // Order test-only default-stream readbacks before nonblocking-stream deallocation.
  rmm::cuda_stream_default.synchronize();
  return equal;
}

bool strings_equal_completed(cudf::column_view expected,
                             cudf::column_view actual,
                             rmm::cuda_stream_view stream)
{
  auto const equal = strings_equal(expected, actual, stream);
  rmm::cuda_stream_default.synchronize();
  return equal;
}

void verify_projection(cudf::table_view expected,
                       cudf::table_view actual,
                       std::span<std::size_t const> indices)
{
  expect(actual.num_columns() == static_cast<cudf::size_type>(indices.size()),
         "projection column count");
  for (std::size_t i = 0; i < indices.size(); ++i) {
    expect(columns_equal_completed(expected.column(indices[i]), actual.column(i)),
           "projection data/type/null mismatch");
  }
}

void test_table_contracts(rmm::device_async_resource_ref mr)
{
  std::array<char const*, 8> const plans{
    "input -> identity\n",
    "input -> bitpack\n",
    "input -> delta -> differences\ndelta.differences -> bitpack\n",
    "input -> rle -> values, runs\nrle.values -> bitpack\nrle.runs -> bitpack\n",
    "input -> for -> deltas, references\nfor.deltas -> bitpack\nfor.references -> bitpack\n",
    "input -> zigzag -> zigzag\nzigzag.zigzag -> bitpack\n",
    "input -> rle -> values, runs\n",
    "input -> bitpack\n"};
  std::array<std::size_t, 8> const all{0, 1, 2, 3, 4, 5, 6, 7};
  std::array<std::size_t, 5> const selected{7, 4, 1, 4, 0};
  for (int rows : {13, 1037, 32781}) {
    std::vector<std::unique_ptr<cudf::column>> columns;
    for (int c = 0; c < 8; ++c) {
      auto fixture = c == 1 || c == 4 ? make_int64_table(1, rows, 1013 * c + 11)
                                      : make_int32_table(1, rows, 1013 * c + 11);
      columns.push_back(std::move(fixture->release().front()));
      if (c == 1) {
        std::vector<std::int64_t> host(rows);
        for (int i = 0; i < rows; ++i)
          host[i] = std::bit_cast<std::int64_t>(std::uint64_t(i) * 0x9e3779b97f4a7c15ULL);
        host[0]     = std::numeric_limits<std::int64_t>::min();
        host.back() = std::numeric_limits<std::int64_t>::max();
        cuda_check(cudaMemcpy(columns.back()->mutable_view().head<void>(),
                              host.data(),
                              host.size() * sizeof(host[0]),
                              cudaMemcpyHostToDevice));
      } else if (c == 2 || c == 3 || c == 5 || c == 6) {
        std::vector<std::int32_t> host(rows);
        for (int i = 0; i < rows; ++i)
          host[i] = ((i / 7 + c * 13) % 127) - 63;
        cuda_check(cudaMemcpy(columns.back()->mutable_view().head<void>(),
                              host.data(),
                              host.size() * sizeof(host[0]),
                              cudaMemcpyHostToDevice));
      }
      if (c == 0) {
        columns.back()->set_null_mask(cudf::create_null_mask(rows, cudf::mask_state::ALL_VALID), 0);
        cudf::set_null_mask(columns.back()->mutable_view().null_mask(), 5, 6, false);
        columns.back()->set_null_count(1);
      }
    }
    cudf::table input(std::move(columns));
    cudf::get_default_stream().synchronize();
    for (int streams : {1, 4}) {
      simpatico::stream_pool pool;
      expect(pool.init(streams), "pool init");
      simpatico::compressed_table compressed;
      for (int c = 0; c < 8; ++c) {
        if (c == 0) {
          // The decoder supports an existing nullable identity representation; the compressor's
          // null-input policy remains unchanged.
          auto tree = simpatico::plan_tree_from_dsl(plans[c]);
          expect(tree && tree->nodes.size() == 2 && tree->nodes[1].op == "identity",
                 "identity fixture plan shape");
          tree->nodes[1].rep = std::make_unique<simpatico::identity_compressed_representation>(
            std::make_unique<cudf::column>(input.view().column(c), cudf::get_default_stream(), mr));
          simpatico::compressed_column column;
          column.dtype     = input.view().column(c).type();
          column.num_rows  = rows;
          column.plan_tree = std::make_unique<simpatico::PlanTree>(std::move(*tree));
          compressed.columns.push_back(std::move(column));
        } else {
          auto single = simpatico::compress_with_plan(
            cudf::table_view{{input.view().column(c)}}, plans[c], cudf::get_default_stream(), mr);
          compressed.columns.push_back(std::move(single.columns.front()));
        }
      }
      cudf::get_default_stream().synchronize();
      auto reference = simpatico::decompress(compressed, cudf::get_default_stream(), mr);
      verify_projection(input.view(), reference->view(), all);
      auto full = simpatico::decompress(compressed, pool, mr);
      verify_projection(input.view(), full->view(), all);
      auto projected = simpatico::decompress(compressed, selected, pool, mr);
      verify_projection(input.view(), projected->view(), selected);
      expect(projected->view().column(1).head<void>() != projected->view().column(3).head<void>(),
             "duplicate projection aliases output storage");
      std::array<std::size_t, 1> const one{4};
      auto single = simpatico::decompress(compressed, one, streams, mr);
      verify_projection(input.view(), single->view(), one);
      auto empty = simpatico::decompress(compressed, std::span<std::size_t const>{}, pool, mr);
      expect(empty->num_columns() == 0, "empty projection not empty");
      std::array<std::size_t, 2> const invalid{0, 8};
      expect_failure([&] { simpatico::decompress(compressed, invalid, pool, mr); },
                     "invalid projection accepted");
      simpatico::stream_pool empty_pool;
      expect_failure([&] { simpatico::decompress(compressed, empty_pool, mr); },
                     "empty pool accepted");
      auto saved = std::move(compressed.columns[7].plan_tree);
      expect_failure([&] { simpatico::decompress(compressed, pool, mr); }, "null plan accepted");
      compressed.columns[7].plan_tree = std::move(saved);
      auto edge = compressed.columns[7].plan_tree->nodes.front().children.front();
      compressed.columns[7].plan_tree->nodes.front().children.front().child = 99999;
      expect_failure([&] { simpatico::decompress(compressed, pool, mr); },
                     "malformed later plan accepted");
      compressed.columns[7].plan_tree->nodes.front().children.front() = edge;
      auto recovered = simpatico::decompress(compressed, pool, mr);
      compressed.columns.clear();
      rmm::cuda_stream consumer(rmm::cuda_stream::flags::non_blocking);
      cudf::table consumed(recovered->view(), consumer.view(), mr);
      consumer.synchronize();
      verify_projection(input.view(), consumed.view(), all);
    }
  }

  simpatico::stream_pool pool;
  expect(pool.init(4), "zero-row pool init");
  auto zero       = make_int32_table(8, 0, 1);
  auto compressed = simpatico::compress_with_plan(
    zero->view(), repeated_plan("input -> identity\n", 8), cudf::get_default_stream(), mr);
  auto decoded = simpatico::decompress(compressed, pool, mr);
  expect(decoded->num_rows() == 0 && decoded->num_columns() == 8, "zero-row table shape");
  simpatico::compressed_table empty;
  expect(simpatico::decompress(empty, pool, mr)->num_columns() == 0, "empty table shape");

  auto mixed_input = make_int32_table(3, 1037, 31);
  auto mixed =
    simpatico::compress_with_plan(mixed_input->view(),
                                  "input -> bitpack\n---\ninput -> ans\n---\ninput -> identity\n",
                                  cudf::get_default_stream(),
                                  mr);
  auto mixed_output = simpatico::decompress(mixed, pool, mr);
  std::array<std::size_t, 3> const mixed_indices{0, 1, 2};
  verify_projection(mixed_input->view(), mixed_output->view(), mixed_indices);
}

void test_completed_public_return(rmm::device_async_resource_ref upstream)
{
  auto input      = make_int32_table(8, 65549, 53);
  auto compressed = simpatico::compress_with_plan(
    input->view(), repeated_plan("input -> identity\n", 8), cudf::get_default_stream(), upstream);
  std::array<std::size_t, 8> const all{0, 1, 2, 3, 4, 5, 6, 7};
  std::array<std::size_t, 5> const selected{7, 4, 1, 4, 0};
  enum class api { pooled, single_stream, projected, column, standalone };
  for (auto const mode :
       {api::pooled, api::single_stream, api::projected, api::column, api::standalone}) {
    simpatico::stream_pool pool;
    expect(pool.init(mode == api::pooled || mode == api::projected ? 4 : 1),
           "public completion pool init");
    {
      auto warm = simpatico::decompress(compressed, pool, upstream);
    }
    cuda_check(pool.sync_all());
    event_markers markers(pool);
    stream_gate gate(pool.streams.back());
    markers.record();
    gate.release_after_delay.store(true);
    std::unique_ptr<cudf::table> table;
    std::unique_ptr<cudf::column> column;
    switch (mode) {
      case api::pooled: table = simpatico::decompress(compressed, pool, upstream); break;
      case api::single_stream:
        table =
          simpatico::decompress(compressed, rmm::cuda_stream_view{pool.streams.front()}, upstream);
        break;
      case api::projected:
        table = simpatico::decompress(compressed, selected, pool, upstream);
        break;
      case api::column:
        column = simpatico::decompress_column(
          *compressed.columns[0].plan_tree, pool.streams.front(), upstream, nullptr);
        break;
      case api::standalone: {
        auto const* rep = dynamic_cast<simpatico::standalone_compressed_representation const*>(
          compressed.columns[0].plan_tree->nodes[1].rep.get());
        expect(rep != nullptr, "standalone identity fixture missing");
        column = rep->decompress(pool.streams.front(), upstream);
        break;
      }
    }
    // Identity has no post-join scratch frees, so readiness directly checks completed copies before
    // verification can hide a missing wait.
    expect(gate.released.load(), "public decode returned before its gated stream completed");
    expect(markers.complete(), "public decode returned before all pool markers completed");
    for (auto stream : pool.streams)
      expect(cudaStreamQuery(stream) == cudaSuccess, "public decode returned with unfinished work");
    expect(!gate.timed_out.load(), "public completion watchdog expired");
    if (table) {
      verify_projection(input->view(),
                        table->view(),
                        mode == api::projected ? std::span<std::size_t const>{selected}
                                               : std::span<std::size_t const>{all});
    } else {
      expect(column != nullptr, "completed direct decode returned no column");
      expect(columns_equal_completed(input->view().column(0), column->view()),
             "completed direct output mismatch");
    }
  }
}

void test_identity_owned_children(rmm::device_async_resource_ref upstream)
{
  rmm::cuda_stream input_stream(rmm::cuda_stream::flags::non_blocking);
  std::vector<std::unique_ptr<cudf::column>> inputs;
  inputs.push_back(make_strings_column(
    {"alpha", "ignored", "", "omega"}, {true, false, true, true}, input_stream.view()));
  inputs.push_back(cudf::make_empty_column(cudf::data_type{cudf::type_id::STRING}));
  auto offsets = [&](std::vector<std::int32_t> const& values) {
    auto column = cudf::make_numeric_column(cudf::data_type{cudf::type_id::INT32},
                                            static_cast<cudf::size_type>(values.size()),
                                            cudf::mask_state::UNALLOCATED,
                                            input_stream.view(),
                                            upstream);
    cuda_check(cudaMemcpyAsync(column->mutable_view().head<void>(),
                               values.data(),
                               values.size() * sizeof(values.front()),
                               cudaMemcpyHostToDevice,
                               input_stream.value()));
    input_stream.synchronize();
    return column;
  };
  auto strings = make_strings_column(
    {"aa", "b", "ccc", "ignored"}, {true, true, true, false}, input_stream.view());
  auto inner =
    cudf::make_lists_column(2, offsets({0, 1, 4}), std::move(strings), 0, rmm::device_buffer{});
  inputs.push_back(
    cudf::make_lists_column(2, offsets({0, 1, 2}), std::move(inner), 0, rmm::device_buffer{}));

  for (auto& input : inputs) {
    auto expected = std::make_unique<cudf::column>(*input, input_stream.view(), upstream);
    input_stream.synchronize();
    auto representation =
      std::make_unique<simpatico::identity_compressed_representation>(std::move(input));
    rmm::cuda_stream stream(rmm::cuda_stream::flags::non_blocking);
    checked_resource supplied(upstream), current(upstream);
    current_resource_guard current_guard(current);
    stream_gate gate(stream.value());
    gate.release_after_delay.store(true);
    auto output = representation->decompress(stream.view(), supplied);
    // Observe completed return before any verification readback can hide an unfinished copy.
    expect(gate.released.load() && cudaStreamQuery(stream.value()) == cudaSuccess,
           "owning identity copy returned before nested data completed");
    auto verify = [&](auto&& self, cudf::column_view source, cudf::column_view copied) -> void {
      expect(source.type() == copied.type() && source.size() == copied.size() &&
               source.null_count() == copied.null_count() &&
               source.num_children() == copied.num_children(),
             "owning identity copy changed column metadata");
      if (source.head<void>())
        expect(source.head<void>() != copied.head<void>(), "identity copy aliases child data");
      if (source.null_mask())
        expect(source.null_mask() != copied.null_mask(), "identity copy aliases validity");
      if (source.type().id() == cudf::type_id::STRING) {
        expect(strings_equal_completed(source, copied, stream.view()),
               "identity copy changed nullable string values");
      } else if (source.type().id() != cudf::type_id::LIST) {
        expect(columns_equal_completed(source, copied), "identity copy changed child values");
      }
      for (cudf::size_type child = 0; child < source.num_children(); ++child)
        self(self, source.child(child), copied.child(child));
    };
    verify(verify, representation->channels_[0]->view(), output->view());
    representation.reset();
    verify(verify, expected->view(), output->view());
    if (output->size() > 0)
      expect(!supplied.observations->attempts.empty(), "identity copy ignored the supplied MR");
    expect(current.observations->attempts.empty(), "identity children used the current MR");
    output.reset();
    supplied.check();
    current.check();
    expect(!gate.timed_out.load(), "owning identity copy watchdog expired");
  }
}

void test_dictionary_width_metadata(rmm::device_async_resource_ref mr)
{
  struct fixture {
    std::vector<std::string> values;
    std::vector<bool> valid;
    std::int64_t width;
  };
  std::array<fixture, 8> const fixtures{{
    {{"aa", "bb", "aa", "cc"}, {}, 2},
    {{"only", "only", "only"}, {}, 4},
    {{"a", "bbb", "", "cc"}, {}, 0},
    {{"aa", "ignored", "bb", "aa"}, {true, false, true, true}, 2},
    {{"a", "ignored", "bbb", "a"}, {true, false, true, true}, 0},
    {{"", "", ""}, {}, 0},
    {{}, {}, 0},
    {{"aa", "b", ""}, {false, false, false}, 0},
  }};
  rmm::cuda_stream stream(rmm::cuda_stream::flags::non_blocking);
  simpatico::stream_pool pool;
  expect(pool.init(2), "dictionary width pool init");
  for (auto const& fixture : fixtures) {
    auto input = make_strings_column(fixture.values, fixture.valid, stream.view());
    // Use the codec directly so nullable inputs exercise dictionary metadata without changing
    // the table compressor's independent null-input policy.
    auto encoded = simpatico::dictionary_compressor{}.compress(input->view(), stream.view(), mr);
    auto const* original =
      dynamic_cast<simpatico::dictionary_compressed_representation const*>(encoded.get());
    expect(original != nullptr && original->dict_column != nullptr,
           "dictionary width fixture missing representation");
    expect(original->constant_key_width == fixture.width,
           "original dictionary did not publish eager key width");

    auto copied = std::make_unique<cudf::column>(original->dict_column->view(), stream.view(), mr);
    {
      stream_gate gate(stream.view());
      gate.release_after_delay.store(true);
      auto published = simpatico::dictionary_compressed_representation::from_encoded_column(
        std::move(copied), stream.view(), mr);
      expect(gate.released.load() && published->constant_key_width == fixture.width,
             "dictionary publication did not complete prior work and metadata");
      expect(!gate.timed_out.load(), "dictionary publication watchdog expired");
    }

    // The direct constructor also supports frame-local reconstruction with unknown metadata.
    // Decode must measure locally without turning that representation into a mutable cache.
    simpatico::dictionary_compressed_representation reconstructed(
      std::make_unique<cudf::column>(original->dict_column->view(), stream.view(), mr));
    expect(reconstructed.constant_key_width == -1, "reconstructed dictionary width is not unknown");
    std::vector<std::string> channel_names;
    std::vector<std::unique_ptr<cudf::column>> channels;
    for (auto const& channel : original->named_channels(stream.view())) {
      channel_names.push_back(channel.name);
      channels.push_back(std::make_unique<cudf::column>(channel.view, stream.view(), mr));
    }
    std::string error;
    auto imported = simpatico::dictionary_compressed_representation::from_outputs(
      channel_names, std::move(channels), stream.view(), mr, &error);
    auto const* imported_dictionary =
      dynamic_cast<simpatico::dictionary_compressed_representation const*>(imported.get());
    expect(imported_dictionary != nullptr && error.empty(), "dictionary channel import failed");
    expect(imported_dictionary->constant_key_width == fixture.width,
           "long-lived dictionary import did not publish eager key width");
    stream.synchronize();

    auto tree = simpatico::plan_tree_from_dsl("input -> dictionary\n");
    expect(tree && tree->nodes.size() == 2 && tree->nodes[1].op == "dictionary",
           "dictionary width fixture plan shape");
    tree->nodes[1].rep = std::move(encoded);
    simpatico::compressed_table compressed;
    simpatico::compressed_column column;
    column.dtype     = input->type();
    column.num_rows  = input->size();
    column.plan_tree = std::make_unique<simpatico::PlanTree>(std::move(*tree));
    compressed.columns.push_back(std::move(column));

    for (int repeat = 0; repeat < 3; ++repeat) {
      auto output = simpatico::decompress(compressed, pool, mr);
      expect(output->num_columns() == 1 && output->view().column(0).type() == input->type(),
             "eager-width dictionary output shape/type mismatch");
      expect(strings_equal_completed(input->view(), output->view().column(0), stream.view()),
             "eager-width dictionary output mismatch");
      expect(original->constant_key_width == fixture.width,
             "decode changed original dictionary width metadata");

      auto rebuilt = reconstructed.decompress(stream.view(), mr);
      expect(rebuilt != nullptr && rebuilt->type() == input->type(),
             "unknown-width dictionary output type mismatch");
      expect(strings_equal_completed(input->view(), rebuilt->view(), stream.view()),
             "unknown-width dictionary output mismatch");
      expect(reconstructed.constant_key_width == -1,
             "decode cached reconstructed dictionary width metadata");

      auto loaded = imported_dictionary->decompress(stream.view(), mr);
      expect(loaded != nullptr && loaded->type() == input->type(),
             "imported dictionary output type mismatch");
      expect(strings_equal_completed(input->view(), loaded->view(), stream.view()),
             "imported eager-width dictionary output mismatch");
      expect(imported_dictionary->constant_key_width == fixture.width,
             "decode changed imported dictionary width metadata");
    }
    std::array<std::size_t, 2> const duplicate{0, 0};
    auto duplicated = simpatico::decompress(compressed, duplicate, pool, mr);
    expect(duplicated->num_columns() == 2, "duplicate dictionary projection shape mismatch");
    for (int index = 0; index < 2; ++index)
      expect(
        strings_equal_completed(input->view(), duplicated->view().column(index), stream.view()),
        "duplicate dictionary projection output mismatch");
    expect(original->constant_key_width == fixture.width,
           "duplicate projection changed shared dictionary width metadata");
  }
}

void test_dictionary_width_sliced_input(rmm::device_async_resource_ref mr)
{
  rmm::cuda_stream stream(rmm::cuda_stream::flags::non_blocking);
  auto input =
    make_strings_column({"a", "long-prefix", "aa", "bb", "aa", "suffix"}, {}, stream.view());
  auto const sliced = cudf::slice(input->view(), {2, 5}, stream.view()).front();
  auto encoded      = simpatico::dictionary_compressor{}.compress(sliced, stream.view(), mr);
  auto const* dictionary =
    dynamic_cast<simpatico::dictionary_compressed_representation const*>(encoded.get());
  expect(dictionary != nullptr && dictionary->constant_key_width == 2,
         "dictionary width inspected the sliced input's prefix");
  auto decoded = dictionary->decompress(stream.view(), mr);
  expect(strings_equal_completed(sliced, decoded->view(), stream.view()),
         "sliced dictionary roundtrip mismatch");
}

void test_dictionary_width_large_keys(rmm::device_async_resource_ref mr)
{
  enum class key_shape { fixed, last_wide, first_empty };
  rmm::cuda_stream stream(rmm::cuda_stream::flags::non_blocking);
  // Few output rows isolate the full key-set reduction from row-count/cardinality policy.
  constexpr cudf::size_type key_count = 1 << 20;
  std::array<int32_t, 3> const codes{0, key_count / 2, key_count - 1};
  for (bool wide_offsets : {false, true}) {
    for (auto shape : {key_shape::fixed, key_shape::last_wide, key_shape::first_empty}) {
      std::vector<int64_t> offsets{0};
      std::string chars;
      std::vector<std::string> expected_values;
      offsets.reserve(key_count + 1);
      chars.reserve(static_cast<std::size_t>(key_count) * 7 + 1);
      for (cudf::size_type key = 0; key < key_count; ++key) {
        auto value = std::to_string(key);
        value.insert(0, 7 - value.size(), '0');
        if (shape == key_shape::first_empty && key == 0) value.clear();
        if (shape == key_shape::last_wide && key == key_count - 1) value += 'x';
        if (std::find(codes.begin(), codes.end(), key) != codes.end())
          expected_values.push_back(value);
        chars += value;
        offsets.push_back(static_cast<int64_t>(chars.size()));
      }
      auto keys_offsets = cudf::make_numeric_column(
        cudf::data_type{wide_offsets ? cudf::type_id::INT64 : cudf::type_id::INT32},
        key_count + 1,
        cudf::mask_state::UNALLOCATED,
        stream.view(),
        mr);
      std::vector<int32_t> offsets32;
      if (wide_offsets) {
        cuda_check(cudaMemcpyAsync(keys_offsets->mutable_view().head<int64_t>(),
                                   offsets.data(),
                                   offsets.size() * sizeof(int64_t),
                                   cudaMemcpyHostToDevice,
                                   stream.value()));
      } else {
        offsets32.assign(offsets.begin(), offsets.end());
        cuda_check(cudaMemcpyAsync(keys_offsets->mutable_view().head<int32_t>(),
                                   offsets32.data(),
                                   offsets32.size() * sizeof(int32_t),
                                   cudaMemcpyHostToDevice,
                                   stream.value()));
      }
      rmm::device_buffer keys_chars(chars.size(), stream.view(), mr);
      cuda_check(cudaMemcpyAsync(
        keys_chars.data(), chars.data(), chars.size(), cudaMemcpyHostToDevice, stream.value()));
      auto keys = cudf::make_strings_column(
        key_count, std::move(keys_offsets), std::move(keys_chars), 0, rmm::device_buffer{});
      auto indices = cudf::make_numeric_column(cudf::data_type{cudf::type_id::INT32},
                                               codes.size(),
                                               cudf::mask_state::UNALLOCATED,
                                               stream.view(),
                                               mr);
      cuda_check(cudaMemcpyAsync(indices->mutable_view().head<int32_t>(),
                                 codes.data(),
                                 sizeof(codes),
                                 cudaMemcpyHostToDevice,
                                 stream.value()));
      auto dictionary =
        cudf::make_dictionary_column(std::move(keys), std::move(indices), rmm::device_buffer{}, 0);
      auto prepared = simpatico::dictionary_compressed_representation::from_encoded_column(
        std::move(dictionary), stream.view(), mr);
      auto const expected_width = shape == key_shape::fixed ? 7 : 0;
      expect(prepared->constant_key_width == expected_width,
             "dictionary width missed a key outside the first reduction block");
      auto expected = make_strings_column(expected_values, {}, stream.view());
      auto decoded  = prepared->decompress(stream.view(), mr);
      expect(strings_equal_completed(expected->view(), decoded->view(), stream.view()),
             "large-key dictionary roundtrip mismatch");

      simpatico::dictionary_compressed_representation unknown(
        std::make_unique<cudf::column>(prepared->dict_column->view(), stream.view(), mr));
      auto fallback = unknown.decompress(stream.view(), mr);
      expect(strings_equal_completed(expected->view(), fallback->view(), stream.view()),
             "large-key unknown-width dictionary roundtrip mismatch");
      expect(unknown.constant_key_width == -1, "large-key fallback mutated width metadata");
    }
  }
}

void test_dictionary_width_failures(rmm::device_async_resource_ref upstream)
{
  simpatico::stream_pool pool;
  expect(pool.init(1), "dictionary observation failure pool init");
  rmm::cuda_stream_view const stream{pool.streams.front()};
  auto input   = make_strings_column({"aa", "bb", "cc", "aa"}, {}, stream);
  auto encoded = simpatico::dictionary_compressor{}.compress(input->view(), stream, upstream);
  auto const* original =
    dynamic_cast<simpatico::dictionary_compressed_representation const*>(encoded.get());
  expect(original != nullptr, "dictionary observation failure fixture missing");
  checked_resource resource(upstream);
  checked_resource default_resource(upstream);
  current_resource_guard current(default_resource);
  {
    auto copied   = std::make_unique<cudf::column>(original->dict_column->view(), stream, upstream);
    auto observed = simpatico::dictionary_compressed_representation::from_encoded_column(
      std::move(copied), stream, resource);
    expect(observed->constant_key_width == 2, "explicit-resource dictionary width mismatch");
  }
  resource.check();
  auto const allocation_count = resource.observations->attempts.size();
  expect(allocation_count == 1, "dictionary observation did not use one device scratch allocation");
  expect(default_resource.observations->attempts.empty(),
         "dictionary observation bypassed supplied resource");
  for (std::size_t fail_at = 0; fail_at < allocation_count; ++fail_at) {
    resource.reset();
    // Allocate the encoded-column copy upstream first, so injected failures target only the
    // width observation's device scratch allocation.
    auto copied = std::make_unique<cudf::column>(original->dict_column->view(), stream, upstream);
    resource.observations->fail_at = fail_at;
    event_markers markers(pool);
    stream_gate gate(stream);
    markers.record();
    gate.release_after_delay.store(true);
    bool injected = false;
    try {
      (void)simpatico::dictionary_compressed_representation::from_encoded_column(
        std::move(copied), stream, resource);
    } catch (injected_out_of_memory const& error) {
      injected = error.requested_bytes == resource.observations->failed_request_bytes &&
                 error.requested_bytes > 0;
    }
    expect(injected, "dictionary observation lost OOM subtype or requested bytes");
    // Check before any verification copy or test synchronization can hide a missing drain.
    expect(gate.released.load() && markers.complete(),
           "dictionary observation failure escaped with pending stream work");
    resource.check();
    expect(default_resource.observations->attempts.empty(),
           "failed dictionary observation bypassed supplied resource");
    expect(!gate.timed_out.load(), "dictionary observation failure watchdog expired");
  }
  // Unknown metadata uses the same reduction with frame-owned device result and scratch storage.
  resource.reset();
  {
    simpatico::dictionary_compressed_representation unknown(
      std::make_unique<cudf::column>(original->dict_column->view(), stream, upstream));
    resource.observations->fail_at = 0;
    event_markers markers(pool);
    stream_gate gate(stream);
    markers.record();
    gate.release_after_delay.store(true);
    bool injected = false;
    try {
      (void)unknown.decompress(stream, resource);
    } catch (injected_out_of_memory const& error) {
      injected = error.requested_bytes == resource.observations->failed_request_bytes &&
                 error.requested_bytes >= sizeof(int64_t);
    }
    expect(injected, "unknown dictionary width lost its metadata allocation error");
    expect(gate.released.load() && markers.complete(),
           "unknown dictionary width failure escaped before prior stream work drained");
    expect(unknown.constant_key_width == -1, "failed dictionary fallback mutated metadata");
    resource.check();
    expect(!gate.timed_out.load(), "unknown dictionary width failure watchdog expired");
  }
  expect(default_resource.observations->attempts.empty(),
         "dictionary failure cleanup bypassed supplied resource");
  default_resource.check();
}

void test_dictionary_pinned_observation(rmm::device_async_resource_ref upstream)
{
  simpatico::stream_pool pool;
  expect(pool.init(2), "dictionary pinned observation pool init");
  rmm::cuda_stream_view const first{pool.streams.front()};
  pinned_scalar_resource pinned(first);
  checked_resource device(upstream);
  std::array<std::vector<std::string>, 5> const values{
    {{"aa", "bb"}, {"abcd", "efgh"}, {"a", "bbb"}, {}, {"ignored", "ignored"}}};
  std::array<int64_t, 5> const widths{2, 4, 0, 0, 0};
  std::vector<std::unique_ptr<simpatico::compressed_representation>> encoded;
  for (std::size_t i = 0; i < values.size(); ++i) {
    auto input = make_strings_column(
      values[i], i == 4 ? std::vector<bool>{false, false} : std::vector<bool>{}, first);
    encoded.push_back(simpatico::dictionary_compressor{}.compress(input->view(), first, upstream));
  }
  for (std::size_t repeat = 0; repeat < 3; ++repeat) {
    for (std::size_t i = 0; i < encoded.size(); ++i) {
      rmm::cuda_stream_view const stream{pool.streams[(repeat + i) % pool.streams.size()]};
      auto const& original =
        dynamic_cast<simpatico::dictionary_compressed_representation const&>(*encoded[i]);
      auto copied = std::make_unique<cudf::column>(original.dict_column->view(), stream, upstream);
      auto const pinned_attempts = pinned.observations->attempts;
      auto const device_attempts = device.observations->attempts.size();
      stream_gate gate(stream);
      gate.release_after_delay.store(true);
      pinned_resource_guard override(pinned);
      auto prepared = simpatico::dictionary_compressed_representation::from_encoded_column(
        std::move(copied), stream, device);
      expect(gate.released.load() && prepared->constant_key_width == widths[i],
             "dictionary read an unfinished or recycled pinned scalar");
      auto const expected_allocations = i < 3 ? 1U : 0U;
      expect(pinned.observations->attempts - pinned_attempts == expected_allocations &&
               device.observations->attempts.size() - device_attempts == expected_allocations,
             "dictionary empty/nonempty observation allocation count mismatch");
      expect(!pinned.observations->live && !pinned.observations->invalid_release,
             "dictionary returned pinned staging with invalid ownership");
      device.check();
      expect(!gate.timed_out.load(), "dictionary pinned observation watchdog expired");
    }
  }

  auto const& original =
    dynamic_cast<simpatico::dictionary_compressed_representation const&>(*encoded.front());
  {
    auto copied = std::make_unique<cudf::column>(original.dict_column->view(), first, upstream);
    auto const pinned_attempts   = pinned.observations->attempts;
    device.observations->fail_at = device.observations->attempts.size();
    event_markers markers(pool);
    stream_gate gate(first);
    markers.record();
    gate.release_after_delay.store(true);
    pinned_resource_guard override(pinned);
    bool injected = false;
    try {
      (void)simpatico::dictionary_compressed_representation::from_encoded_column(
        std::move(copied), first, device);
    } catch (injected_out_of_memory const& error) {
      injected = error.requested_bytes == device.observations->failed_request_bytes &&
                 error.requested_bytes > 0;
    }
    device.observations->fail_at.reset();
    expect(injected && pinned.observations->attempts == pinned_attempts,
           "dictionary allocated pinned output before device scratch or lost device OOM details");
    expect(gate.released.load() && markers.complete(),
           "dictionary device OOM escaped before prior stream work drained");
    expect(!pinned.observations->live && !pinned.observations->invalid_release,
           "dictionary device OOM changed pinned ownership");
    device.check();
    expect(!gate.timed_out.load(), "dictionary device OOM watchdog expired");
  }
  auto copied = std::make_unique<cudf::column>(original.dict_column->view(), first, upstream);
  auto const device_attempts     = device.observations->attempts.size();
  auto const pinned_attempts     = pinned.observations->attempts;
  pinned.observations->fail_next = true;
  event_markers markers(pool);
  stream_gate gate(first);
  markers.record();
  release_observation device_release(device, gate.released);
  gate.release_after_delay.store(true);
  {
    pinned_resource_guard override(pinned);
    bool injected = false;
    try {
      (void)simpatico::dictionary_compressed_representation::from_encoded_column(
        std::move(copied), first, device);
    } catch (injected_out_of_memory const& error) {
      injected = error.requested_bytes == sizeof(int64_t);
    }
    expect(injected && !pinned.observations->fail_next,
           "dictionary pinned OOM subtype/request size was lost");
    expect(device.observations->attempts.size() == device_attempts + 1 &&
             pinned.observations->attempts == pinned_attempts + 1,
           "dictionary pinned OOM did not follow exactly one scratch allocation");
    expect(gate.released.load() && markers.complete(),
           "dictionary pinned OOM escaped before queued work drained");
    expect(!pinned.observations->live && !pinned.observations->invalid_release,
           "dictionary pinned OOM leaked staging");
    device.check();
  }
  expect(!gate.timed_out.load(), "dictionary pinned OOM watchdog expired");
  cuda_check(pool.sync_all());
}

void test_submission_and_kernel_lifetime(rmm::device_async_resource_ref upstream)
{
  auto input      = make_int32_table(2, 65549, 59);
  auto compressed = simpatico::compress_with_plan(
    input->view(), repeated_plan("input -> bitpack\n", 2), cudf::get_default_stream(), upstream);
  simpatico::stream_pool pool;
  expect(pool.init(2), "gate pool init");
  checked_resource resource(upstream);
  auto& cache = codegen::jit::KernelCache::instance();
  cache.clear();
  {
    auto warm = simpatico::decompress(compressed, pool, resource);
  }
  cuda_check(pool.sync_all());
  resource.check();

  auto shape = codegen::jit::FusedTree::make(codegen::OpKind::Bitpack);
  auto spec =
    codegen::decode::jit::render(*shape, "int32_t", codegen::num_chunks_for(input->num_rows()));
  codegen::jit::CompileOptions options;
  options.arch_cc        = codegen::jit::arch_cc_for_current_device();
  options.default_device = true;
  auto handle            = cache.get_or_compile_plain(spec.source, spec.entry_symbol, options);
  std::weak_ptr<codegen::jit::CompiledKernel const> retained = handle;
  handle.reset();

  auto streams = stream_views(pool);
  simpatico::decode_session session(streams, resource);
  event_markers markers(pool);
  // Gate destruction releases on exceptions before the session unwinds.
  stream_gate gate(pool.streams[0]);
  while (!gate.entered.load() && !gate.timed_out.load())
    std::this_thread::yield();
  std::array<std::size_t, 4> const requested{0, 1, 0, 1};
  for (auto index : requested) {
    session.append(value_request(compressed.columns[index]));
    expect(!gate.released.load(), "column submission waited for the gated lane");
  }
  expect(session.stats().pressure_waits == 0, "small submissions triggered memory pressure");
  markers.record();
  cache.clear();
  expect(!retained.expired(), "cache clear unloaded a pending kernel");
  expect(!gate.released.load(), "cache clear waited for a pending kernel");
  gate.released.store(true);
  auto outputs = session.finish();
  // These observations precede verification copies and any extra test synchronization.
  expect(markers.complete(), "session finish missed submitted work");
  expect(retained.expired(), "finished session retained compiled kernels");
  expect(outputs.size() == requested.size(), "gated result count");
  expect(outputs[0]->view().head<void>() != outputs[2]->view().head<void>(),
         "repeated session request aliases output storage");
  for (std::size_t i = 0; i < requested.size(); ++i) {
    expect(columns_equal_completed(input->view().column(requested[i]), outputs[i]->view()),
           "gated output mismatch");
  }
  outputs.clear();
  resource.check();
  expect(!gate.timed_out.load(), "submission watchdog expired");
}

void test_abandoned_session(rmm::device_async_resource_ref upstream)
{
  auto input      = make_int32_table(1, 65549, 67);
  auto compressed = simpatico::compress_with_plan(
    input->view(), "input -> bitpack\n", cudf::get_default_stream(), upstream);
  simpatico::stream_pool pool;
  expect(pool.init(1), "abandon pool init");
  checked_resource resource(upstream);
  {
    auto warm = simpatico::decompress(compressed, pool, resource);
  }
  cuda_check(pool.sync_all());
  resource.check();
  auto streams = stream_views(pool);
  std::optional<simpatico::decode_session> session(std::in_place, streams, resource);
  event_markers markers(pool);
  stream_gate gate(pool.streams[0]);
  session->append(value_request(compressed.columns[0]));
  markers.record();
  release_observation release_guard(resource, gate.released);
  gate.release_after_delay.store(true);
  session.reset();
  expect(gate.released.load(), "abandoned session returned while stream remained gated");
  expect(markers.complete(), "abandoned session did not drain its stream");
  resource.check();
  expect(!gate.timed_out.load(), "abandon watchdog expired");
}

void test_session_state_contracts(rmm::device_async_resource_ref upstream)
{
  auto input      = make_int32_table(1, 13, 69);
  auto compressed = simpatico::compress_with_plan(
    input->view(), "input -> identity\n", cudf::get_default_stream(), upstream);
  rmm::cuda_stream stream(rmm::cuda_stream::flags::non_blocking);
  std::array const streams{stream.view()};
  simpatico::decode_session empty(streams, upstream);
  expect(empty.finish().empty(), "empty session produced output");
  expect_failure([&] { empty.finish(); }, "second session finish accepted");
  expect_failure([&] { empty.append(value_request(compressed.columns[0])); },
                 "append after finish accepted");

  simpatico::PlanTree malformed;
  simpatico::decode_session failed(streams, upstream);
  failed.append(value_request(compressed.columns[0]));
  expect_failure(
    [&] { failed.append(simpatico::column_decode_request{.source = std::cref(malformed)}); },
    "malformed request accepted");
  expect(cudaStreamQuery(stream.value()) == cudaSuccess,
         "failed append returned before prior work completed");
  expect_failure([&] { failed.finish(); }, "failed session published partial outputs");
  expect_failure([&] { failed.append(value_request(compressed.columns[0])); },
                 "failed session accepted another request");
}

void test_pressure_waits_for_external_tail(rmm::device_async_resource_ref upstream)
{
  auto input      = make_int32_table(2, 65549, 73);
  auto compressed = simpatico::compress_with_plan(input->view(),
                                                  "input -> bitpack\n---\ninput -> identity\n",
                                                  cudf::get_default_stream(),
                                                  upstream);
  simpatico::stream_pool pool;
  expect(pool.init(1), "tail-pressure pool init");
  checked_resource resource(upstream);
  auto& cache = codegen::jit::KernelCache::instance();
  cache.clear();
  {
    auto warm = simpatico::decompress(compressed, pool, resource);
  }
  cuda_check(pool.sync_all());
  resource.check();

  auto shape = codegen::jit::FusedTree::make(codegen::OpKind::Bitpack);
  auto spec =
    codegen::decode::jit::render(*shape, "int32_t", codegen::num_chunks_for(input->num_rows()));
  codegen::jit::CompileOptions options;
  options.arch_cc        = codegen::jit::arch_cc_for_current_device();
  options.default_device = true;
  auto handle            = cache.get_or_compile_plain(spec.source, spec.entry_symbol, options);
  std::weak_ptr<codegen::jit::CompiledKernel const> first_kernel = handle;
  handle.reset();

  auto streams = stream_views(pool);
  simpatico::decode_session session(streams, resource);
  session.append(value_request(compressed.columns[0]));
  cache.clear();
  stream_gate gate(pool.streams[0]);
  while (!gate.entered.load() && !gate.timed_out.load())
    std::this_thread::yield();
  expect(!gate.timed_out.load(), "tail-pressure gate did not start");
  // Frame A is complete, but its stream tail is gated. Under-limit submission neither queries
  // nor reclaims that prefix; pressure must wait for the independently progressing controller.
  constexpr std::size_t window_frames = 64;
  for (std::size_t i = 1; i < window_frames; ++i) {
    session.append(value_request(compressed.columns[1]));
    expect(!gate.released.load(), "under-limit append waited for the external tail");
  }
  expect(session.stats().retirement_stream_queries == 0 && session.stats().pressure_waits == 0,
         "under-limit appends performed retirement work");
  expect(!first_kernel.expired(), "pending session lost its loaded kernel");
  release_observation observation(resource, gate.released);
  gate.release_after_delay.store(true);
  session.append(value_request(compressed.columns[1]));
  expect(gate.released.load() && session.stats().pressure_waits > 0,
         "pressure reclaimed a completed prefix without completing its stream tail");
  expect(
    session.stats().retirement_stream_queries > 0 && session.stats().peak_frames <= window_frames,
    "tail retirement skipped pressure accounting");
  expect(!first_kernel.expired(), "tail retirement unloaded a session-pinned module");
  expect(session.stats().peak_retained_device_bytes <
           static_cast<std::size_t>(input->num_rows()) * sizeof(std::int32_t),
         "terminal outputs were charged as retained scratch");
  auto outputs = session.finish();
  expect(first_kernel.expired(), "completed session retained its kernel module");
  expect(outputs.size() == window_frames + 1, "tail retirement lost an output");
  expect(columns_equal_completed(input->view().column(0), outputs[0]->view()),
         "retired frame output mismatch");
  for (std::size_t i = 1; i < outputs.size(); ++i)
    expect(columns_equal_completed(input->view().column(1), outputs[i]->view()),
           "post-tail output mismatch");
  outputs.clear();
  resource.check();
  expect(!gate.timed_out.load(), "tail-pressure watchdog expired");
}

void test_frame_window_backpressure(rmm::device_async_resource_ref upstream)
{
  auto input      = make_int32_table(1, 1037, 79);
  auto compressed = simpatico::compress_with_plan(
    input->view(), "input -> identity\n", cudf::get_default_stream(), upstream);
  simpatico::stream_pool pool;
  expect(pool.init(1), "backpressure pool init");
  checked_resource resource(upstream);
  auto streams = stream_views(pool);
  simpatico::decode_session session(streams, resource);
  stream_gate gate(pool.streams[0]);
  constexpr std::size_t window_frames = 64;
  for (std::size_t i = 0; i < window_frames; ++i) {
    session.append(value_request(compressed.columns[0]));
    expect(!gate.released.load(), "lane reuse waited before reaching the frame window");
  }
  expect(session.stats().pressure_waits == 0, "under-window submissions waited for pressure");
  expect(session.stats().retirement_stream_queries == 0,
         "under-window submissions queried stream completion");
  gate.release_after_delay.store(true);
  session.append(value_request(compressed.columns[0]));
  expect(gate.released.load(), "over-window request did not wait for pending ownership");
  expect(session.stats().pressure_waits > 0, "frame window did not report its pressure wait");
  expect(session.stats().peak_frames <= window_frames, "frame window allowed unbounded admission");
  auto output = session.finish();
  expect(output.size() == window_frames + 1, "backpressure lost an output");
  for (auto const& column : output) {
    expect(columns_equal_completed(input->view().column(0), column->view()),
           "backpressure output mismatch");
  }
  output.clear();
  resource.check();
  expect(!gate.timed_out.load(), "backpressure watchdog expired");
}

void test_stream_completion_failures(rmm::device_async_resource_ref upstream)
{
  auto input      = make_int32_table(1, 1037, 83);
  auto compressed = simpatico::compress_with_plan(
    input->view(), "input -> identity\n", cudf::get_default_stream(), upstream);
  simpatico::stream_pool pool;
  expect(pool.init(3), "stream failure pool init");
  checked_resource resource(upstream);
  // The third supplied handle receives no request in the finish-failure case, but still has work.
  std::array const streams{rmm::cuda_stream_view{pool.streams[0]},
                           rmm::cuda_stream_view{pool.streams[1]},
                           rmm::cuda_stream_view{pool.streams[0]},
                           rmm::cuda_stream_view{pool.streams[1]},
                           rmm::cuda_stream_view{pool.streams[2]}};
  for (bool pressure : {false, true}) {
    for (auto operation : {stream_fault::operation::query, stream_fault::operation::synchronize}) {
      event_markers markers(pool);
      stream_gate first_gate(pool.streams[0]);
      stream_gate second_gate(pool.streams[1]);
      stream_gate external_gate(pool.streams[2]);
      std::optional<simpatico::decode_session> session(std::in_place, streams, resource);
      auto const count = pressure ? std::size_t{64} : std::size_t{2};
      for (std::size_t i = 0; i < count; ++i)
        session->append(value_request(compressed.columns[0]));
      markers.record();
      release_observation observation(resource, first_gate.released, &second_gate.released);
      first_gate.release_after_delay.store(true);
      second_gate.release_after_delay.store(true);
      external_gate.release_after_delay.store(true);
      bool propagated = false;
      {
        stream_observation_scope observed;
        stream_failure_scope fault(operation, pool.streams[0]);
        try {
          if (pressure)
            session->append(value_request(compressed.columns[0]));
          else
            (void)session->finish();
        } catch (std::runtime_error const& error) {
          propagated = std::string(error.what()) == cudaGetErrorString(cudaErrorInvalidValue);
        }
        expect(stream_fault::next == stream_fault::operation::none,
               "stream failure injection was not consumed");
        // Pressure's failed wait follows a query sweep; require a fresh abort query in that case.
        auto const minimum_queries =
          pressure && operation == stream_fault::operation::synchronize ? 2U : 1U;
        for (auto stream : pool.streams)
          expect(observed.count(stream).queries >= minimum_queries,
                 "stream failure cleanup skipped a supplied physical stream");
      }
      expect(propagated, "stream completion failure lost the original CUDA error");
      expect(first_gate.released.load() && second_gate.released.load() && markers.complete(),
             "stream failure escaped before all supplied lanes completed");
      expect(!resource.observations->early_release,
             "stream failure released an owner before all lanes completed");
      expect_failure([&] { session->finish(); }, "failed session published partial output");
      expect_failure([&] { session->append(value_request(compressed.columns[0])); },
                     "stream failure left a reusable session");
      session.reset();
      resource.check();
      expect(!first_gate.timed_out.load() && !second_gate.timed_out.load() &&
               !external_gate.timed_out.load(),
             "stream failure watchdog expired");
      {
        auto recovered = simpatico::decompress(compressed, pool, resource);
        expect(columns_equal_completed(input->view().column(0), recovered->view().column(0)),
               "pool was not reusable after recoverable stream API failure");
      }
      resource.check();
    }
  }
}

class retirement_probe_representation final
  : public simpatico::standalone_compressed_representation {
 public:
  retirement_probe_representation()
    : standalone_compressed_representation(cudf::data_type{cudf::type_id::UINT8}, 1)
  {
  }
  void decompress(simpatico::decode_frame& frame,
                  simpatico::decode_column_slot output) const override
  {
    output.adopt(cudf::make_numeric_column(cudf::data_type{cudf::type_id::UINT8},
                                           1,
                                           cudf::mask_state::UNALLOCATED,
                                           frame.stream(),
                                           frame.mr()));
    auto& scratch = frame.allocate_buffer(16);
    cuda_check(cudaMemsetAsync(scratch.data(), 0x2a, scratch.size(), frame.stream().value()));
    cuda_check(cudaMemcpyAsync(output->mutable_view().head<void>(),
                               scratch.data(),
                               1,
                               cudaMemcpyDeviceToDevice,
                               frame.stream().value()));
  }
};

void test_ready_stream_and_alias_retirement(rmm::device_async_resource_ref upstream)
{
  for (bool aliases : {false, true}) {
    simpatico::stream_pool pool;
    expect(pool.init(2), "ready-stream pool init");
    std::vector<rmm::cuda_stream_view> streams{pool.streams[0], pool.streams[1]};
    if (aliases) streams.insert(streams.begin(), pool.streams[0]);
    auto const blocked = pool.streams[aliases ? 1 : 0];
    auto const ready   = pool.streams[aliases ? 0 : 1];
    checked_resource resource(upstream);
    retirement_probe_representation representation;
    simpatico::column_decode_request const request{
      .source = std::cref(
        static_cast<simpatico::standalone_compressed_representation const&>(representation))};
    simpatico::decode_session session(streams, resource);
    stream_gate blocked_gate(blocked);
    std::size_t ready_frames = 0;
    for (std::size_t i = 0; i < 64; ++i) {
      session.append(request);
      if (streams[i % streams.size()].value() == ready) ++ready_frames;
    }
    expect(
      resource.observations->attempts.size() == 128 && resource.observations->live.size() == 128,
      "retirement probe did not retain one output and one scratch allocation per frame");
    for (std::size_t i = 0; i < 64; ++i) {
      auto const assigned = streams[i % streams.size()].value();
      expect(resource.observations->attempts[2 * i] == assigned &&
               resource.observations->attempts[2 * i + 1] == assigned,
             "duplicate handles changed round-robin weighting");
    }
    cuda_check(__real_cudaStreamSynchronize(ready));
    {
      stream_observation_scope observed;
      session.append(request);
      expect(observed.count(ready).queries == 1 && observed.count(blocked).queries == 1,
             "pressure did not query each eligible physical stream exactly once");
    }
    expect(!blocked_gate.released.load() && session.stats().pressure_waits == 0,
           "ready-stream reclamation waited for another stream's tail");
    expect(session.stats().retirement_stream_queries == 2 && session.stats().peak_frames == 64,
           "ready-stream reclamation lost query or frame accounting");
    expect(resource.observations->live.size() == 130 - ready_frames,
           "completed physical stream did not retire every sealed alias, or released outputs");

    // In the non-alias case this handle has no remaining frames. Its new external tail must
    // nevertheless be included in final completion, not cached from the pressure query above.
    stream_gate ready_tail(ready);
    event_markers markers(pool);
    markers.record();
    blocked_gate.release_after_delay.store(true);
    ready_tail.release_after_delay.store(true);
    std::vector<std::unique_ptr<cudf::column>> output;
    {
      stream_observation_scope observed;
      output = session.finish();
      for (auto handle : pool.streams)
        expect(observed.count(handle).queries == 1 && observed.count(handle).synchronizations <= 1,
               "finish did not deduplicate supplied physical handles");
    }
    expect(blocked_gate.released.load() && ready_tail.released.load() && markers.complete(),
           "finish skipped an external tail on a retired stream");
    expect(output.size() == 65, "ready-stream retirement lost outputs");
    rmm::cuda_stream consumer(rmm::cuda_stream::flags::non_blocking);
    for (auto const& column : output) {
      expect(column->type().id() == cudf::type_id::UINT8 && column->size() == 1,
             "retirement probe output shape mismatch");
      std::uint8_t byte = 0;
      cuda_check(cudaMemcpyAsync(
        &byte, column->view().head<void>(), 1, cudaMemcpyDeviceToHost, consumer.value()));
      consumer.synchronize();
      expect(byte == 0x2a, "ready-stream retirement corrupted an output");
    }
    output.clear();
    resource.check();
    expect(!blocked_gate.timed_out.load() && !ready_tail.timed_out.load(),
           "ready-stream retirement watchdog expired");
  }
}

class retained_scratch_representation final
  : public simpatico::standalone_compressed_representation {
 public:
  static constexpr std::size_t output_bytes    = 40U << 20;
  static constexpr std::size_t device_bytes    = 40U << 20;
  static constexpr std::size_t host_bytes      = 5U << 20;
  static constexpr cudf::size_type output_rows = static_cast<cudf::size_type>(output_bytes);

  explicit retained_scratch_representation(bool host)
    : standalone_compressed_representation(cudf::data_type{cudf::type_id::UINT8}, output_rows),
      host_(host)
  {
  }

  void decompress(simpatico::decode_frame& frame,
                  simpatico::decode_column_slot output) const override
  {
    // Make the large terminal output first. Neither its active ownership nor the already-sealed
    // previous column's output belongs in the temporary-retirement budget.
    output.adopt(cudf::make_numeric_column(cudf::data_type{cudf::type_id::UINT8},
                                           output_rows,
                                           cudf::mask_state::UNALLOCATED,
                                           frame.stream(),
                                           frame.mr()));
    auto* destination = output->mutable_view().head<std::uint8_t>();
    if (host_) {
      auto upload = frame.host_array<std::uint8_t>(host_bytes);
      std::fill(upload.begin(), upload.end(), std::uint8_t{0x2a});
      // Exercise retained upload accounting without a pageable H2D call that could implicitly
      // wait on the deliberately gated stream before the frame is sealed.
      cuda_check(
        cudaMemsetAsync(destination, upload.front(), output_bytes, frame.stream().value()));
    } else {
      auto& scratch = frame.allocate_buffer(device_bytes);
      cuda_check(cudaMemsetAsync(scratch.data(), 0x2a, scratch.size(), frame.stream().value()));
      cuda_check(cudaMemcpyAsync(destination,
                                 scratch.data(),
                                 output_bytes,
                                 cudaMemcpyDeviceToDevice,
                                 frame.stream().value()));
    }
  }

 private:
  bool host_;
};

void test_sealed_frame_byte_accounting(rmm::device_async_resource_ref upstream)
{
  for (bool host_pressure : {false, true}) {
    simpatico::stream_pool pool;
    expect(pool.init(2), "sealed-frame accounting pool init");
    checked_resource resource(upstream);
    auto streams = stream_views(pool);
    retained_scratch_representation representation(host_pressure);
    simpatico::column_decode_request const request{
      .source = std::cref(
        static_cast<simpatico::standalone_compressed_representation const&>(representation))};
    simpatico::decode_session session(streams, resource);
    stream_gate first_gate(pool.streams[0]);
    session.append(request);
    expect(!first_gate.released.load() && session.stats().pressure_waits == 0,
           "first frame counted its terminal output as retained scratch");
    release_observation observation(resource, first_gate.released);
    first_gate.release_after_delay.store(true);
    session.append(request);
    expect(first_gate.released.load() && session.stats().pressure_waits > 0,
           "second allocation omitted the first sealed frame's retained bytes");
    auto const& stats = session.stats();
    expect(stats.peak_retained_device_bytes ==
             (host_pressure ? 0 : retained_scratch_representation::device_bytes),
           "sealed-frame accounting included output bytes or exceeded the device window");
    expect(stats.peak_retained_host_bytes ==
             (host_pressure ? retained_scratch_representation::host_bytes : 0),
           "sealed-frame accounting omitted uploads or exceeded the host window");
    expect(stats.peak_retained_device_bytes <= (64U << 20) &&
             stats.peak_retained_host_bytes <= (8U << 20),
           "two ordinary frames exceeded the retained-byte windows");
    auto output = session.finish();
    expect(output.size() == 2, "sealed-frame pressure lost a terminal output");
    expect(output[0]->view().head<void>() != output[1]->view().head<void>(),
           "sealed-frame pressure aliased terminal outputs");
    std::vector<std::uint8_t> bytes(retained_scratch_representation::output_bytes);
    rmm::cuda_stream consumer(rmm::cuda_stream::flags::non_blocking);
    for (auto const& column : output) {
      expect(column->type().id() == cudf::type_id::UINT8 &&
               column->size() == retained_scratch_representation::output_rows,
             "sealed-frame output metadata mismatch");
      cuda_check(cudaMemcpyAsync(bytes.data(),
                                 column->view().head<void>(),
                                 bytes.size(),
                                 cudaMemcpyDeviceToHost,
                                 consumer.value()));
      consumer.synchronize();
      expect(std::all_of(bytes.begin(), bytes.end(), [](auto byte) { return byte == 0x2a; }),
             "sealed-frame pressure corrupted a terminal output");
    }
    output.clear();
    resource.check();
    expect(!first_gate.timed_out.load(), "sealed-frame pressure watchdog expired");
  }
}

void test_external_phase_tail(rmm::device_async_resource_ref upstream)
{
  auto input      = make_int32_table(1, 1037, 89);
  auto compressed = simpatico::compress_with_plan(
    input->view(), "input -> identity\n", cudf::get_default_stream(), upstream);
  simpatico::stream_pool pool;
  expect(pool.init(1), "phase-tail pool init");
  checked_resource resource(upstream);
  auto streams = stream_views(pool);
  simpatico::decode_session session(streams, resource);
  session.append(value_request(compressed.columns[0]));
  // The request's own work is complete before this external tail starts. Final completion must
  // observe the current supplied stream, not an earlier frame or pressure observation.
  stream_gate gate(pool.streams[0]);
  while (!gate.entered.load() && !gate.timed_out.load())
    std::this_thread::yield();
  expect(!gate.timed_out.load(), "external tail gate did not start");
  event_markers markers(pool);
  markers.record();
  gate.release_after_delay.store(true);
  auto output = session.finish();
  expect(gate.released.load() && markers.complete(), "finish missed external phase work");
  expect(columns_equal_completed(input->view().column(0), output.front()->view()),
         "external-tail output mismatch");
  output.clear();
  resource.check();
  expect(!gate.timed_out.load(), "external tail watchdog expired");
}

class request_copy_failure : public std::bad_alloc {
 public:
  char const* what() const noexcept override { return "injected decode request copy failure"; }
};

struct throwing_probe_copy {
  std::shared_ptr<bool> fail;
  explicit throwing_probe_copy(std::shared_ptr<bool> value) : fail(std::move(value)) {}
  throwing_probe_copy(throwing_probe_copy const& other) : fail(other.fail)
  {
    if (*fail) throw request_copy_failure{};
  }
  std::unique_ptr<cudf::column> operator()(cudf::column_view,
                                           rmm::cuda_stream_view,
                                           rmm::device_async_resource_ref) const
  {
    throw std::logic_error("copy-failure probe must never execute");
  }
};

void test_request_copy_failure(rmm::device_async_resource_ref upstream)
{
  auto input      = make_int32_table(1, 1037, 97);
  auto compressed = simpatico::compress_with_plan(
    input->view(), "input -> identity\n", cudf::get_default_stream(), upstream);
  simpatico::stream_pool pool;
  expect(pool.init(1), "request-copy pool init");
  checked_resource resource(upstream);
  auto streams = stream_views(pool);
  auto fail    = std::make_shared<bool>(false);
  auto const mask_bytes =
    sirius::codegen::selection_mask::AllocWordsFor(input->num_rows()) * sizeof(std::uint32_t);
  rmm::device_buffer mask(mask_bytes, streams.front(), upstream);
  simpatico::mask_decode_request request{
    *compressed.columns[0].plan_tree,
    simpatico::membership_source{throwing_probe_copy{fail}, input->view().column(0).type()},
    {static_cast<std::uint32_t*>(mask.data()), input->num_rows()}};
  event_markers markers(pool);
  stream_gate gate(pool.streams[0]);
  std::optional<simpatico::decode_session> session(std::in_place, streams, resource);
  session->append(value_request(compressed.columns[0]));
  markers.record();
  release_observation observation(resource, gate.released);
  *fail = true;
  gate.release_after_delay.store(true);
  bool propagated = false;
  try {
    // An lvalue is intentional: copying must occur inside append's protected
    // boundary, not in argument construction before the session can drain.
    session->append(request);
  } catch (request_copy_failure const&) {
    propagated = true;
  }
  expect(propagated, "request copy failure subtype was lost");
  expect(gate.released.load() && markers.complete(),
         "host bookkeeping failure escaped before prior work completed");
  expect_failure([&] { session->finish(); }, "request copy failure published partial results");
  session.reset();
  resource.check();
  expect(!gate.timed_out.load(), "request copy failure watchdog expired");
}

void test_byte_window_and_active_frame(rmm::device_async_resource_ref upstream)
{
  auto input      = make_int32_table(1, 1037, 101);
  auto compressed = simpatico::compress_with_plan(
    input->view(), "input -> identity\n", cudf::get_default_stream(), upstream);
  for (bool host_pressure : {false, true}) {
    simpatico::stream_pool pool;
    expect(pool.init(2), "byte-pressure pool init");
    checked_resource resource(upstream);
    auto streams = stream_views(pool);
    simpatico::decode_session session(streams, resource);
    stream_gate first_gate(pool.streams[0]);
    session.append(value_request(compressed.columns[0]));
    // The low-level fixture is session-owned but remains active throughout these allocations.
    auto& frame = simpatico::decode_session_test_access::frame(session);
    first_gate.release_after_delay.store(true);
    constexpr std::size_t device_bytes = 80U << 20;
    constexpr std::size_t host_bytes   = 9U << 20;
    if (host_pressure) {
      auto storage = frame.host_array<std::uint64_t>(host_bytes / sizeof(std::uint64_t));
      expect(reinterpret_cast<std::uintptr_t>(storage.data()) % alignof(std::uint64_t) == 0,
             "upload storage is misaligned");
      storage.front() = 17;
      storage.back()  = 23;
    } else {
      (void)frame.allocate_buffer(device_bytes);
    }
    expect(first_gate.released.load() && session.stats().pressure_waits > 0,
           "retained-byte pressure did not retire prior pending work");
    // Once the sealed predecessor is gone, intrinsic active-frame pressure cannot query or wait
    // on this frame's own stream. Its storage may still be used by later work in the same request.
    stream_gate active_gate(pool.streams[1]);
    auto const queries = session.stats().retirement_stream_queries;
    auto& scratch      = frame.allocate_buffer(16);
    cuda_check(cudaMemsetAsync(scratch.data(), 0, scratch.size(), streams[1].value()));
    expect(!active_gate.released.load(), "active intrinsic frame caused a lane-tail wait");
    expect(session.stats().retirement_stream_queries == queries,
           "active-only stream was queried for retirement");
    auto const& stats = session.stats();
    expect(host_pressure ? stats.peak_retained_host_bytes >= host_bytes
                         : stats.peak_retained_device_bytes >= device_bytes,
           "oversized active-frame bytes were missing from accounting");
    release_observation observation(resource, active_gate.released);
    active_gate.release_after_delay.store(true);
    auto output = session.finish();
    expect(active_gate.released.load(), "finish released an active frame early");
    expect(output.size() == 1 &&
             columns_equal_completed(input->view().column(0), output.front()->view()),
           "pressure lost the earlier frame's terminal output");
    output.clear();
    resource.check();
    expect(!first_gate.timed_out.load() && !active_gate.timed_out.load(),
           "byte-pressure watchdog expired");
  }
}

void test_active_frame_on_retired_alias(rmm::device_async_resource_ref upstream)
{
  auto input      = make_int32_table(1, 1037, 103);
  auto compressed = simpatico::compress_with_plan(
    input->view(), "input -> identity\n", cudf::get_default_stream(), upstream);
  simpatico::stream_pool pool;
  expect(pool.init(1), "active-alias pool init");
  std::array const streams{rmm::cuda_stream_view{pool.streams[0]},
                           rmm::cuda_stream_view{pool.streams[0]}};
  checked_resource resource(upstream);
  simpatico::decode_session session(streams, resource);
  session.append(value_request(compressed.columns[0]));
  cuda_check(__real_cudaStreamSynchronize(pool.streams[0]));
  auto& active     = simpatico::decode_session_test_access::frame(session);
  auto sentinel    = active.host_array<std::uint64_t>(1);
  sentinel.front() = 0x12345678;
  auto& large      = active.allocate_buffer(80U << 20);
  expect(session.stats().retirement_stream_queries == 1 && session.stats().pressure_waits == 0,
         "ready sealed alias was not reclaimed with one physical-stream query");
  expect(sentinel.front() == 0x12345678 && large.size() == (80U << 20),
         "ready-stream reclamation destroyed its active alias");

  stream_gate gate(pool.streams[0]);
  release_observation observation(resource, gate.released);
  auto const queries = session.stats().retirement_stream_queries;
  auto& scratch      = active.allocate_buffer(16);
  cuda_check(cudaMemsetAsync(scratch.data(), 0x2a, scratch.size(), active.stream().value()));
  expect(!gate.released.load() && session.stats().retirement_stream_queries == queries &&
           session.stats().pressure_waits == 0,
         "intrinsically oversized active alias queried or waited on itself");
  gate.release_after_delay.store(true);
  auto output = session.finish();
  expect(gate.released.load() && output.size() == 1,
         "finish lost the sealed output or failed to drain its active alias");
  expect(columns_equal_completed(input->view().column(0), output.front()->view()),
         "active-alias retirement corrupted the sealed output");
  output.clear();
  resource.check();
  expect(!gate.timed_out.load(), "active-alias watchdog expired");
}

void test_frame_owner_growth(rmm::device_async_resource_ref upstream)
{
  rmm::cuda_stream stream(rmm::cuda_stream::flags::non_blocking);
  std::array const streams{stream.view()};
  checked_resource resource(upstream);
  simpatico::decode_session session(streams, resource);
  auto& frame       = simpatico::decode_session_test_access::frame(session);
  auto first_column = frame.make_column();
  first_column.adopt(cudf::make_numeric_column(cudf::data_type{cudf::type_id::UINT8},
                                               1,
                                               cudf::mask_state::UNALLOCATED,
                                               frame.stream(),
                                               frame.mr()));
  auto* const column_owner = &first_column.get();
  auto& first_buffer       = frame.allocate_buffer(16);
  auto& first_output       = frame.allocate_output_buffer(16);
  auto* const buffer_data  = first_buffer.data();
  auto* const output_data  = first_output.data();
  auto upload              = frame.host_array<std::uint64_t>(2);
  upload.front()           = 0x12345678;
  upload.back()            = 0xabcdef;
  auto* const upload_data  = upload.data();

  // Retain handles while each owner collection grows and the upload-record vector reallocates.
  for (std::size_t i = 0; i < 96; ++i) {
    (void)frame.make_column();
    (void)frame.allocate_buffer(16);
    (void)frame.allocate_output_buffer(16);
    auto extra    = frame.host_array<std::uint64_t>(2);
    extra.front() = i;
  }
  expect(&first_column.get() == column_owner && first_column->size() == 1,
         "owner growth invalidated a column slot");
  expect(first_buffer.data() == buffer_data && first_buffer.size() == 16 &&
           first_output.data() == output_data && first_output.size() == 16,
         "owner growth invalidated a buffer reference");
  expect(upload.data() == upload_data && upload.front() == 0x12345678 && upload.back() == 0xabcdef,
         "owner growth invalidated an upload span");
  expect(session.stats().retirement_stream_queries == 0 && session.stats().pressure_waits == 0,
         "small active-owner growth performed retirement work");

  cuda_check(cudaMemsetAsync(first_column->mutable_view().head<void>(), 0x2a, 1, stream.value()));
  cuda_check(cudaMemcpyAsync(
    first_buffer.data(), upload.data(), 16, cudaMemcpyHostToDevice, stream.value()));
  cuda_check(cudaMemcpyAsync(
    first_output.data(), first_buffer.data(), 16, cudaMemcpyDeviceToDevice, stream.value()));
  auto const column_value = frame.read_scalar(first_column->view().head<std::uint8_t>());
  auto const buffer_value =
    frame.read_scalar(static_cast<std::uint64_t const*>(first_output.data()));
  expect(column_value == 0x2a && buffer_value == upload.front(),
         "grown frame owners lost queued data");
  expect(session.finish().empty(), "raw owner-growth fixture published an output");
  resource.check();
}

struct estimate_only_representation : simpatico::compressed_representation {
  std::vector<simpatico::compressible_output> named_channels(rmm::cuda_stream_view) const override
  {
    throw std::logic_error("byte estimate requested channel views");
  }
};

void test_representation_byte_estimate(rmm::device_async_resource_ref upstream)
{
  rmm::cuda_stream stream(rmm::cuda_stream::flags::non_blocking);
  std::array const streams{stream.view()};
  checked_resource resource(upstream);
  checked_resource default_resource(upstream);
  current_resource_guard current(default_resource);
  {
    // Record sizes and capacities before cuDF hides the RMM owners. Its alloc_size() reports buffer
    // sizes, not the capacity retained after shrinking these deliberate slack fixtures.
    struct buffer_extent {
      std::size_t size;
      std::size_t capacity;
    };
    std::vector<buffer_extent> recorded_buffers;
    auto column = [&](std::size_t data_capacity, std::size_t mask_capacity = 0) {
      rmm::device_buffer data(data_capacity, stream.view(), resource);
      data.resize(1, stream.view());
      rmm::device_buffer mask(mask_capacity, stream.view(), resource);
      if (mask_capacity) mask.resize(sizeof(cudf::bitmask_type), stream.view());
      recorded_buffers.push_back({data.size(), data.capacity()});
      recorded_buffers.push_back({mask.size(), mask.capacity()});
      return std::make_unique<cudf::column>(
        cudf::data_type{cudf::type_id::UINT8}, 1, std::move(data), std::move(mask), 0);
    };
    auto base = std::make_unique<estimate_only_representation>();
    std::vector<std::unique_ptr<cudf::column>> children;
    children.push_back(column(128, 64));
    children.push_back(column(256));
    base->channels_.push_back(std::make_unique<cudf::column>(cudf::data_type{cudf::type_id::STRUCT},
                                                             1,
                                                             rmm::device_buffer{},
                                                             rmm::device_buffer{},
                                                             0,
                                                             std::move(children)));
    base->channels_.push_back(nullptr);
    base->channels_.push_back(cudf::make_empty_column(cudf::data_type{cudf::type_id::UINT8}));
    std::size_t base_buffer_sizes      = 0;
    std::size_t base_buffer_capacities = 0;
    for (auto const& buffer : recorded_buffers) {
      base_buffer_sizes += buffer.size;
      base_buffer_capacities += buffer.capacity;
    }
    expect(base_buffer_capacities == 448 && base_buffer_sizes == 6,
           "slack fixture must retain 448 capacity bytes behind 6 buffer-size bytes");
    constexpr std::size_t base_bytes = 6;

    auto empty_dict = std::make_unique<simpatico::dictionary_compressed_representation>(
      cudf::make_empty_column(cudf::data_type{cudf::type_id::DICTIONARY32}));
    auto input = make_strings_column({"aa", "bb", "aa"}, {true, false, true}, stream.view());
    auto dictionary =
      simpatico::dictionary_compressor{}.compress(input->view(), stream.view(), resource);
    auto& dict = dynamic_cast<simpatico::dictionary_compressed_representation&>(*dictionary);
    auto const dictionary_column_bytes = dict.dict_column->alloc_size();
    dict.channels_.push_back(column(64));
    dict.keys_chars_copy        = column(128);
    dict.keys_offsets_synth     = column(256);
    dict.indices_synth          = column(512);
    dict.null_mask_copy         = column(1024);
    auto const dictionary_bytes = dictionary_column_bytes + 5;
    auto const dictionary_width = dict.constant_key_width;

    auto fused = std::make_unique<simpatico::codegen_fused_representation>(
      simpatico::OpId::Bitpack, cudf::data_type{cudf::type_id::INT32}, 1);
    fused->channels_.push_back(column(64));
    fused->buffers.emplace_back("payload", column(512, 128));
    fused->buffers.emplace_back("empty", nullptr);
    constexpr std::size_t fused_bytes = 6;
    std::vector<std::unique_ptr<cudf::column>> fields;
    fields.push_back(column(256));
    fields.push_back(nullptr);
    auto extracted = std::make_unique<simpatico::bitextract_compressed_representation>(
      simpatico::bitextract_spec_result{}, std::move(fields));
    extracted->channels_.push_back(column(128));
    constexpr std::size_t extracted_bytes = 2;

    std::array<simpatico::compressed_representation const*, 5> const representations{
      base.get(), empty_dict.get(), dictionary.get(), fused.get(), extracted.get()};
    std::array<std::size_t, 5> expected{
      base_bytes, 0, dictionary_bytes, fused_bytes, extracted_bytes};
    auto* const lazy_dict       = empty_dict.get();
    auto* const fused_owner     = fused.get();
    auto* const extracted_owner = extracted.get();
    simpatico::decode_session session(streams, resource);
    auto& frame = simpatico::decode_session_test_access::frame(session);
    frame.keep_representation(std::move(base));
    frame.keep_representation(std::move(empty_dict));
    frame.keep_representation(std::move(dictionary));
    frame.keep_representation(std::move(fused));
    frame.keep_representation(std::move(extracted));
    frame.allocate_buffer(640).resize(1, stream.view());
    frame.keep_column(column(768, 64));
    frame.memo_column(1).adopt(column(1024));
    frame.terminal_memo(2);
    frame.memo_column(2).adopt(column(2048));
    frame.allocate_output_buffer(4096);
    frame.output().adopt(column(8192));
    (void)frame.host_array<std::byte>(17);
    constexpr std::size_t other_temporary_bytes = 640 + 5 + 1;

    auto observe = [&] {
      auto const allocations         = resource.observations->attempts.size();
      auto const default_allocations = default_resource.observations->attempts.size();
      stream_observation_scope observed;
      cuda_check(cudaStreamBeginCapture(stream.value(), cudaStreamCaptureModeGlobal));
      cudaGraph_t graph = nullptr;
      try {
        std::size_t total = other_temporary_bytes;
        for (std::size_t i = 0; i < representations.size(); ++i) {
          auto const actual = representations[i]->owned_device_bytes_estimate();
          if (actual != expected[i])
            throw std::runtime_error(
              "representation byte estimate mismatch at index " + std::to_string(i) +
              ": expected " + std::to_string(expected[i]) + ", actual " + std::to_string(actual));
          total += expected[i];
        }
        expect(frame.retained_device_bytes() == total && frame.retained_host_bytes() == 17,
               "representation observer changed frame totals or output exclusions");
      } catch (...) {
        (void)cudaStreamEndCapture(stream.value(), &graph);
        if (graph) (void)cudaGraphDestroy(graph);
        throw;
      }
      cuda_check(cudaStreamEndCapture(stream.value(), &graph));
      std::size_t nodes         = 0;
      auto const count_status   = cudaGraphGetNodes(graph, nullptr, &nodes);
      auto const destroy_status = cudaGraphDestroy(graph);
      cuda_check(count_status);
      cuda_check(destroy_status);
      expect(nodes == 0, "byte estimate enqueued CUDA work");
      for (auto const& calls : stream_fault::counts)
        expect(calls.queries == 0 && calls.synchronizations == 0,
               "byte estimate queried or synchronized a stream");
      expect(resource.observations->attempts.size() == allocations &&
               default_resource.observations->attempts.size() == default_allocations,
             "byte estimate allocated device storage");
      expect(!lazy_dict->keys_chars_copy && !lazy_dict->keys_offsets_synth &&
               !lazy_dict->indices_synth && !lazy_dict->null_mask_copy &&
               lazy_dict->constant_key_width == -1 && dict.constant_key_width == dictionary_width,
             "byte estimate mutated lazy dictionary state");
    };
    observe();
    // Moved-out owners and empty columns no longer contribute to the frame's estimate.
    auto moved_field  = std::move(extracted_owner->fields.front());
    auto moved_buffer = fused_owner->buffers.front().second->release();
    auto moved_output = frame.release(frame.output());
    expected[3] -= 5;
    expected[4] -= 1;
    observe();
    expect(session.stats().retirement_stream_queries == 0 && session.stats().pressure_waits == 0,
           "small estimate fixtures changed retirement policy");
    expect(session.finish().empty(), "estimate fixture unexpectedly published a result");
  }
  resource.check();
  default_resource.check();
  stream.synchronize();
}

void test_single_op_byte_estimate(rmm::device_async_resource_ref upstream)
{
  auto input = make_int32_table(1, 1037, 107);
  cudf::get_default_stream().synchronize();
  rmm::cuda_stream stream(rmm::cuda_stream::flags::non_blocking);
  std::array const streams{stream.view()};
  checked_resource resource(upstream);
  for (auto const* op : {"delta", "for", "rle"}) {
    {
      std::string error;
      auto rep =
        simpatico::compress_single_op(op, input->view().column(0), stream.view(), resource, &error);
      expect(rep != nullptr && error.empty(), "single-op estimate fixture compression failed");
      auto const plan_estimate = rep->owned_device_bytes_estimate();
      expect(plan_estimate > 0, "single-op estimate omitted its owned plan");
      auto const channels = rep->named_channels(stream.view());
      expect(!channels.empty(), "single-op estimate fixture has no borrowed output channels");
      expect(rep->owned_device_bytes_estimate() == plan_estimate,
             "exposing borrowed channels changed the single-op byte estimate");
      // The wrapper's own channels are normally empty; exercise the base-owner contract too.
      rep->channels_.push_back(cudf::make_numeric_column(cudf::data_type{cudf::type_id::UINT8},
                                                         17,
                                                         cudf::mask_state::UNALLOCATED,
                                                         stream.view(),
                                                         resource));
      expect(rep->channels_.front()->alloc_size() == 17,
             "single-op inherited column fixture has an unexpected buffer size");
      auto const expected               = plan_estimate + 17;
      std::size_t live_allocation_bytes = 0;
      for (auto const& [pointer, allocation] : resource.observations->live)
        live_allocation_bytes += allocation.bytes;
      expect(expected <= live_allocation_bytes,
             "single-op byte estimate exceeded its live allocation ledger");
      auto const allocations = resource.observations->attempts.size();
      {
        stream_observation_scope observed;
        for (int i = 0; i < 3; ++i)
          expect(rep->owned_device_bytes_estimate() == expected,
                 "single-op byte estimate lost its inherited contribution or changed on repeat");
        for (auto const& calls : stream_fault::counts)
          expect(calls.queries == 0 && calls.synchronizations == 0,
                 "single-op byte estimate queried or synchronized a stream");
      }
      expect(resource.observations->attempts.size() == allocations,
             "single-op byte estimate allocated device storage");
      auto const channels_after = rep->named_channels(stream.view());
      expect(channels_after.size() == channels.size(),
             "single-op byte estimate changed borrowed channel count");
      for (std::size_t i = 0; i < channels.size(); ++i)
        expect(channels_after[i].name == channels[i].name &&
                 channels_after[i].view.head<void>() == channels[i].view.head<void>() &&
                 channels_after[i].view.size() == channels[i].view.size(),
               "single-op byte estimate changed a borrowed channel view");
      simpatico::decode_session session(streams, resource);
      auto& frame = simpatico::decode_session_test_access::frame(session);
      frame.keep_representation(std::move(rep));
      expect(frame.retained_device_bytes() == expected,
             "single-op plan estimate did not reach frame accounting");
      expect(session.finish().empty(), "single-op estimate fixture published a result");
    }
    resource.check();
    resource.reset();
  }
  stream.synchronize();
}

void test_resources_and_failures(rmm::device_async_resource_ref upstream)
{
  auto input      = make_int32_table(8, 65549, 71);
  auto compressed = simpatico::compress_with_plan(
    input->view(), repeated_plan("input -> bitpack\n", 8), cudf::get_default_stream(), upstream);
  simpatico::stream_pool pool;
  expect(pool.init(4), "failure pool init");
  checked_resource explicit_resource(upstream);
  checked_resource default_resource(upstream);
  current_resource_guard current(default_resource);
  std::array<std::size_t, 8> const all{0, 1, 2, 3, 4, 5, 6, 7};
  {
    auto warm = simpatico::decompress(compressed, pool, explicit_resource);
    verify_projection(input->view(), warm->view(), all);
  }
  cuda_check(pool.sync_all());
  explicit_resource.check();
  expect(default_resource.observations->attempts.empty(),
         "decode scratch bypassed caller resource");
  auto const attempts = explicit_resource.observations->attempts;
  expect(attempts.size() > 8, "explicit resource observed outputs but no decode scratch");
  auto const later = std::find_if(
    attempts.begin(), attempts.end(), [&](auto stream) { return stream != attempts.front(); });
  expect(later != attempts.end(), "multiple column streams not observed");
  std::array<std::size_t, 4> const failures{
    0, 1, static_cast<std::size_t>(later - attempts.begin()) + 1, attempts.size() - 1};
  event_markers markers(pool);
  for (auto fail_at : failures) {
    explicit_resource.reset();
    explicit_resource.observations->fail_at         = fail_at;
    explicit_resource.observations->failure_markers = &markers;
    bool injected                                   = false;
    // Fail the first allocation before any codec work, then also fail cleanup's query on
    // another supplied lane. The original allocation exception must retain priority.
    std::optional<stream_failure_scope> cleanup_failure;
    if (fail_at == 0) cleanup_failure.emplace(stream_fault::operation::query, pool.streams[1]);
    stream_observation_scope observed;
    try {
      (void)simpatico::decompress(compressed, pool, explicit_resource);
    } catch (injected_out_of_memory const& error) {
      injected = std::string(error.what()).find("async decode injected allocation failure") !=
                   std::string::npos &&
                 error.requested_bytes == explicit_resource.observations->failed_request_bytes;
    }
    expect(injected, "allocation failure subtype, requested bytes, or message was lost");
    if (cleanup_failure) {
      expect(stream_fault::next == stream_fault::operation::none,
             "allocation cleanup did not consume its injected stream error");
      for (auto handle : pool.streams)
        expect(observed.count(handle).queries > 0,
               "allocation cleanup skipped a supplied lane after a stream error");
      cleanup_failure.reset();
    }
    // Query before any test-side synchronization or verification copies.
    expect(markers.complete(), "allocation failure escaped before all submitted streams drained");
    explicit_resource.check();
    explicit_resource.reset();
    {
      auto recovered = simpatico::decompress(compressed, pool, explicit_resource);
      verify_projection(input->view(), recovered->view(), all);
    }
    explicit_resource.check();
  }
  expect(default_resource.observations->attempts.empty(),
         "failure/retry scratch bypassed caller resource");
  default_resource.check();
  cuda_check(pool.sync_all());
}

}  // namespace

int main()
{
  try {
    cuda_check(cudaSetDevice(0));
    // cuDF intentionally retains its default pinned pool until process exit.
    // Keep this fixture's pinned allocations individually freed and leak-checkable.
    setenv("LIBCUDF_PINNED_POOL_SIZE", "0", 1);
    setenv("LIBCUDF_PINNED_POOL_MAX_SIZE", "0", 1);
    expect(cudf::config_default_pinned_memory_resource({.pool_size = 0}),
           "pinned resource was initialized before test configuration");
    rmm::mr::cuda_async_memory_resource upstream(128ULL << 20, 128ULL << 20);
    current_resource_guard current(upstream);
    test_table_contracts(upstream);
    test_completed_public_return(upstream);
    test_identity_owned_children(upstream);
    test_dictionary_width_metadata(upstream);
    test_dictionary_width_sliced_input(upstream);
    test_dictionary_width_large_keys(upstream);
    test_dictionary_width_failures(upstream);
    test_dictionary_pinned_observation(upstream);
    test_submission_and_kernel_lifetime(upstream);
    test_abandoned_session(upstream);
    test_session_state_contracts(upstream);
    test_pressure_waits_for_external_tail(upstream);
    test_frame_window_backpressure(upstream);
    test_stream_completion_failures(upstream);
    test_ready_stream_and_alias_retirement(upstream);
    test_sealed_frame_byte_accounting(upstream);
    test_external_phase_tail(upstream);
    test_request_copy_failure(upstream);
    test_byte_window_and_active_frame(upstream);
    test_active_frame_on_retired_alias(upstream);
    test_frame_owner_growth(upstream);
    test_representation_byte_estimate(upstream);
    test_single_op_byte_estimate(upstream);
    test_resources_and_failures(upstream);
    cuda_check(cudaDeviceSynchronize());
    std::puts("test_async_decode: OK");
    return 0;
  } catch (std::exception const& error) {
    std::fprintf(stderr, "test_async_decode: FAIL: %s\n", error.what());
    return 1;
  }
}
