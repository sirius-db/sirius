// SPDX-License-Identifier: Apache-2.0
#include "api/simpatico_codegen.hpp"
#include "codegen/decode/jit/renderer.hpp"
#include "codegen/jit/kernel_cache.hpp"
#include "decode/decode_session.hpp"
#include "decode_session_test_access.hpp"
#include "operators/constant_width_offsets.hpp"
#include "test_utils.hpp"
#include "util/host_observation.hpp"

#include <cudf/binaryop.hpp>
#include <cudf/copying.hpp>
#include <cudf/dictionary/dictionary_factories.hpp>
#include <cudf/dictionary/encode.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/utilities/pinned_memory.hpp>

#include <rmm/cuda_stream.hpp>
#include <rmm/error.hpp>
#include <rmm/mr/cuda_async_memory_resource.hpp>
#include <rmm/mr/per_device_resource.hpp>

#include <cuda/memory_resource>

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <charconv>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <limits>
#include <map>
#include <mutex>
#include <numeric>
#include <optional>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <variant>

// Linker wrappers affect only explicit calls from this executable. Injection is scoped to the
// submitting thread and one raw stream; event markers and independent gate controllers stay real.
// A pending synchronize fault also makes the target stream report itself busy, so a caller that
// waits only after a busy query still reaches the wait where the fault fires, whether or not the
// stream has work (a gated stream reports busy anyway).
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
bool pending(operation expected, cudaStream_t stream) noexcept
{
  return next == expected && stream == target;
}
bool consume(operation expected, cudaStream_t stream) noexcept
{
  if (!pending(expected, stream)) return false;
  next = operation::none;
  return true;
}
}  // namespace stream_fault

extern "C" cudaError_t __real_cudaStreamQuery(cudaStream_t);
extern "C" cudaError_t __real_cudaStreamSynchronize(cudaStream_t);
extern "C" cudaError_t __wrap_cudaStreamQuery(cudaStream_t stream)
{
  stream_fault::observe(stream_fault::operation::query, stream);
  if (stream_fault::consume(stream_fault::operation::query, stream)) return cudaErrorInvalidValue;
  if (stream_fault::pending(stream_fault::operation::synchronize, stream)) return cudaErrorNotReady;
  return __real_cudaStreamQuery(stream);
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

class event_markers {
 public:
  explicit event_markers(simpatico::stream_pool const& pool) : event_markers(pool.streams) {}
  explicit event_markers(std::vector<cudaStream_t> streams) : streams_(std::move(streams))
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
// Byte counters change when allocate and deallocate are called, not when the GPU reaches them.
class checked_resource {
 public:
  struct allocation {
    std::size_t bytes;
    cudaStream_t stream;
  };
  // Everything reset() clears between cases.
  struct counters {
    std::vector<cudaStream_t> attempts;
    std::optional<std::size_t> fail_at;
    std::size_t failed_request_bytes                 = 0;
    event_markers* failure_markers                   = nullptr;
    std::atomic<bool> const* release_required        = nullptr;
    std::atomic<bool> const* second_release_required = nullptr;
    // Set by the first deallocation, so a test can order a release before a stream gate opens.
    std::atomic<bool>* signal_on_deallocate = nullptr;
    std::size_t live_bytes                  = 0;
    std::size_t peak_live_bytes             = 0;
    bool wrong_thread                       = false;
    bool wrong_stream                       = false;
    bool early_release                      = false;
    bool synchronous_access                 = false;
  };
  struct state : counters {
    std::mutex mutex;
    std::thread::id owner = std::this_thread::get_id();
    std::map<void*, allocation> live;
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
    if (ptr) {
      observations->live_bytes += bytes;
      observations->peak_live_bytes =
        std::max(observations->peak_live_bytes, observations->live_bytes);
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
    if (observations->signal_on_deallocate) observations->signal_on_deallocate->store(true);
    if (ptr) {
      auto const found = observations->live.find(ptr);
      observations->wrong_stream |= found == observations->live.end() ||
                                    found->second.stream != stream.get() ||
                                    found->second.bytes != bytes;
      if (found != observations->live.end()) {
        observations->live_bytes -= found->second.bytes;
        observations->live.erase(found);
      }
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
    static_cast<counters&>(*observations) = counters{};
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

// The deadlock watchdog's budget. SIMPATICO_TEST_GATE_TIMEOUT_MS overrides the default for slow
// machines; the variable is read once.
std::chrono::milliseconds gate_timeout()
{
  static std::chrono::milliseconds const timeout = [] {
    std::chrono::milliseconds const fallback{10'000};
    char const* const value = std::getenv("SIMPATICO_TEST_GATE_TIMEOUT_MS");
    if (!value) return fallback;
    std::string_view const text{value};
    long long milliseconds  = 0;
    auto const [end, error] = std::from_chars(text.data(), text.data() + text.size(), milliseconds);
    if (error != std::errc{} || end != text.data() + text.size() || milliseconds <= 0)
      throw std::invalid_argument(
        "SIMPATICO_TEST_GATE_TIMEOUT_MS must be a positive number of milliseconds");
    return std::chrono::milliseconds{milliseconds};
  }();
  return timeout;
}

// compute-sanitizer maps its collection libraries into the target process; their presence in
// /proc/self/maps identifies a run under the tool.
bool compute_sanitizer_mapped()
{
  std::ifstream maps("/proc/self/maps");
  for (std::string line; std::getline(maps, line);) {
    if (line.find("libsanitizer-collection.so") != std::string::npos ||
        line.find("libsanitizer-public.so") != std::string::npos)
      return true;
  }
  return false;
}

// Whether this process arms its stream gates (see stream_gate). SIMPATICO_TEST_STREAM_GATES decides
// when set: `on` and `off` force the mode, `auto` and an unset variable arm the gates unless
// compute-sanitizer is mapped into the process, and any other value is an error. The variable is
// read and /proc/self/maps scanned once; an inert decision prints one notice on stderr.
bool stream_gates_armed()
{
  static bool const armed = [] {
    if (char const* const value = std::getenv("SIMPATICO_TEST_STREAM_GATES")) {
      std::string_view const text{value};
      if (text == "on") return true;
      if (text == "off") {
        std::fputs(
          "stream_gate: stream gates are inert (SIMPATICO_TEST_STREAM_GATES=off); no-wait "
          "assertions are not verified in this run\n",
          stderr);
        return false;
      }
      if (text != "auto")
        throw std::invalid_argument("SIMPATICO_TEST_STREAM_GATES must be on, off, or auto");
    }
    if (!compute_sanitizer_mapped()) return true;
    std::fputs(
      "stream_gate: compute-sanitizer detected; stream gates are inert because the tool blocks "
      "kernel launches queued behind a blocked stream once a budget of pending launches is "
      "exceeded, so no-wait assertions are not verified in this run (native runs verify them)\n",
      stderr);
    return false;
  }();
  return armed;
}

// Holds a stream so that work queued behind it stays pending: a host function on `stream` spins
// until `released` is set, and a call that returns while the gate is held is known not to have
// waited for that stream. The callback only touches atomics. A controller thread sets `released`
// on request (`release_after_delay`, after 50 ms) or when the deadlock watchdog expires, in which
// case it also sets `timed_out` and reports the trip on stderr with the elapsed time so that a
// watchdog release is never mistaken for a real wait; gate_timeout() sets the budget.
//
// Under compute-sanitizer the gate is inert. The tool blocks the host inside cuLaunchKernel once
// the launches pending behind a blocked stream exceed a history-dependent budget, so a held gate
// would deadlock any test that keeps appending, and whether a call waited for its lane cannot be
// observed at all. An inert gate queues nothing and never sets `released`, `entered`, or
// `timed_out`; the expect_* members then assert nothing and wait_until_entered() returns at once,
// so a test's gate-dependent checks are visibly skipped while its ordering, ledger, and result
// checks keep running. stream_gates_armed() decides the mode once per process, with the
// SIMPATICO_TEST_STREAM_GATES override. The destructor releases, joins the controller, and
// synchronizes the stream in either mode.
class stream_gate {
 public:
  explicit stream_gate(::cuda::stream_ref stream) : stream_(stream)
  {
    if (!armed_) return;
    auto const deadline = started_ + gate_timeout();
    controller_         = std::thread([this, deadline] {
      while (!released.load()) {
        if (release_after_delay.load()) {
          std::this_thread::sleep_for(std::chrono::milliseconds(50));
          released.store(true);
          break;
        }
        if (std::chrono::steady_clock::now() >= deadline) {
          timed_out.store(true);
          released.store(true);
          std::fprintf(stderr, "stream_gate: watchdog timed_out (%s)\n", status().c_str());
          break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
      }
    });
    try {
      cuda_check(cudaLaunchHostFunc(
        stream.get(),
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
    if (controller_.joinable()) controller_.join();
    (void)cudaStreamSynchronize(stream_.get());
  }
  stream_gate(stream_gate const&)            = delete;
  stream_gate& operator=(stream_gate const&) = delete;

  [[nodiscard]] bool armed() const noexcept { return armed_; }

  // Spins until the callback has started on the stream, which means all work queued before the
  // gate has completed, or until the watchdog trips.
  void wait_until_entered() const
  {
    if (!armed_) return;
    while (!entered.load() && !timed_out.load())
      std::this_thread::yield();
  }

  // The gate-dependent assertions. Each checks its condition only when the gate is armed and then
  // appends status() to the failure text; an inert gate makes them no-ops.
  void expect_if_armed(bool condition, char const* what) const
  {
    if (!armed_) return;
    expect(condition, (std::string{what} + " (" + status() + ")").c_str());
  }
  void expect_not_released(char const* what) const { expect_if_armed(!released.load(), what); }
  void expect_completed(char const* what) const { expect_if_armed(released.load(), what); }
  void expect_not_timed_out(char const* what) const { expect_if_armed(!timed_out.load(), what); }

  // Watchdog state for a failure message: release and timed_out flags, the time since construction,
  // and the configured budget.
  std::string status() const
  {
    auto const elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                           std::chrono::steady_clock::now() - started_)
                           .count();
    return "released=" + std::to_string(released.load()) +
           " timed_out=" + std::to_string(timed_out.load()) +
           " elapsed_ms=" + std::to_string(elapsed) +
           " timeout_ms=" + std::to_string(gate_timeout().count());
  }

  std::atomic<bool> released{false};
  std::atomic<bool> entered{false};
  std::atomic<bool> timed_out{false};
  std::atomic<bool> release_after_delay{false};

 private:
  ::cuda::stream_ref stream_;
  bool armed_                                    = stream_gates_armed();
  std::chrono::steady_clock::time_point started_ = std::chrono::steady_clock::now();
  std::thread controller_;
};

// Flags a deallocation on `resource` that precedes the release of `gate` (and of `second`, when
// given) as an early release. An inert gate holds nothing, so it imposes no ordering and is
// skipped.
class release_observation {
 public:
  release_observation(checked_resource& resource,
                      stream_gate const& gate,
                      stream_gate const* second = nullptr)
    : resource_(resource)
  {
    resource_.observations->release_required = gate.armed() ? &gate.released : nullptr;
    resource_.observations->second_release_required =
      (second && second->armed()) ? &second->released : nullptr;
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

std::string repeated_plan(std::string const& plan, int columns)
{
  std::string result;
  for (int i = 0; i < columns; ++i) {
    if (i) result += "\n---\n";
    result += plan;
  }
  return result;
}

// A `dictionary` plan node carrying `hint` as its key width, for reconstruction fixtures.
simpatico::PlanNode dictionary_plan_node(std::int64_t hint = -1)
{
  simpatico::PlanNode node;
  node.op                        = "dictionary";
  node.dictionary_key_width_hint = hint;
  return node;
}

// A standalone representation decoded through the public single-request helper, completed.
std::unique_ptr<cudf::column> decode_completed(simpatico::compressed_representation const& rep,
                                               ::cuda::stream_ref stream,
                                               rmm::device_async_resource_ref mr)
{
  return simpatico::decompress_standalone_representation(&rep, stream, mr, nullptr);
}

// An int32 table of `columns` columns compressed with `plan` each, with `lanes` streams and a
// checked resource to decode it.
struct decode_fixture {
  decode_fixture(rmm::device_async_resource_ref upstream,
                 std::string const& plan,
                 int columns,
                 int rows,
                 int seed,
                 int lanes)
    : input(make_int32_table(columns, rows, seed)),
      compressed(simpatico::compress_with_plan(
        input->view(), repeated_plan(plan, columns), cudf::get_default_stream(), upstream)),
      resource(upstream)
  {
    expect(pool.init(lanes), "decode fixture lanes");
  }

  // Decode once, so kernels are compiled before a test observes waits, and check the resource.
  void warm()
  {
    {
      auto warm = simpatico::decompress(compressed, pool, resource);
    }
    cuda_check(pool.sync_all());
    resource.check();
  }

  std::unique_ptr<cudf::table> input;
  simpatico::compressed_table compressed;
  simpatico::stream_pool pool;
  checked_resource resource;
};

simpatico::column_decode_request value_request(simpatico::compressed_column const& column)
{
  return {.source = std::cref(*column.plan_tree), .result = simpatico::value_result{column.dtype}};
}

bool columns_equal_completed(cudf::column_view expected, cudf::column_view actual)
{
  auto const equal = columns_equal(expected, actual);
  // Order test-only default-stream readbacks before nonblocking-stream deallocation.
  ::cuda::stream_ref{cudaStream_t{}}.sync();
  return equal;
}

bool strings_equal_completed(cudf::column_view expected,
                             cudf::column_view actual,
                             ::cuda::stream_ref stream)
{
  auto const equal = strings_equal(expected, actual, stream);
  ::cuda::stream_ref{cudaStream_t{}}.sync();
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
      expect(throws([&] { simpatico::decompress(compressed, invalid, pool, mr); }),
             "invalid projection accepted");
      simpatico::stream_pool empty_pool;
      expect(throws([&] { simpatico::decompress(compressed, empty_pool, mr); }),
             "empty pool accepted");
      auto saved = std::move(compressed.columns[7].plan_tree);
      expect(throws([&] { simpatico::decompress(compressed, pool, mr); }), "null plan accepted");
      compressed.columns[7].plan_tree = std::move(saved);
      auto edge = compressed.columns[7].plan_tree->nodes.front().children.front();
      compressed.columns[7].plan_tree->nodes.front().children.front().child = 99999;
      expect(throws([&] { simpatico::decompress(compressed, pool, mr); }),
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
          simpatico::decompress(compressed, ::cuda::stream_ref{pool.streams.front()}, upstream);
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
        column = decode_completed(*rep, pool.streams.front(), upstream);
        break;
      }
    }
    // Identity has no post-join scratch frees, so readiness directly checks completed copies before
    // verification can hide a missing wait.
    gate.expect_completed("public decode returned before its gated stream completed");
    expect(markers.complete(), "public decode returned before all pool markers completed");
    for (auto stream : pool.streams)
      expect(cudaStreamQuery(stream) == cudaSuccess, "public decode returned with unfinished work");
    gate.expect_not_timed_out("public completion watchdog expired");
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

// The span overloads borrow their streams. Repeated streams are accepted and give the same result
// as the stream_pool overloads; the call waits on each distinct stream at most once, and that wait
// also covers work others queued there; the streams stay usable afterwards.
void test_borrowed_streams(rmm::device_async_resource_ref upstream)
{
  auto input      = make_int32_table(6, 65549, 59);
  auto compressed = simpatico::compress_with_plan(
    input->view(),
    repeated_plan("input -> delta -> differences\ndelta.differences -> bitpack\n", 6),
    cudf::get_default_stream(),
    upstream);
  std::array<std::size_t, 4> const selected{5, 2, 2, 0};
  std::array<simpatico::decode_predicate, 4> const inactive{};
  simpatico::stream_pool pool;
  expect(pool.init(2), "borrowed-stream pool init");
  std::array const borrowed{::cuda::stream_ref{pool.streams[0]},
                            ::cuda::stream_ref{pool.streams[1]},
                            ::cuda::stream_ref{pool.streams[0]},
                            ::cuda::stream_ref{pool.streams[0]}};
  // Also compiles the kernels, so the observed calls below make no first-use waits.
  auto const pooled = simpatico::decompress(compressed, selected, pool, upstream);
  verify_projection(input->view(), pooled->view(), selected);
  for (bool const predicated : {false, true}) {
    cuda_check(pool.sync_all());
    stream_gate gate(pool.streams[1]);
    event_markers markers(pool);
    markers.record();
    gate.release_after_delay.store(true);
    std::unique_ptr<cudf::table> table;
    {
      stream_observation_scope observed;
      table = predicated ? simpatico::decompress(compressed, selected, inactive, borrowed, upstream)
                         : simpatico::decompress(compressed, selected, borrowed, upstream);
      for (auto handle : pool.streams)
        expect(observed.count(handle).queries <= 1 && observed.count(handle).synchronizations <= 1,
               "borrowed streams were waited on beyond the final wait");
    }
    gate.expect_completed("the final wait skipped work queued on a borrowed stream");
    expect(markers.complete(), "the final wait skipped work queued on a borrowed stream");
    gate.expect_not_timed_out("borrowed-stream watchdog expired");
    verify_projection(input->view(), table->view(), selected);
  }
  for (auto handle : pool.streams)
    cuda_check(cudaStreamSynchronize(handle));
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
    auto output = decode_completed(*representation, stream.view(), supplied);
    // Observe completed return before any verification readback can hide an unfinished copy.
    gate.expect_completed("owning identity copy returned before nested data completed");
    expect(cudaStreamQuery(stream.value()) == cudaSuccess,
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
    gate.expect_not_timed_out("owning identity copy watchdog expired");
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
      gate.expect_completed("dictionary publication did not complete prior work");
      expect(published->constant_key_width == fixture.width,
             "dictionary publication did not complete its metadata");
      gate.expect_not_timed_out("dictionary publication watchdog expired");
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
    auto imported = simpatico::reconstruct_representation(
      "dictionary", channel_names, std::move(channels), stream.view(), mr, &error);
    auto const* imported_dictionary =
      dynamic_cast<simpatico::dictionary_compressed_representation const*>(imported.get());
    expect(imported_dictionary != nullptr && error.empty(), "dictionary channel import failed");
    expect(imported_dictionary->constant_key_width == fixture.width,
           "long-lived dictionary import did not publish eager key width");
    stream.synchronize();

    {
      std::array const streams{::cuda::stream_ref{stream}};
      simpatico::decode_session session(streams, mr);
      auto& frame = simpatico::decode_session_test_access::frame(session);
      std::vector<std::unique_ptr<cudf::column>> decoded;
      for (auto const& channel : original->named_channels(stream.view()))
        decoded.push_back(std::make_unique<cudf::column>(channel.view, stream.view(), mr));
      auto const node    = dictionary_plan_node();
      auto const rebuilt = simpatico::reconstruct_decode_representation(
        node, channel_names, std::move(decoded), frame);
      expect(throws([&] {
               (void)simpatico::reconstruct_decode_representation(
                 node,
                 channel_names,
                 std::vector<std::unique_ptr<cudf::column>>(channel_names.size()),
                 frame);
             }),
             "dictionary reconstruction accepted missing channels");
      auto const output = simpatico::decode_standalone(*rebuilt, frame);
      stream.synchronize();
      expect(strings_equal_completed(input->view(), output->view(), stream.view()),
             "frame-reconstructed dictionary roundtrip mismatch");
      expect(session.finish().empty(), "raw dictionary fixture published a session result");
    }

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

      auto rebuilt = decode_completed(reconstructed, stream.view(), mr);
      expect(rebuilt != nullptr && rebuilt->type() == input->type(),
             "unknown-width dictionary output type mismatch");
      expect(strings_equal_completed(input->view(), rebuilt->view(), stream.view()),
             "unknown-width dictionary output mismatch");
      expect(reconstructed.constant_key_width == -1,
             "decode cached reconstructed dictionary width metadata");

      auto loaded = decode_completed(*imported_dictionary, stream.view(), mr);
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

// The hint reaches the frame-local representation at construction, so a hinted decode observes
// nothing on the host, while a reconstruction without a hint still measures through the frame; a
// hint that contradicts the key channels is rejected as corrupt metadata. What the compress walk
// and the readers publish is covered by test_compressed_table_io.
void test_dictionary_key_width_hint(rmm::device_async_resource_ref mr)
{
  rmm::cuda_stream stream(rmm::cuda_stream::flags::non_blocking);
  constexpr cudf::size_type rows = 1037;
  std::vector<std::string> const keys{"A", "N", "R"};
  std::vector<std::string> values(rows);
  for (std::size_t i = 0; i < values.size(); ++i)
    values[i] = keys[(i * 7 + i / 3) % keys.size()];
  auto input   = make_strings_column(values, {}, stream.view());
  auto encoded = simpatico::dictionary_compressor{}.compress(input->view(), stream.view(), mr);
  auto const& original =
    dynamic_cast<simpatico::dictionary_compressed_representation const&>(*encoded);
  std::vector<std::string> names;
  for (auto const& channel : original.named_channels(stream.view()))
    names.push_back(channel.name);
  auto const copy_channels = [&] {
    std::vector<std::unique_ptr<cudf::column>> channels;
    for (auto const& channel : original.named_channels(stream.view()))
      channels.push_back(std::make_unique<cudf::column>(channel.view, stream.view(), mr));
    return channels;
  };
  stream.synchronize();
  std::array const streams{::cuda::stream_ref{stream}};
  for (std::int64_t const hint : {std::int64_t{1}, std::int64_t{-1}}) {
    auto const node = dictionary_plan_node(hint);
    simpatico::decode_session session(streams, mr);
    auto& frame               = simpatico::decode_session_test_access::frame(session);
    auto const uploads_before = simpatico::decode_session_test_access::host_uploads(session);
    auto const rebuilt =
      simpatico::reconstruct_decode_representation(node, names, copy_channels(), frame);
    auto const& dictionary =
      dynamic_cast<simpatico::dictionary_compressed_representation const&>(*rebuilt);
    expect(dictionary.constant_key_width == hint,
           "node hint did not reach the reconstructed dictionary");
    std::unique_ptr<cudf::column> output;
    {
      // A published width leaves no host observation, so the decode returns while its lane is
      // still gated; an unknown width measures and returns only once the gate has released.
      stream_gate gate(stream.view());
      if (hint < 0) gate.release_after_delay.store(true);
      output = simpatico::decode_standalone(*rebuilt, frame);
      if (hint > 0)
        gate.expect_not_released("known key width waited for its lane");
      else
        gate.expect_completed("unknown key width returned before its measurement completed");
      expect(simpatico::decode_session_test_access::host_uploads(session) == uploads_before,
             "dictionary decode changed the frame's retained uploads");
      gate.release_after_delay.store(true);
      expect(session.finish().empty(), "hint fixture published a session result");
      gate.expect_not_timed_out("dictionary hint watchdog expired");
    }
    expect(dictionary.constant_key_width == hint,
           "decode changed the reconstructed dictionary width");
    expect(strings_equal_completed(input->view(), output->view(), stream.view()),
           "hinted reconstruction roundtrip mismatch");
  }
  // A hint that does not describe the stored key chars is corrupt metadata, not a decline.
  for (std::int64_t const hint : {std::int64_t{3}, std::int64_t{-2}}) {
    auto const node = dictionary_plan_node(hint);
    simpatico::decode_session session(streams, mr);
    auto& frame = simpatico::decode_session_test_access::frame(session);
    expect(throws([&] {
             (void)simpatico::reconstruct_decode_representation(
               node, names, copy_channels(), frame);
           }),
           "inconsistent dictionary key width hint was accepted");
    expect(session.finish().empty(), "rejected hint fixture published a session result");
  }
}

// BOOL8 equality on valid rows plus identical validity; bytes under nulls are unspecified. Like the
// other *_completed helpers, the default-stream readbacks are completed before the caller's
// nonblocking-stream deallocations.
bool bool8_equal_where_valid_completed(cudf::column_view expected, cudf::column_view actual)
{
  bool equal = expected.type().id() == cudf::type_id::BOOL8 && actual.type() == expected.type() &&
               expected.size() == actual.size() && validity_equal(expected, actual);
  if (equal) {
    auto const n = static_cast<std::size_t>(expected.size());
    std::vector<std::uint8_t> a(n), b(n);
    cuda_check(cudaMemcpy(a.data(), expected.head<std::uint8_t>(), n, cudaMemcpyDeviceToHost));
    cuda_check(cudaMemcpy(b.data(), actual.head<std::uint8_t>(), n, cudaMemcpyDeviceToHost));
    auto const valid = host_validity_bits(expected);
    for (std::size_t i = 0; i < n && equal; ++i)
      equal = !valid[i] || (a[i] != 0) == (b[i] != 0);
  }
  ::cuda::stream_ref{cudaStream_t{}}.sync();
  return equal;
}

// The dictionary predicate answers equals_any from a lookup table over the keys that is built
// without a cuDF scalar: the needles upload from frame-owned host storage and nothing on the path
// waits for the stream. Its semantics are cuDF's EQUAL folded with LOGICAL_OR over the decoded
// strings.
void test_dictionary_predicate_lookup(rmm::device_async_resource_ref mr)
{
  rmm::cuda_stream stream(rmm::cuda_stream::flags::non_blocking);
  std::array const streams{::cuda::stream_ref{stream}};
  std::vector<std::string> const key_pool{
    "DELIVER IN PERSON", "COLLECT COD", "NONE", "TAKE BACK RETURN", ""};
  constexpr cudf::size_type rows = 1037;

  auto const evaluate = [&](simpatico::dictionary_compressed_representation const& dictionary,
                            std::vector<std::string> const& needles) {
    simpatico::decode_session session(streams, mr);
    auto& frame = simpatico::decode_session_test_access::frame(session);
    simpatico::decode_predicate predicate;
    predicate.equals_any = needles;
    auto result          = dictionary.decompress_predicate(predicate, frame);
    stream.synchronize();
    expect(session.finish().empty(), "predicate fixture published a session result");
    return result;
  };
  auto const reference = [&](cudf::column_view strings, std::vector<std::string> const& needles) {
    auto const bool_type = cudf::data_type{cudf::type_id::BOOL8};
    std::unique_ptr<cudf::column> mask;
    for (auto const& needle : needles) {
      cudf::string_scalar const value(needle, true, stream.view(), mr);
      auto hit = cudf::binary_operation(
        strings, value, cudf::binary_operator::EQUAL, bool_type, stream.view(), mr);
      mask = mask ? cudf::binary_operation(mask->view(),
                                           hit->view(),
                                           cudf::binary_operator::LOGICAL_OR,
                                           bool_type,
                                           stream.view(),
                                           mr)
                  : std::move(hit);
    }
    stream.synchronize();
    return mask;
  };
  auto const make_input = [&](std::size_t key_count, bool nullable) {
    std::vector<std::string> values(rows);
    std::vector<bool> valid;
    if (nullable) valid.assign(rows, true);
    for (std::size_t i = 0; i < values.size(); ++i) {
      values[i] = key_pool[(i * 7 + i / 3) % key_count];
      if (nullable && i % 11 == 0) valid[i] = false;
    }
    return make_strings_column(values, valid, stream.view());
  };

  for (std::size_t key_count : {std::size_t{1}, std::size_t{2}, std::size_t{4}, std::size_t{5}}) {
    for (bool nullable : {false, true}) {
      auto input   = make_input(key_count, nullable);
      auto encoded = simpatico::dictionary_compressor{}.compress(input->view(), stream.view(), mr);
      auto const& dictionary =
        dynamic_cast<simpatico::dictionary_compressed_representation const&>(*encoded);
      auto const& first = key_pool.front();
      auto const& last  = key_pool[key_count - 1];
      std::vector<std::vector<std::string>> const needle_sets{
        {first},
        {first, last},
        {"absent"},
        {""},
        {first + " AND MORE BYTES THAN ANY KEY"},
        {first.substr(0, first.size() - 1)},
        {"absent", last, first},
      };
      for (auto const& needles : needle_sets) {
        auto result = evaluate(dictionary, needles);
        expect(result != nullptr, "dictionary predicate declined a supported shape");
        expect(result->type().id() == cudf::type_id::BOOL8 && result->size() == rows,
               "dictionary predicate result shape");
        auto expected = reference(input->view(), needles);
        expect(bool8_equal_where_valid_completed(expected->view(), result->view()),
               "dictionary predicate lookup differs from the cuDF reference");
      }
    }
  }

  // Without a cuDF scalar the lookup table needs no host wait, so the predicate returns while the
  // frame's stream is still gated.
  {
    auto input   = make_input(4, false);
    auto encoded = simpatico::dictionary_compressor{}.compress(input->view(), stream.view(), mr);
    auto const& dictionary =
      dynamic_cast<simpatico::dictionary_compressed_representation const&>(*encoded);
    std::vector<std::string> const needles{key_pool[1], key_pool[3]};
    simpatico::decode_session session(streams, mr);
    auto& frame = simpatico::decode_session_test_access::frame(session);
    simpatico::decode_predicate predicate;
    predicate.equals_any = needles;
    std::unique_ptr<cudf::column> result;
    {
      stream_gate gate(stream.view());
      result = dictionary.decompress_predicate(predicate, frame);
      expect(result != nullptr, "dictionary predicate declined the gated shape");
      gate.expect_not_released("dictionary predicate waited for its lane");
      gate.release_after_delay.store(true);
      expect(session.finish().empty(), "gated predicate fixture published a session result");
      gate.expect_not_timed_out("dictionary predicate watchdog expired");
    }
    auto expected = reference(input->view(), needles);
    expect(bool8_equal_where_valid_completed(expected->view(), result->view()),
           "gated dictionary predicate lookup differs from the cuDF reference");
  }

  // Zero keys: an all-null column has no key set, so every comparison is null.
  {
    auto input = make_strings_column(
      std::vector<std::string>(rows, "x"), std::vector<bool>(rows, false), stream.view());
    auto encoded = simpatico::dictionary_compressor{}.compress(input->view(), stream.view(), mr);
    auto const& dictionary =
      dynamic_cast<simpatico::dictionary_compressed_representation const&>(*encoded);
    auto result = evaluate(dictionary, {"x"});
    expect(result != nullptr && result->size() == rows && result->null_count() == rows,
           "all-null dictionary predicate is not all null");
  }
  // Indices narrower than INT32 are left to the generic path.
  {
    auto input  = make_strings_column({"a", "b", "a"}, {}, stream.view());
    auto narrow = cudf::dictionary::encode(
      input->view(), cudf::data_type{cudf::type_id::INT16}, stream.view(), mr);
    stream.synchronize();
    simpatico::dictionary_compressed_representation const dictionary(std::move(narrow));
    expect(evaluate(dictionary, {"a"}) == nullptr,
           "narrow-index dictionary predicate was not declined");
  }
}

void test_dictionary_offsets_validation(rmm::device_async_resource_ref mr)
{
  struct fixture {
    cudf::size_type offsets;
    cudf::mask_state mask;
    bool accepted;
  };
  std::array<fixture, 4> const fixtures{{
    {0, cudf::mask_state::UNALLOCATED, false},
    {2, cudf::mask_state::ALL_NULL, false},
    {1, cudf::mask_state::UNALLOCATED, true},
    {2, cudf::mask_state::ALL_VALID, true},
  }};
  rmm::cuda_stream stream(rmm::cuda_stream::flags::non_blocking);
  std::array const streams{::cuda::stream_ref{stream}};
  std::vector<std::string> const names{"keys_offsets", "keys_chars", "indices"};
  for (auto type : {cudf::type_id::INT32, cudf::type_id::INT64}) {
    for (auto const& fixture : fixtures) {
      simpatico::decode_session session(streams, mr);
      auto& frame = simpatico::decode_session_test_access::frame(session);
      std::vector<std::unique_ptr<cudf::column>> channels;
      channels.push_back(cudf::make_numeric_column(
        cudf::data_type{type}, fixture.offsets, fixture.mask, stream.view(), mr));
      if (fixture.offsets > 0) {
        cuda_check(cudaMemsetAsync(channels[0]->mutable_view().head<void>(),
                                   0,
                                   fixture.offsets * cudf::size_of(cudf::data_type{type}),
                                   stream.value()));
      }
      channels.push_back(cudf::make_empty_column(cudf::data_type{cudf::type_id::UINT8}));
      channels.push_back(cudf::make_empty_column(cudf::data_type{cudf::type_id::INT32}));
      stream.synchronize();
      {
        stream_observation_scope observed;
        bool accepted = false;
        try {
          auto const rebuilt = simpatico::reconstruct_decode_representation(
            dictionary_plan_node(), names, std::move(channels), frame);
          auto const& dictionary =
            dynamic_cast<simpatico::dictionary_compressed_representation const&>(*rebuilt);
          auto const keys = cudf::dictionary_column_view(dictionary.dict_column->view()).keys();
          expect(keys.size() == fixture.offsets - 1 && keys.child(0).null_count() == 0,
                 "dictionary reconstruction changed key offsets metadata");
          expect(keys.child(0).nullable() == (fixture.mask == cudf::mask_state::ALL_VALID),
                 "dictionary reconstruction lost an all-valid offsets mask");
          accepted = true;
        } catch (std::invalid_argument const& error) {
          expect(
            std::string(error.what()) == "dictionary: key offsets must be nonempty and null-free",
            "dictionary offsets validation lost its diagnostic");
        }
        expect(accepted == fixture.accepted, "dictionary offsets acceptance mismatch");
        for (auto const& calls : stream_fault::counts) {
          expect(calls.queries == 0 && calls.synchronizations == 0,
                 "dictionary offsets validation queried or synchronized a stream");
        }
      }
      expect(session.finish().empty(), "offsets validation fixture published a session result");
    }
  }
}

// A dictionary the decoder rebuilds keeps its decoded INT32 indices, type and buffer, so the
// constant-width gather and the lookup-table predicate, which both require INT32 codes, still
// apply.
void test_dictionary_rebuild_keeps_indices(rmm::device_async_resource_ref mr)
{
  rmm::cuda_stream stream(rmm::cuda_stream::flags::non_blocking);
  std::array const streams{::cuda::stream_ref{stream}};
  std::vector<std::string> const keys{"AA", "BB", "CC"};
  std::vector<std::string> values;
  for (int row = 0; row < 1037; ++row)
    values.push_back(keys[(row * 7) % keys.size()]);
  auto input   = make_strings_column(values, {}, stream.view());
  auto encoded = simpatico::dictionary_compressor{}.compress(input->view(), stream.view(), mr);
  std::vector<std::string> names;
  std::vector<std::unique_ptr<cudf::column>> channels;
  for (auto const& channel : encoded->named_channels(stream.view())) {
    names.push_back(channel.name);
    channels.push_back(std::make_unique<cudf::column>(channel.view, stream.view(), mr));
  }
  expect(channels[2]->type().id() == cudf::type_id::INT32, "fixture indices are not INT32");
  auto const* const decoded_indices = channels[2]->view().head<void>();
  stream.synchronize();

  simpatico::decode_session session(streams, mr);
  auto& frame        = simpatico::decode_session_test_access::frame(session);
  auto const rebuilt = simpatico::reconstruct_decode_representation(
    dictionary_plan_node(2), names, std::move(channels), frame);
  auto const& dictionary =
    dynamic_cast<simpatico::dictionary_compressed_representation const&>(*rebuilt);
  auto const indices = cudf::dictionary_column_view(dictionary.dict_column->view()).indices();
  expect(indices.type().id() == cudf::type_id::INT32,
         "decoder rebuild changed the dictionary index type");
  expect(indices.head<void>() == decoded_indices, "decoder rebuild copied the dictionary indices");
  auto const hits = dictionary.decompress_predicate(simpatico::decode_predicate{{"BB"}}, frame);
  expect(hits != nullptr, "rebuilt dictionary declined the lookup-table predicate");
  expect(session.finish().empty(), "rebuild fixture published a session result");
}

// Unsigned indices, such as a narrow field a bitjoin decodes, take cuDF's dictionary factory on
// both the decoder and the loader. cuDF retags them as the signed type of the same width, so these
// codes stay within that range; UINT8 codes above 127 would read as negative on every path, a known
// limitation tracked outside this test.
void test_dictionary_unsigned_indices(rmm::device_async_resource_ref mr)
{
  rmm::cuda_stream stream(rmm::cuda_stream::flags::non_blocking);
  std::array const streams{::cuda::stream_ref{stream}};
  constexpr int key_count = 120;
  std::vector<std::string> keys;
  std::vector<std::int32_t> offsets{0};
  std::string chars;
  for (int key = 0; key < key_count; ++key) {
    keys.push_back("k" + std::to_string(1000 + key));
    chars += keys.back();
    offsets.push_back(static_cast<std::int32_t>(chars.size()));
  }
  std::vector<std::uint8_t> codes;
  std::vector<std::string> expected_values;
  for (int row = 0; row < 3 * key_count; ++row) {
    codes.push_back(static_cast<std::uint8_t>((row * 7) % key_count));
    expected_values.push_back(keys[codes.back()]);
  }
  auto const upload = [&](auto const& host, cudf::type_id type) {
    auto column = cudf::make_numeric_column(cudf::data_type{type},
                                            static_cast<cudf::size_type>(host.size()),
                                            cudf::mask_state::UNALLOCATED,
                                            stream.view(),
                                            mr);
    cuda_check(cudaMemcpyAsync(column->mutable_view().head<void>(),
                               host.data(),
                               host.size() * sizeof(host[0]),
                               cudaMemcpyHostToDevice,
                               stream.value()));
    return column;
  };
  auto const channels = [&] {
    std::vector<std::unique_ptr<cudf::column>> result;
    result.push_back(upload(offsets, cudf::type_id::INT32));
    result.push_back(upload(chars, cudf::type_id::UINT8));
    result.push_back(upload(codes, cudf::type_id::UINT8));
    stream.synchronize();
    return result;
  };
  std::vector<std::string> const names{"keys_offsets", "keys_chars", "indices"};
  auto const expected = make_strings_column(expected_values, {}, stream.view());

  std::unique_ptr<cudf::column> decoded;
  {
    simpatico::decode_session session(streams, mr);
    auto& frame        = simpatico::decode_session_test_access::frame(session);
    auto const rebuilt = simpatico::reconstruct_decode_representation(
      dictionary_plan_node(), names, channels(), frame);
    decoded = simpatico::decode_standalone(*rebuilt, frame);
    expect(session.finish().empty(), "unsigned-index fixture published a session result");
  }
  expect(strings_equal_completed(expected->view(), decoded->view(), stream.view()),
         "decode reconstruction changed unsigned dictionary indices");

  std::string error;
  auto const loaded = simpatico::reconstruct_representation(
    "dictionary", names, channels(), stream.view(), mr, &error);
  expect(loaded != nullptr && error.empty(), "loader rejected unsigned dictionary indices");
  auto const loaded_output = decode_completed(*loaded, stream.view(), mr);
  expect(strings_equal_completed(expected->view(), loaded_output->view(), stream.view()),
         "loader changed unsigned dictionary indices");
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
  auto decoded = decode_completed(*dictionary, stream.view(), mr);
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
      auto decoded  = decode_completed(*prepared, stream.view(), mr);
      expect(strings_equal_completed(expected->view(), decoded->view(), stream.view()),
             "large-key dictionary roundtrip mismatch");

      simpatico::dictionary_compressed_representation unknown(
        std::make_unique<cudf::column>(prepared->dict_column->view(), stream.view(), mr));
      auto fallback = decode_completed(unknown, stream.view(), mr);
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
  ::cuda::stream_ref const stream{pool.streams.front()};
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
    gate.expect_completed("dictionary observation failure escaped with pending stream work");
    expect(markers.complete(), "dictionary observation failure escaped with pending stream work");
    resource.check();
    expect(default_resource.observations->attempts.empty(),
           "failed dictionary observation bypassed supplied resource");
    gate.expect_not_timed_out("dictionary observation failure watchdog expired");
  }
  // Unknown metadata uses the same reduction with local device result and scratch storage.
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
      (void)decode_completed(unknown, stream, resource);
    } catch (injected_out_of_memory const& error) {
      injected = error.requested_bytes == resource.observations->failed_request_bytes &&
                 error.requested_bytes >= sizeof(int64_t);
    }
    expect(injected, "unknown dictionary width lost its metadata allocation error");
    gate.expect_completed(
      "unknown dictionary width failure escaped before prior stream work drained");
    expect(markers.complete(),
           "unknown dictionary width failure escaped before prior stream work drained");
    expect(unknown.constant_key_width == -1, "failed dictionary fallback mutated metadata");
    resource.check();
    gate.expect_not_timed_out("unknown dictionary width failure watchdog expired");
  }
  expect(default_resource.observations->attempts.empty(),
         "dictionary failure cleanup bypassed supplied resource");
  default_resource.check();
}

// Publication measures the key width on whichever stream it is given, with one device allocation
// for the result and its scratch and none for an empty key set, and returns only after that stream
// completes.
void test_dictionary_width_across_streams(rmm::device_async_resource_ref upstream)
{
  simpatico::stream_pool pool;
  expect(pool.init(2), "dictionary width stream pool init");
  ::cuda::stream_ref const first{pool.streams.front()};
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
      ::cuda::stream_ref const stream{pool.streams[(repeat + i) % pool.streams.size()]};
      auto const& original =
        dynamic_cast<simpatico::dictionary_compressed_representation const&>(*encoded[i]);
      auto copied = std::make_unique<cudf::column>(original.dict_column->view(), stream, upstream);
      auto const device_attempts = device.observations->attempts.size();
      stream_gate gate(stream);
      gate.release_after_delay.store(true);
      auto prepared = simpatico::dictionary_compressed_representation::from_encoded_column(
        std::move(copied), stream, device);
      gate.expect_completed("dictionary publication returned before its stream completed");
      expect(prepared->constant_key_width == widths[i], "dictionary published a wrong key width");
      expect(device.observations->attempts.size() - device_attempts == (i < 3 ? 1U : 0U),
             "dictionary empty/nonempty measurement allocation count mismatch");
      device.check();
      gate.expect_not_timed_out("dictionary width watchdog expired");
    }
  }
  cuda_check(pool.sync_all());
}

void test_submission_and_kernel_lifetime(rmm::device_async_resource_ref upstream)
{
  decode_fixture fixture(upstream, "input -> bitpack\n", 2, 65549, 59, 2);
  auto& [input, compressed, pool, resource] = fixture;
  auto& cache                               = codegen::jit::KernelCache::instance();
  cache.clear();
  fixture.warm();

  auto shape = codegen::jit::FusedTree::make(codegen::OpKind::Bitpack);
  auto spec =
    codegen::decode::jit::render(*shape, "int32_t", codegen::num_chunks_for(input->num_rows()));
  codegen::jit::CompileOptions options;
  options.arch_cc = codegen::jit::arch_cc_for_current_device();
  auto handle     = cache.get_or_compile_plain(spec.source, spec.entry_symbol, options);
  std::weak_ptr<codegen::jit::CompiledKernel const> retained = handle;
  handle.reset();

  auto streams = pool.refs();
  simpatico::decode_session session(streams, resource);
  event_markers markers(pool);
  // Gate destruction releases on exceptions before the session unwinds.
  stream_gate gate(pool.streams[0]);
  gate.wait_until_entered();
  std::array<std::size_t, 4> const requested{0, 1, 0, 1};
  for (auto index : requested) {
    session.append(value_request(compressed.columns[index]));
    gate.expect_not_released("column submission waited for the gated lane");
  }
  markers.record();
  cache.clear();
  expect(!retained.expired(), "cache clear unloaded a pending kernel");
  gate.expect_not_released("cache clear waited for a pending kernel");
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
  gate.expect_not_timed_out("submission watchdog expired");
}

void test_abandoned_session(rmm::device_async_resource_ref upstream)
{
  decode_fixture fixture(upstream, "input -> bitpack\n", 1, 65549, 67, 1);
  auto& [input, compressed, pool, resource] = fixture;
  fixture.warm();
  auto streams = pool.refs();
  std::optional<simpatico::decode_session> session(std::in_place, streams, resource);
  event_markers markers(pool);
  stream_gate gate(pool.streams[0]);
  session->append(value_request(compressed.columns[0]));
  markers.record();
  release_observation release_guard(resource, gate);
  gate.release_after_delay.store(true);
  session.reset();
  gate.expect_completed("abandoned session returned while stream remained gated");
  expect(markers.complete(), "abandoned session did not drain its stream");
  resource.check();
  gate.expect_not_timed_out("abandon watchdog expired");
}

void test_session_state_contracts(rmm::device_async_resource_ref upstream)
{
  auto input      = make_int32_table(1, 13, 69);
  auto compressed = simpatico::compress_with_plan(
    input->view(), "input -> identity\n", cudf::get_default_stream(), upstream);
  rmm::cuda_stream stream(rmm::cuda_stream::flags::non_blocking);
  std::array const streams{::cuda::stream_ref{stream}};
  simpatico::decode_session empty(streams, upstream);
  expect(empty.finish().empty(), "empty session produced output");
  expect(throws([&] { empty.finish(); }), "second session finish accepted");
  expect(throws([&] { empty.append(value_request(compressed.columns[0])); }),
         "append after finish accepted");

  simpatico::PlanTree malformed;
  simpatico::decode_session failed(streams, upstream);
  failed.append(value_request(compressed.columns[0]));
  expect(throws([&] {
           failed.append(simpatico::column_decode_request{.source = std::cref(malformed)});
         }),
         "malformed request accepted");
  expect(cudaStreamQuery(stream.value()) == cudaSuccess,
         "failed append returned before prior work completed");
  expect(throws([&] { failed.finish(); }), "failed session published partial outputs");
  expect(throws([&] { failed.append(value_request(compressed.columns[0])); }),
         "failed session accepted another request");
}

void test_stream_completion_failures(rmm::device_async_resource_ref upstream)
{
  decode_fixture fixture(upstream, "input -> identity\n", 1, 1037, 83, 3);
  auto& [input, compressed, pool, resource] = fixture;
  // The third supplied handle receives no request, so completion leaves its work alone.
  std::array const streams{::cuda::stream_ref{pool.streams[0]},
                           ::cuda::stream_ref{pool.streams[1]},
                           ::cuda::stream_ref{pool.streams[0]},
                           ::cuda::stream_ref{pool.streams[1]},
                           ::cuda::stream_ref{pool.streams[2]}};
  for (auto operation : {stream_fault::operation::query, stream_fault::operation::synchronize}) {
    event_markers markers({pool.streams[0], pool.streams[1]});
    stream_gate first_gate(pool.streams[0]);
    stream_gate second_gate(pool.streams[1]);
    stream_gate external_gate(pool.streams[2]);
    std::optional<simpatico::decode_session> session(std::in_place, streams, resource);
    for (std::size_t i = 0; i < 2; ++i)
      session->append(value_request(compressed.columns[0]));
    markers.record();
    release_observation observation(resource, first_gate, &second_gate);
    first_gate.release_after_delay.store(true);
    second_gate.release_after_delay.store(true);
    external_gate.release_after_delay.store(true);
    bool propagated = false;
    {
      stream_observation_scope observed;
      stream_failure_scope fault(operation, pool.streams[0]);
      try {
        (void)session->finish();
      } catch (std::runtime_error const& error) {
        propagated =
          std::string_view(error.what()).ends_with(cudaGetErrorString(cudaErrorInvalidValue));
      }
      expect(stream_fault::next == stream_fault::operation::none,
             "stream failure injection was not consumed");
      for (std::size_t lane = 0; lane < 2; ++lane)
        expect(observed.count(pool.streams[lane]).queries >= 1,
               "stream failure cleanup skipped a lane that received a request");
      auto const unused = observed.count(pool.streams[2]);
      expect(unused.queries == 0 && unused.synchronizations == 0,
             "completion waited for a supplied stream that received no request");
    }
    expect(propagated, "stream completion failure lost the original CUDA error");
    first_gate.expect_completed("stream failure escaped before the first lane completed");
    second_gate.expect_completed("stream failure escaped before the second lane completed");
    expect(markers.complete(), "stream failure escaped before its request lanes completed");
    first_gate.expect_if_armed(!resource.observations->early_release,
                               "stream failure released an output before all lanes completed");
    expect(throws([&] { session->finish(); }), "failed session published partial output");
    expect(throws([&] { session->append(value_request(compressed.columns[0])); }),
           "stream failure left a reusable session");
    session.reset();
    resource.check();
    first_gate.expect_not_timed_out("stream failure watchdog expired");
    second_gate.expect_not_timed_out("stream failure watchdog expired");
    external_gate.expect_not_timed_out("stream failure watchdog expired");
    {
      auto recovered = simpatico::decompress(compressed, pool, resource);
      expect(columns_equal_completed(input->view().column(0), recovered->view().column(0)),
             "pool was not reusable after recoverable stream API failure");
    }
    resource.check();
  }
}

class probe_failure final : public std::runtime_error {
 public:
  probe_failure() : std::runtime_error("injected probe failure after enqueue") {}
};

// A one-byte output derived from device scratch that is released when decompress() returns, while
// the work reading it may still be queued.
class scratch_probe_representation final : public simpatico::standalone_compressed_representation {
 public:
  explicit scratch_probe_representation(std::size_t scratch_bytes = 16,
                                        std::uint8_t value        = 0x2a,
                                        bool fail                 = false)
    : standalone_compressed_representation(cudf::data_type{cudf::type_id::UINT8}, 1),
      scratch_bytes_(scratch_bytes),
      value_(value),
      fail_(fail)
  {
  }
  [[nodiscard]] std::unique_ptr<cudf::column> decompress(
    simpatico::decode_frame& frame) const override
  {
    auto output = cudf::make_numeric_column(cudf::data_type{cudf::type_id::UINT8},
                                            1,
                                            cudf::mask_state::UNALLOCATED,
                                            frame.stream(),
                                            frame.mr());
    rmm::device_buffer scratch(scratch_bytes_, frame.stream(), frame.mr());
    cuda_check(cudaMemsetAsync(scratch.data(), value_, scratch.size(), frame.stream().get()));
    if (fail_) throw probe_failure{};
    cuda_check(cudaMemcpyAsync(output->mutable_view().head<void>(),
                               scratch.data(),
                               1,
                               cudaMemcpyDeviceToDevice,
                               frame.stream().get()));
    return output;
  }

 private:
  std::size_t scratch_bytes_;
  std::uint8_t value_;
  bool fail_;
};

simpatico::column_decode_request probe_request(scratch_probe_representation const& representation)
{
  return {.source = std::cref(
            static_cast<simpatico::standalone_compressed_representation const&>(representation))};
}

void expect_probe_outputs(std::vector<std::unique_ptr<cudf::column>> const& outputs,
                          std::vector<std::uint8_t> const& expected)
{
  expect(outputs.size() == expected.size(), "scratch probe output count mismatch");
  rmm::cuda_stream consumer(rmm::cuda_stream::flags::non_blocking);
  for (std::size_t i = 0; i < outputs.size(); ++i) {
    expect(outputs[i]->type().id() == cudf::type_id::UINT8 && outputs[i]->size() == 1,
           "scratch probe output shape mismatch");
    std::uint8_t byte = 0;
    cuda_check(cudaMemcpyAsync(
      &byte, outputs[i]->view().head<void>(), 1, cudaMemcpyDeviceToHost, consumer.value()));
    consumer.synchronize();
    expect(byte == expected[i], "scratch probe output lost its queued value");
  }
}

void test_duplicate_stream_handles(rmm::device_async_resource_ref upstream)
{
  for (bool aliases : {false, true}) {
    simpatico::stream_pool pool;
    expect(pool.init(2), "duplicate-handle pool init");
    std::vector<::cuda::stream_ref> streams{pool.streams[0], pool.streams[1]};
    if (aliases) streams.insert(streams.begin(), pool.streams[0]);
    auto const blocked = pool.streams[aliases ? 1 : 0];
    auto const ready   = pool.streams[aliases ? 0 : 1];
    checked_resource resource(upstream);
    scratch_probe_representation const representation;
    auto const request = probe_request(representation);
    simpatico::decode_session session(streams, resource);
    stream_gate blocked_gate(blocked);
    for (std::size_t i = 0; i < 64; ++i)
      session.append(request);
    blocked_gate.expect_not_released("submission waited for a gated lane");
    // Each request's scratch is released during its append, on its allocation stream.
    expect(
      resource.observations->attempts.size() == 128 && resource.observations->live.size() == 64,
      "scratch probe did not allocate one output and one scratch buffer, or kept its scratch");
    expect(!resource.observations->wrong_stream, "scratch was released on another stream");
    for (std::size_t i = 0; i < 64; ++i) {
      auto const assigned = streams[i % streams.size()].get();
      expect(resource.observations->attempts[2 * i] == assigned &&
               resource.observations->attempts[2 * i + 1] == assigned,
             "duplicate handles changed round-robin weighting");
    }

    // The ready handle's requests have completed. Work queued on it afterwards must still be
    // included in final completion.
    cuda_check(__real_cudaStreamSynchronize(ready));
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
    blocked_gate.expect_completed("finish skipped the blocked lane");
    ready_tail.expect_completed("finish skipped an external tail on a completed stream");
    expect(markers.complete(), "finish skipped an external tail on a completed stream");
    expect_probe_outputs(output, std::vector<std::uint8_t>(64, 0x2a));
    output.clear();
    resource.check();
    blocked_gate.expect_not_timed_out("duplicate-handle watchdog expired");
    ready_tail.expect_not_timed_out("duplicate-handle watchdog expired");
  }
}

void test_external_phase_tail(rmm::device_async_resource_ref upstream)
{
  decode_fixture fixture(upstream, "input -> identity\n", 1, 1037, 89, 1);
  auto& [input, compressed, pool, resource] = fixture;
  auto streams                              = pool.refs();
  simpatico::decode_session session(streams, resource);
  session.append(value_request(compressed.columns[0]));
  // The request's own work is complete before this external tail starts. Final completion must
  // observe the current supplied stream, not a host observation made during submission.
  stream_gate gate(pool.streams[0]);
  gate.wait_until_entered();
  gate.expect_not_timed_out("external tail gate did not start");
  event_markers markers(pool);
  markers.record();
  gate.release_after_delay.store(true);
  auto output = session.finish();
  gate.expect_completed("finish missed external phase work");
  expect(markers.complete(), "finish missed external phase work");
  expect(columns_equal_completed(input->view().column(0), output.front()->view()),
         "external-tail output mismatch");
  output.clear();
  resource.check();
  gate.expect_not_timed_out("external tail watchdog expired");
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
                                           ::cuda::stream_ref,
                                           rmm::device_async_resource_ref) const
  {
    throw std::logic_error("copy-failure probe must never execute");
  }
};

void test_request_copy_failure(rmm::device_async_resource_ref upstream)
{
  decode_fixture fixture(upstream, "input -> identity\n", 1, 1037, 97, 1);
  auto& [input, compressed, pool, resource] = fixture;
  auto streams                              = pool.refs();
  auto fail                                 = std::make_shared<bool>(false);
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
  release_observation observation(resource, gate);
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
  gate.expect_completed("host bookkeeping failure escaped before prior work completed");
  expect(markers.complete(), "host bookkeeping failure escaped before prior work completed");
  expect(throws([&] { session->finish(); }), "request copy failure published partial results");
  session.reset();
  resource.check();
  gate.expect_not_timed_out("request copy failure watchdog expired");
}

void test_host_upload_growth(rmm::device_async_resource_ref upstream)
{
  rmm::cuda_stream stream(rmm::cuda_stream::flags::non_blocking);
  std::array const streams{::cuda::stream_ref{stream}};
  constexpr std::size_t uploads = 97;
  rmm::device_buffer device(uploads * 2 * sizeof(std::uint64_t), stream.view(), upstream);
  {
    simpatico::decode_session session(streams, upstream);
    auto& frame = simpatico::decode_session_test_access::frame(session);
    auto first  = frame.host_array<std::uint64_t>(2);
    expect(reinterpret_cast<std::uintptr_t>(first.data()) % alignof(std::max_align_t) == 0,
           "upload storage is not fundamentally aligned");
    first.front()      = 0x12345678;
    first.back()       = 0xabcdef;
    auto* const origin = first.data();
    std::vector<std::span<std::uint64_t>> spans{first};
    // Keep earlier spans while the frame's upload records grow and reallocate.
    for (std::size_t i = 1; i < uploads; ++i) {
      auto extra    = frame.host_array<std::uint64_t>(2);
      extra.front() = i;
      extra.back()  = ~i;
      spans.push_back(extra);
    }
    expect(first.data() == origin && first.front() == 0x12345678 && first.back() == 0xabcdef,
           "upload growth invalidated an upload span");
    auto* const destination = static_cast<std::uint64_t*>(device.data());
    for (std::size_t i = 0; i < uploads; ++i) {
      cuda_check(cudaMemcpyAsync(destination + 2 * i,
                                 spans[i].data(),
                                 spans[i].size_bytes(),
                                 cudaMemcpyHostToDevice,
                                 stream.value()));
    }
    expect(session.finish().empty(), "raw upload fixture published an output");
  }
  std::vector<std::uint64_t> host(uploads * 2);
  cuda_check(cudaMemcpyAsync(
    host.data(), device.data(), device.size(), cudaMemcpyDeviceToHost, stream.value()));
  stream.synchronize();
  bool equal = host[0] == 0x12345678 && host[1] == 0xabcdef;
  for (std::size_t i = 1; i < uploads; ++i)
    equal = equal && host[2 * i] == i && host[2 * i + 1] == ~i;
  expect(equal, "grown upload storage lost queued data");
}

// An asynchronous copy from pageable memory may read its source after the call returns, so upload
// storage must outlive its copy. The first part is a smoke test of the frame's storage: it detects
// only storage released before the session drains, and only where the driver defers pageable
// host-to-device copies; a driver that stages the copy before returning makes it pass trivially.
// The second part checks that a real leaf (nvCOMP) keeps its uploaded chunk tables in the frame
// until the drain. Whether every leaf uploads only from frame storage or the request copy is
// enforced by review, not by this test.
void test_host_uploads_survive_until_completion(rmm::device_async_resource_ref upstream)
{
  rmm::cuda_stream stream(rmm::cuda_stream::flags::non_blocking);
  std::array const streams{::cuda::stream_ref{stream}};
  // Below the default mmap threshold, so freed storage returns to the heap the churn draws from.
  constexpr std::size_t words = 4096;
  auto const pattern          = [](std::size_t i) { return 0x9e3779b97f4a7c15ULL * (i + 1); };
  rmm::device_buffer device(words * sizeof(std::uint64_t), stream.view(), upstream);
  cuda_check(cudaMemsetAsync(device.data(), 0, device.size(), stream.value()));
  stream.synchronize();
  {
    simpatico::decode_session session(streams, upstream);
    auto& frame = simpatico::decode_session_test_access::frame(session);
    stream_gate gate(stream.view());
    // A copy that waits for the stream before returning must not outlive the watchdog.
    gate.release_after_delay.store(true);
    auto upload = frame.host_array<std::uint64_t>(words);
    for (std::size_t i = 0; i < words; ++i)
      upload[i] = pattern(i);
    cuda_check(cudaMemcpyAsync(
      device.data(), upload.data(), upload.size_bytes(), cudaMemcpyHostToDevice, stream.value()));
    std::vector<std::unique_ptr<std::uint64_t[]>> churn;
    for (int block = 0; block < 64; ++block) {
      churn.push_back(std::make_unique_for_overwrite<std::uint64_t[]>(words));
      std::fill_n(churn.back().get(), words, ~std::uint64_t{0});
    }
    expect(session.finish().empty(), "raw upload fixture published an output");
    gate.expect_completed("upload gate did not release before the session drained");
    gate.expect_not_timed_out("upload watchdog expired");
  }
  std::vector<std::uint64_t> host(words);
  cuda_check(cudaMemcpyAsync(
    host.data(), device.data(), device.size(), cudaMemcpyDeviceToHost, stream.value()));
  stream.synchronize();
  bool equal = true;
  for (std::size_t i = 0; i < words; ++i)
    equal = equal && host[i] == pattern(i);
  expect(equal, "host upload storage was reused before its copy completed");

  auto const input      = make_int32_table(1, 65549, 127);
  auto const compressed = simpatico::compress_with_plan(
    input->view(), "input -> ans\n", cudf::get_default_stream(), upstream);
  cudf::get_default_stream().synchronize();
  std::vector<std::unique_ptr<cudf::column>> outputs;
  {
    simpatico::decode_session session(streams, upstream);
    session.append(value_request(compressed.columns[0]));
    // One block holds the compressed sizes, compressed and output pointers, and output sizes.
    expect(simpatico::decode_session_test_access::host_uploads(session) == 1,
           "an nvCOMP leaf did not keep its chunk tables until the session drains");
    outputs = session.finish();
    expect(simpatico::decode_session_test_access::host_uploads(session) == 0,
           "a finished session kept host uploads");
  }
  expect(columns_equal_completed(input->view().column(0), outputs.front()->view()),
         "nvCOMP decode with retained uploads mismatch");
}

// Host observations stage through the thread's pinned slab below its cap and copy straight into
// the caller's storage above it; either way the bytes are exact and the frame's stream has
// completed before the call returns.
void test_host_observation_staging(rmm::device_async_resource_ref upstream)
{
  simpatico::stream_pool pool;
  expect(pool.init(1), "observation pool init");
  checked_resource resource(upstream);
  auto const streams = pool.refs();
  {
    simpatico::decode_session session(streams, resource);
    auto& frame                 = simpatico::decode_session_test_access::frame(session);
    auto const uploads_before   = simpatico::decode_session_test_access::host_uploads(session);
    constexpr std::size_t cap   = simpatico::pinned_staging_cap_bytes;
    constexpr std::size_t total = cap + 4096;
    std::vector<std::uint8_t> source(total);
    for (std::size_t i = 0; i < total; ++i)
      source[i] = static_cast<std::uint8_t>((i * 7 + 3) & 0xff);
    rmm::device_buffer device(total, frame.stream(), resource);
    cuda_check(cudaMemcpyAsync(
      device.data(), source.data(), total, cudaMemcpyHostToDevice, frame.stream().get()));
    auto const* base = static_cast<std::uint8_t const*>(device.data());

    // Sizes below the initial slab, across its growth, exactly at the cap, and above it.
    for (std::size_t const bytes : {std::size_t{1},
                                    std::size_t{16},
                                    std::size_t{4096},
                                    (std::size_t{64} << 10) + 1,
                                    std::size_t{1} << 20,
                                    cap,
                                    cap + 1,
                                    total}) {
      std::size_t const offset = total - bytes;
      std::vector<std::uint8_t> destination(bytes, 0xee);
      frame.read_bytes(destination.data(), base + offset, bytes);
      expect(std::equal(destination.begin(),
                        destination.end(),
                        source.begin() + static_cast<std::ptrdiff_t>(offset)),
             "host observation bytes differ from the device source");
    }
    // A staged read still waits for the frame's stream tail before returning.
    {
      stream_gate gate(pool.streams[0]);
      gate.release_after_delay.store(true);
      std::uint8_t byte = 0;
      frame.read_bytes(&byte, base + 5, 1);
      gate.expect_completed("host observation returned before its stream completed");
      expect(byte == source[5], "gated host observation byte mismatch");
      gate.expect_not_timed_out("observation watchdog expired");
    }
    std::int64_t const scalar_value = -0x1122334455667788LL;
    rmm::device_buffer scalar_storage(sizeof scalar_value, frame.stream(), resource);
    cuda_check(cudaMemcpyAsync(scalar_storage.data(),
                               &scalar_value,
                               sizeof scalar_value,
                               cudaMemcpyHostToDevice,
                               frame.stream().get()));
    expect(
      frame.read_scalar(static_cast<std::int64_t const*>(scalar_storage.data())) == scalar_value,
      "scalar observation mismatch");
    expect(frame.read_scalar(base + total - 1) == source[total - 1],
           "byte scalar observation mismatch");
    expect(simpatico::decode_session_test_access::host_uploads(session) == uploads_before,
           "host observations changed the frame's retained uploads");
    expect(session.finish().empty(), "observation fixture published a result");
  }
  resource.check();
}

// Device temporaries are released during the append that allocates them, so the call-time ledger
// holds only prior outputs plus the active request's working set. Library temporaries count too,
// because the ledger is also the current device resource.
void test_temporaries_released_at_submission(rmm::device_async_resource_ref upstream)
{
  struct fixture {
    char const* plan;
    bool strings;
    bool observes_host;  ///< Needs a host readback, so its stream cannot stay gated.
  };
  std::array<fixture, 6> const fixtures{{
    {"input -> bitpack\n", false, false},
    {"input -> rle -> values, runs\nrle.values -> bitpack\nrle.runs -> bitpack\n", false, false},
    {"input -> for -> deltas, references\nfor.deltas -> bitpack\nfor.references -> bitpack\n",
     false,
     false},
    {"input -> dictionary -> keys_offsets, keys_chars, indices\ndictionary.indices -> bitpack\n",
     true,
     true},
    {"input -> str_split -> offsets, chars\nstr_split.offsets -> delta -> differences\n"
     "str_split.offsets.differences -> bitpack\n",
     true,
     true},
    // ANS, like LZ4, exercises nvCOMP's metadata and scratch uploads; nvCOMP's LZ4 path also tries
    // a hardware decompression call that compute-sanitizer reports independently of this code.
    {"input -> ans\n", false, true},
  }};
  constexpr int rows = 65549;
  auto const numbers = make_int32_table(1, rows, 113);
  auto const words   = make_string_table(rows, cudf::get_default_stream());
  std::vector<simpatico::compressed_column> columns;
  for (auto const& fixture : fixtures) {
    auto const& input = fixture.strings ? *words : *numbers;
    auto compressed   = simpatico::compress_with_plan(
      input.view(), fixture.plan, cudf::get_default_stream(), upstream);
    columns.push_back(std::move(compressed.columns.front()));
  }
  cudf::get_default_stream().synchronize();
  auto const expected = [&](std::size_t i) {
    return fixtures[i].strings ? words->view().column(0) : numbers->view().column(0);
  };
  auto const verify = [&](std::vector<std::size_t> const& order,
                          std::vector<std::unique_ptr<cudf::column>> const& outputs) {
    expect(outputs.size() == order.size(), "temporary fixture lost an output");
    for (std::size_t i = 0; i < order.size(); ++i) {
      expect(fixtures[order[i]].strings
               ? strings_equal_completed(
                   expected(order[i]), outputs[i]->view(), cudf::get_default_stream())
               : columns_equal_completed(expected(order[i]), outputs[i]->view()),
             "temporary fixture output mismatch");
    }
  };

  simpatico::stream_pool pool;
  expect(pool.init(2), "temporary fixture pool init");
  auto const streams = pool.refs();
  checked_resource ledger(upstream);
  current_resource_guard current(ledger);

  // Each request's ledger peak and output bytes when it is decoded alone.
  std::vector<std::size_t> solo_peak(fixtures.size());
  std::vector<std::size_t> output_bytes(fixtures.size());
  for (std::size_t i = 0; i < fixtures.size(); ++i) {
    std::vector<std::unique_ptr<cudf::column>> outputs;
    {
      simpatico::decode_session session(streams, ledger);
      session.append(value_request(columns[i]));
      outputs = session.finish();
    }
    solo_peak[i]    = ledger.observations->peak_live_bytes;
    output_bytes[i] = ledger.observations->live_bytes;
    expect(output_bytes[i] > 0 && solo_peak[i] > output_bytes[i],
           "temporary fixture has no decode temporaries");
    outputs.clear();
    ledger.check();
    ledger.reset();
  }

  // Every request in one session: after each append only the outputs so far remain allocated.
  std::vector<std::size_t> all(fixtures.size());
  std::iota(all.begin(), all.end(), std::size_t{0});
  std::size_t prior_outputs = 0;
  std::size_t bound         = 0;
  {
    simpatico::decode_session session(streams, ledger);
    for (auto const i : all) {
      bound = std::max(bound, prior_outputs + solo_peak[i]);
      session.append(value_request(columns[i]));
      prior_outputs += output_bytes[i];
      expect(ledger.observations->live_bytes == prior_outputs,
             "an append kept device temporaries beyond its call");
    }
    expect(ledger.observations->peak_live_bytes <= bound,
           "session peak exceeded prior outputs plus one request's working set");
    expect(!ledger.observations->wrong_stream, "a temporary was released on another stream");
    verify(all, session.finish());
  }
  ledger.check();
  ledger.reset();

  // Requests without host readbacks never wait: temporaries are released while every lane is
  // gated, so the work that reads them is still queued.
  std::vector<std::size_t> gated_order;
  for (std::size_t repeat = 0; repeat < 2; ++repeat)
    for (auto const i : all)
      if (!fixtures[i].observes_host) gated_order.push_back(i);
  {
    simpatico::decode_session session(streams, ledger);
    stream_gate first_gate(pool.streams[0]);
    stream_gate second_gate(pool.streams[1]);
    prior_outputs = 0;
    {
      stream_observation_scope observed;
      for (auto const i : gated_order) {
        session.append(value_request(columns[i]));
        prior_outputs += output_bytes[i];
        expect(ledger.observations->live_bytes == prior_outputs,
               "a gated append kept device temporaries beyond its call");
      }
      for (auto const& calls : stream_fault::counts)
        expect(calls.queries == 0 && calls.synchronizations == 0,
               "an append without host readbacks queried or synchronized a stream");
    }
    first_gate.expect_not_released("an append without host readbacks waited for its lane");
    second_gate.expect_not_released("an append without host readbacks waited for its lane");
    expect(!ledger.observations->wrong_stream, "a gated temporary was released on another stream");
    first_gate.release_after_delay.store(true);
    second_gate.release_after_delay.store(true);
    verify(gated_order, session.finish());
    first_gate.expect_not_timed_out("gated temporary watchdog expired");
    second_gate.expect_not_timed_out("gated temporary watchdog expired");
  }
  ledger.check();
}

// Submission never waits to release memory: 256 requests with 32 MiB of scratch each queue behind
// one gated lane, while the call-time ledger never holds more than one request's scratch. Each
// request writes its own byte, so a scratch block reused out of stream order would show up in a
// neighbour's output.
void test_submission_never_waits_for_memory(rmm::device_async_resource_ref upstream)
{
  constexpr std::size_t requests      = 256;
  constexpr std::size_t scratch_bytes = 32U << 20;
  rmm::cuda_stream stream(rmm::cuda_stream::flags::non_blocking);
  std::array const streams{::cuda::stream_ref{stream}};
  checked_resource resource(upstream);
  std::vector<std::unique_ptr<scratch_probe_representation>> representations;
  std::vector<std::uint8_t> expected;
  for (std::size_t i = 0; i < requests; ++i) {
    expected.push_back(static_cast<std::uint8_t>(i));
    representations.push_back(
      std::make_unique<scratch_probe_representation>(scratch_bytes, expected.back()));
  }
  simpatico::decode_session session(streams, resource);
  stream_gate gate(stream.view());
  {
    stream_observation_scope observed;
    for (auto const& representation : representations)
      session.append(probe_request(*representation));
    for (auto const& calls : stream_fault::counts)
      expect(calls.queries == 0 && calls.synchronizations == 0,
             "submission queried or synchronized a stream to release memory");
  }
  gate.expect_not_released("submission waited for its gated lane");
  expect(resource.observations->live.size() == requests,
         "submission kept scratch beyond the appending call");
  expect(resource.observations->peak_live_bytes <= requests + scratch_bytes,
         "submission held more than one request's scratch at a time");
  gate.release_after_delay.store(true);
  expect_probe_outputs(session.finish(), expected);
  gate.expect_not_timed_out("gated scratch probe watchdog expired");
}

// A failure after work was queued releases the request's temporaries during unwinding, on their
// allocation stream and before the stream completes; the session then drains before rethrowing.
void test_failure_after_enqueue_unwinds_stream_ordered(rmm::device_async_resource_ref upstream)
{
  simpatico::stream_pool pool;
  expect(pool.init(1), "failure-unwind pool init");
  auto const streams = pool.refs();
  checked_resource resource(upstream);
  scratch_probe_representation const representation(1U << 20, 0x2a, true);
  std::optional<simpatico::decode_session> session(std::in_place, streams, resource);
  stream_gate gate(pool.streams[0]);
  event_markers markers(pool);
  markers.record();
  release_observation observation(resource, gate);
  // The first release opens the gate after a delay, so every release precedes GPU completion.
  resource.observations->signal_on_deallocate = &gate.release_after_delay;
  bool propagated                             = false;
  try {
    session->append(probe_request(representation));
  } catch (probe_failure const&) {
    propagated = true;
  }
  resource.observations->signal_on_deallocate = nullptr;
  expect(propagated, "failure after enqueue lost its exception subtype");
  gate.expect_completed("failure after enqueue escaped before its lane drained");
  expect(markers.complete(), "failure after enqueue escaped before its lane drained");
  gate.expect_if_armed(resource.observations->early_release,
                       "failure after enqueue did not release temporaries before GPU completion");
  expect(resource.observations->live.empty(),
         "failure after enqueue kept device storage past the throwing append");
  expect(!resource.observations->wrong_stream && !resource.observations->wrong_thread,
         "failure after enqueue released a temporary on another stream or thread");
  expect(throws([&] { session->finish(); }), "failed session published a result");
  expect(throws([&] { session->append(probe_request(representation)); }),
         "failed session accepted another request");
  session.reset();
  expect(resource.observations->live.empty(), "failed session leaked device storage");
  gate.expect_not_timed_out("failure-unwind watchdog expired");
}

// A selection over `rows` rows, built on the host and uploaded, so no CNT wave runs.
class host_selection {
 public:
  host_selection(std::int64_t rows,
                 std::function<bool(std::int64_t)> const& keep,
                 ::cuda::stream_ref stream,
                 rmm::device_async_resource_ref mr)
  {
    using sirius::codegen::selection_mask;
    std::vector<std::uint32_t> words(static_cast<std::size_t>(selection_mask::WordsFor(rows)));
    std::vector<std::uint32_t> offsets(static_cast<std::size_t>(selection_mask::ChunksFor(rows)) +
                                       1);
    for (std::int64_t row = 0; row < rows; ++row) {
      if (!keep(row)) continue;
      words[static_cast<std::size_t>(row / 32)] |= 1U << (row % 32);
      ++offsets[static_cast<std::size_t>(row / sirius::codegen::SELECTION_CHUNK_ROWS) + 1];
      survivors.push_back(static_cast<std::int32_t>(row));
    }
    std::partial_sum(offsets.begin(), offsets.end(), offsets.begin());
    words_   = rmm::device_buffer(words.data(), words.size() * sizeof(words[0]), stream, mr);
    offsets_ = rmm::device_buffer(offsets.data(), offsets.size() * sizeof(offsets[0]), stream, mr);
    indices_ =
      rmm::device_buffer(survivors.data(), survivors.size() * sizeof(survivors[0]), stream, mr);
    stream.sync();
    mask_ = {static_cast<std::uint32_t*>(words_.data()),
             rows,
             static_cast<std::int64_t>(survivors.size()),
             static_cast<std::uint32_t*>(offsets_.data())};
  }
  host_selection(host_selection const&)            = delete;
  host_selection& operator=(host_selection const&) = delete;

  [[nodiscard]] simpatico::decode_selection selection(sirius::codegen::decode_route route) const
  {
    simpatico::decode_selection result;
    result.mask             = &mask_;
    result.survivor_count   = mask_.survivor_count;
    result.route            = route;
    result.survivor_indices = cudf::column_view{cudf::data_type{cudf::type_id::INT32},
                                                static_cast<cudf::size_type>(survivors.size()),
                                                indices_.data(),
                                                nullptr,
                                                0};
    return result;
  }

  std::vector<std::int32_t> survivors;

 private:
  rmm::device_buffer words_;
  rmm::device_buffer offsets_;
  rmm::device_buffer indices_;
  sirius::codegen::selection_mask mask_;
};

// Out-of-range arguments are rejected on the host, before any allocation or device work.
void test_constant_width_offsets_bounds(rmm::device_async_resource_ref upstream)
{
  constexpr auto max_rows = std::numeric_limits<cudf::size_type>::max();
  expect(throws<std::invalid_argument>([&] {
           (void)simpatico::make_constant_width_offsets(
             -1, 4, cudf::get_default_stream(), upstream);
         }),
         "negative row count accepted");
  expect(throws<std::invalid_argument>([&] {
           (void)simpatico::make_constant_width_offsets(
             4, -1, cudf::get_default_stream(), upstream);
         }),
         "negative width accepted");
  expect(throws<std::overflow_error>([&] {
           (void)simpatico::make_constant_width_offsets(
             max_rows / 2 + 1, 2, cudf::get_default_stream(), upstream);
         }),
         "total bytes beyond INT32 accepted");
  expect(throws<std::overflow_error>([&] {
           (void)simpatico::make_constant_width_offsets(
             max_rows, 0, cudf::get_default_stream(), upstream);
         }),
         "offset count beyond INT32 accepted");
}

// The routes whose temporaries changed most, each submitted through sessions: predicates on a
// stored dictionary, a rebuilt dictionary, and a generic (str_split) plan; the dictionary gather
// specialization and its fallback; selected str_split; full-width decode plus gather; and
// membership and range masks. After each append only earlier outputs remain charged; values must
// match. Use after release on these paths is visible only to tools such as compute-sanitizer, not
// to this test.
void test_selection_and_predicate_routes(rmm::device_async_resource_ref upstream)
{
  namespace sc                    = sirius::codegen;
  constexpr cudf::size_type rows  = 5000;
  ::cuda::stream_ref const stream = cudf::get_default_stream();
  std::array<char const*, 10> const words{
    "apple", "banana", "cherry", "apple", "date", "banana", "apple", "elderberry", "fig", "banana"};
  std::vector<std::string> varied(rows);
  std::vector<std::string> fixed(rows);
  for (cudf::size_type row = 0; row < rows; ++row) {
    varied[row] = words[row % words.size()];
    fixed[row]  = std::string{"k"} + static_cast<char>('a' + row % 7);
  }
  auto const varied_input = make_strings_table(varied, {}, stream);
  auto const fixed_input  = make_strings_table(fixed, {}, stream);
  auto const numbers      = make_int32_table(1, rows, 131);
  std::vector<std::int32_t> values(rows);
  cuda_check(cudaMemcpy(values.data(),
                        numbers->view().column(0).head<void>(),
                        values.size() * sizeof(values[0]),
                        cudaMemcpyDeviceToHost));

  auto const compress = [&](cudf::table const& input, char const* plan) {
    auto compressed = simpatico::compress_with_plan(input.view(), plan, stream, upstream);
    return std::move(compressed.columns.front());
  };
  char const* const rebuilt_dictionary =
    "input -> dictionary -> keys_offsets, keys_chars, indices\ndictionary.indices -> bitpack\n";
  auto const stored_dict = compress(*varied_input, "input -> dictionary\n");
  auto const varied_dict = compress(*varied_input, rebuilt_dictionary);
  auto const fixed_dict  = compress(*fixed_input, rebuilt_dictionary);
  auto const split =
    compress(*varied_input, "input -> str_split -> offsets, chars\nstr_split.offsets -> bitpack\n");
  auto const packed = compress(*numbers, "input -> bitpack\n");
  // Chunk 2 keeps no rows.
  host_selection const selected(
    rows, [](std::int64_t row) { return row / 1024 != 2 && row % 3 == 0; }, stream, upstream);

  auto const mask_words = static_cast<std::size_t>(sc::selection_mask::WordsFor(rows));
  std::array<rmm::device_buffer, 3> destinations{
    rmm::device_buffer(mask_words * sizeof(std::uint32_t), stream, upstream),
    rmm::device_buffer(mask_words * sizeof(std::uint32_t), stream, upstream),
    rmm::device_buffer(mask_words * sizeof(std::uint32_t), stream, upstream)};
  auto const destination = [&](std::size_t index) {
    return simpatico::mask_destination{static_cast<std::uint32_t*>(destinations[index].data()),
                                       rows};
  };
  // The membership probe compares with a device column, so the probe itself uploads nothing.
  auto const threshold = values[rows / 2];
  std::vector<std::int32_t> const thresholds(rows, threshold);
  auto const threshold_column = cudf::make_numeric_column(
    cudf::data_type{cudf::type_id::INT32}, rows, cudf::mask_state::UNALLOCATED, stream, upstream);
  cuda_check(cudaMemcpy(threshold_column->mutable_view().head<void>(),
                        thresholds.data(),
                        thresholds.size() * sizeof(thresholds[0]),
                        cudaMemcpyHostToDevice));
  auto const threshold_view = threshold_column->view();
  simpatico::membership_source const below_threshold{
    [threshold_view](
      cudf::column_view keys, ::cuda::stream_ref probe_stream, rmm::device_async_resource_ref mr) {
      return cudf::binary_operation(keys,
                                    threshold_view,
                                    cudf::binary_operator::LESS,
                                    cudf::data_type{cudf::type_id::BOOL8},
                                    probe_stream,
                                    mr);
    },
    cudf::data_type{cudf::type_id::INT32}};
  sc::range_predicate const range{threshold - 200, threshold + 200};
  for (auto& buffer : destinations)
    cuda_check(cudaMemsetAsync(buffer.data(), 0, buffer.size(), stream.get()));
  stream.sync();

  auto const plan = [](simpatico::compressed_column const& column) -> simpatico::PlanTree const& {
    return *column.plan_tree;
  };
  auto const predicate = [&](simpatico::compressed_column const& column,
                             std::vector<std::string> equals_any,
                             std::optional<simpatico::mask_destination> ballot) {
    return simpatico::column_decode_request{
      .source = std::cref(plan(column)),
      .result =
        simpatico::predicate_result{simpatico::decode_predicate{std::move(equals_any)}, ballot}};
  };
  auto const select = [&](simpatico::compressed_column const& column, sc::decode_route route) {
    return simpatico::column_decode_request{
      .source    = std::cref(plan(column)),
      .selection = simpatico::validated_selection(plan(column), selected.selection(route))};
  };
  using request = std::variant<simpatico::column_decode_request, simpatico::mask_decode_request>;
  std::vector<request> const requests{
    predicate(stored_dict, {"apple", "fig"}, destination(0)),
    predicate(varied_dict, {"banana"}, std::nullopt),
    predicate(split, {"cherry", "date"}, std::nullopt),
    select(fixed_dict, sc::decode_route::dict_codes),
    select(varied_dict, sc::decode_route::dict_codes),
    select(split, sc::decode_route::str_split),
    select(packed, sc::decode_route::full),
    simpatico::mask_decode_request{plan(packed), below_threshold, destination(1)},
    simpatico::mask_decode_request{plan(packed), range, destination(2)},
  };
  auto const append = [](simpatico::decode_session& session, request const& item) {
    std::visit([&](auto const& typed) { (void)session.append(typed); }, item);
  };

  simpatico::stream_pool pool;
  expect(pool.init(2), "route coverage pool init");
  auto const streams = pool.refs();
  checked_resource ledger(upstream);
  current_resource_guard current(ledger);
  std::vector<std::size_t> solo_peak(requests.size());
  std::vector<std::size_t> output_bytes(requests.size());
  for (std::size_t i = 0; i < requests.size(); ++i) {
    std::vector<std::unique_ptr<cudf::column>> outputs;
    {
      simpatico::decode_session session(streams, ledger);
      append(session, requests[i]);
      outputs = session.finish();
    }
    solo_peak[i]    = ledger.observations->peak_live_bytes;
    output_bytes[i] = ledger.observations->live_bytes;
    outputs.clear();
    ledger.check();
    ledger.reset();
  }

  std::vector<std::unique_ptr<cudf::column>> outputs;
  {
    simpatico::decode_session session(streams, ledger);
    std::size_t prior_outputs = 0;
    std::size_t bound         = 0;
    for (std::size_t i = 0; i < requests.size(); ++i) {
      bound = std::max(bound, prior_outputs + solo_peak[i]);
      append(session, requests[i]);
      prior_outputs += output_bytes[i];
      expect(ledger.observations->live_bytes == prior_outputs,
             "a routed append kept device temporaries beyond its call");
    }
    expect(ledger.observations->peak_live_bytes <= bound,
           "routed session peak exceeded prior outputs plus one request's working set");
    expect(!ledger.observations->wrong_stream, "a routed temporary was released on another stream");
    outputs = session.finish();
  }

  auto const in = [&](std::vector<std::string> set) {
    return [&varied, set = std::move(set)](std::int64_t row) {
      return std::find(set.begin(), set.end(), varied[static_cast<std::size_t>(row)]) != set.end();
    };
  };
  auto const flags_match = [&](cudf::column_view column, auto const& expected) {
    if (column.type().id() != cudf::type_id::BOOL8 || column.size() != rows ||
        column.null_count() != 0)
      return false;
    std::vector<std::uint8_t> host(rows);
    cuda_check(cudaMemcpy(host.data(), column.head<void>(), host.size(), cudaMemcpyDeviceToHost));
    for (cudf::size_type row = 0; row < rows; ++row)
      if ((host[row] != 0) != expected(row)) return false;
    return true;
  };
  auto const mask_matches = [&](std::size_t index, auto const& expected) {
    std::vector<std::uint32_t> host(mask_words);
    cuda_check(cudaMemcpy(host.data(),
                          destinations[index].data(),
                          host.size() * sizeof(host[0]),
                          cudaMemcpyDeviceToHost));
    for (std::size_t word = 0; word < host.size(); ++word) {
      std::uint32_t want = 0;
      for (int bit = 0; bit < 32; ++bit) {
        auto const row = static_cast<std::int64_t>(word * 32 + bit);
        if (row < rows && expected(row)) want |= 1U << bit;
      }
      if (host[word] != want) return false;
    }
    return true;
  };
  auto const selected_strings = [&](std::vector<std::string> const& source) {
    std::vector<std::string> subset;
    for (auto const row : selected.survivors)
      subset.push_back(source[static_cast<std::size_t>(row)]);
    return make_strings_column(subset, {}, stream);
  };
  std::vector<std::int32_t> selected_values;
  for (auto const row : selected.survivors)
    selected_values.push_back(values[static_cast<std::size_t>(row)]);
  auto const expected_values =
    cudf::make_numeric_column(cudf::data_type{cudf::type_id::INT32},
                              static_cast<cudf::size_type>(selected_values.size()),
                              cudf::mask_state::UNALLOCATED,
                              stream,
                              upstream);
  cuda_check(cudaMemcpy(expected_values->mutable_view().head<void>(),
                        selected_values.data(),
                        selected_values.size() * sizeof(selected_values[0]),
                        cudaMemcpyHostToDevice));

  expect(outputs.size() == 7, "routed session lost a column output");
  expect(
    flags_match(outputs[0]->view(), in({"apple", "fig"})) && mask_matches(0, in({"apple", "fig"})),
    "stored-dictionary predicate or its ballot mismatch");
  expect(flags_match(outputs[1]->view(), in({"banana"})), "rebuilt-dictionary predicate mismatch");
  expect(flags_match(outputs[2]->view(), in({"cherry", "date"})), "generic predicate mismatch");
  expect(strings_equal_completed(selected_strings(fixed)->view(), outputs[3]->view(), stream),
         "dictionary gather specialization mismatch");
  expect(strings_equal_completed(selected_strings(varied)->view(), outputs[4]->view(), stream),
         "dictionary codes fallback mismatch");
  expect(strings_equal_completed(selected_strings(varied)->view(), outputs[5]->view(), stream),
         "selected str_split mismatch");
  expect(columns_equal_completed(expected_values->view(), outputs[6]->view()),
         "full-width decode plus gather mismatch");
  expect(mask_matches(1, [&](std::int64_t row) { return values[row] < threshold; }),
         "membership mask mismatch");
  expect(mask_matches(
           2, [&](std::int64_t row) { return values[row] >= range.lo && values[row] <= range.hi; }),
         "range mask mismatch");
  outputs.clear();
  ledger.check();
}

// The filtered decode completes the lanes that carry its own work, including a BOOL8 gather that no
// session request covers, and leaves a supplied lane that received no work alone.
void test_scan_filter_phase_lanes(rmm::device_async_resource_ref upstream)
{
  namespace sc                   = sirius::codegen;
  constexpr cudf::size_type rows = 1037;
  std::vector<std::string> strings(rows, "other");
  cudf::size_type matches = 0;
  for (cudf::size_type row = 0; row < rows; row += 97, ++matches)
    strings[row] = "match";
  auto const numbers = make_int32_table(1, rows, 137);
  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.push_back(std::make_unique<cudf::column>(numbers->view().column(0)));
  columns.push_back(make_strings_column(strings, {}, cudf::get_default_stream()));
  cudf::table const input{std::move(columns)};
  auto const compressed = simpatico::compress_with_plan(
    input.view(),
    "input -> bitpack\n---\n"
    "input -> dictionary -> keys_offsets, keys_chars, indices\ndictionary.indices -> bitpack\n",
    cudf::get_default_stream(),
    upstream);
  cudf::get_default_stream().synchronize();

  // The range source (every row) runs on lane 0 and the BOOL8 source on lane 1, where its survivor
  // gather also runs; lane 2 receives nothing.
  simpatico::stream_pool pool;
  expect(pool.init(3), "phase lane pool init");
  auto const lanes = pool.refs();
  std::array<std::size_t, 2> const selected{0, 1};
  sc::scan_filter_request request;
  request.routes = {sc::decode_route::bitpack_mask, sc::decode_route::dict_codes};
  request.filters.push_back(
    {0, {std::numeric_limits<std::int32_t>::min(), std::numeric_limits<std::int32_t>::max()}});
  request.bool8_filters.push_back({1, {"match"}});
  rmm::cuda_stream out(rmm::cuda_stream::flags::non_blocking);
  sc::scan_filter_result result;
  // Compiles and loads the kernels first: a module load can wait for every stream.
  auto output =
    simpatico::decompress_scan_filter(compressed, selected, request, result, lanes, out, upstream);
  cuda_check(pool.sync_all());
  output.reset();
  {
    stream_gate idle(pool.streams[2]);
    {
      stream_observation_scope observed;
      output = simpatico::decompress_scan_filter(
        compressed, selected, request, result, lanes, out, upstream);
      auto const unused = observed.count(pool.streams[2]);
      expect(unused.queries == 0 && unused.synchronizations == 0,
             "filtered decode waited for a lane without work");
    }
    idle.expect_not_released("filtered decode waited for a lane without work");
    for (std::size_t lane = 0; lane < 2; ++lane)
      expect(cudaStreamQuery(pool.streams[lane]) == cudaSuccess,
             "filtered decode returned with its own lane work pending");
    idle.release_after_delay.store(true);
    idle.expect_not_timed_out("phase lane watchdog expired");
  }
  expect(result.applied && output->num_rows() == matches, "phase lane fixture was not filtered");
  auto const flags = output->view().column(1);
  expect(flags.type().id() == cudf::type_id::BOOL8, "phase lane fixture lost its BOOL8 answer");
  std::vector<std::uint8_t> host(static_cast<std::size_t>(flags.size()));
  cuda_check(cudaMemcpy(host.data(), flags.head<void>(), host.size(), cudaMemcpyDeviceToHost));
  expect(std::all_of(host.begin(), host.end(), [](auto flag) { return flag != 0; }),
         "phase lane fixture BOOL8 values");
}

void test_resources_and_failures(rmm::device_async_resource_ref upstream)
{
  decode_fixture fixture(upstream, "input -> bitpack\n", 8, 65549, 71, 4);
  auto& [input, compressed, pool, explicit_resource] = fixture;
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
    // Fail the first allocation before any codec work, then also fail cleanup's query on the lane
    // that request was assigned. The original allocation exception must retain priority.
    std::optional<stream_failure_scope> cleanup_failure;
    if (fail_at == 0) cleanup_failure.emplace(stream_fault::operation::query, pool.streams[0]);
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
      expect(observed.count(pool.streams[0]).queries > 0,
             "allocation cleanup skipped its request's lane after a stream error");
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
    // The scan-filter policy reads its gate once.
    setenv("SIRIUS_EXP_FUSED_SCAN_FILTER", "1", 1);
    setenv("LIBCUDF_PINNED_POOL_SIZE", "0", 1);
    setenv("LIBCUDF_PINNED_POOL_MAX_SIZE", "0", 1);
    expect(cudf::config_default_pinned_memory_resource({.pool_size = 0}),
           "pinned resource was initialized before test configuration");
    rmm::mr::cuda_async_memory_resource upstream(128ULL << 20, 128ULL << 20);
    current_resource_guard current(upstream);
    test_table_contracts(upstream);
    test_completed_public_return(upstream);
    test_borrowed_streams(upstream);
    test_identity_owned_children(upstream);
    test_dictionary_width_metadata(upstream);
    test_dictionary_offsets_validation(upstream);
    test_dictionary_rebuild_keeps_indices(upstream);
    test_dictionary_unsigned_indices(upstream);
    test_dictionary_width_sliced_input(upstream);
    test_dictionary_width_large_keys(upstream);
    test_dictionary_width_failures(upstream);
    test_dictionary_width_across_streams(upstream);
    test_dictionary_key_width_hint(upstream);
    test_dictionary_predicate_lookup(upstream);
    test_submission_and_kernel_lifetime(upstream);
    test_abandoned_session(upstream);
    test_session_state_contracts(upstream);
    test_stream_completion_failures(upstream);
    test_duplicate_stream_handles(upstream);
    test_external_phase_tail(upstream);
    test_request_copy_failure(upstream);
    test_host_upload_growth(upstream);
    test_host_uploads_survive_until_completion(upstream);
    test_host_observation_staging(upstream);
    test_temporaries_released_at_submission(upstream);
    test_submission_never_waits_for_memory(upstream);
    test_failure_after_enqueue_unwinds_stream_ordered(upstream);
    test_constant_width_offsets_bounds(upstream);
    test_selection_and_predicate_routes(upstream);
    test_scan_filter_phase_lanes(upstream);
    test_resources_and_failures(upstream);
    cuda_check(cudaDeviceSynchronize());
    std::puts("test_async_decode: OK");
    return 0;
  } catch (std::exception const& error) {
    std::fprintf(stderr, "test_async_decode: FAIL: %s\n", error.what());
    return 1;
  }
}
