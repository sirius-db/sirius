/* Copyright 2026, Sirius Contributors. SPDX-License-Identifier: Apache-2.0 */
#include "op/scan/arrow_materializer.hpp"

#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/cuda_stream.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/error.hpp>
#include <rmm/mr/cuda_async_memory_resource.hpp>

#include <catch.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <limits>
#include <memory_resource>
#include <thread>

using namespace sirius::op::scan;

namespace {
// Migrated from2216's resource controls; copies of the CUDA resource facade share observations.
class observed_device_resource {
 public:
  struct observations {
    long attempts                = 0;
    long fail_at                 = -1;
    std::size_t live_bytes       = 0;
    cudaStream_t expected_stream = nullptr;
    bool wrong_stream            = false;
    bool delay_deallocation      = false;
    int deallocations            = 0;
    std::atomic<int> completed_deallocations{0};
    cudaError_t callback_error = cudaSuccess;
  };
  explicit observed_device_resource(rmm::device_async_resource_ref upstream)
    : state(std::make_shared<observations>()), upstream_(upstream)
  {
  }
  void* allocate(::cuda::stream_ref stream,
                 std::size_t bytes,
                 std::size_t alignment = alignof(std::max_align_t))
  {
    state->wrong_stream |= stream.get() != state->expected_stream;
    attempt();
    auto* result = upstream_.allocate(stream, bytes, alignment);
    state->live_bytes += bytes;
    return result;
  }
  void deallocate(::cuda::stream_ref stream,
                  void* pointer,
                  std::size_t bytes,
                  std::size_t alignment = alignof(std::max_align_t)) noexcept
  {
    state->wrong_stream |= stream.get() != state->expected_stream;
    ++state->deallocations;
    if (state->delay_deallocation) {
      auto status = cudaLaunchHostFunc(
        stream.get(),
        [](void* data) {
          std::this_thread::sleep_for(std::chrono::milliseconds(10));
          ++static_cast<observations*>(data)->completed_deallocations;
        },
        state.get());
      if (status != cudaSuccess) state->callback_error = status;
    }
    upstream_.deallocate(stream, pointer, bytes, alignment);
    state->live_bytes -= bytes;
  }
  void* allocate_sync(std::size_t bytes, std::size_t alignment = alignof(std::max_align_t))
  {
    state->wrong_stream = true;
    attempt();
    auto* result = upstream_.allocate_sync(bytes, alignment);
    state->live_bytes += bytes;
    return result;
  }
  void deallocate_sync(void* pointer,
                       std::size_t bytes,
                       std::size_t alignment = alignof(std::max_align_t)) noexcept
  {
    state->wrong_stream = true;
    upstream_.deallocate_sync(pointer, bytes, alignment);
    state->live_bytes -= bytes;
  }
  bool operator==(observed_device_resource const& other) const noexcept
  {
    return state == other.state;
  }
  friend void get_property(observed_device_resource const&, ::cuda::mr::device_accessible) noexcept
  {
  }
  std::shared_ptr<observations> state;

 private:
  void attempt()
  {
    auto attempt = state->attempts++;
    if (state->fail_at == -2 || attempt == state->fail_at)
      throw rmm::out_of_memory("injected Arrow allocation refusal");
  }
  rmm::device_async_resource_ref upstream_;
};

struct host_resource_lifetime {
  int destructions                       = 0;
  std::size_t outstanding_at_destruction = 0;
};

class bounded_host_resource final : public std::pmr::memory_resource {
 public:
  explicit bounded_host_resource(
    host_resource_lifetime* lifetime    = nullptr,
    std::pmr::memory_resource* upstream = std::pmr::new_delete_resource())
    : lifetime_(lifetime), upstream_(upstream)
  {
  }
  ~bounded_host_resource() override
  {
    if (lifetime_) {
      ++lifetime_->destructions;
      lifetime_->outstanding_at_destruction = used;
    }
  }
  std::size_t limit = std::numeric_limits<std::size_t>::max();
  std::size_t used  = 0;
  std::size_t peak  = 0;
  long attempts     = 0;
  long fail_at      = -1;

 private:
  host_resource_lifetime* lifetime_;
  std::pmr::memory_resource* upstream_;
  void* do_allocate(std::size_t bytes, std::size_t alignment) override
  {
    auto attempt = attempts++;
    if (attempt == fail_at || bytes > limit - used) throw std::bad_alloc();
    auto* result = upstream_->allocate(bytes, alignment);
    used += bytes;
    peak = std::max(peak, used);
    return result;
  }
  void do_deallocate(void* pointer, std::size_t bytes, std::size_t alignment) override
  {
    upstream_->deallocate(pointer, bytes, alignment);
    used -= bytes;
  }
  bool do_is_equal(std::pmr::memory_resource const& other) const noexcept override
  {
    return this == &other;
  }
};

// Device storage is allocated before the stream gate. This facade only divides
// that bounded storage, so allocation cannot grow or synchronize behind the gate.
class fixed_device_resource {
 public:
  fixed_device_resource(void* storage, std::size_t bytes)
    : arena_(std::make_shared<std::pmr::monotonic_buffer_resource>(
        storage, bytes, std::pmr::null_memory_resource()))
  {
  }
  void* allocate(::cuda::stream_ref, std::size_t bytes, std::size_t alignment)
  {
    return arena_->allocate(bytes, alignment);
  }
  void deallocate(::cuda::stream_ref,
                  void* pointer,
                  std::size_t bytes,
                  std::size_t alignment) noexcept
  {
    arena_->deallocate(pointer, bytes, alignment);
  }
  void* allocate_sync(std::size_t bytes, std::size_t alignment)
  {
    return arena_->allocate(bytes, alignment);
  }
  void deallocate_sync(void* pointer, std::size_t bytes, std::size_t alignment) noexcept
  {
    arena_->deallocate(pointer, bytes, alignment);
  }
  bool operator==(fixed_device_resource const& other) const noexcept
  {
    return arena_ == other.arena_;
  }
  friend void get_property(fixed_device_resource const&, ::cuda::mr::device_accessible) noexcept {}

 private:
  std::shared_ptr<std::pmr::monotonic_buffer_resource> arena_;
};

struct release_counts {
  int arrays  = 0;
  int schemas = 0;
};

struct producer_batch {
  ArrowSchema schema{}, fields[3]{};
  ArrowArray array{}, columns[3]{};
  ArrowSchema* field_ptrs[3]{&fields[0], &fields[1], &fields[2]};
  ArrowArray* column_ptrs[3]{&columns[0], &columns[1], &columns[2]};
  int32_t integers[12]{};
  int32_t offsets[13]{};
  uint8_t booleans[2]{0x55, 0x05};
  uint8_t validity[2]{0xbd, 0x0f};  // Nulls at absolute positions1 and6.
  char characters[13] = "abcdefghijkl";
  void const* buffers[3][3]{
    {validity, integers, nullptr}, {validity, booleans, nullptr}, {validity, offsets, characters}};
  void const* root_buffers[1]{nullptr};

  explicit producer_batch(release_counts& released)
  {
    for (int i = 0; i < 12; ++i)
      integers[i] = i;
    for (int i = 0; i < 13; ++i)
      offsets[i] = i;
    char const* formats[3]{"i", "b", "u"};
    for (int i = 0; i < 3; ++i) {
      fields[i].format      = formats[i];
      fields[i].flags       = ARROW_FLAG_NULLABLE;
      fields[i].release     = [](ArrowSchema* p) { p->release = nullptr; };
      columns[i].length     = 12;
      columns[i].null_count = 2;
      columns[i].n_buffers  = i == 2 ? 3 : 2;
      columns[i].buffers    = buffers[i];
      columns[i].release    = [](ArrowArray* p) { p->release = nullptr; };
    }
    schema.format     = "+s";
    schema.n_children = array.n_children = 3;
    schema.children                      = field_ptrs;
    array.children                       = column_ptrs;
    array.length                         = 12;
    array.n_buffers                      = 1;
    array.buffers                        = root_buffers;
    schema.private_data = array.private_data = &released;
    schema.release                           = [](ArrowSchema* p) {
      ++static_cast<release_counts*>(p->private_data)->schemas;
      p->release = nullptr;
    };
    array.release = [](ArrowArray* p) {
      ++static_cast<release_counts*>(p->private_data)->arrays;
      p->release = nullptr;
    };
  }
  ~producer_batch()
  {
    if (array.release) array.release(&array);
    if (schema.release) schema.release(&schema);
  }
};

materialization_program program_for_batch()
{
  materialization_program program{601,
                                  1,
                                  {cudf::data_type{cudf::type_id::INT32},
                                   cudf::data_type{cudf::type_id::BOOL8},
                                   cudf::data_type{cudf::type_id::STRING}},
                                  0,
                                  false};
  program.input_types = program.output_types;
  return program;
}

// The production completion seam can reject installation/sync without poisoning a CUDA context.
struct completion_fault {
  bool refuse_sync = true;
  int sync_calls   = 0;
  static thread_local completion_fault* active;
  static cudaError_t add(cudaStream_t, cudaStreamCallback_t, void*, unsigned)
  {
    return cudaErrorInvalidValue;
  }
  static cudaError_t synchronize(cudaStream_t stream)
  {
    ++active->sync_calls;
    return active->refuse_sync ? cudaErrorUnknown : cudaStreamSynchronize(stream);
  }
  completion_fault() { active = this; }
  ~completion_fault() { active = nullptr; }
};
thread_local completion_fault* completion_fault::active = nullptr;

// Negative control only; the production importer never changes this process-wide resource.
class current_resource_guard {
 public:
  explicit current_resource_guard(rmm::device_async_resource_ref resource)
    : previous_(cudf::set_current_device_resource(
        ::cuda::mr::any_resource<::cuda::mr::device_accessible>{resource}))
  {
  }
  explicit current_resource_guard(observed_device_resource const& resource)
    : previous_(cudf::set_current_device_resource(
        ::cuda::mr::any_resource<::cuda::mr::device_accessible>{resource}))
  {
  }
  ~current_resource_guard() { (void)cudf::set_current_device_resource(std::move(previous_)); }

 private:
  ::cuda::mr::any_resource<::cuda::mr::device_accessible> previous_;
};
}  // namespace

TEST_CASE("Arrow migration normalizes combined Boolean and UTF8 windows on the selected stream",
          "[materialization][scan][arrow_migration]")
{
  release_counts released;
  bounded_host_resource host;
  rmm::cuda_stream stream;
  sirius::exec::retire_lane lane(stream.value());
  observed_device_resource device(cudf::get_current_device_resource_ref());
  device.state->expected_stream = stream.value();
  auto producer                 = std::make_shared<producer_batch>(released);
  producer->array.offset        = 2;
  producer->array.length        = 7;
  for (auto& child : producer->columns) {
    child.offset = 1;
    child.length = 11;
  }
  auto input =
    std::make_shared<owned_arrow_batch>(producer->schema, producer->array, producer, host);
  auto batch =
    import_arrow_host(input, program_for_batch(), lane, rmm::device_async_resource_ref{device});
  input.reset();
  producer.reset();
  CHECK(host.used > 0);
  CHECK(released.arrays == 0);
  REQUIRE(lane.quiesce() == cudaSuccess);
  CHECK(host.used == 0);
  CHECK(released.arrays == 1);
  CHECK(released.schemas == 1);
  CHECK_FALSE(device.state->wrong_stream);
  REQUIRE(batch.rows == 7);
  auto table = batch.table.view();
  for (auto const& column : table)
    CHECK(column.null_count() == 1);
  std::array<int32_t, 7> integers{};
  std::array<uint8_t, 7> booleans{};
  std::array<int32_t, 8> offsets{};
  std::array<char, 7> characters{};
  uint8_t validity = 0;
  cudf::strings_column_view strings(table.column(2));
  REQUIRE(
    cudaMemcpy(integers.data(), table.column(0).head(), sizeof(integers), cudaMemcpyDeviceToHost) ==
    cudaSuccess);
  REQUIRE(
    cudaMemcpy(booleans.data(), table.column(1).head(), sizeof(booleans), cudaMemcpyDeviceToHost) ==
    cudaSuccess);
  REQUIRE(
    cudaMemcpy(offsets.data(), strings.offsets().head(), sizeof(offsets), cudaMemcpyDeviceToHost) ==
    cudaSuccess);
  REQUIRE(cudaMemcpy(characters.data(),
                     strings.chars_begin(stream),
                     sizeof(characters),
                     cudaMemcpyDeviceToHost) == cudaSuccess);
  REQUIRE(cudaMemcpy(&validity, table.column(0).null_mask(), 1, cudaMemcpyDeviceToHost) ==
          cudaSuccess);
  CHECK(validity == 0x77);
  for (int i = 0; i < 7; ++i) {
    CHECK(integers[i] == i + 3);
    CHECK(booleans[i] == ((i + 3) % 2 == 0));
    CHECK(offsets[i] == i);
    CHECK(characters[i] == 'd' + i);
  }
  CHECK(offsets.back() == 7);
}

TEST_CASE("Arrow migration refuses invalid final columns and host exhaustion before GPU allocation",
          "[materialization][scan][arrow_migration]")
{
  release_counts released;
  bounded_host_resource host;
  rmm::cuda_stream stream;
  sirius::exec::retire_lane lane(stream.value());
  observed_device_resource device(cudf::get_current_device_resource_ref());
  device.state->expected_stream = stream.value();
  auto producer                 = std::make_shared<producer_batch>(released);
  SECTION("invalid final string offsets") { producer->offsets[12] = 0; }
  SECTION("host quota refuses staging") { host.limit = 0; }
  auto input =
    std::make_shared<owned_arrow_batch>(producer->schema, producer->array, producer, host);
  REQUIRE_THROWS(
    import_arrow_host(input, program_for_batch(), lane, rmm::device_async_resource_ref{device}));
  CHECK(device.state->attempts == 0);
  CHECK(host.used == 0);
  CHECK(lane.idle());
  input.reset();
  producer.reset();
  CHECK(released.arrays == 1);
  CHECK(released.schemas == 1);
}

TEST_CASE("Arrow import retains the capacity-owned host resource through host-thread retirement",
          "[materialization][scan][arrow_migration]")
{
  release_counts released;
  host_resource_lifetime lifetime;
  rmm::cuda_stream stream;
  sirius::exec::retire_lane lane(stream.value());
  struct admitted_producer {
    bounded_host_resource host;
    producer_batch roots;
    admitted_producer(release_counts& released, host_resource_lifetime& lifetime)
      : host(&lifetime), roots(released)
    {
    }
  };
  auto producer                         = std::make_shared<admitted_producer>(released, lifetime);
  std::weak_ptr<admitted_producer> weak = producer;
  auto input                            = std::make_shared<owned_arrow_batch>(
    producer->roots.schema, producer->roots.array, producer, producer->host);
  {
    auto batch =
      import_arrow_host(input, program_for_batch(), lane, cudf::get_current_device_resource_ref());
  }
  input.reset();
  producer.reset();
  CHECK_FALSE(weak.expired());
  CHECK(lifetime.destructions == 0);
  cudaError_t status = cudaErrorUnknown;
  std::thread drainer([&] { status = lane.quiesce(); });
  drainer.join();
  REQUIRE(status == cudaSuccess);
  CHECK(weak.expired());
  CHECK(released.arrays == 1);
  CHECK(released.schemas == 1);
  CHECK(lifetime.destructions == 1);
  CHECK(lifetime.outstanding_at_destruction == 0);
}

TEST_CASE("Arrow import retains pinned inputs and staging while H2D copies are pending",
          "[materialization][scan][arrow_migration]")
{
  release_counts released;
  void* pinned_storage = nullptr;
  REQUIRE(cudaMallocHost(&pinned_storage, 65536) == cudaSuccess);
  std::unique_ptr<void, decltype(&cudaFreeHost)> pinned(pinned_storage, cudaFreeHost);
  std::pmr::monotonic_buffer_resource arena(pinned.get(), 65536, std::pmr::null_memory_resource());
  bounded_host_resource host(nullptr, &arena);
  struct gate_state {
    std::atomic<bool> entered{false}, release{false}, timed_out{false}, finished{false};
  } gate;
  rmm::cuda_stream stream;
  rmm::mr::cuda_async_memory_resource upstream;
  rmm::device_buffer backing(1 << 20, stream, rmm::device_async_resource_ref{upstream});
  stream.synchronize();
  fixed_device_resource device_arena(backing.data(), backing.size());
  observed_device_resource device(rmm::device_async_resource_ref{device_arena});
  device.state->expected_stream = stream.value();
  sirius::exec::retire_lane lane(stream.value());
  // This state outlives the lane. Every exit releases the gate before the lane
  // quiesces, and the watchdog bounds any unexpected host-blocking CUDA call.
  std::jthread watchdog([&] {
    auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (!gate.release.load() && std::chrono::steady_clock::now() < deadline)
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    if (!gate.release.load()) {
      gate.timed_out = true;
      gate.release   = true;
    }
  });
  auto attempt = [&] {
    struct release_gate {
      gate_state& gate;
      sirius::exec::retire_lane& lane;
      ~release_gate()
      {
        gate.release = true;
        // The gate callback precedes lane submissions, including failures before
        // begin(). An idle retirement queue is not a callback-completion barrier.
        (void)lane.quiesce();
      }
    } release{gate, lane};
    auto producer = std::allocate_shared<producer_batch>(
      std::pmr::polymorphic_allocator<producer_batch>{&host}, released);
    auto producer_bytes = host.used;
    auto input =
      std::make_shared<owned_arrow_batch>(producer->schema, producer->array, producer, host);
    REQUIRE(cudaLaunchHostFunc(
              stream.value(),
              [](void* state) {
                auto& gate   = *static_cast<gate_state*>(state);
                gate.entered = true;
                while (!gate.release.load())
                  std::this_thread::sleep_for(std::chrono::microseconds(100));
                // Distinguish releasing the gate from completing its callback.
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
                gate.finished = true;
              },
              &gate) == cudaSuccess);
    while (!gate.entered.load() && !gate.timed_out.load())
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    REQUIRE(gate.entered.load());
    std::optional<materialized_batch> batch;
    SECTION("successful submission returns before pending copies complete")
    {
      batch.emplace(import_arrow_host(
        input, program_for_batch(), lane, rmm::device_async_resource_ref{device}));
    }
    SECTION("allocation failure after the first H2D enqueue retains its inputs")
    {
      device.state->fail_at = 1;
      REQUIRE_THROWS_AS(
        import_arrow_host(input, program_for_batch(), lane, rmm::device_async_resource_ref{device}),
        rmm::out_of_memory);
      CHECK(device.state->attempts == 2);
    }
    // Keep this early return last so Catch registers both earlier sibling sections.
    SECTION("host allocation fails before any lane submission")
    {
      host.fail_at = host.attempts;
      REQUIRE_THROWS_AS(
        import_arrow_host(input, program_for_batch(), lane, rmm::device_async_resource_ref{device}),
        std::bad_alloc);
      CHECK(device.state->attempts == 0);
      CHECK(lane.idle());
      return;
    }
    CHECK_FALSE(gate.release.load());
    CHECK_FALSE(gate.timed_out.load());
    CHECK(cudaStreamQuery(stream.value()) == cudaErrorNotReady);
    input.reset();
    producer.reset();
    CHECK(released.arrays == 0);
    CHECK(host.used > producer_bytes);
    gate.release = true;
    REQUIRE(lane.quiesce() == cudaSuccess);
    CHECK(released.arrays == 1);
    CHECK(released.schemas == 1);
    CHECK(host.used == 0);
    if (batch) {
      std::array<int32_t, 12> values{};
      REQUIRE(cudaMemcpy(values.data(),
                         batch->table.view().column(0).head(),
                         sizeof(values),
                         cudaMemcpyDeviceToHost) == cudaSuccess);
      CHECK(values.front() == 0);
      CHECK(values.back() == 11);
    }
    batch.reset();
    REQUIRE(lane.quiesce() == cudaSuccess);
    CHECK(device.state->live_bytes == 0);
    CHECK_FALSE(device.state->wrong_stream);
  };
  attempt();
  CHECK(gate.finished.load());
  // Backstop even a broken cleanup guard, so its negative control can fail
  // without letting callback state or backing storage die while CUDA uses it.
  REQUIRE(lane.quiesce() == cudaSuccess);
}

TEST_CASE("Arrow import validates only the selected UTF8 window including empty inputs",
          "[materialization][scan][arrow_migration]")
{
  release_counts released;
  bounded_host_resource host;
  rmm::cuda_stream stream;
  sirius::exec::retire_lane lane(stream.value());
  auto producer               = std::make_shared<producer_batch>(released);
  producer->schema.n_children = producer->array.n_children = 1;
  producer->field_ptrs[0]                                  = &producer->fields[2];
  producer->column_ptrs[0]                                 = &producer->columns[2];
  SECTION("an empty string column need not supply offsets or characters")
  {
    producer->array.length = producer->columns[2].length = 0;
    producer->columns[2].null_count                      = 0;
    for (auto& buffer : producer->buffers[2])
      buffer = nullptr;
  }
  SECTION("invalid offsets outside the selected window are not imported")
  {
    producer->array.offset = 2;
    producer->array.length = 7;
    producer->offsets[0] = producer->offsets[12] = -1;
  }
  auto rows = producer->array.length;
  auto input =
    std::make_shared<owned_arrow_batch>(producer->schema, producer->array, producer, host);
  materialization_program program{603, 1, {cudf::data_type{cudf::type_id::STRING}}, 0, false};
  program.input_types = program.output_types;
  auto batch = import_arrow_host(input, program, lane, cudf::get_current_device_resource_ref());
  REQUIRE(lane.quiesce() == cudaSuccess);
  CHECK(batch.rows == rows);
  CHECK(batch.table.view().column(0).size() == rows);
  if (rows) {
    std::array<char, 7> characters{};
    cudf::strings_column_view strings(batch.table.view().column(0));
    REQUIRE(cudaMemcpy(characters.data(),
                       strings.chars_begin(stream),
                       characters.size(),
                       cudaMemcpyDeviceToHost) == cudaSuccess);
    for (int row = 0; row < 7; ++row)
      CHECK(characters[row] == 'c' + row);
  }
}

TEST_CASE("Arrow migration Boolean slices cover shifted word and mask boundaries",
          "[materialization][scan][arrow_migration]")
{
  for (int rows : {1, 7, 8, 31, 32, 33, 511, 512, 513}) {
    for (int child_offset : {0, 1, 7}) {
      CAPTURE(rows, child_offset);
      release_counts released;
      bounded_host_resource host;
      rmm::cuda_stream stream;
      sirius::exec::retire_lane lane(stream.value());
      struct producer_storage {
        std::shared_ptr<producer_batch> roots;
        std::vector<uint8_t> values, validity;
      };
      auto producer   = std::make_shared<producer_storage>();
      producer->roots = std::make_shared<producer_batch>(released);
      auto& roots     = *producer->roots;
      auto end        = child_offset + 1 + rows;
      producer->values.resize((end + 7) / 8, 0);
      producer->validity.resize((end + 7) / 8, 0);
      for (int row = 0; row < end; ++row) {
        producer->values[row / 8] |= static_cast<uint8_t>((row % 2 == 0) << (row % 8));
        producer->validity[row / 8] |= static_cast<uint8_t>((row % 5 != 0) << (row % 8));
      }
      roots.schema.n_children = roots.array.n_children = 1;
      roots.fields[0].format                           = "b";
      roots.array.offset                               = 1;
      roots.array.length                               = rows;
      roots.columns[0].offset                          = child_offset;
      roots.columns[0].length                          = rows + 1;
      roots.columns[0].null_count                      = -1;
      roots.buffers[0][0]                              = producer->validity.data();
      roots.buffers[0][1]                              = producer->values.data();
      auto input = std::make_shared<owned_arrow_batch>(roots.schema, roots.array, producer, host);
      materialization_program program{602, 1, {cudf::data_type{cudf::type_id::BOOL8}}, 0, false};
      program.input_types = program.output_types;
      auto batch = import_arrow_host(input, program, lane, cudf::get_current_device_resource_ref());
      input.reset();
      producer.reset();
      REQUIRE(lane.quiesce() == cudaSuccess);
      CHECK(released.arrays == 1);
      CHECK(host.used == 0);
      auto column = batch.table.view().column(0);
      std::vector<uint8_t> actual(rows), mask((rows + 7) / 8, 0xff);
      REQUIRE(cudaMemcpy(actual.data(), column.head(), actual.size(), cudaMemcpyDeviceToHost) ==
              cudaSuccess);
      if (column.null_count())
        REQUIRE(cudaMemcpy(mask.data(), column.null_mask(), mask.size(), cudaMemcpyDeviceToHost) ==
                cudaSuccess);
      for (int row = 0; row < rows; ++row) {
        auto source_row = child_offset + 1 + row;
        CHECK(actual[row] == (source_row % 2 == 0));
        CHECK(((mask[row / 8] >> (row % 8)) & 1U) == (source_row % 5 != 0));
      }
    }
  }
}

TEST_CASE("Arrow migration host and device allocation refusals preserve ownership",
          "[materialization][scan][arrow_migration]")
{
  long allocations      = 0;
  long host_allocations = 0;
  {
    release_counts released;
    bounded_host_resource host;
    rmm::cuda_stream stream;
    sirius::exec::retire_lane lane(stream.value());
    observed_device_resource device(cudf::get_current_device_resource_ref());
    device.state->expected_stream = stream.value();
    auto producer                 = std::make_shared<producer_batch>(released);
    auto input =
      std::make_shared<owned_arrow_batch>(producer->schema, producer->array, producer, host);
    {
      auto batch =
        import_arrow_host(input, program_for_batch(), lane, rmm::device_async_resource_ref{device});
    }
    REQUIRE(lane.quiesce() == cudaSuccess);
    allocations      = device.state->attempts;
    host_allocations = host.attempts;
    REQUIRE(allocations > 3);
    REQUIRE(host_allocations > 3);
    REQUIRE(device.state->live_bytes == 0);
  }
  for (long fail_at = 0; fail_at < allocations; ++fail_at) {
    CAPTURE(fail_at);
    release_counts released;
    bounded_host_resource host;
    rmm::cuda_stream stream;
    sirius::exec::retire_lane lane(stream.value());
    observed_device_resource device(cudf::get_current_device_resource_ref());
    device.state->expected_stream = stream.value();
    device.state->fail_at         = fail_at;
    auto producer                 = std::make_shared<producer_batch>(released);
    auto input =
      std::make_shared<owned_arrow_batch>(producer->schema, producer->array, producer, host);
    REQUIRE_THROWS_AS(
      import_arrow_host(input, program_for_batch(), lane, rmm::device_async_resource_ref{device}),
      rmm::out_of_memory);
    CHECK(device.state->attempts == fail_at + 1);
    input.reset();
    producer.reset();
    CHECK(released.arrays == 0);
    CHECK(host.used > 0);
    REQUIRE(lane.quiesce() == cudaSuccess);
    CHECK(host.used == 0);
    CHECK(released.arrays == 1);
    CHECK(released.schemas == 1);
    CHECK(device.state->live_bytes == 0);
    CHECK_FALSE(device.state->wrong_stream);
  }
  for (long fail_at = 0; fail_at < host_allocations; ++fail_at) {
    CAPTURE(fail_at);
    release_counts released;
    bounded_host_resource host;
    host.fail_at = fail_at;
    rmm::cuda_stream stream;
    sirius::exec::retire_lane lane(stream.value());
    observed_device_resource device(cudf::get_current_device_resource_ref());
    device.state->expected_stream = stream.value();
    auto producer                 = std::make_shared<producer_batch>(released);
    auto input =
      std::make_shared<owned_arrow_batch>(producer->schema, producer->array, producer, host);
    REQUIRE_THROWS_AS(
      import_arrow_host(input, program_for_batch(), lane, rmm::device_async_resource_ref{device}),
      std::bad_alloc);
    CHECK(host.attempts == fail_at + 1);
    CHECK(host.used == 0);
    CHECK(device.state->attempts == 0);
    CHECK(lane.idle());
    input.reset();
    producer.reset();
    CHECK(released.arrays == 1);
    CHECK(released.schemas == 1);
  }
}

TEST_CASE("Arrow migration does not route allocations through the process default resource",
          "[materialization][scan][arrow_migration]")
{
  release_counts released;
  bounded_host_resource host;
  rmm::cuda_stream stream;
  rmm::mr::cuda_async_memory_resource upstream;
  // Match Sirius's resource-ref default. Replacing it must not invalidate our supplied resource.
  current_resource_guard installed_default(rmm::device_async_resource_ref{upstream});
  sirius::exec::retire_lane lane(stream.value());
  observed_device_resource supplied(rmm::device_async_resource_ref{upstream});
  observed_device_resource rejecting(rmm::device_async_resource_ref{upstream});
  supplied.state->expected_stream = stream.value();
  rejecting.state->fail_at        = -2;
  auto producer                   = std::make_shared<producer_batch>(released);
  __int128 decimals[12]{};
  for (int i = 0; i < 12; ++i)
    decimals[i] = i * 125;
  producer->fields[0].format = "d:9,2";
  producer->buffers[0][1]    = decimals;
  auto program               = program_for_batch();
  program.output_types[0]    = cudf::data_type{cudf::type_id::DECIMAL128, -2};
  program.input_types        = program.output_types;
  program.decimal_precisions = {9, 0, 0};
  auto input =
    std::make_shared<owned_arrow_batch>(producer->schema, producer->array, producer, host);
  {
    current_resource_guard guard(rejecting);
    auto batch = import_arrow_host(input, program, lane, rmm::device_async_resource_ref{supplied});
    REQUIRE(batch.table.view().column(0).type() == program.output_types[0]);
    REQUIRE(lane.quiesce() == cudaSuccess);
    __int128 value = 0;
    REQUIRE(cudaMemcpy(
              &value, batch.table.view().column(0).head(), sizeof(value), cudaMemcpyDeviceToHost) ==
            cudaSuccess);
    CHECK(value == decimals[0]);
    CHECK(rejecting.state->attempts == 0);
  }
  stream.synchronize();
  CHECK(supplied.state->attempts > 0);
  CHECK(supplied.state->live_bytes == 0);
}

TEST_CASE("Arrow migration failed completion retains roots and staging until verified recovery",
          "[materialization][scan][arrow_migration]")
{
  release_counts released;
  bounded_host_resource host;
  rmm::cuda_stream stream;
  completion_fault fault;
  observed_device_resource device(cudf::get_current_device_resource_ref());
  device.state->expected_stream = stream.value();
  cucascade::exec::detail::completion_cuda_api api;
  api.add_callback = completion_fault::add;
  api.synchronize  = completion_fault::synchronize;
  sirius::exec::retire_lane lane(stream.value(), api);
  struct restore_sync {
    completion_fault& fault;
    ~restore_sync() { fault.refuse_sync = false; }
  } recovery{fault};
  auto producer = std::make_shared<producer_batch>(released);
  auto input =
    std::make_shared<owned_arrow_batch>(producer->schema, producer->array, producer, host);
  // A failed installation gives no completion frontier; failed sync cannot invent one.
  SECTION("explicit commit fails after output construction")
  {
    REQUIRE_THROWS(
      import_arrow_host(input, program_for_batch(), lane, rmm::device_async_resource_ref{device}));
  }
  SECTION("automatic commit fails while preserving the allocation exception")
  {
    device.state->fail_at =
      1;  // The first validity copy precedes refusal of its values allocation.
    REQUIRE_THROWS_AS(
      import_arrow_host(input, program_for_batch(), lane, rmm::device_async_resource_ref{device}),
      rmm::out_of_memory);
    CHECK(device.state->attempts == 2);
  }
  CHECK(fault.sync_calls == 1);
  input.reset();
  producer.reset();
  CHECK(released.arrays == 0);
  CHECK(host.used > 0);
  CHECK(lane.quiesce() == cudaErrorUnknown);
  CHECK(released.arrays == 0);
  CHECK(host.used > 0);
  fault.refuse_sync = false;
  REQUIRE(lane.quiesce() == cudaSuccess);
  CHECK(released.arrays == 1);
  CHECK(released.schemas == 1);
  CHECK(host.used == 0);
  CHECK(lane.enqueue_error() == cudaErrorInvalidValue);
  CHECK(lane.faulted());
}

TEST_CASE("Arrow migration final quiesce covers frees queued after failed commit",
          "[materialization][scan][arrow_migration]")
{
  release_counts released;
  bounded_host_resource host;
  rmm::cuda_stream stream;
  observed_device_resource device(cudf::get_current_device_resource_ref());
  device.state->expected_stream    = stream.value();
  device.state->delay_deallocation = true;
  completion_fault fault;
  fault.refuse_sync = false;
  cucascade::exec::detail::completion_cuda_api api;
  api.add_callback = completion_fault::add;
  api.synchronize  = completion_fault::synchronize;
  sirius::exec::retire_lane lane(stream.value(), api);
  auto producer = std::make_shared<producer_batch>(released);
  auto input =
    std::make_shared<owned_arrow_batch>(producer->schema, producer->array, producer, host);
  REQUIRE_THROWS(
    import_arrow_host(input, program_for_batch(), lane, rmm::device_async_resource_ref{device}));
  CHECK(fault.sync_calls == 1);  // Commit's recovery sync preceded destruction of its output.
  CHECK(device.state->deallocations > 0);
  input.reset();
  producer.reset();
  REQUIRE(lane.drain() ==
          1);  // Input reads completed, but later output frees need their own barrier.
  CHECK(lane.idle());
  CHECK(released.arrays == 1);
  REQUIRE(lane.quiesce() == cudaSuccess);  // Mandatory even when the retirement queue is idle.
  CHECK(fault.sync_calls == 2);
  CHECK(device.state->callback_error == cudaSuccess);
  CHECK(device.state->completed_deallocations == device.state->deallocations);
  CHECK(cudaStreamQuery(stream.value()) == cudaSuccess);
  CHECK(host.used == 0);
}
