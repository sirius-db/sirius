/* Copyright 2026, Sirius Contributors. SPDX-License-Identifier: Apache-2.0 */
#include "op/scan/arrow_materializer.hpp"

#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/cuda_stream.hpp>
#include <rmm/error.hpp>

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

class bounded_host_resource final : public std::pmr::memory_resource {
 public:
  std::size_t limit = std::numeric_limits<std::size_t>::max();
  std::size_t used  = 0;
  std::size_t peak  = 0;
  long attempts     = 0;
  long fail_at      = -1;

 private:
  void* do_allocate(std::size_t bytes, std::size_t alignment) override
  {
    auto attempt = attempts++;
    if (attempt == fail_at || bytes > limit - used) throw std::bad_alloc();
    auto* result = std::pmr::new_delete_resource()->allocate(bytes, alignment);
    used += bytes;
    peak = std::max(peak, used);
    return result;
  }
  void do_deallocate(void* pointer, std::size_t bytes, std::size_t alignment) override
  {
    std::pmr::new_delete_resource()->deallocate(pointer, bytes, alignment);
    used -= bytes;
  }
  bool do_is_equal(std::pmr::memory_resource const& other) const noexcept override
  {
    return this == &other;
  }
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
          "[r4][scan][arrow_migration]")
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
                     strings.chars_begin(stream.view()),
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
          "[r4][scan][arrow_migration]")
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

TEST_CASE("Arrow migration Boolean slices cover shifted word and mask boundaries",
          "[r4][scan][arrow_migration]")
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
          "[r4][scan][arrow_migration]")
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
          "[r4][scan][arrow_migration]")
{
  release_counts released;
  bounded_host_resource host;
  rmm::cuda_stream stream;
  sirius::exec::retire_lane lane(stream.value());
  auto upstream = cudf::get_current_device_resource_ref();
  observed_device_resource supplied(upstream), rejecting(upstream);
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
          "[r4][scan][arrow_migration]")
{
  release_counts released;
  bounded_host_resource host;
  rmm::cuda_stream stream;
  completion_fault fault;
  sirius::exec::detail::completion_cuda_api api;
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
  REQUIRE_THROWS(
    import_arrow_host(input, program_for_batch(), lane, cudf::get_current_device_resource_ref()));
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
          "[r4][scan][arrow_migration]")
{
  release_counts released;
  bounded_host_resource host;
  rmm::cuda_stream stream;
  observed_device_resource device(cudf::get_current_device_resource_ref());
  device.state->expected_stream    = stream.value();
  device.state->delay_deallocation = true;
  completion_fault fault;
  fault.refuse_sync = false;
  sirius::exec::detail::completion_cuda_api api;
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
