/* Copyright 2026, Sirius Contributors. SPDX-License-Identifier: Apache-2.0 */
#include "catch.hpp"
#include "op/scan/arrow/arrow_batch_producer.hpp"
#include "op/scan/arrow/arrow_staging_resource.hpp"
#include "op/scan/arrow_materializer.hpp"
#include "op/scan/sirius_gpu_scan_operator_data.hpp"
#include "op/scan/table_scan/owned_arrow_scan.hpp"
#include "scan_manager/preparation_fixture.hpp"
#include "transparent/replay_admission.hpp"

#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/traits.hpp>

#include <rmm/cuda_device.hpp>
#include <rmm/cuda_stream.hpp>

#include <array>
#include <atomic>
#include <cstring>
#include <future>
#include <vector>
#if defined(__linux__)
#include <sys/mman.h>
#endif

using namespace sirius::op::scan;
namespace {
void cuda_ok(cudaError_t error)
{
  if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}
struct placed_buffer {
  void* data = nullptr;
  size_t bytes;
  arrow_source_memory memory;
  placed_buffer(std::vector<uint8_t> const& input, arrow_source_memory kind, cudaStream_t stream)
    : bytes(input.size()), memory(kind)
  {
    if (memory == arrow_source_memory::device) {
      cuda_ok(cudaMalloc(&data, bytes));
      cuda_ok(cudaMemcpyAsync(data, input.data(), bytes, cudaMemcpyHostToDevice, stream));
    } else if (memory == arrow_source_memory::pinned_host) {
      cuda_ok(cudaMallocHost(&data, bytes));
      std::memcpy(data, input.data(), bytes);
    } else {
      data = new uint8_t[bytes];
      std::memcpy(data, input.data(), bytes);
    }
  }
  ~placed_buffer()
  {
    if (memory == arrow_source_memory::device)
      cudaFree(data);
    else if (memory == arrow_source_memory::pinned_host)
      cudaFreeHost(data);
    else
      delete[] static_cast<uint8_t*>(data);
  }
};
struct import_tree {
  ArrowSchema schema{}, field{};
  ArrowArray array{}, column{};
  ArrowSchema* fields[1]{&field};
  ArrowArray* columns[1]{&column};
  void const* root_buffers[1]{};
  void const* buffers[3]{};
  std::vector<std::shared_ptr<placed_buffer>> storage;
  std::shared_ptr<std::atomic<unsigned>> released = std::make_shared<std::atomic<unsigned>>(0);
  cudaEvent_t ready                               = nullptr;
  import_tree(char const* format, int rows)
  {
    schema.format      = "+s";
    schema.n_children  = 1;
    schema.children    = fields;
    schema.release     = [](ArrowSchema* s) { s->release = nullptr; };
    field.name         = "value";
    field.format       = format;
    field.flags        = ARROW_FLAG_NULLABLE;
    field.release      = schema.release;
    array.length       = rows;
    array.offset       = 1;
    array.n_children   = 1;
    array.children     = columns;
    array.n_buffers    = 1;
    array.buffers      = root_buffers;
    array.private_data = released.get();
    array.release      = [](ArrowArray* a) {
      ++*static_cast<std::atomic<unsigned>*>(a->private_data);
      a->release = nullptr;
    };
    column.length     = 4;
    column.offset     = 1;
    column.null_count = -1;
    column.release    = [](ArrowArray* a) { a->release = nullptr; };
    column.n_buffers  = std::string_view(format) == "u" || std::string_view(format) == "U" ? 3 : 2;
    column.buffers    = buffers;
  }
  ~import_tree()
  {
    if (ready) cudaEventDestroy(ready);
  }
};
struct pinned_arena {
  void* data = nullptr;
  std::unique_ptr<arrow_staging_resource> resource;
  explicit pinned_arena(size_t bytes)
  {
    cuda_ok(cudaMallocHost(&data, bytes));
    resource = std::make_unique<arrow_staging_resource>(data, bytes);
  }
  ~pinned_arena()
  {
    resource.reset();
    cudaFreeHost(data);
  }
};

struct placed_fixture {
  rmm::cuda_stream producer, consumer;
  std::shared_ptr<import_tree> tree = std::make_shared<import_tree>("i", 3);
  arrow_source_profile profile;
  materialization_program program{701, 4, {cudf::data_type{cudf::type_id::INT32}}, 0, false};
  std::vector<uint8_t> payload;
  std::shared_ptr<pinned_arena> staging;
  explicit placed_fixture(arrow_source_memory memory)
  {
    std::array<int32_t, 5> values{0, 0, 11, 22, 33};
    payload.resize(sizeof(values));
    std::memcpy(payload.data(), values.data(), sizeof(values));
    auto buffer             = std::make_shared<placed_buffer>(payload, memory, producer.value());
    tree->storage           = {buffer};
    tree->buffers[1]        = buffer->data;
    tree->column.null_count = 0;
    profile.memory          = memory;
    if (memory == arrow_source_memory::device) cuda_ok(cudaGetDevice(&profile.device));
    profile.leases        = tree;
    profile.control_bytes = sizeof(import_tree) + 256;
    profile.allocations   = {{1, buffer->data, buffer->bytes}};
    profile.buffers       = {{{}, arrow_buffer_use{1, 0, buffer->bytes}}};
    profile.strings.resize(1);
    cuda_ok(cudaEventCreateWithFlags(&tree->ready, cudaEventDisableTiming));
    cuda_ok(cudaEventRecord(tree->ready, producer.value()));
    profile.ready_event  = tree->ready;
    program.input_types  = program.output_types;
    program.arrow_schema = std::make_shared<arrow_bound_schema const>(
      arrow_bound_schema{{{"value", "i", cudf::data_type{cudf::type_id::INT32}, true}}});
  }
  std::shared_ptr<owned_arrow_batch> input()
  {
    auto estimate =
      validate_arrow_source(tree->schema, tree->array, profile, *program.arrow_schema);
    staging = std::make_shared<pinned_arena>(std::max<uint64_t>(1, estimate.staging));
    auto input =
      std::make_shared<owned_arrow_batch>(tree->schema,
                                          tree->array,
                                          tree,
                                          *staging->resource,
                                          arrow_metadata_lengths{},
                                          std::make_shared<arrow_source_profile const>(profile));
    input->bind_staging(staging, *staging->resource);
    return input;
  }
};
}  // namespace

TEST_CASE("Qualified Arrow waits for the source event before reading on another stream",
          "[arrow_qualified][arrow_ownership]")
{
  using namespace std::chrono_literals;
  auto memory = GENERATE(arrow_source_memory::device,
                         arrow_source_memory::pinned_host,
                         arrow_source_memory::pageable_host);
  placed_fixture f(memory);
  f.producer.synchronize();
  sirius::test::preparation::hold pending;
  std::atomic<bool> timed_out{false};
  std::pair<sirius::test::preparation::hold*, std::atomic<bool>*> state{&pending, &timed_out};
  cuda_ok(cudaLaunchHostFunc(
    f.producer.value(),
    [](void* context) {
      auto& observed = *static_cast<decltype(state)*>(context);
      try {
        observed.first->wait();
      } catch (...) {
        *observed.second = true;
      }
    },
    &state));
  cuda_ok(cudaEventRecord(f.tree->ready, f.producer.value()));
  auto input = f.input();
  sirius::exec::retire_lane lane(f.consumer.value());
  int device = 0;
  cuda_ok(cudaGetDevice(&device));
  auto importing = std::async(std::launch::async, [&] {
    rmm::cuda_set_device_raii guard{rmm::cuda_device_id{device}};
    return import_arrow_host(input, f.program, lane, cudf::get_current_device_resource_ref());
  });
  struct release_hold {
    sirius::test::preparation::hold& hold;
    ~release_hold() { hold.release(); }
  } cleanup{pending};
  REQUIRE(pending.await());
  CHECK(importing.wait_for(30ms) == std::future_status::timeout);
  CHECK(f.tree->released->load() == 0);
  pending.release();
  REQUIRE(importing.wait_for(5s) == std::future_status::ready);
  auto output = importing.get();
  REQUIRE(lane.quiesce() == cudaSuccess);
  std::array<int32_t, 3> actual{};
  cuda_ok(cudaMemcpy(
    actual.data(), output.table.view().column(0).head(), sizeof(actual), cudaMemcpyDeviceToHost));
  CHECK(actual == std::array<int32_t, 3>{11, 22, 33});
  CHECK_FALSE(timed_out.load());
}

TEST_CASE("Arrow discard retains unfinished source writes before releasing callbacks and credits",
          "[arrow_qualified][arrow_ownership][arrow_admission]")
{
  using namespace std::chrono_literals;
  auto action = GENERATE(0, 1, 2);  // cancel, fail, abandoned offered slot
  auto memory = GENERATE(arrow_source_memory::device,
                         arrow_source_memory::pinned_host,
                         arrow_source_memory::pageable_host);
  placed_fixture f(memory);
  f.producer.synchronize();
  sirius::test::preparation::hold pending;
  std::atomic<bool> timed_out{false};
  std::pair<sirius::test::preparation::hold*, std::atomic<bool>*> blocked{&pending, &timed_out};
  cuda_ok(cudaLaunchHostFunc(
    f.producer.value(),
    [](void* context) {
      auto& state = *static_cast<decltype(blocked)*>(context);
      try {
        state.first->wait();
      } catch (...) {
        *state.second = true;
      }
    },
    &blocked));
  cuda_ok(cudaEventRecord(f.tree->ready, f.producer.value()));
  struct release_state {
    cudaEvent_t event;
    arrow_push_queue* queue = nullptr;
    std::atomic<unsigned> calls{0};
    std::atomic<cudaError_t> status{cudaErrorNotReady};
  } released{f.tree->ready};
  auto queue                 = std::make_shared<arrow_push_queue>(f.program, 65536, [](auto) {});
  released.queue             = queue.get();
  auto slot                  = queue->acquire_offered_slot();
  f.tree->array.private_data = &released;
  f.tree->array.release      = [](ArrowArray* array) {
    auto& state = *static_cast<release_state*>(array->private_data);
    (void)state.queue->observe();
    state.status = cudaEventQuery(state.event);
    ++state.calls;
    array->release = nullptr;
  };
  slot.set_input(f.tree->schema, f.tree->array, f.tree, f.profile);
  if (action != 2) REQUIRE(queue->offer(slot) == arrow_push_queue::offer_outcome::accepted);
  REQUIRE(pending.await());
  REQUIRE(cudaEventQuery(f.tree->ready) == cudaErrorNotReady);
  int device = 0;
  cuda_ok(cudaGetDevice(&device));
  auto discarding = std::async(std::launch::async, [&] {
    rmm::cuda_set_device_raii guard{rmm::cuda_device_id{device}};
    if (action == 0)
      queue->cancel();
    else if (action == 1)
      queue->fail(std::make_exception_ptr(std::runtime_error("producer failure")));
    else
      slot = {};
  });
  struct release_hold {
    sirius::test::preparation::hold& hold;
    ~release_hold() { hold.release(); }
  } cleanup{pending};
  CHECK(discarding.wait_for(30ms) == std::future_status::timeout);
  CHECK(released.calls == 0);
  CHECK(queue->observe().accepted == (action == 2 ? 0 : 1));
  CHECK(queue->observe().offered == (action == 2 ? 1 : 0));
  pending.release();
  REQUIRE(discarding.wait_for(5s) == std::future_status::ready);
  discarding.get();
  CHECK(released.calls == 1);
  CHECK(released.status == cudaSuccess);
  CHECK(queue->observe().accepted == 0);
  CHECK(queue->observe().offered == 0);
  CHECK_FALSE(timed_out.load());
}

TEST_CASE("Arrow refuses unavailable source devices before publishing a scan input",
          "[arrow_qualified][arrow_profile]")
{
  placed_fixture f(arrow_source_memory::device);
  cuda_ok(cudaGetDeviceCount(&f.profile.device));
  auto input    = f.input();
  auto metadata = std::make_shared<owned_arrow_scan_info>(
    input, 701, 4, 0, std::make_shared<physical_profile_table>(), 8192);
  try {
    scan_operator_input split(
      metadata, nullptr, 0, 0, scan_operator_input::prefetch_start::deferred);
    FAIL("unavailable source device must be refused before task publication");
  } catch (sirius::transparent::classified_execution_error const& error) {
    CHECK(error.cause == sirius::transparent::late_failure_cause::physical_input);
    CHECK(std::string(error.what()).find("required source device is unavailable") !=
          std::string::npos);
  }
}

TEST_CASE("Arrow string row limits are resource refusals after schema and extent validation",
          "[arrow_qualified][arrow_profile][arrow_admission]")
{
  auto format        = GENERATE("u", "U");
  auto rows          = GENERATE(int64_t{INT32_MAX}, int64_t{INT32_MAX} + 1);
  auto malformed     = GENERATE(0, 1, 2);  // valid, schema drift, invalid buffer extent
  auto tree          = std::make_shared<import_tree>(format, 0);
  tree->array.offset = tree->column.offset = 0;
  tree->array.length = tree->column.length = rows;
  tree->column.null_count                  = 0;
  auto base                                = reinterpret_cast<void const*>(uintptr_t{0x100000});
  auto bytes       = static_cast<uint64_t>(rows + 1) * (std::string_view(format) == "U" ? 8 : 4);
  tree->buffers[1] = base;
  arrow_source_profile profile;
  profile.memory        = arrow_source_memory::device;
  profile.device        = 0;
  profile.completed     = true;
  profile.leases        = tree;
  profile.control_bytes = sizeof(import_tree);
  profile.allocations   = {{1, base, bytes}};
  profile.buffers       = {{{}, arrow_buffer_use{1, 0, bytes}, {}}};
  profile.strings       = {arrow_string_window{0, 0}};
  materialization_program program{701, 4, {cudf::data_type{cudf::type_id::STRING}}, 0, false};
  program.input_types  = program.output_types;
  program.arrow_schema = std::make_shared<arrow_bound_schema const>(
    arrow_bound_schema{{{"value", format, cudf::data_type{cudf::type_id::STRING}, true}}});
  if (malformed == 1) tree->field.name = "wrong";
  if (malformed == 2) --profile.buffers[0][1]->bytes;
  arrow_push_queue queue(program, 64ULL << 30, [](auto) {});
  auto slot = queue.acquire_offered_slot();
  slot.set_input(tree->schema, tree->array, tree, std::move(profile));
  try {
    queue.offer(slot);
    FAIL("oversized string rows must fail without accessing the virtual payload");
  } catch (sirius::transparent::classified_execution_error const& error) {
    CHECK(error.cause == (malformed ? sirius::transparent::late_failure_cause::physical_input
                                    : sirius::transparent::late_failure_cause::resource));
    if (!malformed)
      CHECK(std::string(error.what()).find("arrow_batch_too_large") != std::string::npos);
  }
  CHECK(queue.observe().accepted == 0);
}

TEST_CASE("Arrow character limits preserve malformed extent and schema precedence",
          "[arrow_qualified][arrow_profile][arrow_admission]")
{
  auto kind = GENERATE(0, 1, 2, 3, 4, 5);  // limit, excess, high origin, empty, negative, reversed
  auto malformed =
    GENERATE(0, 1, 2, 3, 4);  // valid, short chars, missing chars, later schema/extent
  struct string_tree : import_tree {
    ArrowSchema second{};
    ArrowArray second_array{};
    ArrowSchema* all_fields[2]{&field, &second};
    ArrowArray* all_columns[2]{&column, &second_array};
    int32_t value = 1;
    void const* second_buffers[2]{nullptr, &value};
    string_tree() : import_tree("U", 1)
    {
      array.offset = column.offset = 0;
      column.length                = 1;
      column.null_count            = 0;
      schema.n_children = array.n_children = 2;
      schema.children                      = all_fields;
      array.children                       = all_columns;
      second                               = field;
      second.name                          = "second";
      second.format                        = "i";
      second_array                         = column;
      second_array.n_buffers               = 2;
      second_array.buffers                 = second_buffers;
    }
  };
  auto tree     = std::make_shared<string_tree>();
  int64_t first = kind == 2 ? int64_t{INT32_MAX} + 17 : 0;
  int64_t last  = kind == 0 ? INT32_MAX : kind == 1 ? int64_t{INT32_MAX} + 1 : first + 7;
  if (kind == 3) {
    tree->array.length = tree->column.length = tree->second_array.length = 0;
    first = last = 0;
  }
  if (kind == 4) first = -1;
  if (kind == 5) first = last + 1;
  std::array<int64_t, 2> offsets{first, last};
  auto chars       = reinterpret_cast<void const*>(uintptr_t{0x1000000000});
  auto bytes       = static_cast<uint64_t>(std::max<int64_t>(last, 1));
  tree->buffers[1] = offsets.data();
  tree->buffers[2] = chars;
  arrow_source_profile profile;
  profile.completed     = true;
  profile.leases        = tree;
  profile.control_bytes = sizeof(string_tree);
  profile.allocations   = {{1, offsets.data(), sizeof(offsets)},
                           {2, chars, bytes},
                           {3, &tree->value, sizeof(tree->value)}};
  profile.buffers = {{{}, arrow_buffer_use{1, 0, sizeof(offsets)}, arrow_buffer_use{2, 0, bytes}},
                     {{}, arrow_buffer_use{3, 0, sizeof(tree->value)}}};
  profile.strings = {arrow_string_window{first, last}, {}};
  if (malformed == 1) {
    profile.buffers[0][2]->bytes  = 0;
    profile.buffers[0][2]->offset = 1;
  }
  if (malformed == 2)
    tree->buffers[2] = nullptr;  // A use without a pointer is malformed even when empty.
  if (malformed == 3) tree->second.name = "wrong";
  if (malformed == 4) profile.buffers[1][1]->offset = 1;
  materialization_program program{
    701,
    4,
    {cudf::data_type{cudf::type_id::STRING}, cudf::data_type{cudf::type_id::INT32}},
    0,
    false};
  program.input_types  = program.output_types;
  program.arrow_schema = std::make_shared<arrow_bound_schema const>(
    arrow_bound_schema{{{"value", "U", cudf::data_type{cudf::type_id::STRING}, true},
                        {"second", "i", cudf::data_type{cudf::type_id::INT32}, true}}});
  bool valid = !malformed && kind < 4;
  if (valid && kind != 1)
    CHECK_NOTHROW(validate_arrow_source(tree->schema, tree->array, profile, *program.arrow_schema));
  // Small windows are refused by the byte budget, so virtual chars are never materialized.
  arrow_push_queue queue(program, kind == 1 ? 64ULL << 30 : 1, [](auto) {});
  auto slot = queue.acquire_offered_slot();
  slot.set_input(tree->schema, tree->array, tree, std::move(profile));
  try {
    queue.offer(slot);
    FAIL("character admission must refuse before payload access");
  } catch (sirius::transparent::classified_execution_error const& error) {
    CHECK(error.cause == (valid ? sirius::transparent::late_failure_cause::resource
                                : sirius::transparent::late_failure_cause::physical_input));
    if (valid) CHECK(std::string(error.what()).find("arrow_batch_too_large") != std::string::npos);
  }
  CHECK(queue.observe().accepted == 0);
}

TEST_CASE("Qualified Arrow rejects actual placement drift before importing payload",
          "[arrow_qualified][arrow_profile]")
{
  auto memory = GENERATE(arrow_source_memory::device,
                         arrow_source_memory::pinned_host,
                         arrow_source_memory::pageable_host);
  placed_fixture f(memory);
  f.profile.memory = memory == arrow_source_memory::pageable_host
                       ? arrow_source_memory::pinned_host
                       : arrow_source_memory::pageable_host;
  f.profile.device = -1;
  auto input       = f.input();
  sirius::exec::retire_lane lane(f.consumer.value());
  CHECK_THROWS_AS(
    import_arrow_host(input, f.program, lane, cudf::get_current_device_resource_ref()),
    std::invalid_argument);
  REQUIRE(lane.quiesce() == cudaSuccess);
}

TEST_CASE("Qualified device Arrow rejects malformed payload after readiness without host staging",
          "[arrow_qualified][arrow_profile]")
{
  auto mode = GENERATE(0, 1, 2);  // descending offsets, window mismatch, decimal overflow
  placed_fixture f(arrow_source_memory::device);
  f.producer.synchronize();
  auto format = mode == 2 ? "d:9,0" : "U";
  auto type   = cudf::data_type{mode == 2 ? cudf::type_id::DECIMAL128 : cudf::type_id::STRING};
  f.tree->field.format     = format;
  f.tree->column.n_buffers = mode == 2 ? 2 : 3;
  std::array<int64_t, 6> offsets{0, 0, 0, mode == 0 ? 2 : 1, 1, 3};
  std::array<__int128_t, 5> decimals{0, 0, 1, 1000000000, 2};
  f.payload.resize(mode == 2 ? sizeof(decimals) : sizeof(offsets));
  std::memcpy(f.payload.data(),
              mode == 2 ? static_cast<void*>(decimals.data()) : offsets.data(),
              f.payload.size());
  auto values =
    std::make_shared<placed_buffer>(f.payload, arrow_source_memory::device, f.producer.value());
  f.tree->storage       = {values};
  f.tree->buffers[1]    = values->data;
  f.profile.allocations = {{1, values->data, values->bytes}};
  f.profile.buffers     = {{{}, arrow_buffer_use{1, 0, values->bytes}}};
  std::vector<uint8_t> chars{'a', 'b', 'c'};
  if (mode != 2) {
    auto characters =
      std::make_shared<placed_buffer>(chars, arrow_source_memory::device, f.producer.value());
    f.tree->storage.push_back(characters);
    f.tree->buffers[2] = characters->data;
    f.profile.allocations.push_back({2, characters->data, characters->bytes});
    f.profile.buffers[0].push_back(arrow_buffer_use{2, 0, characters->bytes});
    f.profile.strings[0] = arrow_string_window{mode == 1 ? 1 : 0, 3};
  } else
    f.program.decimal_precisions = {9};
  f.program.input_types = f.program.output_types = {type};
  f.program.arrow_schema =
    std::make_shared<arrow_bound_schema const>(arrow_bound_schema{{{"value", format, type, true}}});
  cuda_ok(cudaEventRecord(f.tree->ready, f.producer.value()));
  auto input = f.input();
  CHECK(input->estimate().staging == 0);
  sirius::exec::retire_lane lane(f.consumer.value());
  CHECK_THROWS_AS(
    import_arrow_host(input, f.program, lane, cudf::get_current_device_resource_ref()),
    std::invalid_argument);
  REQUIRE(lane.quiesce() == cudaSuccess);
}

TEST_CASE("Device Arrow carries source affinity and refuses import on another GPU",
          "[arrow_qualified][multi_gpu]")
{
  int devices = 0;
  cuda_ok(cudaGetDeviceCount(&devices));
  if (devices < 2) SKIP("requires two GPUs");
  rmm::cuda_set_device_raii source_device{rmm::cuda_device_id{1}};
  placed_fixture f(arrow_source_memory::device);
  auto input    = f.input();
  auto profiles = std::make_shared<physical_profile_table>();
  auto metadata = std::make_unique<owned_arrow_scan_info>(input, 701, 4, 0, profiles, 8192);
  scan_operator_input split(
    std::move(metadata), nullptr, 0, 0, scan_operator_input::prefetch_start::deferred);
  CHECK(split.get_preferred_device_id() == 1);
  {
    rmm::cuda_set_device_raii other_device{rmm::cuda_device_id{0}};
    rmm::cuda_stream stream;
    sirius::exec::retire_lane lane(stream.value());
    CHECK_THROWS_AS(
      import_arrow_host(input, f.program, lane, cudf::get_current_device_resource_ref()),
      std::invalid_argument);
    REQUIRE(lane.quiesce() == cudaSuccess);
  }
  sirius::exec::retire_lane lane(f.consumer.value());
  auto result = import_arrow_host(input, f.program, lane, cudf::get_current_device_resource_ref());
  REQUIRE(lane.quiesce() == cudaSuccess);
  CHECK(result.table.num_rows() == 3);
}

TEST_CASE("Qualified Arrow retains zero-column cardinality on every placement",
          "[arrow_qualified][arrow_profile]")
{
  auto rows   = GENERATE(0, 3);
  auto memory = GENERATE(arrow_source_memory::device,
                         arrow_source_memory::pinned_host,
                         arrow_source_memory::pageable_host);
  rmm::cuda_stream stream;
  auto tree               = std::make_shared<import_tree>("i", rows);
  tree->schema.n_children = tree->array.n_children = 0;
  tree->schema.children                            = nullptr;
  tree->array.children                             = nullptr;
  arrow_source_profile profile;
  profile.memory = memory;
  if (memory == arrow_source_memory::device) cuda_ok(cudaGetDevice(&profile.device));
  profile.completed     = true;
  profile.control_bytes = sizeof(import_tree);
  profile.leases        = tree;
  materialization_program program{701, 4, {}, 0, false};
  program.input_types  = program.output_types;
  program.arrow_schema = std::make_shared<arrow_bound_schema const>();
  auto source          = std::make_shared<arrow_source_profile const>(std::move(profile));
  auto staging         = std::make_shared<pinned_arena>(1);
  auto input           = std::make_shared<owned_arrow_batch>(
    tree->schema, tree->array, tree, *staging->resource, arrow_metadata_lengths{}, source);
  input->bind_staging(staging, *staging->resource);
  sirius::exec::retire_lane lane(stream.value());
  auto result = import_arrow_host(input, program, lane, cudf::get_current_device_resource_ref());
  REQUIRE(lane.quiesce() == cudaSuccess);
  CHECK(result.rows == rows);
  CHECK(result.table.num_rows() == rows);
  CHECK(result.table.n_columns() == (rows ? 1 : 0));
  CHECK(result.row_count_carrier == bool(rows));
}

TEST_CASE("Qualified pageable Arrow imports a proven slice beyond 32-bit source offsets",
          "[arrow_qualified][arrow_profile]")
{
#if defined(__linux__)
  struct virtual_parent {
    uint64_t bytes;
    void* data;
    explicit virtual_parent(uint64_t size)
      : bytes(size),
        data(mmap(nullptr,
                  size,
                  PROT_READ | PROT_WRITE,
                  MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE,
                  -1,
                  0))
    {
      if (data == MAP_FAILED) throw std::bad_alloc();
    }
    ~virtual_parent() { munmap(data, bytes); }
  };
  constexpr int64_t offset = int64_t{INT32_MAX} + 1;
  auto parent              = std::make_shared<virtual_parent>((offset + 3) * sizeof(int32_t));
  std::array<int32_t, 3> expected{11, 22, 33};
  std::memcpy(static_cast<int32_t*>(parent->data) + offset, expected.data(), sizeof(expected));
  auto tree               = std::make_shared<import_tree>("i", 3);
  tree->array.offset      = 0;
  tree->column.offset     = offset;
  tree->column.length     = 3;
  tree->column.null_count = 0;
  tree->buffers[1]        = parent->data;
  arrow_source_profile profile;
  profile.completed     = true;
  profile.control_bytes = sizeof(import_tree);
  profile.leases        = parent;
  profile.allocations   = {{1, parent->data, parent->bytes}};
  profile.buffers       = {{{}, arrow_buffer_use{1, 0, parent->bytes}}};
  profile.strings.resize(1);
  materialization_program program{701, 4, {cudf::data_type{cudf::type_id::INT32}}, 0, false};
  program.input_types  = program.output_types;
  program.arrow_schema = std::make_shared<arrow_bound_schema const>(
    arrow_bound_schema{{{"value", "i", cudf::data_type{cudf::type_id::INT32}, true}}});
  auto staging = std::make_shared<pinned_arena>(4096);
  auto input   = std::make_shared<owned_arrow_batch>(
    tree->schema,
    tree->array,
    tree,
    *staging->resource,
    arrow_metadata_lengths{},
    std::make_shared<arrow_source_profile const>(std::move(profile)));
  input->bind_staging(staging, *staging->resource);
  CHECK(input->estimate().retained == parent->bytes);
  rmm::cuda_stream stream;
  sirius::exec::retire_lane lane(stream.value());
  auto result = import_arrow_host(input, program, lane, cudf::get_current_device_resource_ref());
  REQUIRE(lane.quiesce() == cudaSuccess);
  std::array<int32_t, 3> actual{};
  cuda_ok(cudaMemcpy(
    actual.data(), result.table.view().column(0).head(), sizeof(actual), cudaMemcpyDeviceToHost));
  CHECK(actual == expected);
#else
  SKIP("requires sparse virtual memory mappings");
#endif
}

TEST_CASE("Qualified Arrow imports the full scalar and string matrix from every placement",
          "[arrow_profile][arrow_qualified][materialization]")
{
  using T           = cudf::type_id;
  auto [format, id] = GENERATE(table<char const*, T>({{"b", T::BOOL8},
                                                      {"c", T::INT8},
                                                      {"s", T::INT16},
                                                      {"i", T::INT32},
                                                      {"l", T::INT64},
                                                      {"C", T::UINT8},
                                                      {"S", T::UINT16},
                                                      {"I", T::UINT32},
                                                      {"L", T::UINT64},
                                                      {"f", T::FLOAT32},
                                                      {"g", T::FLOAT64},
                                                      {"tdD", T::TIMESTAMP_DAYS},
                                                      {"tss:", T::TIMESTAMP_SECONDS},
                                                      {"tsm:", T::TIMESTAMP_MILLISECONDS},
                                                      {"tsu:", T::TIMESTAMP_MICROSECONDS},
                                                      {"tsn:", T::TIMESTAMP_NANOSECONDS},
                                                      {"d:9,-3", T::DECIMAL128},
                                                      {"d:9,10,128", T::DECIMAL128},
                                                      {"u", T::STRING},
                                                      {"U", T::STRING}}));
  auto memory       = GENERATE(arrow_source_memory::device,
                         arrow_source_memory::pinned_host,
                         arrow_source_memory::pageable_host);
  auto rows         = GENERATE(0, 3);
  CAPTURE(format, int(memory), rows);
  rmm::cuda_stream producer, consumer;
  auto tree    = std::make_shared<import_tree>(format, rows);
  auto type    = id == T::DECIMAL128
                   ? cudf::data_type{id, std::string_view(format) == "d:9,-3" ? 3 : -10}
                   : cudf::data_type{id};
  bool strings = id == T::STRING;
  auto width   = strings ? 1 : cudf::size_of(type);
  std::vector<uint8_t> values(5 * width, 0x55), offsets, chars{'a', 'b', 'c', 'd', 'e', 'f'};
  std::array<uint8_t, 16> first{}, last{};
  int64_t a = -11, b = 33;
  std::memcpy(first.data(), &a, std::min<size_t>(8, width));
  std::memcpy(last.data(), &b, std::min<size_t>(8, width));
  if (id == T::FLOAT32) {
    float a = -1.25f, b = 3.5f;
    std::memcpy(first.data(), &a, 4);
    std::memcpy(last.data(), &b, 4);
  }
  if (id == T::FLOAT64) {
    double a = -1.25, b = 3.5;
    std::memcpy(first.data(), &a, 8);
    std::memcpy(last.data(), &b, 8);
  }
  if (id == T::DECIMAL128) {
    __int128_t a = 123, b = -456;
    std::memcpy(first.data(), &a, 16);
    std::memcpy(last.data(), &b, 16);
  }
  std::memcpy(values.data() + 2 * width, first.data(), width);
  std::memcpy(values.data() + 4 * width, last.data(), width);
  if (id == T::BOOL8) {
    values   = {0b00001100};
    first[0] = 1;
    last[0]  = 0;
  }
  if (strings) {
    std::array<int32_t, 6> small{0, 0, 0, 1, 3, 6};
    std::array<int64_t, 6> large{0, 0, 0, 1, 3, 6};
    offsets.resize(std::string_view(format) == "U" ? sizeof(large) : sizeof(small));
    std::memcpy(offsets.data(),
                std::string_view(format) == "U" ? static_cast<void*>(large.data()) : small.data(),
                offsets.size());
    values = offsets;
  }
  std::vector<uint8_t> validity{0b00010100};
  arrow_source_profile profile;
  profile.memory = memory;
  if (memory == arrow_source_memory::device) cuda_ok(cudaGetDevice(&profile.device));
  profile.leases        = tree;
  profile.control_bytes = sizeof(import_tree) + 256;
  profile.buffers.resize(1);
  profile.strings.resize(1);
  auto place = [&](std::vector<uint8_t> const& input) {
    auto buffer          = std::make_shared<placed_buffer>(input, memory, producer.value());
    auto index           = tree->storage.size();
    tree->buffers[index] = buffer->data;
    profile.allocations.push_back({index + 1, buffer->data, buffer->bytes});
    profile.buffers[0].push_back(arrow_buffer_use{index + 1, 0, buffer->bytes});
    tree->storage.push_back(std::move(buffer));
  };
  place(validity);
  place(values);
  if (strings) {
    place(chars);
    profile.strings[0] = arrow_string_window{0, rows ? 6 : 0};
  }
  cuda_ok(cudaEventCreateWithFlags(&tree->ready, cudaEventDisableTiming));
  cuda_ok(cudaEventRecord(tree->ready, producer.value()));
  profile.ready_event = tree->ready;
  materialization_program program{701, 4, {type}, 0, false};
  program.input_types = program.output_types;
  if (id == T::DECIMAL128) program.decimal_precisions = {9};
  program.arrow_schema =
    std::make_shared<arrow_bound_schema const>(arrow_bound_schema{{{"value", format, type, true}}});
  auto source   = std::make_shared<arrow_source_profile const>(std::move(profile));
  auto estimate = validate_arrow_source(tree->schema, tree->array, *source, *program.arrow_schema);
  auto staging  = std::make_shared<pinned_arena>(std::max<uint64_t>(1, estimate.staging));
  auto input    = std::make_shared<owned_arrow_batch>(
    tree->schema, tree->array, tree, *staging->resource, arrow_metadata_lengths{}, source);
  input->bind_staging(staging, *staging->resource);
  auto releases = tree->released;
  sirius::exec::retire_lane lane(consumer.value());
  auto output = import_arrow_host(input, program, lane, cudf::get_current_device_resource_ref());
  auto column = output.table.view().column(0);
  CHECK(column.type() == type);
  CHECK(column.size() == rows);
  CHECK(column.null_count() == (rows ? 1 : 0));
  if (rows && !strings) {
    std::vector<uint8_t> actual(3 * width);
    cuda_ok(cudaMemcpyAsync(
      actual.data(), column.head(), actual.size(), cudaMemcpyDeviceToHost, consumer.value()));
    consumer.synchronize();
    CHECK(std::memcmp(actual.data(), first.data(), width) == 0);
    CHECK(std::memcmp(actual.data() + 2 * width, last.data(), width) == 0);
    CHECK(column.head() != tree->buffers[1]);
  } else if (rows) {
    cudf::strings_column_view strings(column);
    std::array<int32_t, 4> actual_offsets{};
    std::array<char, 6> actual_chars{};
    cuda_ok(cudaMemcpyAsync(actual_offsets.data(),
                            strings.offsets().head(),
                            sizeof(actual_offsets),
                            cudaMemcpyDeviceToHost,
                            consumer.value()));
    cuda_ok(cudaMemcpyAsync(actual_chars.data(),
                            strings.chars_begin(consumer),
                            actual_chars.size(),
                            cudaMemcpyDeviceToHost,
                            consumer.value()));
    consumer.synchronize();
    CHECK(actual_offsets == std::array<int32_t, 4>{0, 1, 3, 6});
    CHECK(std::string(actual_chars.data(), actual_chars.size()) == "abcdef");
    CHECK(strings.offsets().head() != tree->buffers[1]);
  }
  source.reset();
  tree.reset();
  input.reset();
  CHECK(*releases == 0);
  REQUIRE(lane.quiesce() == cudaSuccess);
  CHECK(*releases == 1);
  CHECK(output.table.num_rows() == rows);  // Output remains independent after source retirement.
  output.table = {};
  REQUIRE(lane.quiesce() == cudaSuccess);
}
