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

#include "catch.hpp"
#include "op/scan/arrow_materializer.hpp"
#include "pipeline/gpu_pipeline_task.hpp"
#include "pipeline/repository_wiring.hpp"
#include "pipeline/sirius_pipeline.hpp"
#include "scan_manager/preparation_fixture.hpp"
#include "utils/telemetry_utils.hpp"

#include <rmm/cuda_device.hpp>
#include <rmm/cuda_stream.hpp>
#include <rmm/mr/cuda_async_memory_resource.hpp>

#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <memory_resource>

using namespace sirius::scan_manager;
using namespace sirius::test::preparation;
using namespace std::chrono_literals;

namespace {
void require_cuda(cudaError_t status)
{
  if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}

// The callback touches host synchronization only, never CUDA or Catch.
struct gpu_read_hold {
  std::mutex mutex;
  std::condition_variable cv;
  bool entered = false, released = false, timed_out = false;
  static void CUDART_CB wait(void* pointer) noexcept
  {
    auto& self = *static_cast<gpu_read_hold*>(pointer);
    std::unique_lock lock(self.mutex);
    self.entered = true;
    self.cv.notify_all();
    self.timed_out = !self.cv.wait_for(lock, 5s, [&] { return self.released; });
  }
  bool await()
  {
    std::unique_lock lock(mutex);
    return cv.wait_for(lock, 5s, [&] { return entered; });
  }
  void release()
  {
    std::lock_guard lock(mutex);
    released = true;
    cv.notify_all();
  }
};

// The reader detaches storage before publishing; the same fixture feeds Arrow and raw GPU reads.
struct detached_input : tagged {
  std::shared_ptr<charged_block const> storage;
  detached_input(int id, size_t bytes, std::shared_ptr<charged_block const> root)
    : tagged(id, bytes), storage(std::move(root))
  {
  }
};
struct detached_coalescer : accumulator {
  detached_coalescer() : accumulator(16) {}
  cursor_step advance(sirius::op::scan::scan_info& input, size_t& cursor, size_t) override
  {
    auto& data = dynamic_cast<detached_input&>(input);
    cursor += 16;
    return {std::make_unique<detached_input>(static_cast<int>(cursor / 16), 16, data.storage),
            cursor == data.rows};
  }
  std::vector<std::unique_ptr<sirius::op::scan::scan_info>> flush() override { return {}; }
};
struct detached_source_state {
  std::unique_ptr<preparation_ledger> ledger;
  std::array<std::byte, 32> reader_buffer;
  std::weak_ptr<charged_block const> output;
  std::atomic<uint64_t> charge_at_close{0};
  explicit detached_source_state(reservation_provider& provider)
    : ledger(std::make_unique<preparation_ledger>(provider))
  {
    reader_buffer.fill(std::byte{0x2a});
    std::array envelope{scan_envelope{{{{8, 0, 0}, 32}}, {}, true}};
    REQUIRE(ledger->admit(envelope, memory_space_id(cucascade::memory::Tier::HOST, 0)).deferred);
  }
};
preparation_coordinator::source detached_source(std::shared_ptr<detached_source_state> storage,
                                                std::shared_ptr<sink> out,
                                                preparation_coordinator::execution_domain domain)
{
  preparation_coordinator::source source;
  source.execution = domain;
  source.coalescer = std::make_shared<detached_coalescer>();
  source.try_claim = [storage, claimed = false]() mutable -> preparation_coordinator::claim_result {
    if (std::exchange(claimed, true)) return preparation_exhausted{};
    return preparation_coordinator::job{[storage] {
      auto permit = storage->ledger->try_acquire_unit({77, 1}, 32);
      if (!permit) throw std::runtime_error("test output admission failed");
      auto block = storage->ledger->allocator(permit).allocate_retained(32);
      std::copy(storage->reader_buffer.begin(), storage->reader_buffer.end(), block.data());
      auto root       = std::make_shared<charged_block const>(std::move(block));
      storage->output = root;
      // The allocation itself must now retain its account and backing.
      storage->ledger->retire_unit({77, 1});
      return std::make_unique<detached_input>(0, 32, std::move(root));
    }};
  };
  source.construct = [](auto batch) { return make_publication(std::move(batch), 16); };
  source.close     = [storage, out] {
    storage->charge_at_close = storage->ledger->charged_bytes({77, 1});
    storage->ledger.reset();
    storage->reader_buffer.fill(std::byte{0xee});
    out->close();
  };
  return source;
}
std::shared_ptr<detached_input> detached_metadata(sirius::op::operator_data& value)
{
  auto& input = dynamic_cast<sirius::op::scan::scan_operator_input&>(value);
  return std::dynamic_pointer_cast<detached_input>(
    std::get<std::shared_ptr<sirius::op::scan::scan_info>>(input.materialization_info));
}
struct arrow_releases {
  std::atomic<unsigned> arrays{0}, schemas{0};
};

// The capacity owner holds the complete C Data tree and its charged value buffer.
// Release callbacks own no reader or statement state and may run after source close.
struct arrow_storage {
  std::shared_ptr<charged_block const> backing;
  std::shared_ptr<arrow_releases> released;
  std::array<std::byte, 1024> staging_storage;
  std::pmr::monotonic_buffer_resource staging{
    staging_storage.data(), staging_storage.size(), std::pmr::null_memory_resource()};
  ArrowSchema schema{}, child_schema{};
  ArrowArray array{}, child{};
  ArrowSchema* schemas[1]{&child_schema};
  ArrowArray* children[1]{&child};
  void const* buffers[2]{};
  void const* root_buffers[1]{};
  std::array<char, 4> root_metadata{}, child_metadata{};  // zero metadata entries

  arrow_storage(std::shared_ptr<charged_block const> block,
                size_t offset,
                std::shared_ptr<arrow_releases> observation)
    : backing(std::move(block)), released(std::move(observation))
  {
    child_schema.format   = "i";
    child_schema.metadata = child_metadata.data();
    child_schema.release  = [](ArrowSchema* value) { value->release = nullptr; };
    schema.format         = "+s";
    schema.metadata       = root_metadata.data();
    schema.n_children     = 1;
    schema.children       = schemas;
    schema.private_data   = released.get();
    schema.release        = [](ArrowSchema* value) {
      value->children[0]->release(value->children[0]);
      ++static_cast<arrow_releases*>(value->private_data)->schemas;
      value->release = nullptr;
    };
    buffers[1]         = backing->data() + offset;
    child.length       = 4;
    child.n_buffers    = 2;
    child.buffers      = buffers;
    child.release      = [](ArrowArray* value) { value->release = nullptr; };
    array.length       = 4;
    array.n_buffers    = 1;
    array.buffers      = root_buffers;
    array.n_children   = 1;
    array.children     = children;
    array.private_data = released.get();
    array.release      = [](ArrowArray* value) {
      value->children[0]->release(value->children[0]);
      ++static_cast<arrow_releases*>(value->private_data)->arrays;
      value->release = nullptr;
    };
  }
};
struct arrow_input : tagged {
  std::shared_ptr<sirius::op::scan::owned_arrow_batch> batch;
  arrow_input(int id, std::shared_ptr<sirius::op::scan::owned_arrow_batch> value)
    : tagged(id, 16), batch(std::move(value))
  {
  }
};
auto arrow_constructor(std::shared_ptr<arrow_releases> releases, bool invalid_second = false)
{
  return [releases = std::move(releases),
          invalid_second](std::unique_ptr<sirius::op::scan::scan_info> input) {
    auto& data = dynamic_cast<detached_input&>(*input);
    auto owner = std::make_shared<arrow_storage>(data.storage, (data.id - 1) * 16, releases);
    sirius::op::scan::arrow_metadata_lengths lengths{
      {owner->schema.metadata, owner->root_metadata.size()},
      {owner->child_schema.metadata, owner->child_metadata.size()}};
    if (invalid_second && data.id == 2) lengths.erase(owner->child_schema.metadata);
    auto batch = std::make_shared<sirius::op::scan::owned_arrow_batch>(
      owner->schema, owner->array, owner, owner->staging, std::move(lengths));
    return make_publication(std::make_unique<arrow_input>(data.id, std::move(batch)), 16);
  };
}
std::shared_ptr<arrow_input> arrow_metadata(sirius::op::operator_data& value)
{
  auto& input = dynamic_cast<sirius::op::scan::scan_operator_input&>(value);
  return std::dynamic_pointer_cast<arrow_input>(
    std::get<std::shared_ptr<sirius::op::scan::scan_info>>(input.materialization_info));
}

// Delay the import's real CUDA completion callback, without replacing the importer
// or its retirement state. cuDF may synchronize internally before committing it.
thread_local gpu_read_hold* import_gate = nullptr;
cudaError_t gated_import_callback(cudaStream_t stream,
                                  cudaStreamCallback_t callback,
                                  void* data,
                                  unsigned flags)
{
  auto status = cudaLaunchHostFunc(stream, gpu_read_hold::wait, import_gate);
  return status == cudaSuccess ? cudaStreamAddCallback(stream, callback, data, flags) : status;
}
struct import_gate_scope {
  explicit import_gate_scope(gpu_read_hold& gate) { import_gate = &gate; }
  ~import_gate_scope() { import_gate = nullptr; }
};
}  // namespace

TEST_CASE("Arrow reader outputs retain one backing through publication and import retirement",
          "[scan_preparation][coordinator][output_lifetime][arrow_output_lifetime]")
{
  using namespace sirius::op::scan;
  using domain = preparation_coordinator::execution_domain;
  enum class outcome { success, cancel_queued, cancel_import, invalid_metadata };
  auto execution = GENERATE(domain::query_owner, domain::native_reader);
  auto result    = GENERATE(
    outcome::success, outcome::cancel_queued, outcome::cancel_import, outcome::invalid_metadata);
  CAPTURE(execution, result);
  auto pool_workers = GENERATE(0u, 2u);
  coordinator_pool coordinators(pool_workers);
  native_reader_executor native(1, 1);
  fixture f(1, 1, 5s, std::nullopt, &native, pool_workers ? &coordinators : nullptr);
  test::test_reservation_provider provider;
  auto storage         = std::make_shared<detached_source_state>(provider);
  auto releases        = std::make_shared<arrow_releases>();
  auto out             = std::make_shared<sink>();
  auto full_window     = std::make_shared<std::promise<size_t>>();
  auto observed_window = full_window->get_future();
  auto guard           = f.shutdown_guard();
  f.start([&] {
    auto source      = detached_source(storage, out, execution);
    source.construct = arrow_constructor(releases, result == outcome::invalid_metadata);
    f.register_source(std::move(source), out);
    preparation_coordinator::source probe;
    probe.coalescer = std::make_shared<detached_coalescer>();
    probe.try_claim = [out, full_window]() -> preparation_coordinator::claim_result {
      auto count = out->count();
      if (!count) return preparation_pending{};
      full_window->set_value(count);
      return preparation_exhausted{};
    };
    probe.construct = [](auto) -> preparation_coordinator::publication {
      throw std::logic_error("observer must not publish");
    };
    probe.close = [] {};
    f.register_source(std::move(probe), std::make_shared<sink>());
  });
  REQUIRE(observed_window.wait_for(1s) == std::future_status::ready);
  REQUIRE(observed_window.get() == 1);
  auto first = out->connector->get_next_split();
  REQUIRE(first);
  REQUIRE(out->await(2));
  if (result == outcome::cancel_queued) {
    f.coordinator.request_stop(stop_reason::user_cancel);
    f.finish();
  }
  auto second = out->connector->get_next_split();
  REQUIRE(second);
  REQUIRE(out->await_closed());
  f.finish();
  REQUIRE_FALSE(f.owner_error);
  CHECK(out->close_calls == 1);
  CHECK(storage->ledger == nullptr);
  CHECK(storage->charge_at_close == 32);
  CHECK(storage->reader_buffer[0] == std::byte{0xee});
  CHECK(provider.seen->allocations == 1);
  CHECK(provider.seen->requests == 1);

  auto a = arrow_metadata(**first);
  auto b = arrow_metadata(**second);
  REQUIRE(a);
  REQUIRE(b);
  first.reset();
  second.reset();
  CHECK(provider.seen->allocated_bytes == 32);  // Pop returned lookahead, not the backing.
  if (result == outcome::cancel_queued) {
    a.reset();
    CHECK(provider.seen->allocated_bytes == 32);
    b.reset();
  } else {
    rmm::mr::cuda_async_memory_resource output_resource;
    rmm::cuda_stream first_stream, second_stream;
    std::array<materialized_batch, 2> outputs;
    std::array<gpu_read_hold, 2> holds;
    sirius::exec::detail::completion_cuda_api api;
    api.add_callback = &gated_import_callback;
    sirius::exec::retire_lane first_lane(first_stream.value(), api);
    sirius::exec::retire_lane second_lane(second_stream.value(), api);
    // Unblock callbacks, destroy outputs, then cover queued frees before releasing the MR.
    auto release_on_exit = std::shared_ptr<void>(nullptr, [&](void*) noexcept {
      for (auto& hold : holds)
        hold.release();
      outputs     = {};
      auto first  = first_lane.quiesce();
      auto second = second_lane.quiesce();
      if (first != cudaSuccess || second != cudaSuccess) std::terminate();
    });
    materialization_program program{77, 1, {cudf::data_type{cudf::type_id::INT32}}, 0, false};
    program.input_types = program.output_types;
    auto import         = [&](size_t i, auto const& input, auto& lane) {
      import_gate_scope scope(holds[i]);
      return import_arrow_host(
        input->batch, program, lane, rmm::device_async_resource_ref{output_resource});
    };
    outputs[0] = import(0, a, first_lane);
    REQUIRE(holds[0].await());
    a.reset();
    CHECK(first_lane.drain() == 0);
    CHECK(releases->arrays == 0);
    CHECK(provider.seen->allocated_bytes == 32);
    holds[0].release();
    REQUIRE(first_lane.quiesce() == cudaSuccess);
    CHECK(releases->arrays == 1);
    CHECK(provider.seen->allocated_bytes == 32);  // The second published slice still owns it.

    if (result == outcome::invalid_metadata) {
      CHECK_THROWS_AS(import(1, b, second_lane), std::invalid_argument);
      CHECK(second_lane.idle());
      b.reset();
    } else {
      outputs[1] = import(1, b, second_lane);
      REQUIRE(holds[1].await());
      b.reset();  // Only the importer now retains the last published input.
      if (result == outcome::cancel_import) f.coordinator.request_stop(stop_reason::user_cancel);
      CHECK(second_lane.drain() == 0);
      CHECK(releases->arrays == 1);
      CHECK(provider.seen->allocated_bytes == 32);
      holds[1].release();
      REQUIRE(second_lane.quiesce() == cudaSuccess);
    }
    // GPU output remains usable after reader, descriptors and host capacity are gone.
    CHECK(storage->output.expired());
    CHECK(provider.seen->allocated_bytes == 0);
    for (auto& output : outputs) {
      if (!output.rows) continue;
      materialization_steps steps;
      steps.projection = [](materialized_batch&) {};  // The only column is the bound output.
      auto finished    = finish_materialization(std::move(output), program, steps);
      std::array<int32_t, 4> values{};
      require_cuda(cudaMemcpy(values.data(),
                              finished.table.view().column(0).head<int32_t>(),
                              sizeof(values),
                              cudaMemcpyDeviceToHost));
      CHECK(std::all_of(values.begin(), values.end(), [](auto v) { return v == 0x2a2a2a2a; }));
    }
    release_on_exit.reset();
    for (auto& hold : holds)
      CHECK_FALSE(hold.timed_out);
  }
  CHECK(releases->arrays == 2);
  CHECK(releases->schemas == 2);
  CHECK(storage->output.expired());
  CHECK(provider.seen->allocated_bytes == 0);
  CHECK(provider.outstanding() == 0);
  CHECK(f.completion.has_error() ==
        (result == outcome::cancel_queued || result == outcome::cancel_import));
  CHECK(f.coordinator.snapshot().phase == preparation_coordinator::lifecycle::quiescent);
}

TEST_CASE("Unpublished Arrow outputs release backing on cancellation and construction failure",
          "[scan_preparation][coordinator][output_lifetime][arrow_output_lifetime]")
{
  using domain              = preparation_coordinator::execution_domain;
  auto execution            = GENERATE(domain::query_owner, domain::native_reader);
  auto construction_failure = GENERATE(false, true);
  native_reader_executor native(1, 1);
  fixture f(1, 1, 5s, std::nullopt, &native);
  test::test_reservation_provider provider;
  auto storage  = std::make_shared<detached_source_state>(provider);
  auto out      = std::make_shared<sink>();
  auto releases = std::make_shared<arrow_releases>();
  if (!construction_failure) {
    f.coordinator.before_publication_for_testing(
      [&] { f.coordinator.request_stop(stop_reason::user_cancel); });
  }
  auto guard = f.shutdown_guard();
  f.start([&] {
    auto source      = detached_source(storage, out, execution);
    source.construct = [construct = arrow_constructor(releases), construction_failure](auto input) {
      auto output = construct(std::move(input));
      if (construction_failure) throw std::runtime_error("output construction failed");
      return output;
    };
    f.register_source(std::move(source), out);
  });
  REQUIRE(out->await_closed());
  f.finish();
  CHECK(f.completion.has_error());
  if (construction_failure) CHECK(f.completion.failure().detail == "output construction failed");
  CHECK(out->count() == 0);
  CHECK(releases->arrays == 1);
  CHECK(releases->schemas == 1);
  CHECK(out->close_calls == 1);
  CHECK(storage->ledger == nullptr);
  CHECK(storage->output.expired());
  CHECK(provider.seen->allocations == 1);
  CHECK(provider.seen->allocated_bytes == 0);
  CHECK(provider.outstanding() == 0);
  CHECK(f.coordinator.snapshot().phase == preparation_coordinator::lifecycle::quiescent);
}

namespace {
struct gpu_read_observation {
  cudaEvent_t finished = nullptr;
  std::promise<void> submitted;
  std::atomic<bool> input_released{false};
  std::atomic<int> release_status{cudaErrorUnknown};
};

// Decorate the real connector input rather than retaining another descriptor.
// The event observes when production compute_task destroys its input owner.
struct gpu_reader_input : sirius::op::operator_data {
  std::unique_ptr<sirius::op::operator_data> input;
  std::shared_ptr<gpu_read_observation> observed;
  gpu_reader_input(std::unique_ptr<sirius::op::operator_data> value,
                   std::shared_ptr<gpu_read_observation> state)
    : input(std::move(value)), observed(std::move(state))
  {
  }
  ~gpu_reader_input() override
  {
    observed->release_status = cudaEventQuery(observed->finished);
    observed->input_released = true;
  }
};

struct delayed_copy_operator : sirius::op::sirius_physical_operator {
  gpu_read_hold& hold;
  void* device;
  size_t offset;
  bool fail;
  cudaEvent_t finished;
  delayed_copy_operator(
    gpu_read_hold& gate, void* output, size_t start, bool throws, cudaEvent_t event)
    : sirius_physical_operator(sirius::op::SiriusPhysicalOperatorType::FILTER, {}, 0),
      hold(gate),
      device(output),
      offset(start),
      fail(throws),
      finished(event)
  {
  }
  std::string get_name() const override { return "delayed_preparation_consumer"; }
  std::unique_ptr<sirius::op::operator_data> execute(sirius::op::operator_data const& input,
                                                     ::cuda::stream_ref stream) override
  {
    auto const& wrapper = dynamic_cast<gpu_reader_input const&>(input);
    auto metadata       = detached_metadata(*wrapper.input);
    if (!metadata) throw std::runtime_error("missing detached scan metadata");
    require_cuda(cudaLaunchHostFunc(stream.get(), gpu_read_hold::wait, &hold));
    require_cuda(cudaMemcpyAsync(
      device, metadata->storage->data() + offset, 16, cudaMemcpyHostToDevice, stream.get()));
    require_cuda(cudaEventRecord(finished, stream.get()));
    wrapper.observed->submitted.set_value();
    if (fail) throw std::runtime_error("consumer failed after submitting GPU read");
    return std::make_unique<sirius::op::operator_data>();
  }
};

// Release gates before joining futures on both assertion failure and normal exit.
struct gpu_read_resources {
  std::array<int, 2> devices{};
  std::array<gpu_read_hold, 2> holds;
  std::array<cudaStream_t, 2> streams{};
  std::array<cudaEvent_t, 2> events{};
  std::array<void*, 2> buffers{};
  ~gpu_read_resources()
  {
    release();
    for (size_t i = 0; i < streams.size(); ++i) {
      rmm::cuda_set_device_raii device{rmm::cuda_device_id{devices[i]}};
      if (streams[i]) cudaStreamSynchronize(streams[i]);
      if (buffers[i]) cudaFree(buffers[i]);
      if (events[i]) cudaEventDestroy(events[i]);
      if (streams[i]) cudaStreamDestroy(streams[i]);
    }
  }
  void initialize(std::array<int, 2> selected_devices)
  {
    devices = selected_devices;
    for (size_t i = 0; i < streams.size(); ++i) {
      rmm::cuda_set_device_raii device{rmm::cuda_device_id{devices[i]}};
      require_cuda(cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking));
      require_cuda(cudaEventCreateWithFlags(&events[i], cudaEventDisableTiming));
      require_cuda(cudaMalloc(&buffers[i], 16));
    }
  }
  void release()
  {
    for (auto& hold : holds)
      hold.release();
  }
};
// Observe the actual allocation owner, without retaining a second root outside
// the descriptors. CUDA resources outlive the provider, ledger and all consumers.
struct gpu_backing_observation {
  std::array<std::atomic<int>, 2> release_status{cudaErrorUnknown, cudaErrorUnknown};
  std::array<std::atomic<int>, 2> stream_status{cudaErrorUnknown, cudaErrorUnknown};
  std::atomic<int> cleanup_status{cudaSuccess};
  std::atomic<unsigned> releases{0};
};
struct registered_gpu_allocation {
  backing_allocation allocation;
  gpu_read_resources& gpu;
  std::shared_ptr<gpu_backing_observation> observed;

  registered_gpu_allocation(backing_allocation value,
                            uint64_t bytes,
                            gpu_read_resources& resources,
                            std::shared_ptr<gpu_backing_observation> observation)
    : allocation(std::move(value)), gpu(resources), observed(std::move(observation))
  {
    rmm::cuda_set_device_raii device{rmm::cuda_device_id{gpu.devices[0]}};
    require_cuda(cudaHostRegister(allocation.data, bytes, cudaHostRegisterPortable));
  }
  registered_gpu_allocation(registered_gpu_allocation const&)            = delete;
  registered_gpu_allocation& operator=(registered_gpu_allocation const&) = delete;
  ~registered_gpu_allocation() noexcept
  {
    auto record_error = [&](cudaError_t status) {
      if (status != cudaSuccess) observed->cleanup_status = status;
    };
    int previous_device = -1;
    record_error(cudaGetDevice(&previous_device));
    // Snapshot BEFORE the safety wait: a broken task must still fail this test.
    bool pending = false;
    for (size_t i = 0; i < gpu.events.size(); ++i) {
      record_error(cudaSetDevice(gpu.devices[i]));
      auto event_status           = cudaEventQuery(gpu.events[i]);
      auto stream_status          = cudaStreamQuery(gpu.streams[i]);
      observed->release_status[i] = event_status;
      observed->stream_status[i]  = stream_status;
      pending |= event_status != cudaSuccess || stream_status != cudaSuccess;
    }
    // A negative control may retire the last owner on either worker. Open both
    // gates before waiting, so that test teardown cannot deadlock that worker.
    if (pending) gpu.release();
    // Wait actual streams too: submission can fail before recording the event.
    // The prior snapshots remain the verdict; this is only a test safety guard.
    for (size_t i = 0; i < gpu.streams.size(); ++i) {
      record_error(cudaSetDevice(gpu.devices[i]));
      record_error(cudaStreamSynchronize(gpu.streams[i]));
    }
    record_error(cudaSetDevice(gpu.devices[0]));
    record_error(cudaHostUnregister(allocation.data));
    if (previous_device >= 0) record_error(cudaSetDevice(previous_device));
    allocation.owner.reset();
    ++observed->releases;
  }
};
class gpu_observed_backing final : public reservation_backing {
 public:
  gpu_observed_backing(std::shared_ptr<reservation_backing> backing,
                       gpu_read_resources& gpu,
                       std::shared_ptr<gpu_backing_observation> observed)
    : backing_(std::move(backing)), gpu_(gpu), observed_(std::move(observed))
  {
  }
  uint64_t charge_size(uint64_t bytes) const override { return backing_->charge_size(bytes); }
  backing_allocation allocate(uint64_t bytes) override
  {
    auto owner = std::make_shared<registered_gpu_allocation>(
      backing_->allocate(bytes), bytes, gpu_, observed_);
    return {owner->allocation.data, std::move(owner)};
  }

 private:
  std::shared_ptr<reservation_backing> backing_;
  gpu_read_resources& gpu_;
  std::shared_ptr<gpu_backing_observation> observed_;
};
struct gpu_observed_provider final : reservation_provider {
  test::test_reservation_provider& backing;
  gpu_read_resources& gpu;
  std::shared_ptr<gpu_backing_observation> observed = std::make_shared<gpu_backing_observation>();
  gpu_observed_provider(test::test_reservation_provider& source, gpu_read_resources& resources)
    : backing(source), gpu(resources)
  {
  }
  std::optional<reservation_grant> request(memory_space_id space, uint64_t bytes) override
  {
    auto grant = backing.request(space, bytes);
    if (grant) grant->handle = std::make_shared<gpu_observed_backing>(grant->handle, gpu, observed);
    return grant;
  }
};

void check_gpu_backing_retirement(std::array<int, 2> devices)
{
  using domain   = preparation_coordinator::execution_domain;
  auto execution = GENERATE(domain::query_owner, domain::native_reader);
  bool fail      = GENERATE(false, true);
  CAPTURE(execution, fail, devices[0], devices[1]);
  rmm::cuda_set_device_raii device{rmm::cuda_device_id{devices[0]}};
  gpu_read_resources gpu;
  gpu.initialize(devices);
  native_reader_executor native(1, 1);
  fixture f(1, 1, 5s, std::nullopt, &native);
  test::test_reservation_provider provider;
  gpu_observed_provider registered_provider(provider, gpu);
  auto storage  = std::make_shared<detached_source_state>(registered_provider);
  auto out      = std::make_shared<sink>();
  auto shutdown = f.shutdown_guard();
  auto closing  = f.make_hold();
  f.start([&] {
    auto source  = detached_source(storage, out, execution);
    source.close = [close = std::move(source.close), closing]() mutable {
      closing->wait();
      close();
    };
    f.register_source(std::move(source), out);
  });
  REQUIRE(out->await(1));
  auto first = out->connector->get_next_split();
  REQUIRE(first);
  REQUIRE(out->await(2));
  auto second = out->connector->get_next_split();
  REQUIRE(second);
  REQUIRE(closing->await());

  {
    std::array<std::future<void>, 2> workers;
    auto release_on_exit = std::shared_ptr<gpu_read_resources>(&gpu, [](auto* p) { p->release(); });
    std::array<std::shared_ptr<gpu_read_observation>, 2> observed;
    std::array<std::unique_ptr<sirius::op::operator_data>, 2> inputs{std::move(*first),
                                                                     std::move(*second)};
    first.reset();
    second.reset();
    for (size_t i = 0; i < workers.size(); ++i) {
      observed[i]           = std::make_shared<gpu_read_observation>();
      observed[i]->finished = gpu.events[i];
      auto input            = std::make_unique<gpu_reader_input>(std::move(inputs[i]), observed[i]);
      auto submitted        = observed[i]->submitted.get_future();
      workers[i] = std::async(std::launch::async, [&, i, input = std::move(input)]() mutable {
        rmm::cuda_set_device_raii device{rmm::cuda_device_id{devices[i]}};
        delayed_copy_operator op(gpu.holds[i], gpu.buffers[i], i * 16, fail, gpu.events[i]);
        auto pipeline = std::make_shared<sirius::pipeline::sirius_pipeline>(
          sirius::pipeline::pipeline_build_context{nullptr, true});
        pipeline->set_pipeline_id(i + 1);
        sirius::pipeline::sirius_pipeline_build_state build;
        build.set_pipeline_source(*pipeline, op);
        build.add_pipeline_operator(*pipeline, op);
        sirius::pipeline::assign_operator_ids({pipeline});
        auto global = std::make_shared<sirius::pipeline::gpu_pipeline_task_global_state>(
          pipeline, sirius::test::make_test_telemetry_context());
        sirius::pipeline::gpu_pipeline_task task(
          i,
          {},
          std::make_unique<sirius::pipeline::gpu_pipeline_task_local_state>(std::move(input)),
          global);
        // Follow the executor telemetry transitions before entering the compute seam.
        task.telemetry_fsm().queued(
          {.queue = {.target = quent::task_queue::TaskQueueId{quent::nil_uuid()},
                     .data   = {.entries = 1}}});
        task.telemetry_fsm().reserving(
          {.requested_bytes      = 0,
           .input_basis          = 0,
           .peak_estimate        = 0,
           .bytes_to_materialize = 0,
           .manager_thread       = {
                   .target = quent::task_manager_loop_thread::TaskManagerLoopThreadId{quent::nil_uuid()},
                   .data   = {}}});
        task.telemetry_fsm().preparing(
          {.origin_tier     = {},
           .target_tier     = "GPU",
           .input_bytes     = 0,
           .executor_thread = {.target =
                                 quent::executor_thread::ExecutorThreadId{quent::nil_uuid()},
                               .data = {}},
           .reservation     = {.target = quent::memory_space::MemorySpaceId{quent::nil_uuid()},
                               .data   = {.bytes = 0}}});
        auto result = task.compute_task(::cuda::stream_ref{gpu.streams[i]});
      });
      CAPTURE(i);
      auto submission_status = submitted.wait_for(5s);
      if (submission_status != std::future_status::ready &&
          workers[i].wait_for(0s) == std::future_status::ready)
        workers[i].get();
      REQUIRE(submission_status == std::future_status::ready);
      // CUDA may serialize host callbacks across streams. Establish gate 0 first,
      // but require only submission of stream 1 until gate 0 has been released.
      if (i == 0) REQUIRE(gpu.holds[i].await());
    }
    closing->release();
    REQUIRE(out->await_closed());
    f.finish();
    REQUIRE_FALSE(f.owner_error);
    REQUIRE(storage->ledger == nullptr);
    REQUIRE(out->close_calls == 1);
    for (size_t i = 0; i < workers.size(); ++i) {
      CHECK(workers[i].wait_for(100ms) == std::future_status::timeout);
      CHECK_FALSE(observed[i]->input_released);
      CHECK(provider.seen->allocated_bytes == 32);
    }
    for (size_t i = 0; i < workers.size(); ++i) {
      gpu.holds[i].release();
      REQUIRE(workers[i].wait_for(5s) == std::future_status::ready);
      if (fail)
        CHECK_THROWS_WITH(workers[i].get(), "consumer failed after submitting GPU read");
      else
        CHECK_NOTHROW(workers[i].get());
      CHECK(observed[i]->input_released);
      CHECK(observed[i]->release_status == cudaSuccess);
      CHECK_FALSE(gpu.holds[i].timed_out);
      if (i == 0) {
        CHECK_FALSE(observed[1]->input_released);
        CHECK_FALSE(storage->output.expired());
        CHECK(provider.seen->allocated_bytes == 32);
        CHECK(provider.outstanding() == 1);
        CHECK(registered_provider.observed->releases == 0);
      }
    }
    for (size_t i = 0; i < gpu.buffers.size(); ++i) {
      rmm::cuda_set_device_raii device{rmm::cuda_device_id{devices[i]}};
      std::array<std::byte, 16> copied;
      require_cuda(
        cudaMemcpy(copied.data(), gpu.buffers[i], copied.size(), cudaMemcpyDeviceToHost));
      CHECK(registered_provider.observed->release_status[i] == cudaSuccess);
      CHECK(registered_provider.observed->stream_status[i] == cudaSuccess);
      CHECK(std::all_of(copied.begin(), copied.end(), [](auto b) { return b == std::byte{0x2a}; }));
    }
    CHECK(registered_provider.observed->cleanup_status == cudaSuccess);
    CHECK(registered_provider.observed->releases == 1);
    CHECK(provider.seen->requests == 1);
    CHECK(provider.seen->allocations == 1);
    // No external owner may postpone the final consumer's backing release.
    CHECK(storage->output.expired());
    CHECK(provider.seen->allocated_bytes == 0);
    CHECK(provider.outstanding() == 0);
  }
  CHECK(storage->output.expired());
  CHECK(provider.seen->allocated_bytes == 0);
  CHECK(provider.outstanding() == 0);
}

}  // namespace

TEST_CASE("GPU task retirement releases detached backing after its final consumer",
          "[scan_preparation][output_lifetime][gpu_output_lifetime]")
{
  check_gpu_backing_retirement({0, 0});
}

TEST_CASE("Detached preparation backing is shared until both GPU consumers retire",
          "[scan_preparation][output_lifetime][gpu_output_lifetime][multi_gpu]")
{
  int count = 0;
  require_cuda(cudaGetDeviceCount(&count));
  if (count < 2) SKIP("Two GPUs are required to qualify shared preparation backing retirement");
  check_gpu_backing_retirement({0, 1});
}
