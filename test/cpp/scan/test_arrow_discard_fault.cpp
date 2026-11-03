/* Copyright 2026, Sirius Contributors. SPDX-License-Identifier: Apache-2.0 */
#include "catch.hpp"
#include "op/scan/arrow/arrow_batch_producer.hpp"
#include "op/scan/arrow/arrow_device_import.hpp"
#include "utils/cuda_status_fault.hpp"

#include <cudf/utilities/memory_resource.hpp>

#include <rmm/cuda_stream.hpp>
#include <rmm/device_buffer.hpp>

#include <atomic>
#include <future>
#include <optional>

using namespace sirius::op::scan;
namespace {
void cuda_ok(cudaError_t status)
{
  if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
struct root_counts {
  std::atomic<unsigned> live{0}, released{0};
};
struct event_root {
  std::shared_ptr<root_counts> counts;
  ArrowSchema schema{}, field{};
  ArrowArray array{}, column{};
  ArrowSchema* fields[1]{&field};
  ArrowArray* columns[1]{&column};
  int32_t value = 1;
  void const* buffers[2]{nullptr, &value};
  void const* root_buffers[1]{};
  cudaEvent_t ready = nullptr;
  event_root(std::shared_ptr<root_counts> counts, cudaStream_t stream) : counts(std::move(counts))
  {
    cuda_ok(cudaEventCreateWithFlags(&ready, cudaEventDisableTiming));
    cuda_ok(cudaEventRecord(ready, stream));
    ++this->counts->live;
    schema.format      = "+s";
    schema.n_children  = 1;
    schema.children    = fields;
    schema.release     = [](ArrowSchema* s) { s->release = nullptr; };
    field.name         = "value";
    field.format       = "i";
    field.release      = schema.release;
    array.length       = 1;
    array.n_children   = 1;
    array.children     = columns;
    array.n_buffers    = 1;
    array.buffers      = root_buffers;
    array.private_data = this->counts.get();
    array.release      = [](ArrowArray* a) {
      ++static_cast<root_counts*>(a->private_data)->released;
      a->release = nullptr;
    };
    column.length    = 1;
    column.n_buffers = 2;
    column.buffers   = buffers;
    column.release   = [](ArrowArray* a) { a->release = nullptr; };
  }
  ~event_root()
  {
    --counts->live;
    cudaEventDestroy(ready);
  }
};
materialization_program integer_program()
{
  materialization_program program{701, 4, {cudf::data_type{cudf::type_id::INT32}}, 0, false};
  program.input_types  = program.output_types;
  program.arrow_schema = std::make_shared<arrow_bound_schema const>(
    arrow_bound_schema{{{"value", "i", cudf::data_type{cudf::type_id::INT32}, false}}});
  return program;
}
void produce(arrow_push_queue::offered_slot& slot,
             std::shared_ptr<root_counts> const& counts,
             cudaStream_t stream,
             uint64_t id)
{
  auto root = std::make_shared<event_root>(counts, stream);
  arrow_source_profile profile;
  profile.memory        = arrow_source_memory::pageable_host;
  profile.leases        = root;
  profile.ready_event   = root->ready;
  profile.control_bytes = sizeof(event_root);
  profile.allocations   = {{id, &root->value, sizeof(root->value)}};
  profile.buffers       = {{{}, arrow_buffer_use{id, 0, sizeof(root->value)}}};
  profile.strings.resize(1);
  slot.set_input(root->schema, root->array, root, std::move(profile));
}

struct validation_completion_fault {
  static inline thread_local validation_completion_fault* active = nullptr;
  bool refuse_sync                                               = true;
  validation_completion_fault() { active = this; }
  ~validation_completion_fault() { active = nullptr; }
  static cudaError_t add(cudaStream_t, cudaStreamCallback_t, void*, unsigned)
  {
    return cudaErrorInvalidValue;
  }
  static cudaError_t synchronize(cudaStream_t stream)
  {
    return active->refuse_sync ? cudaErrorUnknown : cudaStreamSynchronize(stream);
  }
};
}  // namespace

TEST_CASE("Arrow device validation retains its readback destination through failed completion",
          "[arrow_qualified][arrow_ownership][cuda_status_fault]")
{
  using fault       = sirius::test::scoped_cuda_status_fault;
  auto zero_columns = GENERATE(false, true);
  auto mode         = GENERATE(0, 1, 2, 3);  // success, failed wait, failed copy, malformed payload
  rmm::cuda_stream stream;
  auto mr       = cudf::get_current_device_resource_ref();
  int32_t value = mode == 3 && zero_columns ? 0 : 1;
  rmm::device_buffer data(&value, sizeof(value), stream, mr);
  stream.synchronize();
  auto counts = std::make_shared<root_counts>();
  event_root tree(counts, stream.value());
  auto program              = integer_program();
  arrow_bound_schema schema = *program.arrow_schema;
  arrow_source_profile profile;
  profile.strings.resize(1);
  tree.buffers[1] = data.data();
  if (zero_columns) {
    tree.array.n_children = 0;
    tree.array.null_count = -1;
    tree.root_buffers[0]  = data.data();
    schema.columns.clear();
  } else if (mode == 3) {
    tree.column.null_count = 1;
  }
  unsigned released = 0;
  auto observed     = std::shared_ptr<detail::arrow_device_validation_status>(
    new detail::arrow_device_validation_status{99, 99}, [&](auto* state) {
      ++released;
      delete state;
    });
  std::weak_ptr<detail::arrow_device_validation_status> retained = observed;
  validation_completion_fault completion;
  cucascade::exec::detail::completion_cuda_api api;
  api.add_callback = validation_completion_fault::add;
  api.synchronize  = validation_completion_fault::synchronize;
  sirius::exec::retire_lane lane(stream.value(), api);
  std::exception_ptr error;
  {
    auto submission = lane.begin();
    std::optional<fault> injected;
    if (mode == 1 || mode == 2)
      injected.emplace(mode == 1 ? fault::synchronize_stream : fault::copy_device_to_host,
                       cudaErrorUnknown,
                       0,
                       stream.value());
    try {
      auto output =
        detail::import_arrow_device_columns(tree.array, schema, profile, observed, submission, mr);
      CHECK(output->num_columns() == (zero_columns ? 0 : 1));
    } catch (...) {
      error = std::current_exception();
    }
    if (injected) CHECK(injected->consumed());
  }
  CHECK(bool(error) == (mode != 0));
  if (mode != 2) {
    CHECK(observed->invalid == (zero_columns && mode == 3 ? 1 : 0));
    CHECK(observed->nulls == 0);
  }
  observed.reset();
  CHECK_FALSE(retained.expired());
  CHECK(released == 0);
  CHECK(lane.quiesce() == cudaErrorUnknown);
  CHECK_FALSE(retained.expired());
  completion.refuse_sync = false;
  REQUIRE(lane.quiesce() == cudaSuccess);
  CHECK(retained.expired());
  CHECK(released == 1);
}

TEST_CASE("Arrow offered roots retain capacity after readiness synchronization fails",
          "[arrow_qualified][arrow_admission][cuda_status_fault]")
{
  using namespace std::chrono_literals;
  using fault   = sirius::test::scoped_cuda_status_fault;
  auto accepted = GENERATE(0U, 2U);
  auto discard  = GENERATE(0, 1, 2);  // destructor, move assignment, cancellation then reset
  rmm::cuda_stream stream;
  auto counts = std::make_shared<root_counts>();
  arrow_push_queue queue(integer_program(), 65536, [](auto) {});
  for (unsigned i = 0; i < accepted; ++i) {
    auto slot = queue.acquire_offered_slot();
    produce(slot, counts, stream.value(), i + 1);
    REQUIRE(queue.offer(slot) == arrow_push_queue::offer_outcome::accepted);
  }
  auto slot = queue.acquire_offered_slot();
  produce(slot, counts, stream.value(), accepted + 1);
  if (accepted == 2) REQUIRE(queue.offer(slot) == arrow_push_queue::offer_outcome::capacity_wait);
  stream.synchronize();
  if (discard == 2) queue.cancel();
  {
    fault injected(fault::synchronize_event, cudaErrorUnknown);
    if (discard == 0) {
      auto abandoned = std::move(slot);
    } else {
      slot = {};
    }
    REQUIRE(injected.consumed());
  }
  auto observed = queue.observe();
  CHECK(observed.offered == 1);
  CHECK(observed.accepted == (discard == 2 ? 0 : accepted));
  CHECK(observed.reserved_bytes == (observed.accepted + 1) * 65536);
  CHECK(observed.unique_retained_bytes == (observed.accepted + 1) * sizeof(int32_t));
  CHECK(counts->live == observed.accepted + 1);
  CHECK(counts->released == (discard == 2 ? accepted : 0));

  auto acquiring = std::async(std::launch::async, [&] { return queue.acquire_offered_slot(); });
  if (discard != 2) CHECK(acquiring.wait_for(30ms) == std::future_status::timeout);
  queue.cancel();
  REQUIRE(acquiring.wait_for(5s) == std::future_status::ready);
  auto refused = acquiring.get();
  CHECK_FALSE(refused.held());
  CHECK(refused.status() == arrow_push_queue::capacity_result::failed);
  CHECK(queue.observe().offered == 1);
  CHECK(queue.observe().accepted == 0);
  CHECK(counts->live == 1);
  CHECK(counts->released == accepted);
}

TEST_CASE("Arrow accepted roots retain their converted credit after synchronization fails",
          "[arrow_qualified][arrow_admission][cuda_status_fault]")
{
  using fault = sirius::test::scoped_cuda_status_fault;
  auto fail   = GENERATE(false, true);
  rmm::cuda_stream stream;
  auto counts = std::make_shared<root_counts>();
  arrow_push_queue queue(integer_program(), 65536, [](auto) {});
  for (unsigned i = 0; i < 2; ++i) {
    auto slot = queue.acquire_offered_slot();
    produce(slot, counts, stream.value(), i + 1);
    REQUIRE(queue.offer(slot) == arrow_push_queue::offer_outcome::accepted);
  }
  stream.synchronize();
  {
    fault injected(fault::synchronize_event, cudaErrorUnknown);
    if (fail)
      queue.fail(std::make_exception_ptr(std::runtime_error("producer failure")));
    else
      queue.cancel();
    REQUIRE(injected.consumed());
  }
  auto observed = queue.observe();
  CHECK(observed.offered == 0);
  CHECK(observed.accepted == 1);
  CHECK(observed.reserved_bytes == 65536);
  CHECK(observed.unique_retained_bytes == sizeof(int32_t));
  CHECK(counts->live == 1);
  CHECK(counts->released == 1);
}
