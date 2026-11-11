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
#include "creator/task_creator.hpp"
#include "data/data_batch_utils.hpp"
#include "expression/ast/node.hpp"
#include "helper/type_conversions.hpp"
#include "op/scan/arrow/arrow_import_ingestible.hpp"
#include "op/scan/detail/read_retirement.hpp"
#include "op/scan/sirius_gpu_scan_operator.hpp"
#include "operator/operator_test_utils.hpp"
#include "pipeline/gpu_pipeline_task.hpp"
#include "pipeline/query_retirement_test_utils.hpp"
#include "pipeline/task_scheduler.hpp"
#include "scan_manager/preparation_fixture.hpp"
#include "scan_manager/preparation_identity.hpp"
#include "scan_manager/preparation_slot.hpp"
#include "scan_manager/preparation_test_support.hpp"
#include "sirius_context.hpp"
#include "transparent/plan_source_policy.hpp"
#include "transparent/read_view_registry.hpp"
#include "transparent/replay_admission.hpp"
#include "utils/telemetry_utils.hpp"

#include <cudf/strings/strings_column_view.hpp>

#include <rmm/cuda_stream.hpp>

#include <duckdb/common/exception.hpp>

#include <array>
#include <chrono>
#include <cstring>
#include <exception>
#include <future>
#include <numeric>
#include <system_error>
#include <thread>

using namespace sirius::op::scan;
namespace sm = sirius::scan_manager;
using namespace std::chrono_literals;
namespace {
materialization_program integer_program()
{
  materialization_program result{701, 4, {cudf::data_type{cudf::type_id::INT32}}, 0, false};
  result.input_types = result.output_types;
  return result;
}
std::shared_ptr<physical_check_counters> make_counters(bool track_units = true)
{
  auto counters         = std::make_shared<physical_check_counters>();
  counters->track_units = track_units;
  return counters;
}
template <class Action>
std::exception_ptr capture_rejection(Action&& action)
{
  std::exception_ptr error;
  try {
    action();
  } catch (...) {
    error = std::current_exception();
  }
  REQUIRE(error);
  return error;
}
void check_physical_failure(std::exception_ptr error, std::string const& message, bool drift)
{
  REQUIRE(error);
  try {
    std::rethrow_exception(error);
  } catch (sirius::transparent::classified_execution_error const& e) {
    CHECK(e.cause == sirius::transparent::late_failure_cause::physical_input);
    CHECK(std::string(e.what()) == message);
    CHECK(e.reason == (drift ? sirius::transparent::kArrowSchemaDrift : ""));
  }
}
struct integer_tree {
  ArrowSchema schema{}, field{};
  ArrowArray array{}, column{};
  ArrowSchema* fields[1]{&field};
  ArrowArray* columns[1]{&column};
  void const* root_buffers[1]{nullptr};
  int32_t values[3]{11, 22, 33};
  void const* buffers[2]{nullptr, values};
  integer_tree(int rows = 3)
  {
    schema.format     = "+s";
    schema.n_children = 1;
    schema.children   = fields;
    schema.release    = [](ArrowSchema* p) { p->release = nullptr; };
    field.format      = "i";
    field.release     = schema.release;
    array.length      = rows;
    array.n_children  = 1;
    array.children    = columns;
    array.n_buffers   = 1;
    array.buffers     = root_buffers;
    array.release     = [](ArrowArray* p) { p->release = nullptr; };
    column.length     = rows;
    column.n_buffers  = 2;
    column.buffers    = buffers;
    column.release    = array.release;
  }
  offered_arrow_input offer(std::shared_ptr<integer_tree> const& self)
  {
    return {&schema, &array, self, sizeof(integer_tree), {}};
  }
};
class signaled_staging final : public std::pmr::memory_resource {
 public:
  signaled_staging(void* data, size_t bytes, std::function<void()> signal)
    : arena_(data, bytes, std::pmr::null_memory_resource()), signal_(std::move(signal))
  {
  }

 private:
  void* do_allocate(size_t size, size_t alignment) override
  {
    if (signal_) std::exchange(signal_, {})();
    return arena_.allocate(size, alignment);
  }
  void do_deallocate(void*, size_t, size_t) override {}
  bool do_is_equal(std::pmr::memory_resource const& other) const noexcept override
  {
    return this == &other;
  }
  std::pmr::monotonic_buffer_resource arena_;
  std::function<void()> signal_;
};
struct pull_capacity {
  std::shared_ptr<void> claim;
  sm::charged_block block;
  signaled_staging staging;
  pull_capacity(admitted_arrow_input admitted, std::function<void()> signal)
    : claim(admitted.lifetime),
      block(admitted.allocator.allocate(8192)),
      staging(block.data() + 1024, 7168, std::move(signal))
  {
  }
};
class pull_producer final : public arrow_batch_producer {
 public:
  std::vector<int> rows{3};
  std::function<void()> on_normalize;
  std::function<void()> before_next;
  std::function<void()> before_stop_check;
  std::function<void(ArrowSchema&, ArrowArray&)> observe_tree;
  std::shared_ptr<owned_arrow_batch> next_batch;
  sm::unit_envelope envelope{{8192}, 0};
  int64_t max_rows                = 3;
  uint64_t temporary_probe_before = 0, temporary_probe_after = 0;
  size_t cursor  = 0;
  int preflights = 0, calls = 0, closes = 0;
  bool ready = true, cancelled = false, wrong_schema = false, fail_read = false;
  // Legacy producers reported a token stop as an interrupt; the driver must still classify it.
  bool legacy_interrupt_on_stop           = false;
  sm::preparation_execution_domain domain = sm::preparation_execution_domain::query_owner;
  std::thread::id owner                   = std::this_thread::get_id();
  sm::preparation_execution_domain execution_domain() const noexcept override { return domain; }
  arrow_producer_limits limits() const override { return {max_rows, envelope}; }
  void preflight() override { ++preflights; }
  arrow_readiness readiness() const noexcept override
  {
    return ready ? arrow_readiness::ready : arrow_readiness::pending;
  }
  void bind_wakeup(sm::preparation_wakeup) override {}
  arrow_production next(admitted_arrow_input admitted, sm::preparation_stop_token stop) override
  {
    if (!calls && domain == sm::preparation_execution_domain::native_reader)
      owner = std::this_thread::get_id();
    if (owner != std::this_thread::get_id()) throw std::logic_error("wrong pull owner");
    ++calls;
    if (before_stop_check) before_stop_check();
    if (stop.stop_requested()) {
      if (legacy_interrupt_on_stop) throw duckdb::InterruptException();
      throw std::system_error(std::make_error_code(std::errc::operation_canceled));
    }
    if (cancelled) throw duckdb::InterruptException();
    if (before_next) before_next();
    if (fail_read) throw std::runtime_error("pull failed");
    if (next_batch) return produced_arrow_input{std::exchange(next_batch, {}), {}};
    if (cursor == rows.size()) return arrow_end_of_stream{};
    // Probes retire before next() returns: post-return accounting cannot catch them.
    if (temporary_probe_before) {
      auto probe = admitted.allocator.allocate(temporary_probe_before);
    }
    auto allocator = admitted.allocator;
    auto capacity  = std::make_shared<pull_capacity>(std::move(admitted), on_normalize);
    if (temporary_probe_after) { auto probe = allocator.allocate(temporary_probe_after); }
    auto* tree = new (capacity->block.data()) integer_tree(rows[cursor++]);
    if (wrong_schema) tree->field.format = "f";
    if (observe_tree) observe_tree(tree->schema, tree->array);
    return produced_arrow_input{
      std::make_shared<owned_arrow_batch>(tree->schema, tree->array, capacity, capacity->staging),
      {}};
  }
  void close() override
  {
    if (owner != std::this_thread::get_id()) throw std::logic_error("wrong close owner");
    ++closes;
  }
  void cancel() noexcept override { cancelled = true; }
};
struct harness {
  sm::test::test_reservation_provider backing;
  std::shared_ptr<sm::preparation_ledger> ledger;
  std::shared_ptr<physical_profile_table> profiles = std::make_shared<physical_profile_table>();
  std::shared_ptr<void const> identity             = std::make_shared<int>(4);
  std::shared_ptr<arrow_import_ingestible> driver;
  std::shared_ptr<arrow_input_gate> gate = std::make_shared<arrow_input_gate>();
  harness(std::shared_ptr<arrow_batch_producer> producer,
          materialization_program program                      = integer_program(),
          bool empty_source                                    = true,
          sm::reservation_provider* external                   = nullptr,
          std::shared_ptr<arrow_batch_producer> other          = {},
          uint64_t quantum                                     = 1,
          std::shared_ptr<physical_check_counters> counters    = {},
          std::function<void(arrow_import_binding&)> configure = {})
  {
    auto unique     = std::make_unique<sm::preparation_ledger>(external ? *external : backing);
    auto envelope   = producer->limits().envelope;
    backing.quantum = quantum;
    std::vector scans{arrow_preparation_envelope(*producer)};
    auto w        = envelope.temporary_charge(quantum);
    auto retained = envelope.retained;
    if (other) {
      scans.push_back(arrow_preparation_envelope(*other));
      w = std::max(w, other->limits().envelope.temporary_charge(quantum));
      retained += other->limits().envelope.retained;
    }
    uint32_t permits = 0;
    for (auto const& scan : scans)
      permits += scan.minimum_permits;
    backing.grant_bytes = w * permits + retained;
    auto decision =
      unique->admit(scans, sm::memory_space_id{cucascade::memory::Tier::HOST, 0}, permits);
    REQUIRE(decision.deferred);
    arrow_import_binding binding;
    binding.gate          = gate;
    binding.program       = program;
    binding.profiles      = profiles;
    binding.read_identity = identity;
    binding.display_name  = "test Arrow input";
    binding.column_names.resize(program.input_types->size(), "value");
    binding.output_columns.resize(program.output_types.size());
    std::iota(binding.output_columns.begin(), binding.output_columns.end(), 0);
    binding.emit_empty_source = empty_source;
    binding.admission  = std::make_shared<sm::preparation_admission>(std::move(unique), decision);
    ledger             = std::shared_ptr<sm::preparation_ledger>(binding.admission->consume());
    profiles->counters = counters ? std::move(counters) : make_counters();
    if (configure) configure(binding);
    driver = std::make_shared<arrow_import_ingestible>(std::move(binding), std::move(producer));
    driver->attach_preparation_ledger(ledger);
  }
  harness(std::shared_ptr<physical_check_counters> counters,
          std::shared_ptr<arrow_batch_producer> producer,
          materialization_program program = integer_program())
    : harness(std::move(producer), std::move(program), true, nullptr, {}, 1, std::move(counters))
  {
  }
  std::unique_ptr<scan_info> next(sm::preparation_stop_token stop = {})
  {
    auto claim = driver->try_claim_split({});
    REQUIRE(std::holds_alternative<sm::preparation_job>(claim));
    return std::get<sm::preparation_job>(std::move(claim)).run(stop);
  }
  ~harness()
  {
    driver->finish_preparation();
    ledger->close();
  }
};
struct queue_guard {
  std::shared_ptr<arrow_push_queue> queue;
  ~queue_guard() { queue->cancel(); }
};
void await_ready(arrow_push_queue const& queue)
{
  auto deadline = std::chrono::steady_clock::now() + 5s;
  while (queue.readiness() == arrow_readiness::pending &&
         std::chrono::steady_clock::now() < deadline)
    std::this_thread::yield();
  REQUIRE(queue.readiness() == arrow_readiness::ready);
}
std::shared_ptr<arrow_push_queue> queue(
  std::shared_ptr<physical_check_counters> counters,
  materialization_program p                    = integer_program(),
  std::function<void(std::exception_ptr)> fail = [](auto) {})
{
  return std::make_shared<arrow_push_queue>(
    p, arrow_push_limits{32, 8192, 8192, 8192}, std::move(fail), std::move(counters));
}
}  // namespace

namespace {
materialization_program qualified_integer_program()
{
  auto program         = integer_program();
  program.arrow_schema = std::make_shared<arrow_bound_schema const>(
    arrow_bound_schema{{{"value", "i", cudf::data_type{cudf::type_id::INT32}, false}}});
  return program;
}
struct qualified_integer : integer_tree {
  std::function<void()> released;
  explicit qualified_integer(int rows = 3) : integer_tree(rows)
  {
    field.name         = "value";
    array.private_data = this;
    array.release      = [](ArrowArray* array) {
      auto& root     = *static_cast<qualified_integer*>(array->private_data);
      array->release = nullptr;
      if (root.released) root.released();
    };
  }
  arrow_source_profile profile(std::shared_ptr<qualified_integer> const& self)
  {
    arrow_source_profile result;
    result.completed     = true;
    result.leases        = self;
    result.control_bytes = sizeof(*this);
    auto id              = reinterpret_cast<uintptr_t>(self.get());
    result.allocations   = {{id, self.get(), sizeof(*this)}};
    result.buffers       = {
      {{},
             arrow_buffer_use{
         id, static_cast<uint64_t>(reinterpret_cast<uintptr_t>(values) - id), sizeof(values)}}};
    result.strings.resize(1);
    return result;
  }
};
void fill_slot(arrow_push_queue::offered_slot& slot,
               int rows                       = 3,
               std::function<void()> released = {})
{
  auto tree      = std::make_shared<qualified_integer>(rows);
  tree->released = std::move(released);
  slot.set_input(tree->schema, tree->array, tree, tree->profile(tree));
  CHECK_FALSE(tree->schema.release);
  CHECK_FALSE(tree->array.release);
}
}  // namespace

TEST_CASE("Arrow qualified admission retains Q2 credits across dequeue and shared input owners",
          "[arrow_driver][arrow_admission]")
{
  auto rows    = GENERATE(0, 3);
  auto program = qualified_integer_program();
  auto q       = std::make_shared<arrow_push_queue>(program, 8192, [](auto) {});
  queue_guard guard{q};
  harness h(q, program);
  auto released = std::make_shared<std::atomic<int>>(0);
  for (int i = 0; i < 2; ++i) {
    auto slot = q->acquire_offered_slot();
    REQUIRE(slot.held());
    fill_slot(slot, rows, [q, released] {
      CHECK(q->observe().accepted > 0);
      ++*released;
    });
    REQUIRE(q->offer(slot) == arrow_push_queue::offer_outcome::accepted);
    CHECK_FALSE(slot.held());
  }
  auto first  = h.next();
  auto second = h.next();
  REQUIRE(first);
  REQUIRE(second);
  auto retained = dynamic_cast<owned_arrow_scan_info&>(*first).coalesced_copy();
  CHECK(h.ledger->permits_in_flight() == 2);
  auto slot = q->acquire_offered_slot();
  fill_slot(slot, rows);
  CHECK(q->offer(slot) == arrow_push_queue::offer_outcome::capacity_wait);
  CHECK(q->observe().accepted == 2);
  CHECK(q->observe().offered == 1);
  CHECK(q->observe().reserved_bytes == 3 * 8192);
  auto waiting =
    std::async(std::launch::async, [&] { return q->wait_for_capacity_or_terminal(slot); });
  queue_guard wait_guard{q};
  first.reset();
  CHECK(waiting.wait_for(20ms) == std::future_status::timeout);
  CHECK(*released == 0);
  retained.reset();
  REQUIRE(waiting.wait_for(5s) == std::future_status::ready);
  CHECK(waiting.get() == arrow_push_queue::capacity_result::available);
  CHECK(*released == 1);
  CHECK(q->offer(slot) == arrow_push_queue::offer_outcome::accepted);
  auto third = h.next();
  q->finish();
  second.reset();
  third.reset();
  CHECK(*released == 2);
  CHECK(q->observe().accepted == 0);
  CHECK(q->observe().unique_retained_bytes == 0);
  CHECK_FALSE(h.next());
}

TEST_CASE("Arrow offered permit is shared across threads and all terminal transitions wake it",
          "[arrow_driver][arrow_admission]")
{
  auto mode = GENERATE(0, 1, 2, 3);
  auto q    = std::make_shared<arrow_push_queue>(qualified_integer_program(), 8192, [](auto) {});
  queue_guard guard{q};
  auto slot = q->acquire_offered_slot();
  fill_slot(slot);
  auto waiting = std::async(std::launch::async, [&] { return q->acquire_offered_slot(); });
  queue_guard wait_guard{q};
  CHECK(waiting.wait_for(20ms) == std::future_status::timeout);
  CHECK(q->observe().offered == 1);
  if (mode == 0) slot = {};
  if (mode == 1) q->finish();
  if (mode == 2) q->fail(std::make_exception_ptr(std::runtime_error("failed producer")));
  if (mode == 3) q->cancel();
  REQUIRE(waiting.wait_for(5s) == std::future_status::ready);
  auto next = waiting.get();
  CHECK(next.held() == (mode == 0));
  if (mode == 1) CHECK(next.status() == arrow_push_queue::capacity_result::closed);
  if (mode > 1) CHECK(next.status() == arrow_push_queue::capacity_result::failed);
}

TEST_CASE("Qualified Arrow preserves independent cancellation before and during next",
          "[arrow_driver][arrow_stop]")
{
  auto during           = GENERATE(false, true);
  auto token_stopped    = GENERATE(false, true);
  auto original_failure = GENERATE(false, true);
  auto program          = qualified_integer_program();
  int failures = 0, released = 0;
  auto q = std::make_shared<arrow_push_queue>(program, 8192, [&](auto) { ++failures; });
  queue_guard guard{q};
  harness h(q, program);
  auto slot = q->acquire_offered_slot();
  fill_slot(slot, 3, [&] { ++released; });
  REQUIRE(q->offer(slot) == arrow_push_queue::offer_outcome::accepted);
  std::stop_source stop;
  auto cancel = [&] {
    if (original_failure) q->fail(std::make_exception_ptr(std::runtime_error("original failure")));
    q->cancel();
    if (token_stopped) stop.request_stop();
  };
  std::exception_ptr error;
  if (during) {
    sirius::test::preparation::hold allocating;
    h.backing.seen->before_allocate = [&] { allocating.wait(); };
    auto consuming                  = std::async(std::launch::async, [&] {
      try {
        (void)h.next(sm::preparation_stop_token{stop.get_token()});
      } catch (...) {
        return std::current_exception();
      }
      return std::exception_ptr{};
    });
    bool entered                    = allocating.await();
    cancel();
    allocating.release();
    error = consuming.get();
    REQUIRE(entered);
  } else {
    cancel();
    sm::unit_key key{99, 0};
    auto permit = h.ledger->try_acquire_unit(key, 0);
    REQUIRE(permit);
    try {
      (void)q->next({key, h.ledger->allocator(permit), {}},
                    sm::preparation_stop_token{stop.get_token()});
    } catch (...) {
      error = std::current_exception();
    }
    permit.release();
    h.ledger->retire_unit(key);
  }
  REQUIRE(error);
  if (original_failure) {
    CHECK_THROWS_WITH(std::rethrow_exception(error), "original failure");
  } else if (token_stopped) {
    try {
      std::rethrow_exception(error);
    } catch (std::system_error const& e) {
      CHECK(e.code() == std::errc::operation_canceled);
    }
  } else {
    CHECK_THROWS_AS(std::rethrow_exception(error), duckdb::InterruptException);
  }
  CHECK(failures == (original_failure ? 1 : 0));
  CHECK(released == 1);
  CHECK(q->observe().accepted == 0);
  CHECK(h.ledger->permits_in_flight() == 0);
  CHECK(h.backing.seen->allocated_bytes == 0);
}

TEST_CASE("Arrow accepted roots release outside the queue lock on failure and cancellation",
          "[arrow_driver][arrow_admission][arrow_ownership]")
{
  auto failed = GENERATE(false, true);
  auto q      = std::make_shared<arrow_push_queue>(qualified_integer_program(), 8192, [](auto) {});
  queue_guard guard{q};
  auto released = std::make_shared<std::atomic<int>>(0);
  for (int i = 0; i < 2; ++i) {
    auto slot = q->acquire_offered_slot();
    fill_slot(slot, 3, [q, released] {
      (void)q->observe();
      ++*released;
    });
    REQUIRE(q->offer(slot) == arrow_push_queue::offer_outcome::accepted);
  }
  auto slot = q->acquire_offered_slot();
  fill_slot(slot, 3, [q, released] {
    (void)q->observe();
    ++*released;
  });
  CHECK(q->offer(slot) == arrow_push_queue::offer_outcome::capacity_wait);
  auto waiting =
    std::async(std::launch::async, [&] { return q->wait_for_capacity_or_terminal(slot); });
  queue_guard wait_guard{q};
  CHECK(waiting.wait_for(20ms) == std::future_status::timeout);
  if (failed)
    q->fail(std::make_exception_ptr(std::runtime_error("producer failure")));
  else
    q->cancel();
  REQUIRE(waiting.wait_for(5s) == std::future_status::ready);
  CHECK(waiting.get() == arrow_push_queue::capacity_result::failed);
  CHECK(*released == 2);
  CHECK(q->offer(slot) == arrow_push_queue::offer_outcome::terminal);
  slot = {};
  CHECK(*released == 3);
  CHECK(q->observe().accepted == 0);
}

TEST_CASE("Arrow qualified admission rejects valid oversize and first schema drift without waiting",
          "[arrow_driver][arrow_admission]")
{
  auto mode     = GENERATE(0, 1, 2, 3);
  bool drift    = mode >= 2;
  auto counters = make_counters();
  int failures  = 0;
  std::exception_ptr reported;
  auto q = std::make_shared<arrow_push_queue>(
    qualified_integer_program(),
    arrow_push_limits{mode == 0 ? 2 : 3, mode == 1 || mode == 3 ? 64U : 8192U, 8192, 8192},
    [&](std::exception_ptr failure) {
      ++failures;
      reported = failure;
      CHECK(counters->arrow_schema_drift_refusals == (drift ? 1 : 0));
    },
    counters);
  auto slot = q->acquire_offered_slot();
  auto tree = std::make_shared<qualified_integer>();
  if (drift) tree->field.name = "wrong";
  slot.set_input(tree->schema, tree->array, tree, tree->profile(tree));
  auto error = capture_rejection([&] { q->offer(slot); });
  CHECK(reported == error);
  if (drift) {
    check_physical_failure(error, "Arrow storage, name or nullability changed", true);
  } else {
    try {
      std::rethrow_exception(error);
    } catch (sirius::transparent::classified_execution_error const& e) {
      CHECK(e.cause == sirius::transparent::late_failure_cause::resource);
      CHECK(std::string(e.what()).find("arrow_batch_too_large") != std::string::npos);
    }
  }
  CHECK(capture_rejection([&] { q->check_status(); }) == error);
  CHECK(counters->arrow_schema_drift_refusals == (drift ? 1 : 0));
  CHECK(failures == 1);
  CHECK(q->observe().accepted == 0);
  CHECK(slot.held());
}

TEST_CASE("Arrow admission includes rounded staging before accepting a root",
          "[arrow_driver][arrow_admission]")
{
  auto q = std::make_shared<arrow_push_queue>(qualified_integer_program(), 8192, [](auto) {});
  q->bind_allocation_granularity(8192);
  auto slot = q->acquire_offered_slot();
  fill_slot(slot);
  try {
    q->offer(slot);
    FAIL("rounded staging plus retained source exceeded Bmax");
  } catch (sirius::transparent::classified_execution_error const& error) {
    CHECK(error.cause == sirius::transparent::late_failure_cause::resource);
  }
  CHECK(q->observe().accepted == 0);
}

TEST_CASE("Arrow row admission classifies a valid wide root without allocating its payload",
          "[arrow_driver][arrow_admission]")
{
  auto malformed = GENERATE(false, true);
  auto q =
    std::make_shared<arrow_push_queue>(qualified_integer_program(), 64ULL << 30, [](auto) {});
  auto slot          = q->acquire_offered_slot();
  auto tree          = std::make_shared<qualified_integer>();
  auto profile       = tree->profile(tree);
  tree->array.length = tree->column.length = int64_t{INT32_MAX} + 1;
  auto base                                = reinterpret_cast<void const*>(uintptr_t{0x100000});
  auto bytes       = static_cast<uint64_t>(tree->array.length) * sizeof(int32_t);
  tree->buffers[1] = base;
  if (malformed) tree->field.name = "wrong";
  profile.memory      = arrow_source_memory::device;
  profile.device      = 0;
  profile.allocations = {{1, base, bytes}};
  profile.buffers     = {{{}, arrow_buffer_use{1, 0, bytes}}};
  slot.set_input(tree->schema, tree->array, tree, std::move(profile));
  try {
    q->offer(slot);
    FAIL("row limit must refuse before conversion");
  } catch (sirius::transparent::classified_execution_error const& error) {
    CHECK(error.cause == (malformed ? sirius::transparent::late_failure_cause::physical_input
                                    : sirius::transparent::late_failure_cause::resource));
  }
  CHECK(q->observe().accepted == 0);
}

TEST_CASE("Arrow queue counts a shared parent once while reserving separate root credits",
          "[arrow_driver][arrow_admission]")
{
  auto conflict = GENERATE(false, true);
  auto q = std::make_shared<arrow_push_queue>(qualified_integer_program(), 8192, [](auto) {});
  queue_guard guard{q};
  auto parent = std::make_shared<std::array<int32_t, 3>>(std::array<int32_t, 3>{11, 22, 33});
  for (unsigned i = 0; i < 2; ++i) {
    auto slot           = q->acquire_offered_slot();
    auto tree           = std::make_shared<qualified_integer>();
    tree->buffers[1]    = parent->data();
    auto profile        = tree->profile(tree);
    profile.leases      = parent;
    auto id             = conflict && i ? 19U : 17U;
    profile.allocations = {{id, parent->data(), sizeof(*parent)}};
    profile.buffers     = {{{}, arrow_buffer_use{id, 0, sizeof(*parent)}}};
    slot.set_input(tree->schema, tree->array, tree, std::move(profile));
    if (conflict && i) {
      CHECK_THROWS_AS(q->offer(slot), sirius::transparent::classified_execution_error);
      return;
    }
    REQUIRE(q->offer(slot) == arrow_push_queue::offer_outcome::accepted);
  }
  CHECK(q->observe().unique_retained_bytes == sizeof(*parent));
  CHECK(q->observe().accepted == 2);
  CHECK(q->observe().reserved_bytes == 2 * 8192);
  q->cancel();
  CHECK(q->observe().unique_retained_bytes == 0);
}

TEST_CASE("Arrow pull claims retain admission until their last input owner releases",
          "[arrow_driver]")
{
  auto producer = std::make_shared<pull_producer>();
  harness h(producer);
  auto source = h.driver->make_preparation_source({});
  CHECK(producer->calls == 0);
  CHECK(source.execution == sm::preparation_execution_domain::query_owner);
  h.driver->preflight_for_query();
  CHECK(producer->preflights == 1);
  producer->ready = false;
  CHECK(std::holds_alternative<sm::preparation_pending>(source.try_claim()));
  CHECK(h.ledger->tracked_units() == 0);
  producer->ready = true;
  auto occupied   = h.ledger->try_acquire_unit({999, 0}, 0);
  REQUIRE(occupied);
  CHECK(std::holds_alternative<sm::preparation_pending>(source.try_claim()));
  CHECK(h.ledger->tracked_units() == 1);
  CHECK(producer->calls == 0);
  h.ledger->retire_unit({999, 0});
  occupied.release();
  auto input = h.next();
  auto copy  = dynamic_cast<owned_arrow_scan_info&>(*input).coalesced_copy();
  CHECK(input->preparation_bytes() == 8192);
  CHECK(h.ledger->permits_in_flight() == 1);
  CHECK(std::holds_alternative<sm::preparation_pending>(source.try_claim()));
  input.reset();
  CHECK(h.ledger->permits_in_flight() == 1);
  copy.reset();
  CHECK(h.ledger->permits_in_flight() == 0);
  CHECK(h.ledger->tracked_units() == 0);
  CHECK(h.backing.seen->allocated_bytes == 0);
  CHECK_FALSE(h.next());
  CHECK(std::holds_alternative<sm::preparation_exhausted>(source.try_claim()));
  CHECK(source.coalescer->flush().empty());
  source.close();
  CHECK(producer->closes == 1);
  CHECK_FALSE(h.driver->has_processed_all_metadata());
  CHECK_THROWS(h.driver->next_split_provider({}));
  CHECK_FALSE(h.driver->can_serve_cached_view());
}

TEST_CASE("Arrow synchronous pinning stays refused across source termination", "[arrow_driver]")
{
  auto producer = std::make_shared<pull_producer>();
  harness h(producer);
  auto source        = h.driver->make_preparation_source({});
  auto check_refusal = [&] {
    CHECK_THROWS_WITH(
      h.driver->next_split_provider({}),
      "Arrow import does not support synchronous pinning; admitted claims required");
    CHECK_FALSE(h.driver->has_processed_all_metadata());
    CHECK_FALSE(h.driver->can_serve_cached_view());
  };
  check_refusal();
  SECTION("normal exhaustion without an accepted root")
  {
    producer->rows.clear();
    CHECK_FALSE(h.next());
    auto tail = source.coalescer->flush();
    REQUIRE(tail.size() == 1);
    CHECK(dynamic_cast<owned_arrow_scan_info&>(*tail.front()).rootless());
  }
  SECTION("normal exhaustion after accepted data")
  {
    REQUIRE(h.next());
    CHECK_FALSE(h.next());
    CHECK(source.coalescer->flush().empty());
  }
  SECTION("descriptor source closed") { source.close(); }
  SECTION("producer failure")
  {
    producer->fail_read = true;
    CHECK_THROWS_WITH(h.next(), "pull failed");
    check_refusal();
    source.close();
  }
  SECTION("cancelled with an outstanding input")
  {
    auto input = h.next();
    h.driver->stop_preparation();
    check_refusal();
    CHECK(h.ledger->permits_in_flight() == 1);
    input.reset();
    CHECK(h.ledger->permits_in_flight() == 0);
  }
  SECTION("preparation finished") { h.driver->finish_preparation(); }
  check_refusal();
}

TEST_CASE("Arrow driver preserves empty source and zero-row data as distinct states",
          "[arrow_driver]")
{
  auto mode      = GENERATE(0, 1, 2);
  auto producer  = std::make_shared<pull_producer>();
  producer->rows = mode == 1 ? std::vector<int>{0} : std::vector<int>{};
  harness h(producer, integer_program(), mode != 2);
  auto coalescer = h.driver->create_batch_coalescer();
  auto manager   = sirius::test::operator_utils::initialize_memory_manager();
  auto* space    = manager->get_memory_space(cucascade::memory::Tier::GPU, 0);
  rmm::cuda_stream stream;
  std::unique_ptr<scan_info> input;
  if (mode == 1) {
    input = h.next();
    REQUIRE(input);
    CHECK_FALSE(dynamic_cast<owned_arrow_scan_info&>(*input).rootless());
  } else {
    CHECK_FALSE(h.next());
    auto tail = coalescer->flush();
    REQUIRE(tail.size() == (mode == 2 ? 0 : 1));
    if (mode == 2) return;
    input = std::move(tail.front());
    CHECK(dynamic_cast<owned_arrow_scan_info&>(*input).rootless());
    CHECK(input->preparation_bytes() == 0);
    CHECK(h.backing.seen->allocations == 0);
    CHECK(coalescer->flush().empty());
  }
  auto result = h.driver->materialize_metadata_to_table(*input, *space, stream);
  CHECK(result.table.num_rows() == 0);
  CHECK(result.table.view().num_columns() == 1);
  CHECK(h.profiles->counters->arrow_batches_imported == (mode == 1 ? 1 : 0));
  CHECK(h.profiles->counters->arrow_rows_imported == 0);
  input.reset();
  if (mode == 1) {
    CHECK_FALSE(h.next());
    CHECK(coalescer->flush().empty());
  }
  stream.synchronize();
}

TEST_CASE("Arrow driver rejects foreign read identity and schema drift", "[arrow_driver][fw_s4]")
{
  auto producer = std::make_shared<pull_producer>();
  SECTION("schema drift is a physical input failure")
  {
    auto mode        = GENERATE(0, 1, 2);
    auto track_units = GENERATE(false, true);
    CAPTURE(mode, track_units);
    auto counters        = make_counters(track_units);
    auto program         = integer_program();
    std::string expected = "Arrow schema does not match its bound program";
    if (mode == 0) {
      producer->wrong_schema = true;
    } else if (mode == 1) {
      producer->observe_tree = [](ArrowSchema& schema, ArrowArray& array) {
        schema.n_children = array.n_children = 0;
      };
    } else {
      producer->rows         = {0};
      producer->observe_tree = [](ArrowSchema& schema, ArrowArray&) {
        schema.children[0]->format = "d:10,2";
      };
      program.output_types       = {cudf::data_type{cudf::type_id::DECIMAL128, -2}};
      program.input_types        = program.output_types;
      program.decimal_precisions = {9};
      expected                   = "Arrow decimal precision does not match its bound program";
    }
    harness h(counters, producer, program);
    check_physical_failure(capture_rejection([&] { h.next(); }), expected, true);
    CHECK(h.profiles->counters == counters);
    CHECK(counters->arrow_schema_drift_refusals == 1);
    CHECK(producer->calls == 1);
    CHECK(h.ledger->permits_in_flight() == 0);
    CHECK(h.backing.seen->allocated_bytes == 0);
    h.driver->stop_preparation();
    CHECK(h.driver->create_batch_coalescer()->flush().empty());
    CHECK(counters->arrow_schema_drift_refusals == 1);
  }
  SECTION("row bounds retain precedence over schema comparison")
  {
    producer->rows         = {4};
    producer->observe_tree = [](ArrowSchema& schema, ArrowArray& array) {
      schema.n_children = array.n_children = 0;
    };
    harness h(producer);
    check_physical_failure(capture_rejection([&] { h.next(); }),
                           "Arrow producer exceeded its declared row bound",
                           false);
    CHECK(h.profiles->counters->arrow_schema_drift_refusals == 0);
    CHECK(h.ledger->permits_in_flight() == 0);
    CHECK(h.backing.seen->allocated_bytes == 0);
  }
  SECTION("owned validator retains the schema drift family and reason")
  {
    harness h(producer);
    auto input           = h.next();
    auto program         = integer_program();
    program.output_types = {cudf::data_type{cudf::type_id::FLOAT32}};
    program.input_types  = program.output_types;
    auto validator       = owned_arrow_split_validator(program);
    check_physical_failure(
      capture_rejection([&] {
        validate_split_for_gpu(
          701, check_bit(later_check::host_staged), {}, *input, *validator, 4, h.profiles);
      }),
      "Arrow schema does not match its bound program",
      true);
  }
  SECTION("same schema does not authorize another bound read")
  {
    harness h(producer);
    auto input        = h.next();
    auto const& arrow = dynamic_cast<owned_arrow_scan_info const&>(*input);
    owned_arrow_scan_info foreign(
      arrow.batch(), 701, 4, 0, h.profiles, 8192, std::make_shared<int>(4));
    CHECK_THROWS_AS(validate_split_for_gpu(701,
                                           check_bit(later_check::host_staged),
                                           {},
                                           foreign,
                                           *h.driver->materializer().validator,
                                           4,
                                           h.profiles),
                    certificate_incomplete);
  }
  SECTION("errors do not become empty source")
  {
    harness h(producer);
    producer->fail_read = true;
    CHECK_THROWS_WITH(h.next(), "pull failed");
    CHECK(h.driver->create_batch_coalescer()->flush().empty());
  }
  SECTION("cancel does not become empty source")
  {
    harness h(producer);
    std::stop_source stop;
    stop.request_stop();
    try {
      h.next(sm::preparation_stop_token(stop.get_token()));
      FAIL("stopped input was accepted");
    } catch (std::system_error const& error) {
      CHECK(error.code() == std::errc::operation_canceled);
    }
    CHECK(producer->calls == 0);
    CHECK(h.driver->create_batch_coalescer()->flush().empty());
  }
}

TEST_CASE("Arrow push queue wakes an empty source and copies through admitted backing",
          "[arrow_driver]")
{
  auto counters = make_counters();
  auto q        = queue(counters);
  harness h(counters, q);
  auto gate = std::make_shared<sm::preparation_gate>();
  h.driver->bind_preparation_wakeup(sm::preparation_wakeup(gate));
  CHECK(std::holds_alternative<sm::preparation_pending>(h.driver->try_claim_split({})));
  CHECK(h.ledger->tracked_units() == 0);
  auto owner      = std::make_shared<integer_tree>();
  auto offered    = owner->offer(owner);
  auto submitting = std::async(std::launch::async, [&] { return q->offer(offered); });
  queue_guard guard{q};
  await_ready(*q);
  {
    std::unique_lock lock(gate->mutex);
    REQUIRE(gate->cv.wait_for(lock, 5s, [&] { return gate->wake_generation > 0; }));
  }
  // Concurrent offers never retain a second waiting root.
  CHECK_FALSE(q->offer(offered));
  CHECK(submitting.wait_for(0ms) == std::future_status::timeout);
  auto input = h.next();
  REQUIRE(submitting.wait_for(5s) == std::future_status::ready);
  CHECK(submitting.get());
  owner->values[0] = 999;
  auto manager     = sirius::test::operator_utils::initialize_memory_manager();
  auto* space      = manager->get_memory_space(cucascade::memory::Tier::GPU, 0);
  rmm::cuda_stream stream;
  auto result = h.driver->materialize_metadata_to_table(*input, *space, stream);
  std::array<int32_t, 3> actual{};
  REQUIRE(cudaMemcpyAsync(actual.data(),
                          result.table.view().column(0).data<int32_t>(),
                          sizeof(actual),
                          cudaMemcpyDeviceToHost,
                          stream.value()) == cudaSuccess);
  stream.synchronize();
  CHECK(actual == std::array<int32_t, 3>{11, 22, 33});
  CHECK(h.profiles->counters->arrow_batches_imported == 1);
  q->finish();
  CHECK(std::holds_alternative<sm::preparation_pending>(h.driver->try_claim_split({})));
  input.reset();
  CHECK(h.backing.seen->allocated_bytes == 0);
  CHECK_FALSE(h.next());
  CHECK(std::holds_alternative<sm::preparation_exhausted>(h.driver->try_claim_split({})));
  CHECK_FALSE(q->offer(offered));
}

TEST_CASE("Arrow push terminal transitions release blocked external offers", "[arrow_driver]")
{
  auto mode = GENERATE(0, 1, 2, 3);
  std::atomic<int> failures{0};
  auto counters = make_counters();
  auto q        = queue(counters, integer_program(), [&](auto) { ++failures; });
  harness h(counters, q);
  auto owner      = std::make_shared<integer_tree>();
  auto offered    = owner->offer(owner);
  auto submitting = std::async(std::launch::async, [&] { return q->offer(offered); });
  queue_guard guard{q};
  await_ready(*q);
  if (mode == 0) q->cancel();
  if (mode == 1) q->close();
  if (mode == 2) q->fail(std::make_exception_ptr(std::runtime_error("external failure")));
  if (mode == 3) h.driver->stop_preparation();
  REQUIRE(submitting.wait_for(5s) == std::future_status::ready);
  CHECK_FALSE(submitting.get());
  CHECK(failures == (mode == 2 ? 1 : 0));
  CHECK_THROWS(h.next());
  CHECK(h.driver->create_batch_coalescer()->flush().empty());
  CHECK(h.ledger->tracked_units() == 0);
}

TEST_CASE("Arrow push finish drains queued data before EOS", "[arrow_driver]")
{
  auto counters = make_counters();
  auto q        = queue(counters);
  harness h(counters, q);
  auto owner      = std::make_shared<integer_tree>();
  auto offered    = owner->offer(owner);
  auto submitting = std::async(std::launch::async, [&] { return q->offer(offered); });
  queue_guard guard{q};
  await_ready(*q);
  q->finish();
  auto input = h.next();
  REQUIRE(input);
  REQUIRE(submitting.wait_for(5s) == std::future_status::ready);
  CHECK(submitting.get());
  input.reset();
  CHECK_FALSE(h.next());
  CHECK(h.driver->create_batch_coalescer()->flush().empty());
}

TEST_CASE("Arrow push rejects schema and bound drift before waiting", "[arrow_driver][fw_s4]")
{
  auto mode           = GENERATE(0, 1, 2, 3, 4, 5, 6);
  auto track_units    = GENERATE(false, true);
  auto observe_stored = GENERATE(false, true);
  CAPTURE(mode, track_units, observe_stored);
  duckdb::SiriusContext statistics;
  auto counters         = statistics.physical_counters();
  counters->track_units = track_units;
  auto before           = statistics.get_transparent_execution_stats().arrow_schema_drift_refusals;
  REQUIRE(before == 0);
  int failures             = 0;
  uint64_t count_at_report = UINT64_MAX;
  std::exception_ptr reported;
  auto q = queue(counters, integer_program(), [&](auto e) {
    ++failures;
    reported        = e;
    count_at_report = counters->arrow_schema_drift_refusals.load();
  });
  harness h(counters, q);
  auto owner   = std::make_shared<integer_tree>();
  auto offered = owner->offer(owner);
  if (mode == 0) owner->field.format = "f";
  if (mode == 1) offered.retained_bytes = 8193;
  if (mode == 2) owner->array.length = 33;
  if (mode == 3) owner->field.flags = ARROW_FLAG_NULLABLE, owner->column.null_count = 1;
  if (mode == 4) owner->schema.n_children = owner->array.n_children = 0;
  if (mode == 5) owner->array.n_children = 0;
  if (mode == 6) owner->field.format = "f", offered.retained_bytes = 8193;
  auto error = capture_rejection([&] { q->offer(offered); });
  auto drift = mode == 0 || mode == 4;
  std::string expected =
    mode == 0 ? "Arrow offered schema drift"
              : (mode == 3 || mode == 5 ? "unsupported or invalid arrow_host_flat_v1 input"
                                        : "Arrow offered input exceeds its declared bounds");
  check_physical_failure(error, expected, drift);
  CHECK(reported == error);
  check_physical_failure(reported, expected, drift);
  CHECK(failures == 1);
  CHECK(h.profiles->counters == counters);
  CHECK(count_at_report == (drift ? 1 : 0));
  CHECK(counters->arrow_schema_drift_refusals == count_at_report);
  CHECK(statistics.get_transparent_execution_stats().arrow_schema_drift_refusals ==
        before + (drift ? 1 : 0));
  CHECK(owner->schema.release);
  CHECK(owner->array.release);
  CHECK(owner.use_count() == 2);
  CHECK(h.backing.seen->allocations == 0);
  CHECK(h.ledger->tracked_units() == 0);
  if (observe_stored) {
    auto stored = capture_rejection([&] { h.next(); });
    CHECK(stored == error);
    check_physical_failure(stored, expected, drift);
  } else {
    h.driver->stop_preparation();
  }
  CHECK(failures == 1);
  CHECK(counters->arrow_schema_drift_refusals == count_at_report);
  CHECK(statistics.get_transparent_execution_stats().arrow_schema_drift_refusals ==
        before + (drift ? 1 : 0));
  CHECK(h.ledger->permits_in_flight() == 0);
}

TEST_CASE("Arrow push imports selected strings and preserves zero-column cardinality",
          "[arrow_driver]")
{
  auto mode    = GENERATE(0, 1, 2, 3);
  auto large   = GENERATE(false, true);
  auto program = integer_program();
  auto owner   = std::make_shared<integer_tree>();
  std::array<int32_t, 7> offsets{-1, -1, 0, 2, 4, 6, -1};
  std::array<int64_t, 7> large_offsets{-1, -1, 0, 2, 4, 6, -1};
  std::array<char, 6> chars{'a', 'b', 'c', 'd', 'e', 'f'};
  std::array<void const*, 3> buffers{nullptr, offsets.data(), chars.data()};
  if (large) buffers[1] = large_offsets.data();
  if (mode < 2) {
    program.output_types    = {cudf::data_type{cudf::type_id::STRING}};
    owner->field.format     = large ? "U" : "u";
    owner->column.n_buffers = 3;
    owner->column.buffers   = buffers.data();
    owner->column.offset    = 1;
    owner->column.length    = 4;
    owner->array.offset     = 1;
    owner->array.length     = mode == 0 ? 3 : 0;
    if (mode == 1) buffers[1] = buffers[2] = nullptr;
  } else {
    program.output_types.clear();
    owner->schema.n_children = owner->array.n_children = 0;
    owner->array.length                                = mode == 2 ? 3 : 0;
  }
  program.input_types = program.output_types;
  auto counters       = make_counters();
  auto q              = queue(counters, program);
  harness h(counters, q, program);
  auto offered    = owner->offer(owner);
  auto submitting = std::async(std::launch::async, [&] { return q->offer(offered); });
  queue_guard guard{q};
  await_ready(*q);
  auto input = h.next();
  REQUIRE(input);
  REQUIRE(submitting.wait_for(5s) == std::future_status::ready);
  CHECK(submitting.get());
  auto manager = sirius::test::operator_utils::initialize_memory_manager();
  auto* space  = manager->get_memory_space(cucascade::memory::Tier::GPU, 0);
  rmm::cuda_stream stream;
  auto result = h.driver->materialize_metadata_to_table(*input, *space, stream);
  CHECK(result.table.num_rows() == owner->array.length);
  CHECK(result.table.view().num_columns() == (mode == 3 ? 0 : 1));
  if (mode == 0) {
    cudf::strings_column_view strings(result.table.view().column(0));
    std::array<char, 6> actual{};
    REQUIRE(cudaMemcpyAsync(actual.data(),
                            strings.chars_begin(stream),
                            actual.size(),
                            cudaMemcpyDeviceToHost,
                            stream.value()) == cudaSuccess);
    stream.synchronize();
    CHECK(actual == chars);
  }
  CHECK(h.profiles->counters->arrow_batches_imported == 1);
  q->finish();
  input.reset();
  CHECK_FALSE(h.next());
  CHECK(h.driver->create_batch_coalescer()->flush().empty());
  stream.synchronize();
}

TEST_CASE("Arrow rootless descriptors require normal EOS from the bound source", "[arrow_driver]")
{
  auto producer = std::make_shared<pull_producer>();
  harness h(producer);
  owned_arrow_scan_info forged(integer_program(), 0, h.profiles, h.identity);
  CHECK_THROWS_AS(validate_split_for_gpu(701,
                                         check_bit(later_check::host_staged),
                                         {},
                                         forged,
                                         *h.driver->materializer().validator,
                                         4,
                                         h.profiles),
                  certificate_incomplete);
  CHECK_FALSE(h.driver->has_processed_all_metadata());
  CHECK_THROWS(h.driver->next_split_provider({}));
  h.driver->stop_preparation();
  CHECK_FALSE(h.driver->has_processed_all_metadata());
  CHECK_THROWS(h.driver->next_split_provider({}));
}

TEST_CASE("Arrow copy and normalization bounds are checked before accepting an offer",
          "[arrow_driver][fw_s4]")
{
  auto mode     = GENERATE(0, 1);
  auto counters = make_counters(false);
  int failures  = 0;
  std::exception_ptr reported;
  auto q = std::make_shared<arrow_push_queue>(
    integer_program(),
    arrow_push_limits{3, 8192, mode == 0 ? 16U : 8192U, mode == 1 ? 1U : 8192U},
    [&](auto error) {
      ++failures;
      reported = error;
    },
    counters);
  harness h(counters, q);
  auto owner   = std::make_shared<integer_tree>();
  auto offered = owner->offer(owner);
  auto error   = capture_rejection([&] { q->offer(offered); });
  CHECK(reported == error);
  check_physical_failure(error, "Arrow offered input exceeds copy or normalization bounds", false);
  CHECK(failures == 1);
  CHECK(h.profiles->counters == counters);
  CHECK(counters->arrow_schema_drift_refusals == 0);
  CHECK(owner.use_count() == 2);
}

TEST_CASE("Arrow push rejects unavailable envelopes at construction",
          "[arrow_driver][arrow_profile]")
{
  auto mode = GENERATE(0, 1, 2, 3, 4, 5);
  arrow_push_limits limits{3, 8192, 8192, 8192};
  if (mode == 0) limits.max_offered_bytes = 0;
  if (mode == 1) limits.max_copy_bytes = 0;
  if (mode == 2) limits.staging_bytes = 0;
  if (mode == 3) limits.max_copy_bytes = UINT64_MAX;
  if (mode == 4) limits.staging_bytes = UINT64_MAX;
  if (mode == 5) limits.max_rows = -1;
  try {
    arrow_push_queue q(integer_program(), limits, [](auto) {});
    FAIL("invalid envelope was accepted");
  } catch (sirius::transparent::classified_execution_error const& error) {
    CHECK(error.cause == sirius::transparent::late_failure_cause::resource);
    CHECK(std::string(error.what()).find("arrow_envelope_unavailable") != std::string::npos);
  }
}

TEST_CASE("Qualified Arrow pull classifies valid row-limit refusals after validating the schema",
          "[arrow_driver][arrow_profile]")
{
  auto malformed     = GENERATE(false, true);
  auto producer      = std::make_shared<pull_producer>();
  producer->max_rows = 2;
  auto tree          = std::make_shared<qualified_integer>();
  if (malformed) tree->field.name = "wrong";
  auto profile         = std::make_shared<arrow_source_profile const>(tree->profile(tree));
  producer->next_batch = std::make_shared<owned_arrow_batch>(tree->schema,
                                                             tree->array,
                                                             tree,
                                                             *std::pmr::null_memory_resource(),
                                                             arrow_metadata_lengths{},
                                                             std::move(profile));
  harness h(producer, qualified_integer_program());
  try {
    (void)h.next();
    FAIL("invalid input was accepted");
  } catch (sirius::transparent::classified_execution_error const& error) {
    CHECK(error.cause == (malformed ? sirius::transparent::late_failure_cause::physical_input
                                    : sirius::transparent::late_failure_cause::resource));
    if (!malformed)
      CHECK(std::string(error.what()).find("arrow_batch_too_large") != std::string::npos);
    CHECK(error.reason == (malformed ? sirius::transparent::kArrowSchemaDrift : ""));
  }
  CHECK(h.profiles->counters->arrow_schema_drift_refusals == (malformed ? 1 : 0));
  CHECK(producer->calls == 1);
  CHECK_FALSE(h.gate->active);
  CHECK(h.ledger->permits_in_flight() == 0);
  CHECK(h.backing.seen->allocated_bytes == 0);
}

TEST_CASE("Arrow push preserves timestamp units and signed values through sliced imports",
          "[arrow_driver][arrow_profile]")
{
  using T             = cudf::type_id;
  auto [format, type] = GENERATE(table<char const*, T>({{"tss:", T::TIMESTAMP_SECONDS},
                                                        {"tsm:", T::TIMESTAMP_MILLISECONDS},
                                                        {"tsu:", T::TIMESTAMP_MICROSECONDS},
                                                        {"tsn:", T::TIMESTAMP_NANOSECONDS}}));
  CAPTURE(format);
  auto program         = integer_program();
  program.output_types = {cudf::data_type{type}};
  program.input_types  = program.output_types;
  auto owner           = std::make_shared<integer_tree>();
  std::array<int64_t, 5> values{111, 222, -1234567890123LL, 0, 9876543210123LL};
  uint8_t validity    = 0b00010111;
  owner->field.format = format;
  owner->field.flags  = ARROW_FLAG_NULLABLE;
  owner->buffers[0]   = &validity;
  owner->buffers[1]   = values.data();
  owner->array.offset = owner->column.offset = 1;
  owner->column.length                       = 4;
  owner->column.null_count                   = -1;
  auto counters                              = make_counters();
  auto q                                     = queue(counters, program);
  harness h(counters, q, program);
  auto submitting = std::async(std::launch::async, [&] { return q->offer(owner->offer(owner)); });
  queue_guard guard{q};
  await_ready(*q);
  auto input = h.next();
  REQUIRE(input);
  REQUIRE(submitting.wait_for(5s) == std::future_status::ready);
  CHECK(submitting.get());
  auto manager = sirius::test::operator_utils::initialize_memory_manager();
  auto* space  = manager->get_memory_space(cucascade::memory::Tier::GPU, 0);
  rmm::cuda_stream stream;
  auto result = h.driver->materialize_metadata_to_table(*input, *space, stream);
  auto column = result.table.view().column(0);
  CHECK(column.type() == program.output_types[0]);
  CHECK(column.size() == 3);
  CHECK(column.null_count() == 1);
  std::array<int64_t, 3> actual{};
  uint32_t mask = 0;
  REQUIRE(cudaMemcpyAsync(
            actual.data(), column.head(), sizeof(actual), cudaMemcpyDeviceToHost, stream.value()) ==
          cudaSuccess);
  REQUIRE(cudaMemcpyAsync(
            &mask, column.null_mask(), sizeof(mask), cudaMemcpyDeviceToHost, stream.value()) ==
          cudaSuccess);
  stream.synchronize();
  CHECK(actual.front() == values[2]);
  CHECK(actual.back() == values[4]);
  CHECK((mask & 7) == 5);
  q->finish();
  input.reset();
  CHECK_FALSE(h.next());
}

TEST_CASE("Arrow cancellation during import waits for completion and vetoes its output",
          "[arrow_driver]")
{
  auto producer = std::make_shared<pull_producer>();
  std::promise<void> entered;
  auto entered_future = entered.get_future();
  std::promise<void> resume;
  auto resumed           = resume.get_future().share();
  producer->on_normalize = [&] {
    entered.set_value();
    resumed.wait();
  };
  harness h(producer);
  auto input   = h.next();
  auto manager = sirius::test::operator_utils::initialize_memory_manager();
  auto* space  = manager->get_memory_space(cucascade::memory::Tier::GPU, 0);
  rmm::cuda_stream stream;
  auto importing = std::async(std::launch::async, [&] {
    cudaSetDevice(0);
    return h.driver->materialize_metadata_to_table(*input, *space, stream);
  });
  // Release the worker even if the assertion below fails.
  struct release_guard {
    std::promise<void>& promise;
    bool released = false;
    ~release_guard()
    {
      if (!released) promise.set_value();
    }
  } release{resume};
  REQUIRE(entered_future.wait_for(5s) == std::future_status::ready);
  h.driver->stop_preparation();
  CHECK(h.ledger->permits_in_flight() == 1);
  CHECK(h.backing.seen->allocated_bytes == 8192);
  resume.set_value();
  release.released = true;
  REQUIRE(importing.wait_for(5s) == std::future_status::ready);
  CHECK_THROWS_AS(importing.get(), duckdb::InterruptException);
  stream.synchronize();
  input.reset();
  CHECK(h.ledger->permits_in_flight() == 0);
  CHECK(h.backing.seen->allocated_bytes == 0);
  CHECK(h.driver->create_batch_coalescer()->flush().empty());
}

TEST_CASE("Arrow lanes run and close on their declared owner without publishing after stop",
          "[arrow_driver][arrow_driver_owner]")
{
  auto native      = GENERATE(false, true);
  auto stopped     = GENERATE(false, true);
  auto producer    = std::make_shared<pull_producer>();
  producer->rows   = {3, 0, 3};
  producer->domain = native ? sm::preparation_execution_domain::native_reader
                            : sm::preparation_execution_domain::query_owner;
  harness h(producer);
  sirius::exec::static_thread_pool pool(2);
  sirius::exec::scoped_dispatcher dispatcher(pool);
  sirius::pipeline::completion_handler completion;
  sm::native_reader_executor readers(1, 1);
  sm::preparation_coordinator coordinator(completion, dispatcher, {2, 2, 2, 1}, &readers);
  auto lane      = h.driver->make_preparation_source({});
  auto connector = std::make_shared<sm::split_connector>();
  sm::preparation_coordinator::source source;
  source.execution   = lane.execution;
  source.try_claim   = std::move(lane.try_claim);
  source.close       = std::move(lane.close);
  source.bind_wakeup = std::move(lane.bind_wakeup);
  source.coalescer   = std::move(lane.coalescer);
  source.destination = connector;
  source.construct   = [](auto metadata) {
    auto bytes = metadata->preparation_bytes();
    return sirius::test::preparation::make_publication(std::move(metadata), bytes);
  };
  std::atomic<int> publications{0};
  coordinator.after_publication_for_testing([&](auto, auto const&, auto) { ++publications; });
  if (stopped)
    coordinator.before_publication_for_testing(
      [&] { coordinator.request_stop(sm::stop_reason::user_cancel); });
  coordinator.add_source(std::move(source));
  h.driver->preflight_for_query();
  coordinator.arm();
  coordinator.start();
  auto deadline = std::chrono::steady_clock::now() + 10s;
  while (coordinator.snapshot().phase != sm::preparation_coordinator::lifecycle::quiescent &&
         std::chrono::steady_clock::now() < deadline) {
    coordinator.advance(1ms);
    if (connector->has_more_splits()) (void)connector->get_next_split();
  }
  auto finished = coordinator.snapshot().phase == sm::preparation_coordinator::lifecycle::quiescent;
  if (!finished) coordinator.request_stop(sm::stop_reason::failure_induced);
  coordinator.drain();
  REQUIRE(finished);
  if (!stopped && completion.terminal_error()) std::rethrow_exception(completion.terminal_error());
  CHECK(publications == (stopped ? 0 : 3));
  CHECK(producer->closes == 1);
  CHECK((producer->owner != std::this_thread::get_id()) == native);
  CHECK(h.backing.seen->allocated_bytes == 0);
  CHECK(h.ledger->tracked_units() == 0);
  CHECK_FALSE(h.driver->has_processed_all_metadata());
}

TEST_CASE("Arrow push copies nullable Boolean slices and bound decimal precision", "[arrow_driver]")
{
  auto mode    = GENERATE(0, 1, 2, 3);
  auto boolean = mode == 0;
  auto program = integer_program();
  auto owner   = std::make_shared<integer_tree>();
  std::array<__int128_t, 3> decimals{123, -456, 0};
  uint8_t values = 0b00101000, validity = 0b00011000;
  std::array<void const*, 2> buffers{};
  if (boolean) {
    program.output_types     = {cudf::data_type{cudf::type_id::BOOL8}};
    owner->field.format      = "b";
    owner->field.flags       = ARROW_FLAG_NULLABLE;
    owner->array.offset      = 2;
    owner->column.offset     = 1;
    owner->column.length     = 8;
    owner->column.null_count = 1;
    buffers                  = {&validity, &values};
  } else {
    int32_t scale              = mode == 1 ? 2 : mode == 2 ? -3 : 10;
    program.output_types       = {cudf::data_type{cudf::type_id::DECIMAL128, -scale}};
    program.decimal_precisions = {9};
    owner->field.format        = mode == 1 ? "d:9,2" : mode == 2 ? "d:9,-3" : "d:9,10,128";
    buffers                    = {nullptr, decimals.data()};
  }
  owner->column.buffers = buffers.data();
  program.input_types   = program.output_types;
  auto counters         = make_counters();
  auto q                = queue(counters, program);
  harness h(counters, q, program);
  auto offered    = owner->offer(owner);
  auto submitting = std::async(std::launch::async, [&] { return q->offer(offered); });
  queue_guard guard{q};
  await_ready(*q);
  auto input = h.next();
  REQUIRE(input);
  REQUIRE(submitting.wait_for(5s) == std::future_status::ready);
  CHECK(submitting.get());
  auto manager = sirius::test::operator_utils::initialize_memory_manager();
  auto* space  = manager->get_memory_space(cucascade::memory::Tier::GPU, 0);
  rmm::cuda_stream stream;
  auto result = h.driver->materialize_metadata_to_table(*input, *space, stream);
  auto column = result.table.view().column(0);
  CHECK(column.type() == program.output_types[0]);
  CHECK(column.size() == 3);
  if (boolean) {
    std::array<uint8_t, 3> actual{};
    uint32_t mask = 0;
    REQUIRE(cudaMemcpyAsync(actual.data(),
                            column.head<uint8_t>(),
                            actual.size(),
                            cudaMemcpyDeviceToHost,
                            stream.value()) == cudaSuccess);
    REQUIRE(cudaMemcpyAsync(
              &mask, column.null_mask(), sizeof(mask), cudaMemcpyDeviceToHost, stream.value()) ==
            cudaSuccess);
    stream.synchronize();
    CHECK(actual == std::array<uint8_t, 3>{1, 0, 1});
    CHECK((mask & 7) == 3);
    CHECK(column.null_count() == 1);
  } else {
    std::array<__int128_t, 3> actual{};
    REQUIRE(cudaMemcpyAsync(actual.data(),
                            column.head<uint8_t>(),
                            sizeof(actual),
                            cudaMemcpyDeviceToHost,
                            stream.value()) == cudaSuccess);
    stream.synchronize();
    CHECK(actual == decimals);
  }
  q->finish();
  input.reset();
  CHECK_FALSE(h.next());
}

TEST_CASE("Arrow push refuses decimal precision drift even when the scale matches",
          "[arrow_driver][fw_s4]")
{
  auto track_units           = GENERATE(false, true);
  auto counters              = make_counters(track_units);
  auto program               = integer_program();
  program.output_types       = {cudf::data_type{cudf::type_id::DECIMAL128, -2}};
  program.input_types        = program.output_types;
  program.decimal_precisions = {9};
  auto owner                 = std::make_shared<integer_tree>(0);
  owner->field.format        = "d:10,2";
  int failures               = 0;
  std::exception_ptr reported;
  auto q = queue(counters, program, [&](auto error) {
    ++failures;
    reported = error;
  });
  harness h(counters, q, program);
  auto error = capture_rejection([&] { q->offer(owner->offer(owner)); });
  CHECK(reported == error);
  check_physical_failure(error, "Arrow offered decimal precision drift", true);
  CHECK(failures == 1);
  CHECK(h.profiles->counters == counters);
  CHECK(counters->arrow_schema_drift_refusals == 1);
  CHECK(h.ledger->tracked_units() == 0);
  CHECK(owner->schema.release);
  CHECK(owner->array.release);
}

TEST_CASE("Arrow rootless empty source supports a zero-column binding", "[arrow_driver]")
{
  auto producer = std::make_shared<pull_producer>();
  producer->rows.clear();
  auto program = integer_program();
  program.output_types.clear();
  program.input_types = program.output_types;
  harness h(producer, program);
  CHECK_FALSE(h.next());
  auto tail = h.driver->create_batch_coalescer()->flush();
  REQUIRE(tail.size() == 1);
  auto manager = sirius::test::operator_utils::initialize_memory_manager();
  auto* space  = manager->get_memory_space(cucascade::memory::Tier::GPU, 0);
  rmm::cuda_stream stream;
  auto result = h.driver->materialize_metadata_to_table(*tail.front(), *space, stream);
  CHECK(result.table.num_rows() == 0);
  CHECK(result.table.view().num_columns() == 0);
  CHECK(h.profiles->counters->arrow_batches_imported == 0);
  CHECK(h.backing.seen->allocations == 0);
  stream.synchronize();
}

TEST_CASE("Arrow imported input retires only after its final descriptor owner",
          "[arrow_driver][arrow_ownership]")
{
  auto cancelled = GENERATE(false, true);
  auto producer  = std::make_shared<pull_producer>();
  producer->rows = {3, 3};
  harness h(producer);
  // Observe only input-gate returns; ledger wakeups are a separate event.
  auto returned = std::make_shared<sm::preparation_gate>();
  h.gate->wake  = sm::preparation_wakeup(returned);
  auto returns  = [&] {
    std::lock_guard lock(returned->mutex);
    return returned->wake_generation;
  };
  auto manager = sirius::test::operator_utils::initialize_memory_manager();
  auto* space  = manager->get_memory_space(cucascade::memory::Tier::GPU, 0);
  rmm::cuda_stream stream;
  auto input    = h.next();
  auto retained = dynamic_cast<owned_arrow_scan_info&>(*input).coalesced_copy();
  auto output   = h.driver->materialize_metadata_to_table(*input, *space, stream);
  CHECK(h.profiles->counters->arrow_batches_imported == 1);
  input.reset();
  CHECK(h.gate->active.load());
  CHECK(returns() == 0);
  CHECK(h.ledger->permits_in_flight() == 1);
  CHECK(std::holds_alternative<sm::preparation_pending>(h.driver->try_claim_split({})));
  if (cancelled) h.driver->stop_preparation();
  CHECK(h.ledger->permits_in_flight() == 1);
  retained.reset();
  CHECK_FALSE(h.gate->active.load());
  CHECK(returns() == 1);
  retained.reset();
  CHECK(returns() == 1);
  CHECK(h.ledger->permits_in_flight() == 0);
  CHECK(h.backing.seen->allocated_bytes == 0);
  // The detached GPU output remains usable after every host input owner has gone.
  std::array<int32_t, 3> actual{};
  REQUIRE(cudaMemcpyAsync(actual.data(),
                          output.table.view().column(0).head<int32_t>(),
                          sizeof(actual),
                          cudaMemcpyDeviceToHost,
                          stream.value()) == cudaSuccess);
  stream.synchronize();
  CHECK(actual == std::array<int32_t, 3>{11, 22, 33});
  if (!cancelled) {
    // Dropping a claimed job must also return capacity without running the producer.
    {
      auto abandoned = h.driver->try_claim_split({});
      REQUIRE(std::holds_alternative<sm::preparation_job>(abandoned));
    }
    CHECK(h.ledger->permits_in_flight() == 0);
    CHECK(producer->calls == 1);
    auto next = h.next();
    REQUIRE(next);
    CHECK(producer->calls == 2);
    next.reset();
    CHECK(h.ledger->permits_in_flight() == 0);
  } else {
    CHECK_THROWS_AS(h.driver->try_claim_split({}), duckdb::InterruptException);
  }
}

TEST_CASE("Arrow push copy retains foreign backing after a cancelled offer returns",
          "[arrow_driver][arrow_ownership]")
{
  auto cancelled = GENERATE(false, true);
  auto counters  = make_counters();
  auto q         = queue(counters);
  harness h(counters, q);
  sirius::test::preparation::hold copying;
  h.backing.seen->before_allocate = [&] { copying.wait(); };
  std::promise<std::thread::id> foreign_release;
  auto released = foreign_release.get_future();
  auto owner    = std::shared_ptr<integer_tree>(new integer_tree, [&](integer_tree* tree) {
    delete tree;
    foreign_release.set_value(std::this_thread::get_id());
  });
  std::weak_ptr<integer_tree> weak = owner;
  auto submitting = std::async(std::launch::async, [q, owner = std::move(owner)]() mutable {
    auto id       = std::this_thread::get_id();
    bool accepted = q->offer(owner->offer(owner));
    owner.reset();
    return std::pair{accepted, id};
  });
  queue_guard guard{q};
  await_ready(*q);
  auto claim = h.driver->try_claim_split({});
  REQUIRE(std::holds_alternative<sm::preparation_job>(claim));
  auto copying_job = std::async(std::launch::async,
                                [job = std::get<sm::preparation_job>(std::move(claim))]() mutable {
                                  auto id = std::this_thread::get_id();
                                  try {
                                    return std::pair{job.run({}), id};
                                  } catch (duckdb::InterruptException const&) {
                                    return std::pair{std::unique_ptr<scan_info>{}, id};
                                  }
                                });
  // Declared after both futures so failed assertions unblock before joining them.
  struct unblock {
    sirius::test::preparation::hold& hold;
    ~unblock() { hold.release(); }
  } unblocker{copying};
  REQUIRE(copying.await());
  std::thread::id submit_thread;
  if (cancelled) {
    q->cancel();
    REQUIRE(submitting.wait_for(5s) == std::future_status::ready);
    auto submitted = submitting.get();
    submit_thread  = submitted.second;
    CHECK_FALSE(submitted.first);
    CHECK(copying_job.wait_for(0ms) == std::future_status::timeout);
    CHECK(released.wait_for(0ms) == std::future_status::timeout);
    auto still_readable = weak.lock();
    REQUIRE(still_readable);
    CHECK(still_readable->values[2] == 33);
  }
  copying.release();
  REQUIRE(copying_job.wait_for(5s) == std::future_status::ready);
  auto [input, copy_thread] = copying_job.get();
  if (!cancelled) {
    REQUIRE(submitting.wait_for(5s) == std::future_status::ready);
    auto submitted = submitting.get();
    submit_thread  = submitted.second;
    CHECK(submitted.first);
    REQUIRE(input);
    CHECK(input->preparation_bytes() > 0);
  } else {
    CHECK_FALSE(input);
  }
  REQUIRE(released.wait_for(5s) == std::future_status::ready);
  auto final_thread = released.get();
  CHECK((final_thread == copy_thread || final_thread == submit_thread));
  if (cancelled) CHECK(final_thread == copy_thread);
  CHECK(weak.expired());
  input.reset();
  CHECK(h.ledger->permits_in_flight() == 0);
  CHECK(h.backing.seen->allocated_bytes == 0);
}

TEST_CASE("Arrow returned tree callbacks run once on the final retaining thread",
          "[arrow_driver][arrow_ownership]")
{
  auto producer = std::make_shared<pull_producer>();
  std::thread::id schema_thread, array_thread;
  int schemas = 0, arrays = 0;
  struct observation {
    std::thread::id* thread;
    int* count;
  } schema{&schema_thread, &schemas}, array{&array_thread, &arrays};
  producer->observe_tree = [&](ArrowSchema& s, ArrowArray& a) {
    s.private_data = &schema;
    a.private_data = &array;
    s.release      = [](ArrowSchema* value) {
      auto& seen   = *static_cast<observation*>(value->private_data);
      *seen.thread = std::this_thread::get_id();
      ++*seen.count;
      value->release = nullptr;
    };
    a.release = [](ArrowArray* value) {
      auto& seen   = *static_cast<observation*>(value->private_data);
      *seen.thread = std::this_thread::get_id();
      ++*seen.count;
      value->release = nullptr;
    };
  };
  harness h(producer);
  auto input    = h.next();
  auto retained = dynamic_cast<owned_arrow_scan_info&>(*input).coalesced_copy();
  auto manager  = sirius::test::operator_utils::initialize_memory_manager();
  auto* space   = manager->get_memory_space(cucascade::memory::Tier::GPU, 0);
  rmm::cuda_stream stream;
  auto output = h.driver->materialize_metadata_to_table(*input, *space, stream);
  REQUIRE(output.table.num_rows() == 3);
  input.reset();
  CHECK(schemas == 0);
  CHECK(arrays == 0);
  auto disposing = std::async(std::launch::async, [retained = std::move(retained)]() mutable {
    retained.reset();
    return std::this_thread::get_id();
  });
  auto thread    = disposing.get();
  CHECK(thread != std::this_thread::get_id());
  CHECK(schema_thread == thread);
  CHECK(array_thread == thread);
  CHECK(schemas == 1);
  CHECK(arrays == 1);
  CHECK(h.ledger->permits_in_flight() == 0);
}

namespace {
struct retry_observation {
  std::atomic<unsigned> retries{0};
  owned_arrow_batch const* root = nullptr;
  std::atomic<bool> same_root{true};
  std::function<void()> on_retry;
};
class observed_arrow_task final : public sirius::pipeline::gpu_pipeline_task {
 public:
  observed_arrow_task(uint64_t id,
                      std::unique_ptr<sirius::pipeline::sirius_pipeline_task_local_state> local,
                      std::shared_ptr<sirius::pipeline::sirius_pipeline_task_global_state> global,
                      std::shared_ptr<retry_observation> observation)
    : gpu_pipeline_task(id, {}, std::move(local), std::move(global)),
      observed(std::move(observation))
  {
  }
  std::unique_ptr<sirius::pipeline::gpu_pipeline_task> create_rescheduled_task(
    uint64_t id, std::unique_ptr<sirius::pipeline::sirius_pipeline_task_local_state> local) override
  {
    auto const& state   = local->cast<sirius::pipeline::gpu_pipeline_task_local_state>();
    auto const& input   = dynamic_cast<scan_operator_input const&>(*state._input_data);
    auto const& split   = dynamic_cast<owned_arrow_scan_info const&>(input.get_scan_info());
    observed->same_root = observed->same_root && split.batch().get() == observed->root;
    ++observed->retries;
    if (observed->on_retry) observed->on_retry();
    return std::make_unique<observed_arrow_task>(
      id, std::move(local), get_shared_global_state(), observed);
  }

 private:
  std::shared_ptr<retry_observation> observed;
};
class arrow_engine_sink final : public sirius::op::sirius_physical_operator {
 public:
  arrow_engine_sink()
    : sirius_physical_operator(sirius::op::SiriusPhysicalOperatorType::FILTER, {}, 0)
  {
  }
  sirius::OrderPreservationType output_order_preservation() const override
  {
    return sirius::OrderPreservationType::INSERTION_ORDER;
  }
  void sink(sirius::op::operator_data const& input, ::cuda::stream_ref stream) override
  {
    auto const& batches = dynamic_cast<sirius::op::pipelineable_operator_data const&>(input);
    for (auto const& batch : batches.get_data_batches()) {
      auto view = sirius::get_cudf_table_view(*batch);
      columns   = view.num_columns();
      values.resize(view.num_rows());
      if (!values.empty() && cudaMemcpyAsync(values.data(),
                                             view.column(0).head(),
                                             values.size() * sizeof(int32_t),
                                             cudaMemcpyDeviceToHost,
                                             stream.get()) != cudaSuccess)
        throw std::runtime_error("sink copy failed");
      retained = batch;
      ++publications;
    }
  }
  std::vector<int32_t> values;
  int columns = 0, publications = 0;
  std::shared_ptr<cucascade::data_batch> retained;
};
// A trusted test connector binds a non-replayable stream to the production scan,
// coordinator, scheduler and GPU executor; only root creation and the result sink are fixtures.
struct arrow_engine_fixture {
  decltype(sirius::test::operator_utils::initialize_memory_manager()) manager =
    sirius::test::operator_utils::initialize_memory_manager();
  sm::host_reservation_provider backing{*manager};
  std::shared_ptr<sirius::transparent::read_view_registry> registry =
    std::make_shared<sirius::transparent::read_view_registry>();
  bound_read_view view;
  materialization_program program;
  std::shared_ptr<sirius::pipeline::completion_handler> completion =
    std::make_shared<sirius::pipeline::completion_handler>();
  std::shared_ptr<arrow_push_queue> queue;
  std::unique_ptr<harness> input;
  std::unique_ptr<sirius_gpu_scan_operator> scan;
  arrow_engine_sink sink;
  std::shared_ptr<sirius::pipeline::sirius_pipeline> pipeline;
  std::shared_ptr<sirius::pipeline::sirius_pipeline_task_global_state> global;
  sirius::exec::query_lifecycle_registry lifecycle;
  std::unique_ptr<sirius::pipeline::task_scheduler> scheduler;
  std::unique_ptr<sirius::creator::task_creator> creator;
  std::shared_ptr<retry_observation> retries      = std::make_shared<retry_observation>();
  std::shared_ptr<std::atomic<unsigned>> released = std::make_shared<std::atomic<unsigned>>(0);
  std::vector<scan_execution_record> records;

  arrow_engine_fixture()
  {
    bound_read_identity identity;
    identity.source = {"test_arrow_stream", source_kind::stream_source, "test.arrow.qualified.v1"};
    identity.data_view   = stream_identity{17};
    identity.bound_types = {duckdb::LogicalType::INTEGER};
    identity.bound_names = {"value"};
    view.identity        = make_bound_read_identity(std::move(identity), {});
    view.replay_policy   = {"test_arrow_stream",
                            sirius::transparent::byte_source_class::stream,
                            false,
                            "Arrow stream has no replay authority"};
    program              = qualified_integer_program();
    program.contract =
      allocate_scan_contract(*registry,
                             {},
                             program.generation,
                             1,
                             std::make_shared<bound_read_view const>(view),
                             {},
                             {},
                             {materializer_kind::stream, "arrow_host_flat_v1"},
                             {duckdb::LogicalType::INTEGER, duckdb::LogicalType::INTEGER});
    // Duplicate projection makes skipped post-processing observable; filter removes 11.
    program.output_types.push_back(program.output_types[0]);
    program.predicates = 1;
    auto counters      = make_counters();
    queue              = std::make_shared<arrow_push_queue>(
      // Leave room for the real scan's 8x cold estimate and the executor's 2x retry floor.
      program,
      8ULL << 20,
      [done = completion](auto error) { done->report_error(error); },
      counters);
    input = std::make_unique<harness>(
      queue, program, true, &backing, nullptr, 1, counters, [&](arrow_import_binding& binding) {
        binding.read_identity  = view.identity;
        binding.output_columns = {0, 0};
        auto integer           = sirius::logical_type::make(sirius::type_id::INTEGER);
        binding.predicate      = std::make_shared<sirius::ast::node>(sirius::ast::comparison{
          sirius::comparison_type::gt,
          std::make_unique<sirius::ast::node>(sirius::ast::reference{0, integer}),
          std::make_unique<sirius::ast::node>(sirius::ast::constant{int32_t{11}, integer})});
      });
    registry->profiles = input->profiles;
    registry->bind_split_validator(program.contract, input->driver->materializer().validator);
    certification_result certified;
    certified.verdict      = eligibility_verdict::supported;
    certified.later_checks = check_bit(later_check::host_staged);
    registry->record_verdict(program.contract, certified);
    registry->publish_correspondence(
      certificate_evidence_scope::binding_correspondence, "test stream", std::array{view});
    input->profiles->counters->scan_execution_for_testing =
      [this](bool, scan_execution_record const& record) { records.push_back(record); };
    scan = std::make_unique<sirius_gpu_scan_operator>(
      sirius::from_duckdb_vec(duckdb::vector<duckdb::LogicalType>{duckdb::LogicalType::INTEGER,
                                                                  duckdb::LogicalType::INTEGER}),
      3,
      input->driver,
      program.contract,
      nullptr,
      registry);
    scan->operator_id = 1;
    sink.operator_id  = 2;
    namespace pl      = sirius::pipeline;
    pipeline = std::make_shared<pl::sirius_pipeline>(pl::pipeline_build_context{nullptr, true});
    pipeline->set_pipeline_id(73);
    pl::sirius_pipeline_build_state build;
    build.set_pipeline_source(*pipeline, *scan);
    build.add_pipeline_operator(*pipeline, *scan);
    build.set_pipeline_sink(*pipeline, sink, 1);
    global = std::make_shared<pl::sirius_pipeline_task_global_state>(
      pipeline, sirius::test::make_test_telemetry_context());
    global->set_completion_handler(completion);
  }
  ~arrow_engine_fixture()
  {
    queue->cancel();
    if (scheduler) scheduler->stop();
    sink.retained.reset();  // Free output before executor streams and allocator.
    input->profiles->counters->scan_execution_for_testing = {};
  }
  std::unique_ptr<sirius::op::operator_data> prepare(
    int rows, arrow_source_memory memory = arrow_source_memory::pageable_host)
  {
    if (rows >= 0) {
      auto slot = queue->acquire_offered_slot();
      if (memory == arrow_source_memory::pageable_host) {
        fill_slot(slot, rows, [count = released] { ++*count; });
      } else {
        auto tree      = std::make_shared<qualified_integer>(rows);
        tree->released = [count = released] { ++*count; };
        void* values   = nullptr;
        auto status    = memory == arrow_source_memory::device
                           ? cudaMalloc(&values, sizeof(tree->values))
                           : cudaMallocHost(&values, sizeof(tree->values));
        REQUIRE(status == cudaSuccess);
        std::shared_ptr<void> owner(values, [memory](void* p) {
          if (memory == arrow_source_memory::device)
            cudaFree(p);
          else
            cudaFreeHost(p);
        });
        if (memory == arrow_source_memory::device)
          REQUIRE(cudaMemcpy(values, tree->values, sizeof(tree->values), cudaMemcpyHostToDevice) ==
                  cudaSuccess);
        else
          std::memcpy(values, tree->values, sizeof(tree->values));
        auto profile     = tree->profile(tree);
        tree->buffers[1] = values;
        profile.memory   = memory;
        if (memory == arrow_source_memory::device)
          REQUIRE(cudaGetDevice(&profile.device) == cudaSuccess);
        profile.leases      = std::move(owner);
        profile.allocations = {{1, values, sizeof(tree->values)}};
        profile.buffers     = {{{}, arrow_buffer_use{1, 0, sizeof(tree->values)}}};
        slot.set_input(tree->schema, tree->array, tree, std::move(profile));
      }
      REQUIRE(queue->offer(slot) == arrow_push_queue::offer_outcome::accepted);
    }
    queue->finish();
    sirius::exec::static_thread_pool pool(2);
    sirius::exec::scoped_dispatcher dispatcher(pool);
    sm::preparation_coordinator coordinator(*completion, dispatcher, {2, 2, 2, 1});
    auto lane = input->driver->make_preparation_source({});
    sm::preparation_coordinator::source source;
    source.execution   = lane.execution;
    source.try_claim   = std::move(lane.try_claim);
    source.bind_wakeup = std::move(lane.bind_wakeup);
    source.close       = std::move(lane.close);
    source.coalescer   = std::move(lane.coalescer);
    source.destination = scan->get_shared_split_connector();
    source.construct   = [](auto metadata) {
      auto bytes = metadata->preparation_bytes();
      return sirius::test::preparation::make_publication(std::move(metadata), bytes);
    };
    coordinator.add_source(std::move(source));
    input->driver->preflight_for_query();
    coordinator.arm();
    coordinator.start();
    auto deadline = std::chrono::steady_clock::now() + 5s;
    while (!scan->get_split_connector().has_more_splits() && !completion->has_error() &&
           std::chrono::steady_clock::now() < deadline)
      coordinator.advance(1ms);
    REQUIRE_FALSE(completion->has_error());
    REQUIRE(scan->get_split_connector().has_more_splits());
    auto result = scan->get_next_task_input_data();
    REQUIRE(result);
    auto const& split = dynamic_cast<owned_arrow_scan_info const&>(
      dynamic_cast<scan_operator_input const&>(*result).get_scan_info());
    retries->root = split.batch().get();
    coordinator.drain();
    return result;
  }
  void run(std::unique_ptr<sirius::op::operator_data> data)
  {
    namespace pl = sirius::pipeline;
    lifecycle.open_query(pipeline->get_query_id());
    sirius::exec::thread_pool_config config;
    config.num_threads        = 1;
    config.thread_name_prefix = "arrow-engine";
    scheduler                 = std::make_unique<pl::task_scheduler>(
      lifecycle, config, *manager, sirius::test::make_test_telemetry_context());
    creator = std::make_unique<sirius::creator::task_creator>(
      lifecycle, sirius::creator::task_creator_config{}, *manager);
    creator->set_task_scheduler(*scheduler);
    scheduler->set_task_creator(*creator);
    scheduler->start();
    auto local = std::make_unique<pl::gpu_pipeline_task_local_state>(std::move(data));
    local->set_preferred_device_id(0);
    scheduler->schedule(
      std::make_unique<observed_arrow_task>(1, std::move(local), global, retries));
    auto deadline = std::chrono::steady_clock::now() + 10s;
    while (!pipeline->get_tasks_completed() && !completion->has_error() &&
           std::chrono::steady_clock::now() < deadline)
      std::this_thread::sleep_for(1ms);
    bool finished = pipeline->get_tasks_completed() || completion->has_error();
    sirius::test::retire_scheduler_query(
      lifecycle, *scheduler, pipeline->get_query_id(), false, creator.get());
    REQUIRE(finished);
  }
};

// Keep the production task's reservation, execution, sink and exception cleanup paths.
// Only the operator boundary is supplied by the fixture.
class arrow_task_operator final : public sirius::op::sirius_physical_operator {
 public:
  sirius::OrderPreservationType output_order_preservation() const override
  {
    return sirius::OrderPreservationType::NO_ORDER;
  }

  arrow_import_ingestible& driver;
  cucascade::memory::memory_space& space;
  int mode;
  int32_t* actual = nullptr;
  std::shared_ptr<cucascade::data_batch> published;
  bool reached_sink = false;
  sirius::test::preparation::hold pending_read;
  std::atomic<bool> read_timed_out{false};
  arrow_task_operator(arrow_import_ingestible& d, cucascade::memory::memory_space& s, int m)
    : sirius_physical_operator(
        sirius::op::SiriusPhysicalOperatorType::GPU_SCAN,
        sirius::from_duckdb_vec(duckdb::vector<duckdb::LogicalType>{duckdb::LogicalType::INTEGER}),
        3),
      driver(d),
      space(s),
      mode(m)
  {
    if (cudaMallocHost(reinterpret_cast<void**>(&actual), 3 * sizeof(int32_t)) != cudaSuccess)
      throw std::runtime_error("sink observation allocation failed");
    std::fill_n(actual, 3, 0);
  }
  ~arrow_task_operator() override { cudaFreeHost(actual); }
  std::unique_ptr<sirius::op::operator_data> execute(sirius::op::operator_data const& input,
                                                     ::cuda::stream_ref stream) override
  {
    auto const& split = dynamic_cast<scan_operator_input const&>(input);
    auto result       = driver.materialize_metadata_to_table(split.get_scan_info(), space, stream);
    auto table        = driver.post_filter_and_project(std::move(result), space, stream);
    return std::make_unique<sirius::op::pipelineable_operator_data>(
      std::vector<std::shared_ptr<cucascade::data_batch>>{
        sirius::make_data_batch(std::move(table), space, stream, batch_telemetry())});
  }
  void sink(sirius::op::operator_data const& input, ::cuda::stream_ref stream) override
  {
    reached_sink        = true;
    auto const& batches = dynamic_cast<sirius::op::pipelineable_operator_data const&>(input);
    auto const& batch   = batches.get_data_batches().front();
    auto view           = sirius::get_cudf_table_view(*batch);
    auto status         = cudaLaunchHostFunc(
      stream.get(),
      [](void* data) {
        auto& self = *static_cast<arrow_task_operator*>(data);
        try {
          self.pending_read.wait();
        } catch (...) {
          self.read_timed_out = true;
        }
      },
      this);
    if (status != cudaSuccess) throw std::runtime_error("sink hold failed");
    status = cudaMemcpyAsync(actual,
                             view.column(0).head<int32_t>(),
                             3 * sizeof(int32_t),
                             cudaMemcpyDeviceToHost,
                             stream.get());
    if (status != cudaSuccess) throw std::runtime_error("sink copy failed");
    if (mode == 1) throw std::runtime_error("sink failure after enqueue");
    if (mode == 2) {
      driver.stop_preparation();
      throw duckdb::InterruptException();
    }
    published = batch;
  }
};
}  // namespace

TEST_CASE("Qualified Arrow executes through production scan and executor with the same retry root",
          "[arrow_driver][arrow_engine][arrow_qualified]")
{
  auto rows   = GENERATE(0, 3);
  auto oom    = GENERATE(false, true);
  auto memory = GENERATE(arrow_source_memory::device,
                         arrow_source_memory::pinned_host,
                         arrow_source_memory::pageable_host);
  arrow_engine_fixture f;
  auto data = f.prepare(rows, memory);
  CHECK(f.input->profiles->counters->arrow_batches_imported == 0);
  CHECK(f.released->load() == 0);
  auto injections                       = std::make_shared<test_injections>();
  injections->gpu_task_oom              = oom ? 1 : 0;
  injections->gpu_task_launch_error     = oom ? 0 : 1;
  injections->gpu_task_retry_limit      = 2;
  injections->gpu_task_retry_backoff_ms = 0;
  f.completion->injections              = injections;
  f.run(std::move(data));
  REQUIRE_FALSE(f.completion->has_error());
  CHECK(f.retries->retries == 1);
  CHECK(f.retries->same_root);
  CHECK(f.released->load() == 1);
  CHECK(f.queue->observe().accepted == 0);
  CHECK(f.input->ledger->permits_in_flight() == 0);
  CHECK(f.sink.publications == 1);
  CHECK(f.sink.columns == 2);
  CHECK(f.sink.values == (rows ? std::vector<int32_t>{22, 33} : std::vector<int32_t>{}));
  CHECK(f.input->profiles->counters->arrow_batches_imported == 1);
  REQUIRE(f.records.size() == 2);
  CHECK(f.records[0].producer == execution_record_producer::arrow_import);
  CHECK(f.records[0].authoritative);
  CHECK(f.records[0].predicates.empty());
  CHECK_FALSE(f.records[0].whole_static_complete);
  CHECK_FALSE(f.records[0].projection_complete);
  CHECK(f.records[0].layout_width == 1);
  CHECK(f.records[1].whole_static_complete);
  CHECK(f.records[1].projection_complete);
  CHECK(f.records[1].layout_width == 2);
  for (auto const& record : f.records) {
    CHECK(record.visibility_complete);
    CHECK(record.visibility_applied.empty());
    CHECK(record.positions == row_position_state::none);
  }
  REQUIRE(f.sink.retained);
  CHECK(sirius::get_cudf_table_view(*f.sink.retained).num_rows() == (rows ? 2 : 0));
}

TEST_CASE("Qualified Arrow preserves executor exhaustion and cancellation without CPU replay",
          "[arrow_driver][arrow_engine][arrow_qualified]")
{
  auto mode = GENERATE(0, 1, 2);
  arrow_engine_fixture f;
  auto data                             = f.prepare(3);
  auto injections                       = std::make_shared<test_injections>();
  injections->gpu_task_oom              = mode == 1 ? 0 : 100;
  injections->gpu_task_launch_error     = mode == 1 ? 100 : 0;
  injections->gpu_task_retry_limit      = 1;
  injections->gpu_task_retry_backoff_ms = 0;
  f.completion->injections              = injections;
  if (mode == 2)
    f.retries->on_retry = [&] {
      f.input->driver->stop_preparation();
      f.completion->report_error(std::make_exception_ptr(duckdb::InterruptException()));
    };
  f.run(std::move(data));
  REQUIRE(f.completion->has_error());
  using cause = sirius::transparent::late_failure_cause;
  if (mode == 2)
    CHECK_THROWS_AS(std::rethrow_exception(f.completion->terminal_error()),
                    duckdb::InterruptException);
  else
    CHECK(f.completion->failure().cause ==
          (mode == 0 ? cause::oom_exhausted : cause::retry_exhausted));
  CHECK(f.retries->same_root);
  CHECK(f.sink.publications == 0);
  CHECK(f.released->load() == 1);
  CHECK(f.queue->observe().accepted == 0);
  CHECK(f.input->ledger->permits_in_flight() == 0);
  sirius::transparent::plan_source_policy policy;
  policy.scans = {f.view.replay_policy};
  CHECK_FALSE(policy.cpu_replay_permitted());
  CHECK_THROWS(sirius::transparent::require_cpu_replay(
    policy, "test Arrow input", f.completion->failure().detail));
}

TEST_CASE("Qualified empty Arrow source emits one typed carrier through the real scan",
          "[arrow_driver][arrow_engine]")
{
  arrow_engine_fixture f;
  auto data            = f.prepare(-1);
  auto const& metadata = dynamic_cast<scan_operator_input const&>(*data).get_scan_info();
  CHECK(dynamic_cast<owned_arrow_scan_info const&>(metadata).rootless());
  f.run(std::move(data));
  REQUIRE_FALSE(f.completion->has_error());
  CHECK(f.sink.publications == 1);
  CHECK(f.sink.columns == 2);
  CHECK(f.sink.values.empty());
  CHECK(f.released->load() == 0);
  CHECK(f.input->profiles->counters->arrow_batches_imported == 0);
  CHECK(f.input->ledger->permits_in_flight() == 0);
}

TEST_CASE("Arrow real consumer rejects stale or inconsistent raw execution records",
          "[arrow_driver][arrow_engine]")
{
  auto mode = GENERATE(0, 1, 2, 3, 4, 5);
  arrow_engine_fixture f;
  auto data   = f.prepare(3);
  auto* space = f.manager->get_memory_space(cucascade::memory::Tier::GPU, 0);
  rmm::cuda_stream stream;
  auto const& metadata = dynamic_cast<scan_operator_input const&>(*data).get_scan_info();
  auto raw             = f.input->driver->materialize_metadata_to_table(metadata, *space, stream);
  REQUIRE(raw.execution);
  if (mode == 0) ++raw.execution->generation;
  if (mode == 1) raw.execution->layout = std::make_shared<int>(0);
  if (mode == 2) raw.execution->layout_width = 99;
  if (mode == 3) raw.execution->authoritative = false;
  if (mode == 4) raw.required_visibility = std::make_shared<visibility_bindings const>();
  if (mode == 5) raw.execution->projection_complete = true;
  CHECK_THROWS_AS(f.input->driver->post_filter_and_project(std::move(raw), *space, stream),
                  std::invalid_argument);
  CHECK(f.records.empty());
  stream.synchronize();
}

TEST_CASE("Pending Arrow allows another source to progress and wakes the parked coordinator",
          "[arrow_driver][arrow_admission][arrow_wakeup]")
{
  auto transition    = GENERATE(0, 1, 2);  // accept, finish, fail
  auto program       = qualified_integer_program();
  auto waiting_queue = std::make_shared<arrow_push_queue>(program, 8192, [](auto) {});
  auto ready_queue   = std::make_shared<arrow_push_queue>(program, 8192, [](auto) {});
  queue_guard waiting_guard{waiting_queue}, ready_guard{ready_queue};
  harness waiting(waiting_queue, program), ready(ready_queue, program);
  auto slot = ready_queue->acquire_offered_slot();
  fill_slot(slot);
  REQUIRE(ready_queue->offer(slot) == arrow_push_queue::offer_outcome::accepted);
  ready_queue->finish();
  sirius::exec::static_thread_pool pool(2);
  sirius::exec::scoped_dispatcher dispatcher(pool);
  sirius::pipeline::completion_handler completion;
  sm::preparation_coordinator coordinator(completion, dispatcher, {2, 2, 2, 1});
  auto waiting_output = std::make_shared<sm::split_connector>();
  auto ready_output   = std::make_shared<sm::split_connector>();
  auto add            = [&](harness& h, std::shared_ptr<sm::split_connector> output) {
    auto lane = h.driver->make_preparation_source({});
    sm::preparation_coordinator::source source;
    source.execution   = lane.execution;
    source.try_claim   = std::move(lane.try_claim);
    source.bind_wakeup = std::move(lane.bind_wakeup);
    source.close       = std::move(lane.close);
    source.coalescer   = std::move(lane.coalescer);
    source.destination = std::move(output);
    source.construct   = [](auto metadata) {
      auto bytes = metadata->preparation_bytes();
      return sirius::test::preparation::make_publication(std::move(metadata), bytes);
    };
    coordinator.add_source(std::move(source));
  };
  add(waiting, waiting_output);
  add(ready, ready_output);
  std::promise<void> parked;
  auto idle = parked.get_future();
  std::atomic<bool> signaled{false};
  coordinator.before_control_wait_for_testing([&] {
    if (ready_output->is_discovery_complete() && !signaled.exchange(true)) parked.set_value();
  });
  auto query_owner = std::async(std::launch::async, [&] {
    coordinator.arm();
    coordinator.start();
    while (coordinator.snapshot().phase != sm::preparation_coordinator::lifecycle::quiescent)
      coordinator.advance(std::chrono::hours{1});
    coordinator.drain();
  });
  struct stop_control {
    sm::preparation_coordinator& control;
    std::future<void>& owner;
    ~stop_control()
    {
      control.request_stop(sm::stop_reason::normal_eos);
      if (owner.valid()) owner.wait();
    }
  } cleanup{coordinator, query_owner};
  REQUIRE(idle.wait_for(5s) == std::future_status::ready);
  CHECK(ready_output->has_more_splits());
  CHECK_FALSE(waiting_output->is_discovery_complete());
  CHECK_FALSE(waiting_output->has_more_splits());
  CHECK(waiting.ledger->tracked_units() == 0);
  // No test wake or periodic advance: only the producer transition wakes control.
  if (transition == 0) {
    auto offered = waiting_queue->acquire_offered_slot();
    fill_slot(offered);
    CHECK(waiting_queue->offer(offered) == arrow_push_queue::offer_outcome::accepted);
    waiting_queue->finish();
  } else if (transition == 1)
    waiting_queue->finish();
  else
    waiting_queue->fail(std::make_exception_ptr(std::runtime_error("external Arrow failure")));
  auto deadline = std::chrono::steady_clock::now() + 5s;
  while (!waiting_output->is_discovery_complete() && std::chrono::steady_clock::now() < deadline)
    std::this_thread::sleep_for(1ms);
  REQUIRE(waiting_output->is_discovery_complete());
  ready_output->get_next_split().reset();
  if (transition == 2)
    CHECK(completion.has_error());
  else {
    REQUIRE(waiting_output->has_more_splits());
    auto output = waiting_output->get_next_split();
    REQUIRE(output);
    auto const& input = dynamic_cast<scan_operator_input const&>(**output);
    CHECK(dynamic_cast<owned_arrow_scan_info const&>(input.get_scan_info()).rootless() ==
          (transition == 1));
    output.reset();
    CHECK_FALSE(completion.has_error());
  }
  coordinator.request_stop(sm::stop_reason::normal_eos);
  REQUIRE(query_owner.wait_for(5s) == std::future_status::ready);
  REQUIRE_NOTHROW(query_owner.get());
}

TEST_CASE("Arrow outputs survive input retirement through task success and sink failure",
          "[arrow_driver][arrow_ownership][arrow_task_lifetime]")
{
  auto mode     = GENERATE(0, 1, 2);
  auto producer = std::make_shared<pull_producer>();
  harness h(producer);
  auto manager = sirius::test::operator_utils::initialize_memory_manager();
  auto* space  = manager->get_memory_space(cucascade::memory::Tier::GPU, 0);
  auto* allocator =
    space->get_memory_resource_as<cucascade::memory::reservation_aware_resource_adaptor>();
  REQUIRE(allocator);
  auto baseline = allocator->get_total_allocated_bytes();
  rmm::cuda_stream stream;
  arrow_task_operator op(*h.driver, *space, mode);
  op.operator_id = 1;
  namespace pl   = sirius::pipeline;
  auto pipeline  = std::make_shared<pl::sirius_pipeline>(pl::pipeline_build_context{nullptr});
  pipeline->set_pipeline_id(71);
  pl::sirius_pipeline_build_state build;
  build.set_pipeline_source(*pipeline, op);
  build.add_pipeline_operator(*pipeline, op);
  build.set_pipeline_sink(*pipeline, op, 1);
  auto global = std::make_shared<pl::sirius_pipeline_task_global_state>(
    pipeline, sirius::test::make_test_telemetry_context());
  auto input = std::make_unique<scan_operator_input>(h.next());
  auto local = std::make_unique<pl::gpu_pipeline_task_local_state>(std::move(input));
  auto task  = std::make_unique<pl::gpu_pipeline_task>(
    1, std::vector<cucascade::shared_data_repository*>{}, std::move(local), global);
  auto info        = task->get_estimated_reservation_size_info(space);
  auto reservation = space->make_reservation_or_null(1 << 20);
  REQUIRE(reservation);
  task->local_state()->cast<pl::gpu_pipeline_task_local_state>().set_reservation(
    std::move(reservation), info);
  task->telemetry_fsm().queued(
    {.queue = {.target = quent::task_queue::TaskQueueId{quent::nil_uuid()},
               .data   = {.entries = 1}}});
  task->telemetry_fsm().reserving(
    {.requested_bytes      = 0,
     .input_basis          = 0,
     .peak_estimate        = 0,
     .bytes_to_materialize = 0,
     .manager_thread       = {
             .target = quent::task_manager_loop_thread::TaskManagerLoopThreadId{quent::nil_uuid()},
             .data   = {}}});
  auto executing = std::async(std::launch::async, [&] {
    if (cudaSetDevice(0) != cudaSuccess) throw std::runtime_error("task device unavailable");
    task->execute(stream);
  });
  struct release_read {
    sirius::test::preparation::hold& hold;
    ~release_read() { hold.release(); }
  } release{op.pending_read};
  REQUIRE(op.pending_read.await());
  CHECK(executing.wait_for(50ms) == std::future_status::timeout);
  CHECK(h.ledger->permits_in_flight() == 0);
  CHECK(allocator->get_total_allocated_bytes() > baseline);
  op.pending_read.release();
  REQUIRE(executing.wait_for(5s) == std::future_status::ready);
  if (mode == 0) REQUIRE_NOTHROW(executing.get());
  if (mode == 1) REQUIRE_THROWS_WITH(executing.get(), "sink failure after enqueue");
  if (mode == 2) REQUIRE_THROWS_AS(executing.get(), duckdb::InterruptException);
  CHECK_FALSE(op.read_timed_out.load());
  // No fixture synchronization before these observations: execute must join the sink copy.
  REQUIRE(op.reached_sink);
  CHECK(std::array<int32_t, 3>{op.actual[0], op.actual[1], op.actual[2]} ==
        std::array<int32_t, 3>{11, 22, 33});
  CHECK(h.profiles->counters->arrow_batches_imported == 1);
  CHECK(h.ledger->permits_in_flight() == 0);
  CHECK(h.backing.seen->allocated_bytes == 0);
  task.reset();
  CHECK_FALSE(allocator->is_stream_tracked(stream));
  if (mode == 0) {
    REQUIRE(op.published);
    CHECK(allocator->get_total_allocated_bytes() > baseline);
    CHECK(sirius::get_cudf_table_view(*op.published).num_rows() == 3);
    op.published.reset();
  } else {
    CHECK_FALSE(op.published);
  }
  CHECK(allocator->get_total_allocated_bytes() == baseline);
  // Destruction may enqueue frees. Keep the caller's allocator and stream alive for that
  // completion.
  REQUIRE(cudaStreamSynchronize(stream.value()) == cudaSuccess);
}

TEST_CASE("Arrow failed completion retains real backing and its caller resource owner",
          "[arrow_driver][arrow_ownership][arrow_failed_completion]")
{
  auto import_throws = GENERATE(false, true);
  struct caller_resources {
    decltype(sirius::test::operator_utils::initialize_memory_manager()) manager =
      sirius::test::operator_utils::initialize_memory_manager();
    rmm::cuda_stream stream;
  };
  auto caller                              = std::make_shared<caller_resources>();
  std::weak_ptr<caller_resources> lifetime = caller;
  auto* space = caller->manager->get_memory_space(cucascade::memory::Tier::GPU, 0);
  auto* allocator =
    space->get_memory_resource_as<cucascade::memory::reservation_aware_resource_adaptor>();
  REQUIRE(allocator);
  auto baseline = allocator->get_total_allocated_bytes();
  {
    auto provider = std::make_unique<sm::host_reservation_provider>(*caller->manager, caller);
    auto producer = std::make_shared<pull_producer>();
    if (import_throws) producer->on_normalize = [] { throw std::runtime_error("staging failure"); };
    auto h     = std::make_unique<harness>(producer, integer_program(), true, provider.get());
    auto input = h->next();
    // Reuse the scan retirement fault seam. It reports a failed join only after
    // real completion, so the test can eventually destroy retained resources safely.
    sirius::op::scan::detail::read_retirement_test_hooks hooks(std::nullopt, true);
    CHECK_THROWS_AS(h->driver->materialize_metadata_to_table(*input, *space, caller->stream),
                    sirius::unjoined_gpu_work);
    REQUIRE(hooks.retained.has_value());
    input.reset();
    h->driver->stop_preparation();
    CHECK(h->ledger->permits_in_flight() == 1);
    CHECK(h->profiles->counters->arrow_batches_imported == 0);
    if (!import_throws) CHECK(allocator->get_total_allocated_bytes() > baseline);
    h.reset();
    producer.reset();
    provider.reset();
    caller.reset();
    CHECK_FALSE(lifetime.expired());
    // The retained import now owns the only path to the caller's allocator and stream.
    // The injected failure followed a real import barrier. Destruction releases the
    // table before the lane and input owner; an idle lane adds no final barrier.
  }
  CHECK(lifetime.expired());
}

TEST_CASE("Arrow producers cannot borrow another scans larger temporary envelope",
          "[arrow_driver][arrow_envelope]")
{
  auto small = std::make_shared<pull_producer>();
  auto large = std::make_shared<pull_producer>();
  // Individually rounded blocks permit 12288 bytes, not round(4098) == 8192.
  small->envelope.temporary = {4097, 1};
  large->envelope.temporary = {32768};
  bool exceeds              = false;
  auto tested               = small;
  auto other                = large;
  SECTION("independent rounding permits the declared simultaneous allocations")
  {
    small->temporary_probe_after = 4096;
  }
  SECTION("rounded cumulative allocation exceeds the smaller envelope")
  {
    small->temporary_probe_after = 4097;
    exceeds                      = true;
  }
  SECTION("oversized transient allocation cannot disappear before acceptance")
  {
    small->temporary_probe_before = 12289;
    exceeds                       = true;
  }
  SECTION("larger producer retains its own allowance")
  {
    tested                       = large;
    other                        = small;
    large->temporary_probe_after = 16384;
  }
  harness h(tested, integer_program(), true, nullptr, other, 4096);
  if (exceeds) {
    try {
      h.next();
      FAIL("producer exceeded its temporary envelope");
    } catch (sm::preparation_resource_error const& error) {
      CHECK(error.bound_exceeded);
      CHECK(error.original == nullptr);
    }
    CHECK(h.ledger->bound_exceeded() == 1);
    CHECK(h.ledger->allocation_failures() == 0);
  } else {
    auto input = h.next();
    REQUIRE(input);
    CHECK(h.ledger->bound_exceeded() == 0);
    CHECK(h.ledger->permits_in_flight() == 1);
    input.reset();
  }
  CHECK_FALSE(h.gate->active.load());
  CHECK(h.ledger->permits_in_flight() == 0);
  CHECK(h.ledger->tracked_units() == 0);
  CHECK(h.backing.seen->allocated_bytes == 0);
}

TEST_CASE("Arrow retained budgets are reusable through end of stream",
          "[arrow_driver][arrow_retained]")
{
  auto producer               = std::make_shared<pull_producer>();
  producer->envelope.retained = 64;
  producer->rows              = {2, 2};
  SECTION("one batch") { producer->rows = {2}; }
  SECTION("multiple batches") {}
  harness h(producer);
  for (size_t i = 0; i < producer->rows.size(); ++i) {
    auto input = h.next();
    REQUIRE(input);
    auto retained = dynamic_cast<owned_arrow_scan_info&>(*input).coalesced_copy();
    input.reset();
    CHECK(std::holds_alternative<sm::preparation_pending>(h.driver->try_claim_split({})));
    retained.reset();
    CHECK(h.ledger->tracked_units() == 0);
    CHECK(h.ledger->permits_in_flight() == 0);
  }
  CHECK_FALSE(h.next());
  CHECK(h.ledger->tracked_units() == 0);
  CHECK(h.backing.seen->allocated_bytes == 0);
}

TEST_CASE("Arrow in-flight reads preserve coordinator stop intent", "[arrow_driver][arrow_stop]")
{
  auto kind   = GENERATE(0, 1, 2);  // pull, legacy push, qualified push
  auto push   = kind != 0;
  auto reason = GENERATE(
    sm::stop_reason::normal_eos, sm::stop_reason::user_cancel, sm::stop_reason::failure_induced);
  auto read_fails = GENERATE(false, true);
  sirius::test::preparation::hold reading;
  std::atomic<int> failures{0};
  auto pull     = std::make_shared<pull_producer>();
  pull->rows    = {3, 3};
  pull->domain  = sm::preparation_execution_domain::native_reader;
  auto program  = kind == 2 ? qualified_integer_program() : integer_program();
  auto counters = make_counters();
  auto q        = queue(counters, program, [&](auto) { ++failures; });
  std::shared_ptr<arrow_batch_producer> producer =
    push ? std::static_pointer_cast<arrow_batch_producer>(q)
         : std::static_pointer_cast<arrow_batch_producer>(pull);
  harness h(counters, producer, program);
  int allocations = 0;
  auto pause      = [&] {
    reading.wait();
    if (read_fails) throw std::runtime_error("read failed after stop");
  };
  pull->before_next = [&] {
    if (pull->calls == 2) pause();
  };
  if (push)
    h.backing.seen->before_allocate = [&] {
      if (++allocations == 2) pause();
    };
  std::future<void> submitting;
  sirius::exec::static_thread_pool pool(2);
  sirius::exec::scoped_dispatcher dispatcher(pool);
  sirius::pipeline::completion_handler completion;
  sm::native_reader_executor readers(1, 1);
  sm::preparation_coordinator coordinator(completion, dispatcher, {2, 2, 2, 1}, &readers);
  auto lane      = h.driver->make_preparation_source({});
  auto connector = std::make_shared<sm::split_connector>();
  sm::preparation_coordinator::source source;
  source.execution   = lane.execution;
  source.try_claim   = std::move(lane.try_claim);
  source.close       = std::move(lane.close);
  source.bind_wakeup = std::move(lane.bind_wakeup);
  source.coalescer   = std::move(lane.coalescer);
  source.destination = connector;
  source.construct   = [](auto metadata) {
    auto bytes = metadata->preparation_bytes();
    return sirius::test::preparation::make_publication(std::move(metadata), bytes);
  };
  std::atomic<int> publications{0};
  coordinator.after_publication_for_testing([&](auto, auto const&, auto) { ++publications; });
  coordinator.add_source(std::move(source));
  if (push)
    submitting = std::async(std::launch::async, [&] {
      if (kind == 2) {
        for (int i = 0; i < 2; ++i) {
          auto slot = q->acquire_offered_slot();
          fill_slot(slot);
          REQUIRE(q->offer(slot) == arrow_push_queue::offer_outcome::accepted);
        }
      } else {
        auto owner = std::make_shared<integer_tree>();
        if (q->offer(owner->offer(owner))) (void)q->offer(owner->offer(owner));
      }
    });
  coordinator.arm();
  coordinator.start();
  bool entered  = false;
  auto deadline = std::chrono::steady_clock::now() + 4s;
  while (std::chrono::steady_clock::now() < deadline) {
    coordinator.advance(1ms);
    if (connector->has_more_splits()) (void)connector->get_next_split();
    {
      std::lock_guard lock(reading.mutex);
      entered = reading.entered;
    }
    if (entered || completion.has_error()) break;
  }
  if (reason == sm::stop_reason::failure_induced)
    completion.report_error(std::make_exception_ptr(std::runtime_error("original failure")));
  coordinator.request_stop(reason);
  reading.release();
  coordinator.drain();
  q->cancel();
  if (submitting.valid()) submitting.get();
  REQUIRE(entered);
  CHECK(publications == 1);
  CHECK(completion.has_error() == (read_fails || reason != sm::stop_reason::normal_eos));
  if (reason == sm::stop_reason::user_cancel)
    CHECK_THROWS_AS(std::rethrow_exception(completion.terminal_error()),
                    duckdb::InterruptException);
  if (reason == sm::stop_reason::failure_induced)
    CHECK_THROWS_WITH(std::rethrow_exception(completion.terminal_error()), "original failure");
  CHECK(failures == (push && read_fails ? 1 : 0));
  CHECK_FALSE(h.gate->active.load());
  CHECK(h.ledger->permits_in_flight() == 0);
  CHECK(h.ledger->tracked_units() == 0);
  CHECK(h.backing.seen->allocated_bytes == 0);
}

TEST_CASE("Arrow independent cancellation survives a later coordinator normal stop",
          "[arrow_driver][arrow_stop]")
{
  auto qualified = GENERATE(false, true);
  auto program   = qualified ? qualified_integer_program() : integer_program();
  auto counters  = make_counters();
  std::atomic<int> failures{0}, releases{0}, publications{0};
  auto q = queue(counters, program, [&](auto) { ++failures; });
  harness h(counters, q, program);
  sirius::test::preparation::hold staging, escaped;
  h.backing.seen->before_allocate = [&] { staging.wait(); };
  std::exception_ptr driver_error;
  bool token_was_live = false;
  std::future<bool> submitting;
  queue_guard guard{q};
  sirius::exec::static_thread_pool pool(2);
  sirius::exec::scoped_dispatcher dispatcher(pool);
  sirius::pipeline::completion_handler completion;
  sm::preparation_coordinator coordinator(completion, dispatcher, {1, 2, 2, 1});
  struct unblock {
    sirius::test::preparation::hold &staging, &escaped;
    ~unblock()
    {
      staging.release();
      escaped.release();
    }
  } unblocker{staging, escaped};
  auto lane      = h.driver->make_preparation_source({});
  auto connector = std::make_shared<sm::split_connector>();
  sm::preparation_coordinator::source source;
  source.execution = lane.execution;
  source.try_claim =
    [&, claim = std::move(lane.try_claim)]() mutable -> sm::preparation_coordinator::claim_result {
    auto claimed = claim();
    if (auto* work = std::get_if<sm::preparation_job>(&claimed)) {
      *work = sm::preparation_job{[&, work = std::move(*work)](auto stop) mutable {
        try {
          return work.run(stop);
        } catch (...) {
          driver_error   = std::current_exception();
          token_was_live = !stop.stop_requested();
          escaped.wait();  // Hold the driver error before the coordinator classifies it.
          throw;
        }
      }};
    }
    return claimed;
  };
  source.close       = std::move(lane.close);
  source.bind_wakeup = std::move(lane.bind_wakeup);
  source.coalescer   = std::move(lane.coalescer);
  source.destination = connector;
  source.construct   = [](auto metadata) {
    auto bytes = metadata->preparation_bytes();
    return sirius::test::preparation::make_publication(std::move(metadata), bytes);
  };
  coordinator.after_publication_for_testing([&](auto, auto const&, auto) { ++publications; });
  coordinator.add_source(std::move(source));
  if (qualified) {
    auto slot = q->acquire_offered_slot();
    fill_slot(slot, 3, [&] { ++releases; });
    REQUIRE(q->offer(slot) == arrow_push_queue::offer_outcome::accepted);
  } else {
    submitting = std::async(std::launch::async, [&] {
      auto owner = std::make_shared<integer_tree>();
      return q->offer(owner->offer(owner));
    });
    await_ready(*q);
  }
  coordinator.arm();
  coordinator.start();
  coordinator.advance();
  REQUIRE(staging.await());
  CHECK(h.ledger->permits_in_flight() == 1);
  q->cancel();
  staging.release();
  REQUIRE(escaped.await());
  REQUIRE(driver_error);
  CHECK(token_was_live);
  CHECK_THROWS_AS(std::rethrow_exception(driver_error), duckdb::InterruptException);
  CHECK_FALSE(completion.has_error());
  coordinator.request_stop(sm::stop_reason::normal_eos);
  escaped.release();
  coordinator.drain();
  if (submitting.valid()) CHECK_FALSE(submitting.get());
  REQUIRE(completion.has_error());
  CHECK_THROWS_AS(std::rethrow_exception(completion.terminal_error()), duckdb::InterruptException);
  CHECK(publications == 0);
  CHECK(failures == 0);
  CHECK_FALSE(h.gate->active.load());
  CHECK(h.ledger->permits_in_flight() == 0);
  CHECK(h.ledger->tracked_units() == 0);
  CHECK(h.backing.seen->allocated_bytes == 0);
  if (qualified) {
    CHECK(releases == 1);
    auto observed = q->observe();
    CHECK(observed.offered == 0);
    CHECK(observed.accepted == 0);
    CHECK(observed.reserved_bytes == 0);
    CHECK(observed.unique_retained_bytes == 0);
  }
}

// A pull producer that observes the token inside next() and still reports it as an interrupt
// must not turn normal shutdown into a query error; an interrupt without a token stop must
// keep propagating unchanged.
TEST_CASE("Arrow legacy pull interrupts after an observed stop follow coordinator intent",
          "[arrow_driver][arrow_stop]")
{
  enum class mode { normal_eos, user_cancel, failure_induced, no_token_interrupt };
  auto which =
    GENERATE(mode::normal_eos, mode::user_cancel, mode::failure_induced, mode::no_token_interrupt);
  sirius::test::preparation::hold reading;
  auto pull                      = std::make_shared<pull_producer>();
  pull->rows                     = {3, 3};
  pull->domain                   = sm::preparation_execution_domain::native_reader;
  pull->legacy_interrupt_on_stop = true;
  harness h(pull);
  pull->before_stop_check = [&] {
    if (pull->calls == 2) {
      reading.wait();
      if (which == mode::no_token_interrupt) pull->cancelled = true;
    }
  };
  sirius::exec::static_thread_pool pool(2);
  sirius::exec::scoped_dispatcher dispatcher(pool);
  sirius::pipeline::completion_handler completion;
  sm::native_reader_executor readers(1, 1);
  sm::preparation_coordinator coordinator(completion, dispatcher, {2, 2, 2, 1}, &readers);
  auto lane      = h.driver->make_preparation_source({});
  auto connector = std::make_shared<sm::split_connector>();
  sm::preparation_coordinator::source source;
  source.execution   = lane.execution;
  source.try_claim   = std::move(lane.try_claim);
  source.close       = std::move(lane.close);
  source.bind_wakeup = std::move(lane.bind_wakeup);
  source.coalescer   = std::move(lane.coalescer);
  source.destination = connector;
  source.construct   = [](auto metadata) {
    auto bytes = metadata->preparation_bytes();
    return sirius::test::preparation::make_publication(std::move(metadata), bytes);
  };
  std::atomic<int> publications{0};
  coordinator.after_publication_for_testing([&](auto, auto const&, auto) { ++publications; });
  coordinator.add_source(std::move(source));
  coordinator.arm();
  coordinator.start();
  bool entered  = false;
  auto deadline = std::chrono::steady_clock::now() + 4s;
  while (std::chrono::steady_clock::now() < deadline) {
    coordinator.advance(1ms);
    if (connector->has_more_splits()) (void)connector->get_next_split();
    {
      std::lock_guard lock(reading.mutex);
      entered = reading.entered;
    }
    if (entered || completion.has_error()) break;
  }
  // The producer is inside next(), before its own stop check: the stop is observed by the
  // producer, not by the driver's bracketing checks.
  if (which == mode::failure_induced)
    completion.report_error(std::make_exception_ptr(std::runtime_error("original failure")));
  if (which == mode::normal_eos) coordinator.request_stop(sm::stop_reason::normal_eos);
  if (which == mode::user_cancel) coordinator.request_stop(sm::stop_reason::user_cancel);
  if (which == mode::failure_induced) coordinator.request_stop(sm::stop_reason::failure_induced);
  reading.release();
  coordinator.drain();
  REQUIRE(entered);
  CHECK(publications == 1);
  CHECK(completion.has_error() == (which != mode::normal_eos));
  if (which == mode::user_cancel || which == mode::no_token_interrupt)
    CHECK_THROWS_AS(std::rethrow_exception(completion.terminal_error()),
                    duckdb::InterruptException);
  if (which == mode::failure_induced)
    CHECK_THROWS_WITH(std::rethrow_exception(completion.terminal_error()), "original failure");
  CHECK_FALSE(h.gate->active.load());
  CHECK(h.ledger->permits_in_flight() == 0);
  CHECK(h.ledger->tracked_units() == 0);
  CHECK(h.backing.seen->allocated_bytes == 0);
}

TEST_CASE("Arrow children share preparation identities with retained slots",
          "[arrow_driver][composite_scan]")
{
  bool shared_ids = GENERATE(false, true);
  sm::test::test_reservation_provider provider;
  provider.grant_bytes = 3 * 8192 + 16;
  auto ledger          = std::make_unique<sm::preparation_ledger>(provider);
  auto first           = std::make_shared<pull_producer>();
  auto second          = std::make_shared<pull_producer>();
  std::vector<sm::scan_envelope> scans{arrow_preparation_envelope(*first),
                                       arrow_preparation_envelope(*second),
                                       sm::scan_envelope{{{{8}, 16}}, {}, true, 1}};
  auto decision = ledger->admit(scans, sm::memory_space_id{cucascade::memory::Tier::HOST, 0}, 3);
  REQUIRE(decision.deferred);
  auto admission     = std::make_shared<sm::preparation_admission>(std::move(ledger), decision);
  auto shared_ledger = std::shared_ptr<sm::preparation_ledger>(admission->consume());
  auto ids           = std::make_shared<sm::preparation_identity>();
  auto program       = integer_program();
  sm::preparation_slot slot(shared_ledger, {program.contract, ids->next()}, 16);
  auto make_child = [&](auto producer) {
    arrow_import_binding binding;
    binding.program        = program;
    binding.profiles       = std::make_shared<physical_profile_table>();
    binding.read_identity  = std::make_shared<int>(4);
    binding.column_names   = {"value"};
    binding.output_columns = {0};
    binding.admission      = admission;
    binding.owns_admission = false;
    if (shared_ids) binding.next_unit_id = [ids] { return ids->next(); };
    auto child = std::make_unique<arrow_import_ingestible>(std::move(binding), producer);
    child->attach_preparation_ledger(shared_ledger);
    return child;
  };
  auto a = make_child(first), b = make_child(second);
  if (!shared_ids) {
    CHECK_THROWS_WITH(a->try_claim_split({}), "duplicate preparation unit");
    CHECK(first->calls == 0);
  } else {
    auto claim_a = a->try_claim_split({});
    auto claim_b = b->try_claim_split({});
    REQUIRE(std::holds_alternative<sm::preparation_job>(claim_a));
    REQUIRE(std::holds_alternative<sm::preparation_job>(claim_b));
    auto input_a = std::get<sm::preparation_job>(claim_a).run({});
    auto input_b = std::get<sm::preparation_job>(claim_b).run({});
    REQUIRE(input_a);
    REQUIRE(input_b);
    REQUIRE(input_a->certificates().size() == 1);
    REQUIRE(input_b->certificates().size() == 1);
    CHECK(input_a->certificates().front().split_id == 1);
    CHECK(input_b->certificates().front().split_id == 2);
    CHECK(shared_ledger->tracked_units() == 3);
    input_a.reset();
    input_b.reset();
    CHECK(shared_ledger->tracked_units() == 1);
    auto permit = slot.try_acquire();
    REQUIRE(permit);
    auto bytes = shared_ledger->allocator(permit).allocate_retained(16);
    CHECK(shared_ledger->charged_bytes(slot.key()) == 16);
  }
  a->finish_preparation();
  b->finish_preparation();
}
