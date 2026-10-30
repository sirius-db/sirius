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
#include "data/data_batch_utils.hpp"
#include "helper/type_conversions.hpp"
#include "op/scan/arrow/arrow_import_ingestible.hpp"
#include "op/scan/detail/read_retirement.hpp"
#include "operator/operator_test_utils.hpp"
#include "pipeline/gpu_pipeline_task.hpp"
#include "scan_manager/preparation_fixture.hpp"
#include "scan_manager/preparation_test_support.hpp"
#include "utils/telemetry_utils.hpp"

#include <cudf/strings/strings_column_view.hpp>

#include <rmm/cuda_stream.hpp>

#include <duckdb/common/exception.hpp>

#include <array>
#include <chrono>
#include <cstring>
#include <future>
#include <numeric>
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
  std::function<void(ArrowSchema&, ArrowArray&)> observe_tree;
  sm::unit_envelope envelope{{8192}, 0};
  uint64_t temporary_probe_before = 0, temporary_probe_after = 0;
  size_t cursor  = 0;
  int preflights = 0, calls = 0, closes = 0;
  bool ready = true, cancelled = false, wrong_schema = false, fail_read = false;
  sm::preparation_execution_domain domain = sm::preparation_execution_domain::query_owner;
  std::thread::id owner                   = std::this_thread::get_id();
  sm::preparation_execution_domain execution_domain() const noexcept override { return domain; }
  arrow_producer_limits limits() const override { return {3, envelope}; }
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
    if (cancelled || stop.stop_requested()) throw duckdb::InterruptException();
    if (fail_read) throw std::runtime_error("pull failed");
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
  std::unique_ptr<arrow_import_ingestible> driver;
  std::shared_ptr<arrow_input_gate> gate = std::make_shared<arrow_input_gate>();
  harness(std::shared_ptr<arrow_batch_producer> producer,
          materialization_program program             = integer_program(),
          bool empty_source                           = true,
          sm::reservation_provider* external          = nullptr,
          std::shared_ptr<arrow_batch_producer> other = {},
          uint64_t quantum                            = 1)
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
    backing.grant_bytes = w * scans.size() + retained;
    auto decision =
      unique->admit(scans, sm::memory_space_id{cucascade::memory::Tier::HOST, 0}, scans.size());
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
    profiles->counters = std::make_shared<physical_check_counters>();
    profiles->counters->track_units = true;
    driver = std::make_unique<arrow_import_ingestible>(std::move(binding), std::move(producer));
    driver->attach_preparation_ledger(ledger);
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
  materialization_program p                    = integer_program(),
  std::function<void(std::exception_ptr)> fail = [](auto) {})
{
  return std::make_shared<arrow_push_queue>(
    p, arrow_push_limits{32, 8192, 8192, 8192}, std::move(fail));
}
}  // namespace

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

TEST_CASE("Arrow driver rejects foreign read identity and schema drift", "[arrow_driver]")
{
  auto producer = std::make_shared<pull_producer>();
  harness h(producer);
  SECTION("schema drift is a physical input failure")
  {
    producer->wrong_schema = true;
    try {
      h.next();
      FAIL("expected schema refusal");
    } catch (sirius::transparent::classified_execution_error const& e) {
      CHECK(e.cause == sirius::transparent::late_failure_cause::physical_input);
    }
    CHECK(h.ledger->permits_in_flight() == 0);
    CHECK(h.backing.seen->allocated_bytes == 0);
  }
  SECTION("same schema does not authorize another bound read")
  {
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
    producer->fail_read = true;
    CHECK_THROWS_WITH(h.next(), "pull failed");
    CHECK(h.driver->create_batch_coalescer()->flush().empty());
  }
  SECTION("cancel does not become empty source")
  {
    std::stop_source stop;
    stop.request_stop();
    CHECK_THROWS_AS(h.next(sm::preparation_stop_token(stop.get_token())),
                    duckdb::InterruptException);
    CHECK(producer->calls == 0);
    CHECK(h.driver->create_batch_coalescer()->flush().empty());
  }
}

TEST_CASE("Arrow push queue wakes an empty source and copies through admitted backing",
          "[arrow_driver]")
{
  auto q = queue();
  harness h(q);
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
  auto q = queue(integer_program(), [&](auto) { ++failures; });
  harness h(q);
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
  auto q = queue();
  harness h(q);
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

TEST_CASE("Arrow push rejects schema and bound drift before waiting", "[arrow_driver]")
{
  auto mode    = GENERATE(0, 1, 2, 3);
  int failures = 0;
  auto q       = queue(integer_program(), [&](auto e) {
    ++failures;
    try {
      std::rethrow_exception(e);
    } catch (sirius::transparent::classified_execution_error const& error) {
      CHECK(error.cause == sirius::transparent::late_failure_cause::physical_input);
    }
  });
  harness h(q);
  auto owner   = std::make_shared<integer_tree>();
  auto offered = owner->offer(owner);
  if (mode == 0) owner->field.format = "f";
  if (mode == 1) offered.retained_bytes = 8193;
  if (mode == 2) owner->array.length = 33;
  if (mode == 3) owner->field.flags = ARROW_FLAG_NULLABLE, owner->column.null_count = 1;
  CHECK_THROWS(q->offer(offered));
  CHECK(failures == 1);
  CHECK(owner->schema.release);
  CHECK(owner->array.release);
  CHECK(owner.use_count() == 2);
  CHECK(h.backing.seen->allocations == 0);
  CHECK_THROWS(h.next());
}

TEST_CASE("Arrow push imports selected strings and preserves zero-column cardinality",
          "[arrow_driver]")
{
  auto mode    = GENERATE(0, 1, 2, 3);
  auto program = integer_program();
  auto owner   = std::make_shared<integer_tree>();
  std::array<int32_t, 6> offsets{-1, 0, 2, 4, 6, -1};
  std::array<char, 6> chars{'a', 'b', 'c', 'd', 'e', 'f'};
  std::array<void const*, 3> buffers{nullptr, offsets.data(), chars.data()};
  if (mode < 2) {
    program.output_types    = {cudf::data_type{cudf::type_id::STRING}};
    owner->field.format     = "u";
    owner->column.n_buffers = 3;
    owner->column.buffers   = buffers.data();
    owner->column.length    = 5;
    owner->array.offset     = 1;
    owner->array.length     = mode == 0 ? 3 : 0;
    if (mode == 1) buffers[1] = buffers[2] = nullptr;
  } else {
    program.output_types.clear();
    owner->schema.n_children = owner->array.n_children = 0;
    owner->array.length                                = mode == 2 ? 3 : 0;
  }
  program.input_types = program.output_types;
  auto q              = queue(program);
  harness h(q, program);
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
          "[arrow_driver]")
{
  auto mode    = GENERATE(0, 1);
  int failures = 0;
  auto q       = std::make_shared<arrow_push_queue>(
    integer_program(),
    arrow_push_limits{3, 8192, mode == 0 ? 16U : 8192U, mode == 1 ? 1U : 8192U},
    [&](auto) { ++failures; });
  auto owner   = std::make_shared<integer_tree>();
  auto offered = owner->offer(owner);
  CHECK_THROWS(q->offer(offered));
  CHECK(failures == 1);
  CHECK(owner.use_count() == 2);
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
  auto boolean = GENERATE(false, true);
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
    program.output_types       = {cudf::data_type{cudf::type_id::DECIMAL128, -2}};
    program.decimal_precisions = {9};
    owner->field.format        = "d:9,2";
    buffers                    = {nullptr, decimals.data()};
  }
  owner->column.buffers = buffers.data();
  program.input_types   = program.output_types;
  auto q                = queue(program);
  harness h(q, program);
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
          "[arrow_driver]")
{
  auto program               = integer_program();
  program.output_types       = {cudf::data_type{cudf::type_id::DECIMAL128, -2}};
  program.input_types        = program.output_types;
  program.decimal_precisions = {9};
  auto owner                 = std::make_shared<integer_tree>(0);
  owner->field.format        = "d:10,2";
  int failures               = 0;
  auto q                     = queue(program, [&](auto) { ++failures; });
  CHECK_THROWS(q->offer(owner->offer(owner)));
  CHECK(failures == 1);
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
  auto q         = queue();
  harness h(q);
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
// Keep the production task's reservation, execution, sink and exception cleanup paths.
// Only the operator boundary is supplied by the fixture.
class arrow_task_operator final : public sirius::op::sirius_physical_operator {
 public:
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
  auto pipeline  = std::make_shared<pl::sirius_pipeline>(pl::pipeline_build_context{nullptr, true});
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
    // Its table is destroyed before the retirement lane's final barrier and input owner.
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
