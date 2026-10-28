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
#include "op/scan/arrow/arrow_import_ingestible.hpp"
#include "operator/operator_test_utils.hpp"
#include "scan_manager/preparation_fixture.hpp"
#include "scan_manager/preparation_test_support.hpp"

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
  size_t cursor  = 0;
  int preflights = 0, calls = 0, closes = 0;
  bool ready = true, cancelled = false, wrong_schema = false, fail_read = false;
  sm::preparation_execution_domain domain = sm::preparation_execution_domain::query_owner;
  std::thread::id owner                   = std::this_thread::get_id();
  sm::preparation_execution_domain execution_domain() const noexcept override { return domain; }
  arrow_producer_limits limits() const override { return {3, {{8192}, 0}}; }
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
    auto capacity = std::make_shared<pull_capacity>(std::move(admitted), on_normalize);
    auto* tree    = new (capacity->block.data()) integer_tree(rows[cursor++]);
    if (wrong_schema) tree->field.format = "f";
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
  harness(std::shared_ptr<arrow_batch_producer> producer,
          materialization_program program = integer_program(),
          bool empty_source               = true)
  {
    auto unique   = std::make_unique<sm::preparation_ledger>(backing);
    auto envelope = producer->limits().envelope;
    backing.grant_bytes =
      std::accumulate(envelope.temporary.begin(), envelope.temporary.end(), envelope.retained);
    auto scan = arrow_preparation_envelope(*producer);
    auto decision =
      unique->admit(std::array{scan}, sm::memory_space_id{cucascade::memory::Tier::HOST, 0});
    REQUIRE(decision.deferred);
    arrow_import_binding binding;
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
    std::lock_guard lock(gate->mutex);
    CHECK(gate->wake_generation > 0);
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
