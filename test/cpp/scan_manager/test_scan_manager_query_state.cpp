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

// Per-query scan state (sirius_scan_manager::query_scan_manager_state).
//
// The scan manager used to keep one query's worth of state in bare members, with reset()
// wiping all of it. Two things broke the moment two queries could overlap:
//
//   - the dispatcher was shared, so finishing query A called request_stop() on the dispatcher
//     running query B's scans — B's sequencer died and its split_connector closed mid-scan,
//     which the consumer reads as end-of-stream (silent truncation, not an error);
//   - the coalescer was shared and its slot map is keyed by scan_op->get_operator_id(), and
//     operator ids restart at 0 for every query — so two queries' first scan operators
//     collided on one slot and each could be fed the other's splits.
//
// These gates pin both, plus the lifecycle arithmetic (register / reset / reset_all and the
// no-op paths) that sirius_context's cleanup and failure backstop depend on.

#include "catch.hpp"
#include "memory/topology_index.hpp"
#include "op/scan/duckdb_native_gpu_ingestible.hpp"
#include "op/scan/parquet_gpu_ingestible.hpp"
#include "op/scan/sirius_gpu_scan_operator.hpp"
#include "op/scan/sirius_gpu_scan_operator_data.hpp"
#include "pipeline/repository_wiring.hpp"
#include "pipeline/sirius_pipeline.hpp"
#include "planner/query.hpp"
#include "scan/test_utils.hpp"
#include "scan_manager/preparation_test_support.hpp"
#include "scan_manager/sirius_scan_manager.hpp"
#include "scan_manager/split_connector.hpp"
#include "sirius_context.hpp"
#include "sirius_engine.hpp"
#include "utils/gpu_execution_fixture.hpp"
#include "utils/isolated_checkpoint_test.hpp"
#include "utils/telemetry_utils.hpp"

#include <cudf/column/column_factories.hpp>

#include <cucascade/memory/topology_discovery.hpp>
#include <duckdb/catalog/catalog.hpp>
#include <duckdb/catalog/catalog_entry/duck_table_entry.hpp>
#include <unistd.h>

#include <array>
#include <atomic>
#include <condition_variable>
#include <filesystem>
#include <future>
#include <memory>
#include <string>
#include <thread>
#include <vector>

// Inject a format-neutral query-owner source before arm, without exposing a
// production registration bypass. All execution and retirement use the manager.
namespace sirius::scan_manager {
struct scan_manager_query_test {
  static void register_ioctx(sirius_scan_manager& manager,
                             cucascade::io::io_context_type type,
                             cucascade::io::io_context_registry::scheme_checker_type checker,
                             cucascade::io::io_context_registry::factory_type factory)
  {
    manager._ioctx_registry.register_ioctx(type, std::move(checker), std::move(factory));
  }
  static std::shared_ptr<cucascade::io::ioctx> ioctx_for_path(sirius_scan_manager& manager,
                                                              std::string_view path)
  {
    return manager.ioctx_for_path(path);
  }
  static void advance(sirius_scan_manager& manager, sirius::query_id_t id)
  {
    if (auto state = manager.get_query_state(id); state && state->coordinator)
      state->coordinator->advance(std::chrono::milliseconds(1));
  }
  static void before_native_thread(sirius_scan_manager& manager, std::function<void(size_t)> hook)
  {
    manager._native_readers.before_thread_for_testing(std::move(hook));
  }
  static size_t coordinator_workers(sirius_scan_manager const& manager)
  {
    return manager._coordinators.started_workers();
  }
  static bool contains(sirius_scan_manager& manager, sirius::query_id_t id)
  {
    return manager.get_query_state(id) != nullptr;
  }
  static std::shared_ptr<preparation_gate> install(
    sirius_scan_manager& manager,
    sirius::query_id_t id,
    preparation_coordinator::source source,
    std::shared_ptr<pipeline::completion_handler> completion)
  {
    auto state = std::make_shared<sirius_scan_manager::query_scan_manager_state>(
      manager._config.preparation.resolve(manager._config.thread_pool.num_threads));
    state->completion         = std::move(completion);
    state->dispatcher         = std::make_unique<exec::scoped_dispatcher>(manager._thread_pool);
    state->metadata_processor = std::make_unique<load_balancing_scan_batch_coalescer>();
    state->coordinator        = manager.make_preparation_coordinator(*state);
    state->coordinator->add_source(std::move(source));
    auto gate = state->coordinator->publication_gate();
    state->coordinator->arm();
    std::lock_guard lock(manager._query_states_mutex);
    if (!manager._query_states.emplace(id, std::move(state)).second)
      throw std::logic_error("duplicate test query");
    return gate;
  }
};
}  // namespace sirius::scan_manager

namespace {

namespace scan = sirius::op::scan;
using sirius::scan_manager::sirius_scan_manager;

std::filesystem::path project_root()
{
#ifdef SIRIUS_PROJECT_ROOT
  return std::filesystem::path{SIRIUS_PROJECT_ROOT};
#else
  return std::filesystem::current_path();
#endif
}

/// Local-file scan config: no S3 backend, no prefetch cache, two producer workers.
/// Blocking coalescers use a separate pool so they cannot occupy producer capacity.
sirius::scan_manager::scan_manager_config make_local_config()
{
  sirius::scan_manager::scan_manager_config cfg;
  cfg.thread_pool.num_threads = 2;
  cfg.uring_reactors_per_gpu  = 1;
  cfg.cache.mode              = cucascade::io::cache::cache_mode::none;
  cfg.apply_cache_mode();
  return cfg;
}

std::unique_ptr<scan::parquet_ingestible_table_info> make_table_info(std::string const& file)
{
  auto info                 = std::make_unique<scan::parquet_ingestible_table_info>();
  info->resolved_file_paths = {
    (project_root() / "test/cpp/integration/data/parquet" / file).string()};
  info->names = {"c0"};
  info->returned_types.push_back(sirius::logical_type::make(sirius::type_id::INTEGER));
  info->column_ids.push_back(duckdb::ColumnIndex(0));
  info->projected_output_count = 1;
  return info;
}

struct scan_query {
  std::vector<std::unique_ptr<scan::sirius_gpu_scan_operator>> operators;
  std::vector<std::shared_ptr<sirius::pipeline::sirius_pipeline>> pipelines;
  std::shared_ptr<const sirius::telemetry::telemetry_context> tctx;
  std::unique_ptr<sirius::planner::query> query;

  scan_query(sirius::query_id_t id,
             std::vector<std::shared_ptr<scan::gpu_ingestible>> inputs,
             scan::scan_contract_id first_contract = 1)
  {
    sirius::pipeline::sirius_pipeline_build_state build_state;
    for (auto& input : inputs) {
      auto pipeline = std::make_shared<sirius::pipeline::sirius_pipeline>(
        sirius::pipeline::pipeline_build_context{nullptr, true});
      pipeline->set_pipeline_id(pipelines.size() + 1);
      pipeline->set_query_id(id);
      auto op = std::make_unique<scan::sirius_gpu_scan_operator>(
        duckdb::vector<sirius::logical_type>{sirius::logical_type::make(sirius::type_id::INTEGER)},
        0,
        std::move(input),
        first_contract + pipelines.size());
      build_state.set_pipeline_source(*pipeline, *op);
      build_state.add_pipeline_operator(*pipeline, *op);
      operators.push_back(std::move(op));
      pipelines.push_back(std::move(pipeline));
    }
    sirius::pipeline::assign_operator_ids(pipelines);
    tctx = sirius::test::make_test_telemetry_context();
    sirius::telemetry::query_telemetry_info tinfo{quent::now_v7(), tctx->worker_id().raw(), id};
    query = std::make_unique<sirius::planner::query>(pipelines, tctx->context(), id, tinfo);
  }
};

scan_query make_query(sirius::query_id_t query_id, std::string const& file)
{
  auto info           = make_table_info(file);
  auto const contract = static_cast<scan::scan_contract_id>(sirius::value_of(query_id)) + 1;
  info->contract_id   = contract;
  return {query_id, {scan::make_ingestible(std::move(info))}, contract};
}

cucascade::memory::system_topology_info single_gpu_topology()
{
  cucascade::memory::system_topology_info topology;
  topology.num_gpus = 1;
  cucascade::memory::gpu_topology_info gpu;
  gpu.id        = 0;
  gpu.numa_node = 0;
  topology.gpus.push_back(std::move(gpu));
  return topology;
}

struct fixture {
  std::unique_ptr<sirius::memory::sirius_memory_reservation_manager> memory =
    initialize_memory_manager(1);
  std::shared_ptr<const sirius::memory::topology_index> topology =
    std::make_shared<sirius::memory::topology_index>(single_gpu_topology(), std::vector<int>{0});
};

struct manager_fixture : fixture {
  sirius::exec::query_lifecycle_registry lifecycle;
  sirius_scan_manager manager;
  explicit manager_fixture(sirius::scan_manager::scan_manager_config config = make_local_config())
    : manager(lifecycle, config, *memory, topology)
  {
  }
  template <typename T>
  T collect(std::future<T>& consumer, std::initializer_list<sirius::query_id_t> owners)
  {
    auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    while (consumer.wait_for(std::chrono::milliseconds(0)) != std::future_status::ready) {
      if (std::chrono::steady_clock::now() >= deadline) {
        manager.reset_all();
        FAIL("query owners did not finish scan preparation");
      }
      for (auto id : owners)
        sirius::scan_manager::scan_manager_query_test::advance(manager, id);
    }
    return consumer.get();
  }
};

struct start_failing_ioctx : cucascade::io::ioctx {
  std::atomic<size_t>* starts;
  std::atomic<bool>* fail;

  start_failing_ioctx(std::atomic<size_t>& starts, std::atomic<bool>& fail)
    : starts(&starts), fail(&fail)
  {
  }
  ~start_failing_ioctx() override { pre_destroy(); }
  cucascade::io::io_context_type type() const noexcept override
  {
    return cucascade::io::io_context_type::restful;
  }
  void start() override
  {
    ++*starts;
    if (fail->load()) { throw std::runtime_error("injected start failure"); }
  }
  void shutdown() noexcept override {}
  bool supports(std::string_view) const noexcept override { return true; }
  bool supports_device_read() const noexcept override { return false; }
  bool supports_host_to_device_read() const noexcept override { return false; }
  bool supports_vector_host_read() const noexcept override { return false; }
  bool supports_device_range_read() const noexcept override { return false; }
  std::vector<cucascade::io::byte_range> align_and_coalesce(
    std::span<cucascade::io::byte_range const> ranges,
    std::optional<size_t>) const noexcept override
  {
    return {ranges.begin(), ranges.end()};
  }
  size_t host_read_io(cucascade::io::io_object const&, size_t, size_t, uint8_t*) override
  {
    throw std::runtime_error("test context must not read");
  }
  cucascade::exec::semi_future<size_t> mixed_readv_async_io(
    cucascade::io::io_object const&,
    std::vector<cucascade::io::prepared_io_slice>&&) noexcept override
  {
    return cucascade::exec::make_semi_future<size_t>(
      std::make_exception_ptr(std::runtime_error("test context must not read")));
  }
  std::shared_ptr<cucascade::io::io_object> create_io_object(std::string) override
  {
    throw std::runtime_error("test context must not open");
  }
};

}  // namespace

TEST_CASE("routed ioctx start failure is not cached and the next lookup retries",
          "[scan_manager][query_state][routing]")
{
  using cucascade::io::io_context_type;
  using sirius::scan_manager::scan_manager_query_test;
  std::atomic<size_t> constructions{0};
  std::atomic<size_t> starts{0};
  std::atomic<bool> fail{true};
  manager_fixture f;
  auto const path = "s3://bucket/a.parquet";

  scan_manager_query_test::register_ioctx(
    f.manager,
    io_context_type::restful,
    [](std::string_view p) { return p.starts_with("s3://"); },
    [&](auto const&) {
      ++constructions;
      return std::make_shared<start_failing_ioctx>(starts, fail);
    });

  REQUIRE_THROWS_WITH(scan_manager_query_test::ioctx_for_path(f.manager, path),
                      Catch::Matchers::ContainsSubstring("injected start failure"));
  REQUIRE(constructions == 1);
  REQUIRE(starts == 1);
  REQUIRE_THROWS_WITH(f.manager.create_datasource(path),
                      Catch::Matchers::ContainsSubstring("injected start failure"));
  REQUIRE(constructions == 2);
  REQUIRE(starts == 2);

  fail     = false;
  auto ctx = scan_manager_query_test::ioctx_for_path(f.manager, path);
  REQUIRE(ctx);
  CHECK(ctx->type() == io_context_type::restful);
  CHECK(constructions == 3);
  CHECK(starts == 3);
  CHECK(scan_manager_query_test::ioctx_for_path(f.manager, path) == ctx);
  CHECK(constructions == 3);
  CHECK(starts == 3);
}

TEST_CASE("routed ioctx factory declines remain cached", "[scan_manager][query_state][routing]")
{
  using sirius::scan_manager::scan_manager_query_test;
  std::atomic<size_t> constructions{0};
  manager_fixture f;
  scan_manager_query_test::register_ioctx(
    f.manager,
    cucascade::io::io_context_type::restful,
    [](std::string_view p) { return p.starts_with("s3://"); },
    [&](auto const&) -> std::shared_ptr<cucascade::io::ioctx> {
      ++constructions;
      return nullptr;
    });
  CHECK_FALSE(scan_manager_query_test::ioctx_for_path(f.manager, "s3://bucket/a.parquet"));
  CHECK_FALSE(scan_manager_query_test::ioctx_for_path(f.manager, "s3://bucket/a.parquet"));
  CHECK(constructions == 1);
}

TEST_CASE("two queries register independently and reset drops only one",
          "[scan_manager][query_state]")
{
  manager_fixture f;
  auto& manager = f.manager;

  auto a = make_query(sirius::make_query_id(1), "nation.parquet");
  auto b = make_query(sirius::make_query_id(2), "supplier.parquet");

  manager.prepare_for_query(*a.query, /*enable_pinned_zone_map_pruning=*/true, {});
  REQUIRE(manager.num_active_queries() == 1);
  manager.prepare_for_query(*b.query, /*enable_pinned_zone_map_pruning=*/true, {});
  REQUIRE(manager.num_active_queries() == 2);

  // The gate: tearing down A must not touch B. Before per-query dispatchers this called
  // request_stop() on the one dispatcher driving both, killing B's sequencer.
  manager.reset(a.query->query_id());
  REQUIRE(manager.num_active_queries() == 1);

  // B is still live and still serving: its connector yields its splits and closes normally
  // (a closed-by-A connector would have ended the stream with no splits at all).
  std::future<std::size_t> consumer;
  try {
    consumer = std::async(std::launch::async, [&] {
      std::size_t count = 0;
      while (auto split = b.operators.front()->get_split_connector().get_next_split())
        ++count;
      return count;
    });
    manager.start_preparation(b.query->query_id());
  } catch (...) {
    manager.reset(b.query->query_id());
    throw;
  }
  auto b_splits = f.collect(consumer, {b.query->query_id()});
  REQUIRE(b_splits > 0);

  manager.reset(b.query->query_id());
  REQUIRE(manager.num_active_queries() == 0);
}

TEST_CASE("concurrent queries do not collide on operator id", "[scan_manager][query_state]")
{
  auto config                    = make_local_config();
  config.thread_pool.num_threads = GENERATE(1, 3);
  manager_fixture f(config);
  auto& manager = f.manager;

  // Both scan operators are id 0 — ids restart per query. A shared coalescer keys its slot map
  // by that id, so one query's splits would land in the other's slot.
  auto a = make_query(sirius::make_query_id(1), "nation.parquet");
  auto b = make_query(sirius::make_query_id(2), "supplier.parquet");
  REQUIRE(a.operators.front()->get_operator_id() == b.operators.front()->get_operator_id());

  manager.prepare_for_query(*a.query, true, {});
  manager.prepare_for_query(*b.query, true, {});

  // Each operator receives splits for ITS OWN file, not the other's. The coalescer emits
  // parquet_split_info (row-group slices, possibly spanning files), not the per-file
  // parquet_file_scan_info the pre-coalescer metadata pass produces.
  auto drain_paths = [](scan::sirius_gpu_scan_operator& op) {
    std::vector<std::string> paths;
    while (auto split = op.get_split_connector().get_next_split()) {
      auto* input = dynamic_cast<scan::scan_operator_input*>(split->get());
      if (input == nullptr || !input->has_scan_metadata()) { continue; }
      auto const* split_info =
        dynamic_cast<scan::parquet_split_info const*>(&input->get_scan_info());
      if (split_info == nullptr) { continue; }
      for (auto const& slice : split_info->rg_slices) {
        paths.push_back(slice.file_path);
      }
    }
    return paths;
  };

  std::future<std::vector<std::string>> a_consumer, b_consumer;
  try {
    a_consumer = std::async(std::launch::async, [&] { return drain_paths(*a.operators.front()); });
    b_consumer = std::async(std::launch::async, [&] { return drain_paths(*b.operators.front()); });
    manager.start_preparation(a.query->query_id());
    manager.start_preparation(b.query->query_id());
  } catch (...) {
    manager.reset_all();
    throw;
  }
  auto a_paths = f.collect(a_consumer, {a.query->query_id(), b.query->query_id()});
  auto b_paths = f.collect(b_consumer, {b.query->query_id()});

  REQUIRE_FALSE(a_paths.empty());
  REQUIRE_FALSE(b_paths.empty());
  for (auto const& p : a_paths) {
    REQUIRE(p.find("nation.parquet") != std::string::npos);
  }
  for (auto const& p : b_paths) {
    REQUIRE(p.find("supplier.parquet") != std::string::npos);
  }

  manager.reset_all();
  REQUIRE(manager.num_active_queries() == 0);
}

TEST_CASE("reset is a no-op for unknown and already-reset queries", "[scan_manager][query_state]")
{
  manager_fixture f;
  auto& manager = f.manager;

  // Never prepared: sirius_context's failure backstop calls reset(query_id) unconditionally,
  // so an unknown id has to be harmless.
  REQUIRE_NOTHROW(manager.reset(sirius::make_query_id(99)));
  REQUIRE(manager.num_active_queries() == 0);

  auto a = make_query(sirius::make_query_id(1), "nation.parquet");
  manager.prepare_for_query(*a.query, true, {});
  REQUIRE(manager.num_active_queries() == 1);

  manager.reset(a.query->query_id());
  // Double reset: run_mandatory_cleanup and drop_query_runtime_state_best_effort can both
  // fire for the same query on the failure path.
  REQUIRE_NOTHROW(manager.reset(a.query->query_id()));
  REQUIRE(manager.num_active_queries() == 0);
}

TEST_CASE("reset_all drops every query and stop() still returns", "[scan_manager][query_state]")
{
  manager_fixture f;
  auto& manager = f.manager;

  auto a = make_query(sirius::make_query_id(1), "nation.parquet");
  auto b = make_query(sirius::make_query_id(2), "supplier.parquet");
  manager.prepare_for_query(*a.query, true, {});
  manager.prepare_for_query(*b.query, true, {});
  REQUIRE(manager.num_active_queries() == 2);

  manager.reset_all();
  REQUIRE(manager.num_active_queries() == 0);

  // stop() drains again then stops the pool; a sequencer left parked on a dequeue that will
  // never be satisfied would hang here.
  REQUIRE_NOTHROW(manager.stop());
}

TEST_CASE("a query with no GPU scan operators registers nothing", "[scan_manager][query_state]")
{
  manager_fixture f;
  auto& manager = f.manager;

  auto const query_id = sirius::make_query_id(7);
  scan_query empty{query_id, {}};
  manager.prepare_for_query(*empty.query, true, {});
  // Nothing to tear down, so nothing is registered and the matching reset is a no-op.
  REQUIRE(manager.num_active_queries() == 0);
  REQUIRE_NOTHROW(manager.reset(query_id));
}

namespace {
using namespace std::chrono_literals;
namespace sm = sirius::scan_manager;

// These tests exercise control flow only; no format reader or GPU batch is needed.
struct empty_coalescer final : scan::batch_coalescer {
  size_t target_batch_bytes() const noexcept override { return 1; }
  cursor_step advance(scan::scan_info&, size_t&, size_t) override { return {nullptr, true}; }
  std::vector<std::unique_ptr<scan::scan_info>> push(std::unique_ptr<scan::scan_info>) override
  {
    return {};
  }
  std::vector<std::unique_ptr<scan::scan_info>> flush() override { return {}; }
};
struct owner_observation {
  size_t calls = 0, closes = 0, disposed = 0;
  std::thread::id worker, closer, disposer;
};
struct owner_attempt {
  std::shared_ptr<sirius::pipeline::completion_handler> completion =
    std::make_shared<sirius::pipeline::completion_handler>();
  std::future<void> future                    = completion->get_awaitable();
  std::shared_ptr<owner_observation> observed = std::make_shared<owner_observation>();
};
struct owner_capture {
  std::shared_ptr<owner_observation> observed;
  explicit owner_capture(std::shared_ptr<owner_observation> observed)
    : observed(std::move(observed))
  {
  }
  ~owner_capture()
  {
    ++observed->disposed;
    observed->disposer = std::this_thread::get_id();
  }
};
sm::preparation_coordinator::source owner_source(
  std::shared_ptr<owner_observation> observed,
  std::shared_ptr<sirius::pipeline::completion_handler> completion,
  bool pending = false,
  bool fail    = false)
{
  sm::preparation_coordinator::source source;
  source.execution   = sm::preparation_coordinator::execution_domain::query_owner;
  source.coalescer   = std::make_shared<empty_coalescer>();
  source.destination = std::make_shared<sm::split_connector>();
  source.try_claim   = [observed,
                      pending,
                      fail,
                      claimed = false]() mutable -> sm::preparation_coordinator::claim_result {
    if (pending) return sm::preparation_pending{};
    if (std::exchange(claimed, true)) return sm::preparation_exhausted{};
    return sm::preparation_coordinator::job{
      [observed,
       fail,
       capture = std::make_unique<owner_capture>(observed)]() -> std::unique_ptr<scan::scan_info> {
        ++observed->calls;
        observed->worker = std::this_thread::get_id();
        if (fail) throw std::runtime_error("provider read failed");
        return {};
      }};
  };
  source.construct = [](auto) -> sm::preparation_coordinator::publication {
    throw std::logic_error("control-only source must not publish a batch");
  };
  source.close = [observed, completion] {
    ++observed->closes;
    observed->closer = std::this_thread::get_id();
    completion->mark_completed();
  };
  return source;
}

// A broken wait gets a bounded failure and can still drain on Q, rather than
// leaving a watchdog thread detached or hanging test fixture destruction.
struct owner_retirement_guard {
  sirius_scan_manager& manager;
  sirius::query_id_t id;
  ~owner_retirement_guard() { manager.quiesce(id); }
};

struct completion_watchdog {
  std::mutex mutex;
  std::condition_variable changed;
  bool finished = false;
  std::thread thread;
  explicit completion_watchdog(std::shared_ptr<sirius::pipeline::completion_handler> completion)
    : thread([this, completion] {
        std::unique_lock lock(mutex);
        if (!changed.wait_for(lock, 2s, [&] { return finished; }))
          completion->report_error(
            std::make_exception_ptr(std::runtime_error("owner did not pump")));
      })
  {
  }
  ~completion_watchdog()
  {
    {
      std::lock_guard lock(mutex);
      finished = true;
    }
    changed.notify_all();
    thread.join();
  }
};
}  // namespace

TEST_CASE("external scan retirement keeps the owner reachable until close",
          "[scan_manager][query_state][owner_pump]")
{
  auto config = make_local_config();
  config.preparation.coordinator_workers =
    GENERATE(std::optional<size_t>{0}, std::optional<size_t>{});
  manager_fixture f(config);
  auto& manager                       = f.manager;
  auto id                             = sirius::make_query_id(42);
  auto [completion, future, observed] = owner_attempt{};
  auto gate                           = sm::scan_manager_query_test::install(
    manager, id, owner_source(observed, completion, true), completion);
  bool owner_waits = false;
  SECTION("reset before start with owner completion wait") { owner_waits = true; }
  SECTION("reset during execution with owner retirement") { manager.start_preparation(id); }
  auto external = std::async(std::launch::async, [&] { manager.reset(id); });
  owner_retirement_guard retirement{manager, id};
  bool stop_seen;
  {
    std::unique_lock lock(gate->mutex);
    stop_seen = gate->cv.wait_for(lock, 2s, [&] { return gate->closed; });
  }
  // Even after external reset enters drain, lookup must reach the Q close mailbox.
  auto active_while_closing = sm::scan_manager_query_test::contains(manager, id);
  manager.start_preparation(id);  // A late start must not reopen stopped preparation.
  if (owner_waits) {
    completion_watchdog watchdog(completion);
    manager.await_query_completion(id, future);
  } else {
    // This used to wait for execution_mutex held by the external drain, while
    // that same drain waited for this thread to execute provider close.
    auto second = std::async(std::launch::async, [&] { manager.quiesce(id); });
    manager.quiesce(id);
    REQUIRE(second.wait_for(2s) == std::future_status::ready);
    second.get();
    future.get();
  }
  REQUIRE(external.wait_for(2s) == std::future_status::ready);
  external.get();
  REQUIRE(stop_seen);
  REQUIRE(active_while_closing);
  REQUIRE(observed->calls == 0);
  REQUIRE(observed->closes == 1);
  REQUIRE(observed->closer == std::this_thread::get_id());
  REQUIRE_FALSE(sm::scan_manager_query_test::contains(manager, id));
}

TEST_CASE("provider failure retires on Q without losing the first error",
          "[scan_manager][query_state][owner_pump]")
{
  manager_fixture f;
  auto& manager                       = f.manager;
  auto id                             = sirius::make_query_id(43);
  auto [completion, future, observed] = owner_attempt{};
  auto source                         = owner_source(observed, completion, false, true);
  auto close                          = std::move(source.close);
  source.close                        = [close] {
    close();
    throw std::runtime_error("secondary close failure");
  };
  sm::scan_manager_query_test::install(manager, id, std::move(source), completion);
  manager.start_preparation(id);
  std::string error;
  {
    completion_watchdog watchdog(completion);
    try {
      manager.await_query_completion(id, future);
    } catch (std::exception const& ex) {
      error = ex.what();
    }
  }
  // Error readiness precedes physical close; retirement must continue pumping.
  auto closes_at_error = observed->closes;
  manager.quiesce(id);
  manager.quiesce(id);
  manager.reset(id);
  REQUIRE(error == "provider read failed");
  REQUIRE(closes_at_error == 0);
  REQUIRE(observed->calls == 1);
  REQUIRE(observed->disposed == 1);
  REQUIRE(observed->disposer == std::this_thread::get_id());
  REQUIRE(observed->closes == 1);
  REQUIRE(observed->closer == std::this_thread::get_id());
}

TEST_CASE("owner cancellation and health checks run between provider operations",
          "[scan_manager][query_state][owner_pump]")
{
  manager_fixture f;
  auto& manager                       = f.manager;
  auto id                             = sirius::make_query_id(44);
  auto [completion, future, observed] = owner_attempt{};
  sm::scan_manager_query_test::install(manager, id, owner_source(observed, completion), completion);
  manager.start_preparation(id);
  bool const health_failure = GENERATE(false, true);
  std::string error;
  {
    completion_watchdog watchdog(completion);
    try {
      manager.await_query_completion(id, future, [&] {
        if (!observed->calls) return false;
        if (health_failure) throw std::runtime_error("runtime unavailable");
        return true;
      });
    } catch (duckdb::InterruptException const&) {
      error = "interrupted";
    } catch (std::runtime_error const& ex) {
      error = ex.what();
    }
  }
  manager.reset(id);
  REQUIRE(error == (health_failure ? "runtime unavailable" : "interrupted"));
  REQUIRE_FALSE(future.valid());
  REQUIRE(observed->calls == 1);
  REQUIRE(observed->disposed == 1);
  REQUIRE(observed->closes == 1);
  REQUIRE(observed->closer == std::this_thread::get_id());
}

TEST_CASE("execution error survives cancellation and closes unstarted owner preparation",
          "[scan_manager][query_state][owner_pump]")
{
  manager_fixture f;
  auto& manager                       = f.manager;
  auto id                             = sirius::make_query_id(45);
  auto [completion, future, observed] = owner_attempt{};
  sm::scan_manager_query_test::install(manager, id, owner_source(observed, completion), completion);
  auto report = [&] {
    completion->report_error(
      std::make_exception_ptr(std::runtime_error("original execution error")));
  };
  SECTION("error ready before owner wait")
  {
    report();
    REQUIRE_THROWS_WITH(manager.await_query_completion(id, future), "original execution error");
  }
  SECTION("error reported while checking cancellation")
  {
    REQUIRE_THROWS_WITH(manager.await_query_completion(id,
                                                       future,
                                                       [&] {
                                                         report();
                                                         return true;
                                                       }),
                        "original execution error");
  }
  SECTION("error reported while checking runtime health")
  {
    REQUIRE_THROWS_WITH(
      manager.await_query_completion(id,
                                     future,
                                     [&]() -> bool {
                                       report();
                                       throw std::runtime_error("runtime unavailable");
                                     }),
      "original execution error");
  }
  manager.quiesce(id);
  manager.start_preparation(id);  // Retired state cannot start a producer.
  manager.reset(id);
  REQUIRE(observed->calls == 0);
  REQUIRE(observed->closes == 1);
  REQUIRE(observed->closer == std::this_thread::get_id());
}

TEST_CASE("one external retirement does not block another query owner",
          "[scan_manager][query_state][owner_pump]")
{
  manager_fixture f;
  auto& manager = f.manager;
  auto a = sirius::make_query_id(46), b = sirius::make_query_id(47);
  auto [ca, future_a, oa] = owner_attempt{};
  auto [cb, future_b, ob] = owner_attempt{};
  auto gate = sm::scan_manager_query_test::install(manager, a, owner_source(oa, ca, true), ca);
  sm::scan_manager_query_test::install(manager, b, owner_source(ob, cb), cb);
  auto external = std::async(std::launch::async, [&] { manager.reset(a); });
  owner_retirement_guard retirement{manager, a};
  {
    std::unique_lock lock(gate->mutex);
    REQUIRE(gate->cv.wait_for(lock, 2s, [&] { return gate->closed; }));
  }
  manager.start_preparation(b);
  {
    completion_watchdog watchdog(cb);
    manager.await_query_completion(b, future_b);
  }
  manager.reset(b);
  auto a_still_waiting = external.wait_for(0ms);
  manager.quiesce(a);
  external.get();
  REQUIRE(a_still_waiting == std::future_status::timeout);
  REQUIRE(oa->closes == 1);
  REQUIRE(ob->calls == 1);
  REQUIRE(ob->closes == 1);
  REQUIRE_FALSE(sm::scan_manager_query_test::contains(manager, a));
  REQUIRE_FALSE(sm::scan_manager_query_test::contains(manager, b));
}

TEST_CASE("native reader capacity belongs to the manager across attempts",
          "[scan_manager][query_state][native_runtime]")
{
  auto config                              = make_local_config();
  config.preparation.native_reader_workers = 1;
  config.preparation.max_native_readers    = 1;
  manager_fixture f(config);
  auto& manager = f.manager;
  auto a = sirius::make_query_id(52), b = sirius::make_query_id(53);
  auto ca = std::make_shared<sirius::pipeline::completion_handler>();
  auto cb = std::make_shared<sirius::pipeline::completion_handler>();
  auto oa = std::make_shared<owner_observation>(), ob = std::make_shared<owner_observation>();
  auto sa = owner_source(oa, ca, true), sb = owner_source(ob, cb);
  sa.execution = sb.execution = sm::preparation_coordinator::execution_domain::native_reader;
  sm::scan_manager_query_test::install(manager, a, std::move(sa), ca);
  REQUIRE_THROWS_WITH(sm::scan_manager_query_test::install(manager, b, std::move(sb), cb),
                      "native reader capacity unavailable for complete attempt");
  REQUIRE(sm::scan_manager_query_test::contains(manager, a));
  REQUIRE_FALSE(sm::scan_manager_query_test::contains(manager, b));
  REQUIRE(ob->closes == 0);  // Unadmitted source has no initialized reader to close.
  manager.reset(a);
  REQUIRE(oa->closes == 1);
  REQUIRE(oa->closer != std::this_thread::get_id());

  // A returned its capacity. A fresh attempt can consume it without rebuilding the runtime.
  cb           = std::make_shared<sirius::pipeline::completion_handler>();
  auto future  = cb->get_awaitable();
  sb           = owner_source(ob, cb);
  sb.execution = sm::preparation_coordinator::execution_domain::native_reader;
  sm::scan_manager_query_test::install(manager, b, std::move(sb), cb);
  bool const start = GENERATE(false, true);
  CAPTURE(start);
  if (start) {
    manager.start_preparation(b);
    completion_watchdog watchdog(cb);
    manager.await_query_completion(b, future);
  }
  manager.stop();  // Drains both armed and started readers before joining native workers.
  REQUIRE(ob->calls == (start ? 1 : 0));
  REQUIRE(ob->closes == 1);
  REQUIRE(ob->closer == oa->closer);  // The same leased worker was reused.
  REQUIRE(ob->closer != std::this_thread::get_id());
  if (start) REQUIRE(ob->worker == ob->closer);
  REQUIRE_FALSE(sm::scan_manager_query_test::contains(manager, b));
  REQUIRE_NOTHROW(manager.stop());

  auto late_completion = std::make_shared<sirius::pipeline::completion_handler>();
  auto late            = owner_source(ob, late_completion);
  late.execution       = sm::preparation_coordinator::execution_domain::native_reader;
  REQUIRE_THROWS_WITH(
    sm::scan_manager_query_test::install(manager, b, std::move(late), late_completion),
    "native executor is stopped");
}

TEST_CASE("non-native preparation leaves the native runtime dormant",
          "[scan_manager][query_state][native_runtime]")
{
  manager_fixture f;
  auto& manager = f.manager;
  CHECK(sm::scan_manager_query_test::coordinator_workers(manager) == 0);
  auto id       = sirius::make_query_id(55);
  size_t starts = 0;
  sm::scan_manager_query_test::before_native_thread(manager, [&](size_t) { ++starts; });
  auto [completion, future, observed] = owner_attempt{};
  auto source                         = owner_source(observed, completion);
  SECTION("metadata source")
  {
    source.execution = sm::preparation_coordinator::execution_domain::metadata_worker;
  }
  SECTION("query-owner source")
  {
    source.execution = sm::preparation_coordinator::execution_domain::query_owner;
  }
  auto expected_coordinators =
    source.execution == sm::preparation_coordinator::execution_domain::metadata_worker ? 0u : 1u;
  sm::scan_manager_query_test::install(manager, id, std::move(source), completion);
  manager.start_preparation(id);
  {
    completion_watchdog watchdog(completion);
    manager.await_query_completion(id, future);
  }
  CHECK(sm::scan_manager_query_test::coordinator_workers(manager) == expected_coordinators);
  manager.stop();
  REQUIRE(starts == 0);
  REQUIRE(observed->calls == 1);
  REQUIRE(observed->closes == 1);
}

TEST_CASE("window retirement observes a provider close failure that arrives during drain",
          "[integration][query_state][preparation_cleanup]")
{
  auto const* child = std::getenv("SIRIUS_NATIVE_LEASE_UNAVAILABLE_VARIANT");
  auto variant      = child && *child ? std::stoi(child) : GENERATE(0, 1);
  if (sirius::test::run_isolated(std::to_string(variant))) return;
  sirius::test::GpuExecutionFixture fixture;
  auto& con     = fixture.con;
  auto state    = sirius::test::get_registered_sirius_context(*con);
  auto& manager = state->get_scan_manager();
  auto observed = std::make_shared<owner_observation>();
  sirius::query_id_t id{};
  bool unwind = variant != 0;
  try {
    duckdb::SiriusContext::StandaloneQueryScope window(*state, *con->context, "late_close");
    id              = window.query_id();
    auto completion = state->window_completion(id);
    auto source     = owner_source(observed, completion, true);
    auto close      = std::move(source.close);
    source.close    = [close] {
      close();
      throw std::runtime_error("late provider close failed");
    };
    sm::scan_manager_query_test::install(manager, id, std::move(source), completion);
    completion->report_error(
      std::make_exception_ptr(std::runtime_error("original execution failure")));
    REQUIRE(observed->closes == 0);
    if (unwind) throw std::runtime_error("original execution failure");
    window.finish();
    FAIL("late close failure must prevent replay");
  } catch (std::runtime_error const& error) {
    CHECK(std::string(error.what()) == "original execution failure");
  }
  CHECK(observed->closes == 1);
  CHECK(observed->closer == std::this_thread::get_id());
  CHECK(sm::scan_manager_query_test::contains(manager, id));
  CHECK(state->get_query_lifecycle_registry().size() == 1);
  CHECK(state->get_runtime_health() == duckdb::SiriusContext::runtime_health::UNAVAILABLE);
}

namespace {

// Test providers retain ordinary Parquet plumbing; only their preparation boundaries
// are instrumented. No production DuckLake dependency or reader is introduced.
struct preflight_refused : std::runtime_error {
  using std::runtime_error::runtime_error;
};

struct preflight_source : scan::parquet_gpu_ingestible {
  preflight_source() : parquet_gpu_ingestible(make_table_info("nation.parquet")) {}
  std::function<void()> preflight = [] {};
  std::function<void()> metadata  = [] {};
  void preflight_for_query() override { preflight(); }
  void ensure_metadata_prepared() override { metadata(); }
};

struct domain_source : preflight_source {
  sm::preparation_execution_domain execution;
  std::shared_ptr<sirius::pipeline::completion_handler> completion;
  std::vector<std::shared_ptr<owner_observation>> observations;
  std::atomic<size_t> closed = 0;
  size_t stops               = 0;
  bool stopped_after_close   = true;
  void stop_preparation() noexcept override
  {
    ++stops;
    for (auto const& observed : observations)
      stopped_after_close &= observed->closes == 1;
  }
  std::function<void()> after_claim                       = [] {};
  std::function<void(sm::preparation_stop_token)> on_work = [](auto) {};
  bool fail_close                                         = false;
  bool with_unit                                          = false;
  bool missing_input                                      = false;
  std::unique_ptr<scan::batch_coalescer> create_batch_coalescer() const override
  {
    return std::make_unique<empty_coalescer>();
  }

  preparation_source make_preparation_source(sirius::io::ioctx_resolver) override
  {
    auto observed = std::make_shared<owner_observation>();
    observations.push_back(observed);
    preparation_source source;
    source.execution = execution;
    source.try_claim = [observed,
                        with_unit     = with_unit,
                        missing_input = missing_input,
                        index         = observations.size(),
                        after_claim   = after_claim,
                        on_work       = on_work,
                        claimed       = false]() mutable -> metadata_claim {
      if (std::exchange(claimed, true)) return sm::preparation_exhausted{};
      auto job = metadata_job{
        [observed,
         with_unit,
         missing_input,
         on_work,
         capture = std::make_shared<owner_capture>(observed)](sm::preparation_stop_token stop) {
          ++observed->calls;
          observed->worker = std::this_thread::get_id();
          on_work(stop);
          if (!with_unit) return std::unique_ptr<scan::scan_info>{};
          auto input = std::make_unique<scan::scan_info>();
          scan::split_dependencies deps;
          deps.checkpoint_iteration = 17;
          if (!missing_input) input->set_contract_payload(0, {{}}, {deps});
          return input;
        },
        with_unit ? std::optional{metadata_unit{{1, index},
                                                sm::required_input_set{}.set(static_cast<size_t>(
                                                  sm::required_input::checkpoint_iteration))}}
                  : std::nullopt};
      after_claim();
      return job;
    };
    source.close = [this, observed] {
      ++observed->closes;
      observed->closer = std::this_thread::get_id();
      if (fail_close) throw std::runtime_error("registered provider close failed");
      if (++closed == observations.size()) completion->mark_completed();
    };
    return source;
  }
};

struct mixed_domain_source : domain_source {
  std::vector<preparation_source> make_preparation_sources(
    sirius::io::ioctx_resolver resolve) override
  {
    std::vector<preparation_source> sources;
    for (auto domain : {sm::preparation_execution_domain::metadata_worker,
                        sm::preparation_execution_domain::query_owner,
                        sm::preparation_execution_domain::native_reader}) {
      execution        = domain;
      auto source      = make_preparation_source(resolve);
      source.coalescer = create_batch_coalescer();
      sources.push_back(std::move(source));
    }
    return sources;
  }
};

std::shared_ptr<sm::preparation_admission> make_test_admission(
  sm::test::test_reservation_provider& provider)
{
  auto ledger = std::make_unique<sm::preparation_ledger>(provider);
  std::array envelopes{sm::scan_envelope{{{{8, 8, 16}, 20}}, {}, true}};
  auto decision = ledger->admit(envelopes, sm::memory_space_id{cucascade::memory::Tier::HOST, 0});
  REQUIRE(decision.deferred);
  return std::make_shared<sm::preparation_admission>(std::move(ledger), decision);
}

struct admitted_source : preflight_source {
  std::shared_ptr<sm::preparation_admission> admission;
  std::shared_ptr<sm::preparation_ledger> ledger;
  std::weak_ptr<sm::preparation_ledger> installed;
  bool fail_install = false, stopped = false, finished = false;
  sm::preparation_request request()
  {
    return {sm::scan_envelope{{{{8, 8, 16}, 20}}, {}, true},
            [](sm::preparation_ledger&) {},
            [this](auto value) { admission = std::move(value); },
            [this]() noexcept { admission.reset(); }};
  }
  std::shared_ptr<sm::preparation_admission> preparation_admission() const override
  {
    return admission;
  }
  void attach_preparation_ledger(std::shared_ptr<sm::preparation_ledger> value) override
  {
    ledger    = std::move(value);
    installed = ledger;
    if (fail_install) throw std::runtime_error("provider installation failed");
  }
  void stop_preparation() noexcept override
  {
    stopped = true;
    ledger.reset();
  }
  void finish_preparation() noexcept override
  {
    stop_preparation();
    admission->finish();
    finished = true;
  }
};

struct preflight_files {
  std::filesystem::path path =
    std::filesystem::temp_directory_path() /
    ("sirius_preflight_" + std::to_string(getpid()) + "_" +
     std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
  preflight_files() { std::filesystem::create_directory(path); }
  ~preflight_files()
  {
    std::error_code ignored;
    std::filesystem::remove_all(path, ignored);
  }
};

struct preflight_native_table {
  preflight_files files;
  duckdb::DuckDB db{(files.path / "native.duckdb").string()};
  duckdb::Connection con{db};
  duckdb::DataTable* storage;

  preflight_native_table()
  {
    auto result =
      con.Query("CREATE TABLE t(i INTEGER); INSERT INTO t VALUES (1); CHECKPOINT; BEGIN;");
    if (result->HasError()) throw std::runtime_error(result->GetError());
    auto& catalog = duckdb::Catalog::GetCatalog(*con.context, "");
    storage       = &catalog.GetEntry<duckdb::TableCatalogEntry>(*con.context, "main", "t")
                 .Cast<duckdb::DuckTableEntry>()
                 .GetStorage();
  }
  std::unique_ptr<scan::duckdb_native_ingestible_table_info> info()
  {
    auto result         = std::make_unique<scan::duckdb_native_ingestible_table_info>();
    result->storage     = storage;
    result->context     = con.context.get();
    result->db_path     = (files.path / "native.duckdb").string();
    result->names       = {"i"};
    result->table_name  = "t";
    result->contract_id = 100;
    result->column_ids.push_back(duckdb::ColumnIndex(0));
    result->projected_output_count = 1;
    result->projected_cols         = {{duckdb::StorageIndex(0), false}};
    result->projected_types        = {sirius::logical_type::make(sirius::type_id::INTEGER)};
    result->returned_types.push_back(sirius::logical_type::make(sirius::type_id::INTEGER));
    result->output_types = result->returned_types;
    return result;
  }
};

struct preflight_native_source : scan::duckdb_native_gpu_ingestible {
  explicit preflight_native_source(preflight_native_table& table)
    : duckdb_native_gpu_ingestible(table.info())
  {
  }
  std::function<void()> metadata;
  void ensure_metadata_prepared() override { metadata(); }
};

}  // namespace

TEST_CASE("registered reader work receives the attempt stop reason",
          "[scan_manager][query_state][provider_registration]")
{
  auto domain = GENERATE(sm::preparation_execution_domain::query_owner,
                         sm::preparation_execution_domain::native_reader);
  auto reason = GENERATE(
    sm::stop_reason::normal_eos, sm::stop_reason::failure_induced, sm::stop_reason::user_cancel);
  manager_fixture f;
  auto id            = sirius::make_query_id(608);
  auto source        = std::make_shared<domain_source>();
  source->execution  = domain;
  source->completion = std::make_shared<sirius::pipeline::completion_handler>();
  auto future        = source->completion->get_awaitable();
  sm::preparation_stop_token retained;
  source->on_work = [&](auto stop) {
    retained = stop;
    if (reason == sm::stop_reason::failure_induced)
      source->completion->report_error(
        std::make_exception_ptr(std::runtime_error("reader failure")));
    else if (reason == sm::stop_reason::normal_eos)
      source->completion->mark_completed();
    else
      f.manager.close_preparation(id, reason);
  };
  scan_query query{id, {source}};
  owner_retirement_guard retirement{f.manager, id};
  f.manager.prepare_for_query(*query.query, true, {}, source->completion);
  f.manager.start_preparation(id);
  {
    completion_watchdog watchdog(source->completion);
    if (reason == sm::stop_reason::normal_eos)
      REQUIRE_NOTHROW(f.manager.await_query_completion(id, future));
    else
      REQUIRE_THROWS(f.manager.await_query_completion(id, future));
  }
  f.manager.quiesce(id);
  f.manager.reset(id);
  CHECK(retained.stop_requested());
  CHECK(retained.reason() == reason);
  auto const& observed = *source->observations.front();
  CHECK(observed.calls == 1);
  CHECK(observed.worker == observed.disposer);
  CHECK(observed.worker == observed.closer);
}

TEST_CASE("registered reader claims retain domain ownership when unit admission stops",
          "[scan_manager][query_state][provider_registration]")
{
  auto domain = GENERATE(sm::preparation_execution_domain::metadata_worker,
                         sm::preparation_execution_domain::query_owner,
                         sm::preparation_execution_domain::native_reader);
  bool fail   = GENERATE(false, true);
  manager_fixture f;
  auto id                   = sirius::make_query_id(607);
  auto source               = std::make_shared<domain_source>();
  source->execution         = domain;
  source->with_unit         = true;
  source->completion        = std::make_shared<sirius::pipeline::completion_handler>();
  auto future               = source->completion->get_awaitable();
  std::atomic<bool> claimed = false;
  source->after_claim       = [&] {
    claimed = true;
    if (!fail) f.manager.close_preparation(id, sm::stop_reason::user_cancel);
  };
  scan_query query{id, {source}};
  owner_retirement_guard retirement{f.manager, id};
  f.manager.prepare_for_query(*query.query, true, {}, source->completion, [&] {
    if (fail && claimed) throw std::runtime_error("unit admission failed");
    return false;
  });
  f.manager.start_preparation(id);
  {
    completion_watchdog watchdog(source->completion);
    REQUIRE_THROWS(f.manager.await_query_completion(id, future));
  }
  f.manager.quiesce(id);
  f.manager.reset(id);
  auto const& observed = *source->observations.front();
  CHECK(observed.calls == 0);
  CHECK(observed.disposed == 1);
  CHECK(observed.closes == 1);
  CHECK(observed.disposer == observed.closer);
  CHECK((observed.closer == std::this_thread::get_id()) ==
        (domain != sm::preparation_execution_domain::native_reader));
}

TEST_CASE("scan manager registers independent provider execution and close",
          "[scan_manager][query_state][native_runtime][provider_registration]")
{
  auto const domain = GENERATE(sm::preparation_execution_domain::query_owner,
                               sm::preparation_execution_domain::native_reader);
  enum class ending { exhausted, cancelled, close_failure, unit_ready, unit_missing_input };
  auto const mode = GENERATE(ending::exhausted,
                             ending::cancelled,
                             ending::close_failure,
                             ending::unit_ready,
                             ending::unit_missing_input);
  auto config     = make_local_config();
  config.preparation.coordinator_workers =
    GENERATE(std::optional<size_t>{0}, std::optional<size_t>{});
  manager_fixture f(config);
  auto& manager         = f.manager;
  auto const id         = sirius::make_query_id(51);
  auto source           = std::make_shared<domain_source>();
  source->execution     = domain;
  source->fail_close    = mode == ending::close_failure;
  source->with_unit     = mode == ending::unit_ready || mode == ending::unit_missing_input;
  source->missing_input = mode == ending::unit_missing_input;
  source->completion    = std::make_shared<sirius::pipeline::completion_handler>();
  auto future           = source->completion->get_awaitable();
  scan_query query{id, {source, source}};
  owner_retirement_guard retirement{manager, id};
  manager.prepare_for_query(*query.query, true, {}, source->completion);
  auto const registrations = source->observations.size();
  if (registrations != 2) manager.reset(id);
  REQUIRE(registrations == 2);
  CHECK(sm::scan_manager_query_test::coordinator_workers(manager) ==
        config.preparation.resolved_coordinator_workers());

  if (mode != ending::cancelled) {
    manager.start_preparation(id);
    completion_watchdog watchdog(source->completion);
    if (mode == ending::close_failure) {
      REQUIRE_THROWS_WITH(manager.await_query_completion(id, future),
                          "registered provider close failed");
    } else if (mode == ending::unit_missing_input) {
      REQUIRE_THROWS_WITH(manager.await_query_completion(id, future),
                          "preparation unit requires one dependency record");
    } else {
      REQUIRE_NOTHROW(manager.await_query_completion(id, future));
    }
  }
  manager.quiesce(id);
  manager.reset(id);
  CHECK_FALSE(sm::scan_manager_query_test::contains(manager, id));
  CHECK(source->stops > 0);
  CHECK(source->stopped_after_close);
  for (auto const& observed : source->observations) {
    CHECK(observed->closes == 1);
    CHECK((observed->closer == std::this_thread::get_id()) ==
          (domain == sm::preparation_execution_domain::query_owner));
    if (mode == ending::exhausted || mode == ending::unit_ready) {
      CHECK(observed->calls == 1);
      CHECK(observed->disposed == 1);
      CHECK(observed->worker == observed->closer);
      CHECK(observed->disposer == observed->worker);
    } else if (mode == ending::cancelled) {
      CHECK(observed->calls == 0);
      CHECK(observed->disposed == 0);
    }
  }
  for (auto const& op : query.operators)
    CHECK(op->get_split_connector().is_closed());
  if (mode == ending::close_failure) {
    CHECK_THROWS_WITH(source->completion->throw_if_preparation_cleanup_failed(),
                      "registered provider close failed");
  } else {
    CHECK_NOTHROW(source->completion->throw_if_preparation_cleanup_failed());
  }
  REQUIRE_NOTHROW(manager.stop());
  REQUIRE_NOTHROW(manager.stop());
}

TEST_CASE("scan manager registers several execution domains for one scan",
          "[scan_manager][query_state][preparation_group]")
{
  bool const cancel_before_start = GENERATE(false, true);
  manager_fixture f;
  auto id            = sirius::make_query_id(607);
  auto source        = std::make_shared<mixed_domain_source>();
  source->completion = std::make_shared<sirius::pipeline::completion_handler>();
  auto future        = source->completion->get_awaitable();
  scan_query query{id, {source}};
  owner_retirement_guard retirement{f.manager, id};
  f.manager.prepare_for_query(*query.query, true, {}, source->completion);
  REQUIRE(source->observations.size() == 3);
  if (!cancel_before_start) {
    f.manager.start_preparation(id);
    completion_watchdog watchdog(source->completion);
    REQUIRE_NOTHROW(f.manager.await_query_completion(id, future));
  }
  f.manager.quiesce(id);
  f.manager.reset(id);
  CHECK(source->stopped_after_close);
  for (size_t i = 0; i < source->observations.size(); ++i) {
    auto const& observed = *source->observations[i];
    CHECK(observed.closes == 1);
    CHECK(observed.calls == (cancel_before_start ? 0 : 1));
    CHECK((observed.closer == std::this_thread::get_id()) == (i == 1));
    if (!cancel_before_start) {
      CHECK(observed.disposer == observed.worker);
      if (i != 0) CHECK(observed.worker == observed.closer);
    }
  }
  CHECK(query.operators.front()->get_split_connector().is_closed());
}

TEST_CASE("independent providers compose admission and release partial preparation installations",
          "[scan_manager][query_state][preparation_lifecycle]")
{
  manager_fixture f;
  sm::test::test_reservation_provider provider;
  auto first           = std::make_shared<admitted_source>();
  auto second          = std::make_shared<admitted_source>();
  provider.grant_bytes = 128;
  std::array requests{first->request(), second->request()};
  auto decision = sm::prepare_statement_resources(
    provider, requests, sm::memory_space_id{cucascade::memory::Tier::HOST, 0});
  REQUIRE(decision.deferred);
  CHECK(decision.sigma_retained == 40);
  CHECK(decision.c_requested == 72);
  CHECK(provider.seen->requests == 1);
  bool mismatch = false;
  SECTION("shared admission") {}
  SECTION("second installation fails") { second->fail_install = true; }
  SECTION("foreign statement admission is refused")
  {
    second->admission = make_test_admission(provider);
    mismatch          = true;
  }
  auto const id = sirius::make_query_id(606);
  scan_query query{id, {first, second}};
  owner_retirement_guard retirement{f.manager, id};
  auto completion = std::make_shared<sirius::pipeline::completion_handler>();
  if (second->fail_install || mismatch) {
    CHECK_THROWS_WITH(f.manager.prepare_for_query(*query.query, true, {}, completion),
                      mismatch ? "scans must share the statement preparation admission"
                               : "provider installation failed");
    CHECK_FALSE(sm::scan_manager_query_test::contains(f.manager, id));
  } else {
    REQUIRE_NOTHROW(f.manager.prepare_for_query(*query.query, true, {}, completion));
    REQUIRE(first->ledger);
    CHECK(first->ledger == second->ledger);
    CHECK_FALSE(first->stopped);
    f.manager.quiesce(id);
    f.manager.reset(id);
  }
  CHECK(first->stopped);
  CHECK(second->stopped);
  CHECK(first->installed.expired());
  CHECK(second->installed.expired());
  CHECK_THROWS_AS(first->admission->consume(), std::logic_error);
  CHECK(provider.outstanding() == (mismatch ? 1 : 0));
}

TEST_CASE("engine retires non-Iceberg preparation before releasing an unstarted plan",
          "[integration][query_state][preparation_lifecycle]")
{
  sirius::test::GpuExecutionFixture fixture;
  auto state = sirius::test::get_registered_sirius_context(*fixture.con);
  duckdb::SiriusContext::StandaloneQueryScope window(
    *state, *fixture.con->context, "unstarted_preparation");
  auto source = std::make_shared<admitted_source>();
  // The engine can own a plan before manager registration has installed any attempt.
  sm::test::test_reservation_provider provider;
  source->admission = make_test_admission(provider);
  scan_query query{window.query_id(), {source}};
  {
    sirius::sirius_engine engine(*fixture.con->context, window.query_id());
    engine.sirius_physical_plan = query.operators.front().get();
  }
  CHECK(source->finished);
  CHECK(provider.outstanding() == 0);
}

TEST_CASE("all bound providers preflight before a native scan takes its checkpoint key",
          "[scan_manager][query_state][provider_preflight]")
{
  bool const native_first = GENERATE(false, true);
  preflight_native_table table;
  manager_fixture f;
  auto& manager    = f.manager;
  auto const id    = sirius::make_query_id(600);
  auto const owner = std::this_thread::get_id();
  auto a           = std::make_shared<preflight_source>();
  auto b           = std::make_shared<preflight_source>();
  auto native      = std::make_shared<preflight_native_source>(table);
  int ac = 0, bc = 0;
  a->preflight = [&] {
    REQUIRE(std::this_thread::get_id() == owner);
    REQUIRE(manager.checkpoint_key_count(id) == 0);
    ++ac;
  };
  b->preflight = [&] {
    REQUIRE(std::this_thread::get_id() == owner);
    REQUIRE(manager.checkpoint_key_count(id) == 0);
    ++bc;
  };
  native->metadata = [&] {
    REQUIRE(ac == 1);
    REQUIRE(bc == 1);
    REQUIRE(manager.holds_checkpoint_key(id, native->attached_database()));
    throw std::runtime_error("native metadata boundary reached");
  };
  std::vector<std::shared_ptr<scan::gpu_ingestible>> sources{a, b, a};
  sources.insert(native_first ? sources.begin() : sources.end(), native);
  scan_query query{id, std::move(sources)};
  REQUIRE_THROWS_WITH(manager.prepare_for_query(*query.query, true, {}),
                      "native metadata boundary reached");
  REQUIRE(ac == 1);
  REQUIRE(bc == 1);
  REQUIRE(manager.checkpoint_key_count(id) == 1);
  manager.reset(id);
  REQUIRE(manager.checkpoint_key_count(id) == 0);
}

TEST_CASE("failed or interrupted provider preflight never starts scan preparation",
          "[scan_manager][query_state][provider_preflight]")
{
  int const mode = GENERATE(0, 1, 2, 3, 4);
  manager_fixture f;
  auto& manager = f.manager;
  auto const id = sirius::make_query_id(601);
  auto a        = std::make_shared<preflight_source>();
  auto b        = std::make_shared<preflight_source>();
  int ac = 0, bc = 0, metadata = 0;
  bool interrupted = mode >= 2;
  auto completion  = std::make_shared<sirius::pipeline::completion_handler>();
  if (mode == 2)
    completion->report_error(std::make_exception_ptr(std::runtime_error("original query failure")));
  a->preflight = [&] { ++ac; };
  b->preflight = [&] {
    ++bc;
    if (mode == 0) throw preflight_refused("provider preflight refused");
    interrupted = true;
  };
  a->metadata = b->metadata = [&] {
    ++metadata;
    throw std::runtime_error("metadata ran too early");
  };
  scan_query query{id, {a, b}};
  auto prepare = [&] {
    manager.prepare_for_query(*query.query, true, {}, completion, [&] {
      // The original failure wins even if it commits during the interrupt check.
      if (mode == 4)
        completion->report_error(
          std::make_exception_ptr(std::runtime_error("original query failure")));
      return interrupted;
    });
  };
  if (mode == 0) {
    try {
      prepare();
      FAIL("preflight refusal was swallowed");
    } catch (preflight_refused const& error) {
      REQUIRE(std::string(error.what()) == "provider preflight refused");
    }
  } else if (mode == 2 || mode == 4) {
    REQUIRE_THROWS_WITH(prepare(), "original query failure");
  } else {
    REQUIRE_THROWS_AS(prepare(), duckdb::InterruptException);
  }
  REQUIRE(ac == (mode >= 2 ? 0 : 1));
  REQUIRE(bc == (mode >= 2 ? 0 : 1));
  REQUIRE(metadata == 0);
  REQUIRE(manager.checkpoint_key_count(id) == 0);
  REQUIRE_FALSE(sirius::scan_manager::scan_manager_query_test::contains(manager, id));
  manager.reset(id);
}

TEST_CASE("provider preflight requires only its own attempt to be free of checkpoint keys",
          "[scan_manager][query_state][provider_preflight]")
{
  bool const same_query = GENERATE(false, true);
  preflight_native_table table;
  manager_fixture f;
  auto& manager        = f.manager;
  auto const id        = sirius::make_query_id(603);
  auto const key_owner = same_query ? id : sirius::make_query_id(604);
  manager.acquire_checkpoint_key(key_owner, table.storage->GetAttached());
  auto a       = std::make_shared<preflight_source>();
  int calls    = 0;
  a->preflight = [&] { ++calls; };
  a->metadata  = [] { throw std::runtime_error("metadata boundary reached"); };
  scan_query query{id, {a}};
  REQUIRE_THROWS_WITH(manager.prepare_for_query(*query.query, true, {}),
                      same_query ? "provider preflight requires an attempt without checkpoint keys"
                                 : "metadata boundary reached");
  REQUIRE(calls == (same_query ? 0 : 1));
  REQUIRE(manager.checkpoint_key_count(key_owner) == 1);
  manager.reset(id);
  manager.reset(key_owner);
}

TEST_CASE("shared providers preflight once per attempt with or without cached rows",
          "[scan_manager][query_state][provider_preflight]")
{
  bool cached = GENERATE(false, true);
  manager_fixture f;
  auto& manager     = f.manager;
  auto const id     = sirius::make_query_id(605);
  auto source       = std::make_shared<preflight_source>();
  int calls         = 0;
  source->preflight = [&] { ++calls; };
  if (cached) {
    source->metadata = [] { throw std::runtime_error("cache hit entered disk metadata"); };
    auto* space      = sirius::scan_test_utils::get_space(*f.memory, cucascade::memory::Tier::GPU);
    cudf::data_type const type{cudf::type_id::INT32};
    std::vector<std::unique_ptr<cudf::column>> columns;
    columns.push_back(cudf::make_fixed_width_column(type, 1));
    std::vector<std::unique_ptr<cudf::table>> tables;
    tables.push_back(std::make_unique<cudf::table>(std::move(columns)));
    auto stored = manager.insert_pinned_entry(
      "preflight_pin",
      sirius::scan_manager::cache_entry_info::from(source->table_info()),
      std::move(tables),
      {space},
      {},
      {},
      {{{type, false}}});
    REQUIRE(stored.size() == 1);
  }
  scan_query query{id, {source, source}};
  for (int attempt = 1; attempt <= 2; ++attempt) {
    REQUIRE_NOTHROW(manager.prepare_for_query(*query.query, true, {}));
    REQUIRE(calls == attempt);
    REQUIRE(manager.checkpoint_key_count(id) == 0);
    manager.reset(id);
    REQUIRE_FALSE(sm::scan_manager_query_test::contains(manager, id));
  }
  if (cached) manager.remove_pinned_entry("preflight_pin");
}
