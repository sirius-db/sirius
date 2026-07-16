/*
 * Copyright 2025, Sirius Contributors.
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

#include "operator/operator_test_utils.hpp"
#include "utils/utils.hpp"

#include <catch.hpp>
#include <cucascade/data/data_batch.hpp>
#include <cucascade/data/data_repository.hpp>
#include <data/data_batch_utils.hpp>
#include <exec/exchange_channel.hpp>
#include <exec/stream_session.hpp>
#include <helper/type_conversions.hpp>
#include <op/sirius_physical_streaming_sink.hpp>
#include <op/sirius_physical_streaming_source.hpp>
#include <sirius/exception.hpp>
#include <sirius_context.hpp>

#include <chrono>
#include <filesystem>
#include <memory>
#include <set>
#include <thread>
#include <vector>

using namespace sirius::exec;
using namespace sirius::op;
using namespace cucascade;
using namespace cucascade::memory;
using namespace sirius::test::operator_utils;

namespace {

std::filesystem::path integration_config_path()
{
  // test/cpp/exec/ -> test/cpp/ -> test/cpp/integration/integration.yaml
  return std::filesystem::path(__FILE__).parent_path().parent_path() / "integration" /
         "integration.yaml";
}

duckdb::vector<sirius::logical_type> int_type()
{
  return sirius::from_duckdb_vec(duckdb::vector<duckdb::LogicalType>{duckdb::LogicalType::INTEGER});
}

// A hand-built `STREAMING_SOURCE -> STREAMING_SINK` plan plus its bindings — the identity loopback
// that exercises the whole session path (push -> source -> port -> sink -> pull).
struct loopback_plan {
  std::shared_ptr<exchange_channel> in_ch;
  std::shared_ptr<shared_data_repository> in_repo;
  std::shared_ptr<exchange_channel> out_ch;
  std::shared_ptr<shared_data_repository> out_repo;
  sirius_physical_streaming_source* source{nullptr};  // owned by streaming_plan::root
  sirius_physical_streaming_sink* sink{nullptr};      // owned by streaming_plan::root
  streaming_plan plan;
};

loopback_plan make_loopback(std::size_t in_capacity  = 16,
                            std::size_t out_capacity = 16,
                            std::uint32_t senders    = 1)
{
  loopback_plan lb;
  lb.in_ch = std::make_shared<exchange_channel>(
    exchange_channel::config{.capacity_items = in_capacity, .expected_senders = senders});
  lb.in_repo = std::make_shared<shared_data_repository>();
  lb.out_ch =
    std::make_shared<exchange_channel>(exchange_channel::config{.capacity_items = out_capacity});
  lb.out_repo = std::make_shared<shared_data_repository>();

  auto source =
    std::make_unique<sirius_physical_streaming_source>(int_type(), 0, lb.in_ch, lb.in_repo);
  lb.source = source.get();
  auto sink =
    std::make_unique<sirius_physical_streaming_sink>(int_type(), 0, lb.out_ch, lb.out_repo);
  lb.sink = sink.get();

  // sink is the tree root; the source is its single child (source -> sink).
  sink->children.push_back(std::move(source));

  lb.plan.root = std::move(sink);
  lb.plan.inputs.push_back({/*id=*/1, lb.in_ch, lb.in_repo, lb.source});
  lb.plan.outputs.push_back({/*id=*/2, lb.out_ch, lb.out_repo, lb.sink});
  return lb;
}

}  // namespace

// ============================================================================
// Construction & validation (no execution)
// ============================================================================

TEST_CASE("stream_session: builds a valid source->sink plan", "[stream_session]")
{
  auto [db, con] = sirius::make_test_db_and_connection();
  auto ctx       = sirius::get_sirius_context(con, integration_config_path());

  auto lb = make_loopback();
  REQUIRE_NOTHROW(stream_session(*ctx, std::move(lb.plan)));
}

TEST_CASE("stream_session: rejects malformed plans", "[stream_session]")
{
  auto [db, con] = sirius::make_test_db_and_connection();
  auto ctx       = sirius::get_sirius_context(con, integration_config_path());

  SECTION("duplicate input stream id")
  {
    auto lb = make_loopback();
    lb.plan.inputs.push_back({/*id=*/1, lb.in_ch, lb.in_repo, lb.source});  // id 1 already used
    REQUIRE_THROWS_AS(stream_session(*ctx, std::move(lb.plan)), sirius::invalid_input_exception);
  }

  SECTION("stream id shared between input and output")
  {
    auto lb               = make_loopback();
    lb.plan.outputs[0].id = 1;  // collides with the input id
    REQUIRE_THROWS_AS(stream_session(*ctx, std::move(lb.plan)), sirius::invalid_input_exception);
  }

  SECTION("null binding operator")
  {
    auto lb              = make_loopback();
    lb.plan.inputs[0].op = nullptr;
    REQUIRE_THROWS_AS(stream_session(*ctx, std::move(lb.plan)), sirius::invalid_input_exception);
  }
}

TEST_CASE("stream_session: unknown stream ids are rejected", "[stream_session]")
{
  auto [db, con] = sirius::make_test_db_and_connection();
  auto ctx       = sirius::get_sirius_context(con, integration_config_path());

  auto lb = make_loopback();
  stream_session session(*ctx, std::move(lb.plan));

  REQUIRE_THROWS_AS(session.pull(/*unknown=*/99), sirius::invalid_input_exception);
  REQUIRE_THROWS_AS(session.close_input(/*unknown=*/99), sirius::invalid_input_exception);
  REQUIRE_THROWS_AS(session.wait(/*unknown=*/99), sirius::invalid_input_exception);
}

// ============================================================================
// End-to-end loopback (IT-0): push -> source -> sink -> pull, with EOS.
// ============================================================================

TEST_CASE("stream_session: loopback identity round-trip with EOS", "[stream_session][it0]")
{
  auto mem_mgr    = sirius::test::operator_utils::initialize_memory_manager();
  auto* gpu_space = mem_mgr->get_memory_space(Tier::GPU, 0);
  REQUIRE(gpu_space != nullptr);

  auto [db, con] = sirius::make_test_db_and_connection();
  auto ctx       = sirius::get_sirius_context(con, integration_config_path());
  // The task_creator's client context is wired in SiriusContext::QueryBegin, which only fires for
  // a real DuckDB query. Run a trivial one so create_query() below can prepare the creator.
  con.Query("SELECT 42");

  constexpr int N = 4;
  auto lb         = make_loopback();
  auto stream     = default_stream();

  stream_session session(*ctx, std::move(lb.plan));
  session.start();

  std::set<uint64_t> expected_ids;
  for (int i = 0; i < N; ++i) {
    auto b = make_numeric_batch<int32_t>(*gpu_space, {i}, cudf::type_id::INT32);
    expected_ids.insert(b->get_batch_id());
    session.push(/*stream=*/1, std::move(b));
  }
  session.close_input(/*stream=*/1);

  // Drain the output, driving re-arm through pull(). Bounded by a wall-clock deadline so a wiring
  // bug fails as a timeout instead of hanging the suite.
  std::set<uint64_t> got_ids;
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
  while (!session.drained(/*stream=*/2)) {
    auto batch = session.pull(/*stream=*/2);
    if (batch) {
      got_ids.insert((*batch)->get_batch_id());
    } else {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    REQUIRE(std::chrono::steady_clock::now() < deadline);
  }
  // Drain any batch delivered exactly as the channel closed.
  while (auto batch = session.pull(/*stream=*/2)) {
    got_ids.insert((*batch)->get_batch_id());
  }

  REQUIRE(got_ids == expected_ids);
  REQUIRE(session.finished());
}

TEST_CASE("stream_session: backpressure delivers every batch through a tiny channel",
          "[stream_session][it0]")
{
  auto mem_mgr    = sirius::test::operator_utils::initialize_memory_manager();
  auto* gpu_space = mem_mgr->get_memory_space(Tier::GPU, 0);
  REQUIRE(gpu_space != nullptr);

  auto [db, con] = sirius::make_test_db_and_connection();
  auto ctx       = sirius::get_sirius_context(con, integration_config_path());
  con.Query("SELECT 42");

  constexpr int N = 8;
  auto lb         = make_loopback(/*in_capacity=*/2, /*out_capacity=*/2);  // N >> capacity

  stream_session session(*ctx, std::move(lb.plan));
  session.start();

  // Build all batches up front so the producer thread only pushes.
  std::vector<std::shared_ptr<data_batch>> batches;
  std::set<uint64_t> expected_ids;
  for (int i = 0; i < N; ++i) {
    auto b = make_numeric_batch<int32_t>(*gpu_space, {i}, cudf::type_id::INT32);
    expected_ids.insert(b->get_batch_id());
    batches.push_back(std::move(b));
  }

  // Producer thread: push() blocks while the tiny input channel is full and resumes as the engine
  // drains it. CHECK (not REQUIRE) — a throwing assertion across a thread boundary terminates.
  std::thread producer([&] {
    for (auto& b : batches) {
      try {
        session.push(/*stream=*/1, b);
      } catch (...) {
        CHECK(false);
        return;
      }
    }
    session.close_input(/*stream=*/1);
  });

  std::set<uint64_t> got_ids;
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
  while (!session.drained(/*stream=*/2)) {
    auto batch = session.pull(/*stream=*/2);
    if (batch) {
      got_ids.insert((*batch)->get_batch_id());
    } else {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    REQUIRE(std::chrono::steady_clock::now() < deadline);
  }
  while (auto batch = session.pull(/*stream=*/2)) {
    got_ids.insert((*batch)->get_batch_id());
  }

  producer.join();
  REQUIRE(got_ids == expected_ids);
}

TEST_CASE("stream_session: fan-in input closes only after every sender signals EOS",
          "[stream_session][it0]")
{
  auto mem_mgr    = sirius::test::operator_utils::initialize_memory_manager();
  auto* gpu_space = mem_mgr->get_memory_space(Tier::GPU, 0);
  REQUIRE(gpu_space != nullptr);

  auto [db, con] = sirius::make_test_db_and_connection();
  auto ctx       = sirius::get_sirius_context(con, integration_config_path());
  con.Query("SELECT 42");

  constexpr int N = 3;
  auto lb         = make_loopback(/*in_capacity=*/16, /*out_capacity=*/16, /*senders=*/2);

  stream_session session(*ctx, std::move(lb.plan));
  session.start();

  std::set<uint64_t> expected_ids;
  for (int i = 0; i < N; ++i) {
    auto b = make_numeric_batch<int32_t>(*gpu_space, {i}, cudf::type_id::INT32);
    expected_ids.insert(b->get_batch_id());
    session.push(/*stream=*/1, std::move(b));
  }

  // Only one of the two expected senders has signalled EOS: the input stream stays open, so the
  // output cannot reach end-of-stream no matter how much we drain.
  session.close_input(/*stream=*/1, /*sender=*/0);

  std::set<uint64_t> got_ids;
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
  while (got_ids.size() < static_cast<std::size_t>(N)) {
    auto batch = session.pull(/*stream=*/2);
    if (batch) {
      got_ids.insert((*batch)->get_batch_id());
    } else {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    REQUIRE(std::chrono::steady_clock::now() < deadline);
  }
  REQUIRE(got_ids == expected_ids);
  REQUIRE_FALSE(session.drained(/*stream=*/2));  // still open: sender 1 has not signalled
  REQUIRE_FALSE(session.finished());

  // The final sender signals EOS — now the stream drains to end-of-stream.
  session.close_input(/*stream=*/1, /*sender=*/1);
  const auto eos_deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
  while (!session.drained(/*stream=*/2)) {
    session.pull(/*stream=*/2);
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
    REQUIRE(std::chrono::steady_clock::now() < eos_deadline);
  }
  REQUIRE(session.finished());
}
