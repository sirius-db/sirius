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

// A plan built in one statement and run in another (a prepared statement's validated plan), and
// a failed window's abort reaching the other ranks.

#include "exchange/all_gather.hpp"
#include "exchange/exchange_service.hpp"
#include "exchange/loopback_fabric.hpp"
#include "sirius_context.hpp"

#include <catch.hpp>
#include <duckdb.hpp>
#include <utils/sirius_test_env.hpp>

#include <chrono>
#include <condition_variable>
#include <exception>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <vector>

using window = duckdb::SiriusContext::StandaloneQueryScope;

TEST_CASE("a distributed query planned in an earlier statement", "[exchange][query_windows]")
{
  REQUIRE(sirius::test::g_integration_env != nullptr);
  if (!sirius::test::g_integration_env->is_active()) { sirius::test::g_integration_env->resume(); }
  auto con = sirius::test::g_integration_env->make_connection();
  auto ctx = con.context->registered_state->Get<duckdb::SiriusContext>("sirius_state");
  REQUIRE(ctx);
  auto const planned = ctx->next_distributed_query(*con.context);
  // The next statement starts: what the last one planned is no longer pending.
  REQUIRE_FALSE(con.Query("SELECT 42")->HasError());

  SECTION("is not adopted by the next statement's window")
  {
    window w(*ctx, *con.context, "next_statement");
    CHECK(ctx->distributed_query_of(w.query_id()) == 0);
  }

  SECTION("is adopted by the window that reuses its plan")
  {
    sirius::query_id_t id{};
    {
      window w(*ctx, *con.context, "reuse");
      id = w.query_id();
      ctx->adopt_distributed_query(*con.context, planned);
      CHECK(ctx->distributed_query_of(id) == planned);
    }
    CHECK(ctx->distributed_query_of(id) == 0);
  }
}

TEST_CASE("a failed distributed window aborts its query on the other ranks",
          "[exchange][query_windows]")
{
  using namespace sirius::test::exchange_utils;
  REQUIRE(sirius::test::g_integration_env != nullptr);
  if (!sirius::test::g_integration_env->is_active()) { sirius::test::g_integration_env->resume(); }
  auto con = sirius::test::g_integration_env->make_connection();
  auto ctx = con.context->registered_state->Get<duckdb::SiriusContext>("sirius_state");
  REQUIRE(ctx);
  auto& gpu = *ctx->get_memory_manager().get_memory_space(cucascade::memory::Tier::GPU, 0);

  // This process is rank 0 of a loopback exchange; the test plays rank 1.
  loopback_hub hub(2);
  std::vector<std::unique_ptr<frame_ring>> rings;
  std::vector<std::unique_ptr<fabric>> fabrics;
  std::vector<std::shared_ptr<sirius::exchange::exchange_service>> services;
  for (rank_t r = 0; r < 2; ++r) {
    rings.push_back(std::make_unique<frame_ring>(4, 64 << 10));
    rings.push_back(std::make_unique<frame_ring>(8, 64 << 10));
    fabrics.push_back(hub.attach(r, *rings[2 * r], *rings[2 * r + 1]));
    services.push_back(std::make_shared<sirius::exchange::exchange_service>(
      *fabrics.back(), *rings[2 * r], *rings[2 * r + 1], gpu, nullptr));
  }
  ctx->set_exchange_service_for_testing(services[0]);

  auto const query = ctx->next_distributed_query(*con.context);
  {
    window w(*ctx, *con.context, "fails");
    ctx->abort_distributed_query(w.query_id(),
                                 std::make_exception_ptr(std::runtime_error("rank 0 failed")));
  }
  // Rank 1 waits on rank 0 in this query's all-gather; the abort ends the wait at once.
  auto const started = std::chrono::steady_clock::now();
  CHECK_THROWS_WITH(sirius::exchange::all_gather(*services[1],
                                                 sirius::exchange::make_stream_id(query, 1),
                                                 nullptr,
                                                 gpu,
                                                 rmm::cuda_stream_default,
                                                 std::chrono::seconds{60}),
                    Catch::Matchers::ContainsSubstring("rank 0 failed"));
  CHECK(std::chrono::steady_clock::now() - started < std::chrono::seconds{30});

  ctx->set_exchange_service_for_testing(nullptr);
  services.clear();
}

namespace {

/// Counts the ends a stream's senders sent; waits for the first.
struct counting_inbound : sirius::exchange::inbound {
  std::mutex mutex;
  std::condition_variable ended;
  int ends = 0;
  void on_batch(sirius::exchange::rank_t,
                std::uint64_t,
                std::shared_ptr<cucascade::data_batch>) override
  {
  }
  void on_end(sirius::exchange::rank_t) override
  {
    std::lock_guard lock{mutex};
    ++ends;
    ended.notify_all();
  }
  void on_error(std::exception_ptr) override {}
  bool wait_end()
  {
    std::unique_lock lock{mutex};
    return ended.wait_for(lock, std::chrono::seconds{30}, [this] { return ends > 0; });
  }
  int count()
  {
    std::lock_guard lock{mutex};
    return ends;
  }
};

}  // namespace

TEST_CASE("numbers a gateway skips retire on this rank; a prepared query's number does not",
          "[exchange][query_windows]")
{
  using namespace sirius::test::exchange_utils;
  REQUIRE(sirius::test::g_integration_env != nullptr);
  if (!sirius::test::g_integration_env->is_active()) { sirius::test::g_integration_env->resume(); }
  auto con = sirius::test::g_integration_env->make_connection();
  auto ctx = con.context->registered_state->Get<duckdb::SiriusContext>("sirius_state");
  REQUIRE(ctx);
  auto& gpu = *ctx->get_memory_manager().get_memory_space(cucascade::memory::Tier::GPU, 0);

  // Skipped before the exchange connects: retired once it does.
  auto const base           = ctx->next_distributed_query(*con.context);
  auto const before_connect = base + 2;
  auto const prepared       = ctx->next_distributed_query(*con.context, base + 3);

  loopback_hub hub(2);
  std::vector<std::unique_ptr<frame_ring>> rings;
  std::vector<std::unique_ptr<fabric>> fabrics;
  std::vector<std::shared_ptr<sirius::exchange::exchange_service>> services;
  for (rank_t r = 0; r < 2; ++r) {
    rings.push_back(std::make_unique<frame_ring>(4, 64 << 10));
    rings.push_back(std::make_unique<frame_ring>(8, 64 << 10));
    fabrics.push_back(hub.attach(r, *rings[2 * r], *rings[2 * r + 1]));
    services.push_back(std::make_shared<sirius::exchange::exchange_service>(
      *fabrics.back(), *rings[2 * r], *rings[2 * r + 1], gpu, nullptr));
  }
  ctx->set_exchange_service_for_testing(services[0]);

  // The prepared statement runs later; the statements in between skip numbers here.
  REQUIRE_FALSE(con.Query("SELECT 42")->HasError());
  auto const skipped = prepared + 4;
  auto const later   = ctx->next_distributed_query(*con.context, prepared + 10);

  // Rank 1 ends a stream of each; rank 0 attaches late, after they have all landed.
  for (auto query : {before_connect, skipped, prepared, later}) {
    services[1]->close(sirius::exchange::make_stream_id(query, 1), 0);
  }
  auto const last_landed = std::make_shared<counting_inbound>();
  services[0]->attach(sirius::exchange::make_stream_id(later, 1), last_landed);
  REQUIRE(last_landed->wait_end());
  auto const count_ends = [&](std::uint64_t query) {
    auto const receiver = std::make_shared<counting_inbound>();
    services[0]->attach(sirius::exchange::make_stream_id(query, 1), receiver);
    return receiver->count();
  };
  CHECK(count_ends(before_connect) == 0);
  CHECK(count_ends(skipped) == 0);
  CHECK(count_ends(prepared) == 1);

  // A plan rebuilt within the statement that numbered it gets a clear error.
  CHECK_THROWS_WITH(ctx->next_distributed_query(*con.context, later),
                    Catch::Matchers::ContainsSubstring("retry the statement"));

  ctx->set_exchange_service_for_testing(nullptr);
  services.clear();
}
