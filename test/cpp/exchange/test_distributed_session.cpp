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

#include "exchange/all_gather.hpp"
#include "exchange/distributed_session.hpp"
#include "exchange/exchange_service.hpp"
#include "exchange/exchange_test_utils.hpp"
#include "exchange/loopback_fabric.hpp"
#include "sirius/exception.hpp"

#include <catch.hpp>

#include <chrono>
#include <exception>
#include <memory>
#include <stdexcept>
#include <vector>

using sirius::exchange::distributed_session;

namespace {

using namespace sirius::test::exchange_utils;

/// Two ranks in one process, each with an exchange service over the loopback fabric.
struct two_ranks {
  loopback_hub hub{2};
  std::vector<std::unique_ptr<frame_ring>> rings;
  std::vector<std::unique_ptr<fabric>> fabrics;
  std::vector<std::shared_ptr<sirius::exchange::exchange_service>> services;

  two_ranks()
  {
    for (rank_t r = 0; r < 2; ++r) {
      rings.push_back(std::make_unique<frame_ring>(4, 64 * KiB));
      rings.push_back(std::make_unique<frame_ring>(8, 64 * KiB));
      fabrics.push_back(hub.attach(r, *rings[2 * r], *rings[2 * r + 1]));
      services.push_back(std::make_shared<sirius::exchange::exchange_service>(
        *fabrics.back(), *rings[2 * r], *rings[2 * r + 1], *env().gpu, env().host));
    }
  }
  ~two_ranks() { services.clear(); }
};

/// Records whether its stream was failed.
struct recording : sirius::exchange::inbound {
  bool failed = false;
  void on_batch(rank_t, std::uint64_t, std::shared_ptr<cucascade::data_batch>) override {}
  void on_end(rank_t) override {}
  void on_error(std::exception_ptr) override { failed = true; }
};

}  // namespace

TEST_CASE("a distributed session numbers one connection's queries and hands over on close",
          "[exchange][distributed_session]")
{
  distributed_session session;
  CHECK(session.next_query(7) == 1);
  CHECK(session.next_query(7) == 2);
  CHECK_THROWS_AS(session.next_query(8), sirius::invalid_input_exception);
  // Another connection's close changes nothing.
  session.release_connection(8);
  CHECK_THROWS_AS(session.next_query(8), sirius::invalid_input_exception);
  // The owner's close hands numbering on, where it stopped.
  session.release_connection(7);
  CHECK(session.next_query(8) == 3);
}

TEST_CASE("a distributed session ties windows to queries until they end",
          "[exchange][distributed_session]")
{
  distributed_session session;
  CHECK(session.query_of(5) == 0);
  session.tie(5, 11);
  session.tie(6, 12);
  CHECK(session.query_of(5) == 11);
  CHECK(session.query_of(6) == 12);
  session.forget_window(5);
  CHECK(session.query_of(5) == 0);
  CHECK(session.query_of(6) == 12);
}

TEST_CASE("a distributed session without an exchange aborts and retires nothing",
          "[exchange][distributed_session]")
{
  distributed_session session;
  session.tie(5, 11);
  CHECK(session.service() == nullptr);
  session.abort(5, std::make_exception_ptr(std::runtime_error("failed")));
  session.retire(5);
  session.shutdown();
  CHECK(session.query_of(5) == 11);
}

TEST_CASE("a distributed session connects until a connect yields an exchange",
          "[exchange][distributed_session]")
{
  distributed_session session;
  int connects       = 0;
  auto const connect = [&] {
    ++connects;
    return std::shared_ptr<sirius::exchange::exchange_service>{};
  };
  distributed_session::settings const wanted{.peers = {"a:1", "b:2"}, .rank = 1};
  CHECK(session.service(wanted, connect) == nullptr);
  CHECK(session.service(wanted, connect) == nullptr);
  CHECK(connects == 2);
  // A connect that fails leaves no exchange behind.
  CHECK_THROWS(session.service(wanted, []() -> std::shared_ptr<sirius::exchange::exchange_service> {
    throw std::runtime_error("no peers");
  }));
  CHECK(session.service() == nullptr);
}

TEST_CASE("a window that fails to begin without an exchange is only untied",
          "[exchange][distributed_session]")
{
  distributed_session session;
  session.tie(5, 11);
  session.tie(6, 12);
  session.fail_window_begin(5, std::make_exception_ptr(std::runtime_error("begin failed")));
  CHECK(session.query_of(5) == 0);
  CHECK(session.query_of(6) == 12);
  // A window with no distributed query: nothing to do.
  session.fail_window_begin(7, std::make_exception_ptr(std::runtime_error("begin failed")));
}

TEST_CASE("a window that fails to begin aborts its query on every rank and retires it here",
          "[exchange][distributed_session]")
{
  two_ranks c;
  distributed_session session;
  session.set_service_for_testing(c.services[0]);
  auto const error = std::make_exception_ptr(std::runtime_error("begin failed"));

  session.tie(5, 11);
  session.fail_window_begin(5, error);
  CHECK(session.query_of(5) == 0);
  // Rank 1 waits on rank 0 in that query's all-gather; the abort ends the wait at once.
  auto const started = std::chrono::steady_clock::now();
  CHECK_THROWS_WITH(sirius::exchange::all_gather(*c.services[1],
                                                 sirius::exchange::make_stream_id(11, 1),
                                                 nullptr,
                                                 *env().gpu,
                                                 utils::default_stream(),
                                                 std::chrono::seconds{60}),
                    Catch::Matchers::ContainsSubstring("begin failed"));
  CHECK(std::chrono::steady_clock::now() - started < std::chrono::seconds{30});

  // Retired here: the exchange keeps nothing of the query, so a receiver attached now is not
  // failed. One attached to a query that was only aborted is.
  auto const forgotten = std::make_shared<recording>();
  c.services[0]->attach(sirius::exchange::make_stream_id(11, 2), forgotten);
  CHECK_FALSE(forgotten->failed);
  session.tie(6, 12);
  session.abort(6, error);
  CHECK(session.query_of(6) == 12);
  auto const aborted = std::make_shared<recording>();
  c.services[0]->attach(sirius::exchange::make_stream_id(12, 2), aborted);
  CHECK(aborted->failed);
}
