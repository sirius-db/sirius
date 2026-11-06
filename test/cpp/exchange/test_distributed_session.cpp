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

#include "exchange/distributed_session.hpp"
#include "sirius/exception.hpp"

#include <catch.hpp>

#include <stdexcept>

using sirius::exchange::distributed_session;

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
