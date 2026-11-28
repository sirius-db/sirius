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
#include "op/scan/paimon/paimon_batch_cursor.hpp"
#include "scan_manager/preparation_ledger.hpp"

#include <duckdb/common/arrow/arrow.hpp>
#include <paimon/reader/batch_reader.h>

#include <functional>
#include <stop_token>
#include <string_view>
#include <system_error>
using namespace sirius::op::scan;
namespace {
struct observation {
  int reads = 0, closes = 0, releases = 0, active_roots = 0, destroyed = 0;
  paimon::Status read_status;
  bool read_throw = false, close_error = false, eof_schema = false;
  std::weak_ptr<void> owners;
  bool check_owners = false;
  std::vector<int64_t> rows{0, 2};
  std::function<void()> on_read;
};
paimon::BatchReader::ReadBatch roots(int64_t rows, observation& state)
{
  auto array          = std::make_unique<ArrowArray>();
  auto schema         = std::make_unique<ArrowSchema>();
  array->length       = rows;
  array->private_data = schema->private_data = &state;
  array->release                             = [](ArrowArray* root) {
    auto& s = *static_cast<observation*>(root->private_data);
    ++s.releases;
    --s.active_roots;
    root->release = nullptr;
  };
  schema->format  = "+s";
  schema->release = [](ArrowSchema* root) {
    auto& s = *static_cast<observation*>(root->private_data);
    ++s.releases;
    --s.active_roots;
    root->release = nullptr;
  };
  state.active_roots += 2;
  return {std::move(array), std::move(schema)};
}
class reader final : public paimon::BatchReader {
 public:
  explicit reader(std::shared_ptr<observation> value) : state(std::move(value)) {}
  ~reader() override
  {
    if (state->check_owners) REQUIRE_FALSE(state->owners.expired());
    ++state->destroyed;
  }
  paimon::Result<ReadBatch> NextBatch() override
  {
    REQUIRE(state->active_roots == 0);
    ++state->reads;
    if (state->on_read) state->on_read();
    if (state->read_throw) throw std::runtime_error("selected thrown read error");
    if (!state->read_status.ok()) return state->read_status;
    if (cursor < state->rows.size()) return roots(state->rows[cursor++], *state);
    if (state->eof_schema) {
      auto eof = roots(0, *state);
      eof.first->release(eof.first.get());
      eof.first.reset();
      return eof;
    }
    return MakeEofBatch();
  }
  void Close() override
  {
    if (state->check_owners) REQUIRE_FALSE(state->owners.expired());
    ++state->closes;
    REQUIRE(state->active_roots == 0);
    if (state->close_error) throw std::runtime_error("selected close error");
  }
  std::shared_ptr<paimon::Metrics> GetReaderMetrics() const override { return {}; }
  std::shared_ptr<observation> state;
  size_t cursor = 0;
};
paimon_batch_cursor make_cursor(std::shared_ptr<observation> const& state)
{
  return paimon_batch_cursor(std::make_unique<reader>(state), state);
}
}  // namespace
TEST_CASE("Paimon cursor preserves zero rows and retires EOF once", "[paimon_cursor]")
{
  auto state  = std::make_shared<observation>();
  auto cursor = make_cursor(state);
  std::vector<int64_t> rows;
  auto accept = [&](ArrowSchema const&, ArrowArray const& array) { rows.push_back(array.length); };
  REQUIRE(cursor.consume({}, 2, accept));
  REQUIRE(state->releases == 2);
  REQUIRE(cursor.consume({}, 2, accept));
  REQUIRE(state->releases == 4);
  REQUIRE_FALSE(cursor.consume({}, 2, accept));
  REQUIRE(rows == std::vector<int64_t>{0, 2});
  REQUIRE(state->closes == 1);
  REQUIRE_FALSE(cursor.consume({}, 2, accept));
  cursor.close();
  REQUIRE(state->reads == 3);
  REQUIRE(state->closes == 1);
}
TEST_CASE("Paimon cursor releases roots on normalization failure and never retries",
          "[paimon_cursor]")
{
  auto state  = std::make_shared<observation>();
  auto cursor = make_cursor(state);
  auto fail   = [](ArrowSchema const&, ArrowArray const&) {
    throw std::runtime_error("copy failure");
  };
  REQUIRE_THROWS_WITH(cursor.consume({}, 2, fail), "copy failure");
  REQUIRE(state->active_roots == 0);
  REQUIRE(state->releases == 2);
  REQUIRE_THROWS_WITH(cursor.consume({}, 2, fail), "copy failure");
  REQUIRE(state->reads == 1);
  cursor.close();
  REQUIRE(state->closes == 1);
}
TEST_CASE("Paimon cursor latches native errors without reporting EOF", "[paimon_cursor]")
{
  auto state = std::make_shared<observation>();
  SECTION("returned status")
  {
    state->read_status = paimon::Status::IOError("selected read error");
  }
  SECTION("thrown exception") { state->read_throw = true; }
  auto cursor = make_cursor(state);
  auto accept = [](ArrowSchema const&, ArrowArray const&) {};
  REQUIRE_THROWS(cursor.consume({}, 2, accept));
  REQUIRE_THROWS(cursor.consume({}, 2, accept));
  REQUIRE(state->reads == 1);
  cursor.close();
}
TEST_CASE("Paimon cursor preserves SDK resource and cancellation status", "[paimon_cursor]")
{
  auto state  = std::make_shared<observation>();
  auto cursor = make_cursor(state);
  auto accept = [](ArrowSchema const&, ArrowArray const&) { FAIL("failed batch was delivered"); };
  auto const code    = GENERATE(paimon::StatusCode::CapacityError,
                             paimon::StatusCode::OutOfMemory,
                             paimon::StatusCode::Cancelled);
  state->read_status = paimon::Status(code, "selected status");
  for (int attempt = 0; attempt != 2; ++attempt) {
    try {
      (void)cursor.consume({}, 2, accept);
      FAIL("failed native batch became data or EOF");
    } catch (sirius::scan_manager::preparation_resource_error const& failure) {
      CHECK(code != paimon::StatusCode::Cancelled);
      CHECK(failure.bound_exceeded == (code == paimon::StatusCode::CapacityError));
      CHECK(std::string_view(failure.what()).find("selected status") != std::string_view::npos);
    } catch (std::system_error const& failure) {
      CHECK(code == paimon::StatusCode::Cancelled);
      CHECK(failure.code() == std::errc::operation_canceled);
    }
  }
  CHECK(state->reads == 1);
  CHECK(state->active_roots == 0);
  cursor.close();
  CHECK(state->closes == 1);
}
TEST_CASE("Paimon cursor stop before and after native call prevents output", "[paimon_cursor]")
{
  auto state  = std::make_shared<observation>();
  auto cursor = make_cursor(state);
  std::stop_source stop;
  bool accepted = false;
  auto accept   = [&](ArrowSchema const&, ArrowArray const&) { accepted = true; };
  SECTION("before") { stop.request_stop(); }
  SECTION("during")
  {
    state->on_read = [&] { stop.request_stop(); };
  }
  try {
    cursor.consume(sirius::scan_manager::preparation_stop_token(stop.get_token()), 2, accept);
    FAIL("stopped read must fail");
  } catch (std::system_error const& error) {
    REQUIRE(error.code() == std::errc::operation_canceled);
  }
  REQUIRE_FALSE(accepted);
  REQUIRE(state->active_roots == 0);
  REQUIRE_THROWS(cursor.consume({}, 2, accept));
  cursor.close();
}
TEST_CASE("Paimon cursor rejects oversized output and reports close failure", "[paimon_cursor]")
{
  auto state  = std::make_shared<observation>();
  auto accept = [](ArrowSchema const&, ArrowArray const&) {};
  SECTION("row limit")
  {
    state->rows = {3};
    auto cursor = make_cursor(state);
    REQUIRE_THROWS_AS(cursor.consume({}, 2, accept),
                      sirius::scan_manager::preparation_resource_error);
    REQUIRE(state->active_roots == 0);
    cursor.close();
  }
  SECTION("negative row count")
  {
    state->rows = {-1};
    auto cursor = make_cursor(state);
    REQUIRE_THROWS(cursor.consume({}, 2, accept));
    REQUIRE(state->active_roots == 0);
    cursor.close();
  }
  SECTION("explicit close error")
  {
    state->close_error = true;
    auto cursor        = make_cursor(state);
    REQUIRE_THROWS_WITH(cursor.close(), "selected close error");
    REQUIRE_THROWS_WITH(cursor.consume({}, 2, accept), "selected close error");
    cursor.close();
    REQUIRE(state->closes == 1);
    REQUIRE(state->destroyed == 1);
  }
  SECTION("close error")
  {
    state->rows.clear();
    state->close_error = true;
    auto cursor        = make_cursor(state);
    REQUIRE_THROWS_WITH(cursor.consume({}, 2, accept), "selected close error");
    REQUIRE_THROWS_WITH(cursor.consume({}, 2, accept), "selected close error");
    cursor.close();
    REQUIRE(state->closes == 1);
    REQUIRE(state->destroyed == 1);
  }
  SECTION("EOF schema cleanup")
  {
    state->rows.clear();
    state->eof_schema = true;
    auto cursor       = make_cursor(state);
    REQUIRE_FALSE(cursor.consume({}, 2, accept));
    REQUIRE(state->active_roots == 0);
    REQUIRE(state->releases == 2);
  }
}

TEST_CASE("Paimon cursor cleans a reader rejected during construction", "[paimon_cursor]")
{
  auto state = std::make_shared<observation>();
  REQUIRE_THROWS_AS(paimon_batch_cursor(std::make_unique<reader>(state), {}),
                    std::invalid_argument);
  REQUIRE(state->closes == 1);
  REQUIRE(state->destroyed == 1);
}
TEST_CASE("Paimon cursor early close forbids further native reads", "[paimon_cursor]")
{
  auto state  = std::make_shared<observation>();
  auto cursor = make_cursor(state);
  cursor.close();
  REQUIRE_THROWS_AS(cursor.consume({}, 2, [](ArrowSchema const&, ArrowArray const&) {}),
                    std::logic_error);
  REQUIRE(state->reads == 0);
  REQUIRE(state->closes == 1);
}

TEST_CASE("Paimon cursor holds native owners through reader teardown", "[paimon_cursor]")
{
  auto state          = std::make_shared<observation>();
  auto owners         = std::make_shared<int>(0);
  state->owners       = owners;
  state->check_owners = true;
  {
    paimon_batch_cursor cursor(std::make_unique<reader>(state), owners);
    owners.reset();
    cursor.close();
    REQUIRE(state->destroyed == 1);
    REQUIRE_FALSE(state->owners.expired());
  }
  REQUIRE(state->owners.expired());
}
TEST_CASE("Paimon cursor stop during detach releases roots and latches failure", "[paimon_cursor]")
{
  auto state  = std::make_shared<observation>();
  auto cursor = make_cursor(state);
  std::stop_source stop;
  auto detach = [&](ArrowSchema const&, ArrowArray const&) { stop.request_stop(); };
  REQUIRE_THROWS_AS(
    cursor.consume(sirius::scan_manager::preparation_stop_token(stop.get_token()), 2, detach),
    std::system_error);
  REQUIRE(state->active_roots == 0);
  REQUIRE_THROWS_AS(cursor.consume({}, 2, detach), std::system_error);
  REQUIRE(state->reads == 1);
  cursor.close();
}
