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

#include "../operator/operator_test_utils.hpp"

#include <catch.hpp>
#include <cucascade/data/data_batch.hpp>
#include <cucascade/data/data_repository.hpp>
#include <data/data_batch_utils.hpp>
#include <exec/stream_session.hpp>
#include <helper/type_conversions.hpp>
#include <sirius/exception.hpp>

#include <memory>
#include <set>
#include <vector>

using namespace sirius::exec;
using namespace sirius::op;
using namespace cucascade;
using namespace cucascade::memory;

namespace {

using namespace sirius::test::operator_utils;

/// A streaming source plus the repository behind it, so a test can inspect where a routed push
/// actually landed.
struct source_fixture {
  std::shared_ptr<sirius_physical_streaming_source> source;
  std::shared_ptr<cucascade::shared_data_repository> repo;
};

source_fixture make_source(std::set<sender_id_t> expected = {0})
{
  auto repo   = std::make_shared<cucascade::shared_data_repository>();
  auto source = std::make_shared<sirius_physical_streaming_source>(
    sirius::from_duckdb_vec(duckdb::vector<duckdb::LogicalType>{duckdb::LogicalType::INTEGER}),
    0,
    repo,
    std::move(expected));
  return {std::move(source), std::move(repo)};
}

struct sink_fixture {
  std::shared_ptr<sirius_physical_streaming_sink> sink;
  std::vector<std::shared_ptr<cucascade::shared_data_repository>> repos;
};

/// Single-destination sink (INTEGER payload, no partitioning).
sink_fixture make_sink()
{
  auto repo = std::make_shared<cucascade::shared_data_repository>();
  auto sink = std::make_shared<sirius_physical_streaming_sink>(
    sirius::from_duckdb_vec(duckdb::vector<duckdb::LogicalType>{duckdb::LogicalType::INTEGER}),
    0,
    repo);
  return {std::move(sink), {std::move(repo)}};
}

/// Partitioned sink over `n` destinations, routing on column 0 of a (BIGINT, INTEGER) schema.
sink_fixture make_partitioned_sink(std::size_t n)
{
  std::vector<std::shared_ptr<cucascade::shared_data_repository>> repos;
  for (std::size_t i = 0; i < n; ++i) {
    repos.push_back(std::make_shared<cucascade::shared_data_repository>());
  }
  duckdb::vector<duckdb::LogicalType> types{duckdb::LogicalType::BIGINT,
                                            duckdb::LogicalType::INTEGER};
  auto sink = std::make_shared<sirius_physical_streaming_sink>(
    sirius::from_duckdb_vec(types), 0, repos, partition_spec{{0}, {}});
  return {std::move(sink), std::move(repos)};
}

/// Feed one batch through a sink exactly as publish_output() would.
void sink_one(sirius_physical_streaming_sink& sink, std::shared_ptr<cucascade::data_batch> batch)
{
  pipelineable_operator_data data{
    std::vector<std::shared_ptr<cucascade::data_batch>>{std::move(batch)}};
  sink.sink(data, default_stream());
}

}  // namespace

// ============================================================================
// SESS-1: registration is reflected in the two id namespaces
// ============================================================================

TEST_CASE("stream_session SESS-1: input and output ids are separate namespaces", "[stream_session]")
{
  auto in  = make_source();
  auto out = make_sink();

  stream_session session;
  // Deliberately the same numeric id on both sides: direction, not the number, disambiguates.
  session.add_source(7, *in.source);
  session.add_sink({7}, *out.sink);

  REQUIRE(session.input_streams() == std::vector<stream_id_t>{7});
  REQUIRE(session.output_streams() == std::vector<stream_id_t>{7});
}

// ============================================================================
// SESS-2: push routes to the addressed source, and only that one
// ============================================================================

TEST_CASE("stream_session SESS-2: push reaches only the addressed source", "[stream_session]")
{
  auto mem_mgr    = sirius::test::operator_utils::initialize_memory_manager();
  auto* gpu_space = mem_mgr->get_memory_space(Tier::GPU, 0);
  REQUIRE(gpu_space != nullptr);

  auto a = make_source();
  auto b = make_source();

  stream_session session;
  session.add_source(10, *a.source);
  session.add_source(20, *b.source);

  REQUIRE(session.push(10, make_numeric_batch<int32_t>(*gpu_space, {1}, cudf::type_id::INT32)));
  REQUIRE(a.repo->total_size() == 1);
  REQUIRE(b.repo->total_size() == 0);

  REQUIRE(session.push(20, make_numeric_batch<int32_t>(*gpu_space, {2}, cudf::type_id::INT32)));
  REQUIRE(a.repo->total_size() == 1);
  REQUIRE(b.repo->total_size() == 1);
}

// ============================================================================
// SESS-3: close_input marks one sender of one stream
// ============================================================================

TEST_CASE("stream_session SESS-3: close_input is scoped to its stream and sender",
          "[stream_session]")
{
  auto a = make_source({0, 1});
  auto b = make_source({0});

  stream_session session;
  session.add_source(10, *a.source);
  session.add_source(20, *b.source);

  session.close_input(10, 0);
  REQUIRE(a.source->stream().sender_closed(0));
  REQUIRE_FALSE(a.source->stream().terminal());  // still waiting on sender 1
  REQUIRE_FALSE(b.source->stream().terminal());  // untouched

  session.close_input(20, 0);
  REQUIRE(b.source->stream().terminal());
  REQUIRE_FALSE(a.source->stream().terminal());
}

// ============================================================================
// SESS-4: pull / wait / drained read only the addressed output stream
// ============================================================================

TEST_CASE("stream_session SESS-4: output calls resolve to one sink partition", "[stream_session]")
{
  auto mem_mgr    = sirius::test::operator_utils::initialize_memory_manager();
  auto* gpu_space = mem_mgr->get_memory_space(Tier::GPU, 0);
  REQUIRE(gpu_space != nullptr);

  auto a = make_sink();
  auto b = make_sink();

  stream_session session;
  session.add_sink({100}, *a.sink);
  session.add_sink({200}, *b.sink);

  auto batch    = make_numeric_batch<int32_t>(*gpu_space, {42}, cudf::type_id::INT32);
  auto batch_id = batch->get_batch_id();
  sink_one(*a.sink, batch);

  REQUIRE_FALSE(session.pull(200).has_value());
  auto pulled = session.pull(100);
  REQUIRE(pulled.has_value());
  REQUIRE((*pulled)->get_batch_id() == batch_id);

  a.sink->finalize_operator();
  REQUIRE(session.drained(100));
  REQUIRE_FALSE(session.drained(200));
}

// ============================================================================
// SESS-5: a partitioned sink registers one id per destination, positionally
// ============================================================================

TEST_CASE("stream_session SESS-5: each sink partition gets its own stream id", "[stream_session]")
{
  auto mem_mgr    = sirius::test::operator_utils::initialize_memory_manager();
  auto* gpu_space = mem_mgr->get_memory_space(Tier::GPU, 0);
  REQUIRE(gpu_space != nullptr);

  constexpr std::size_t N = 3;
  auto out                = make_partitioned_sink(N);

  stream_session session;
  const std::vector<stream_id_t> ids{300, 301, 302};
  session.add_sink(ids, *out.sink);
  REQUIRE(session.output_streams() == ids);

  std::vector<int64_t> keys;
  std::vector<int32_t> vals;
  for (int i = 0; i < 32; ++i) {
    keys.push_back(i);
    vals.push_back(i);
  }
  sink_one(*out.sink,
           make_two_column_batch<int64_t, int32_t>(*gpu_space, keys, vals, cudf::type_id::INT32));
  out.sink->finalize_operator();

  // ids[i] must address repository i and nothing else.
  for (std::size_t i = 0; i < N; ++i) {
    const bool repo_has_data = !out.repos[i]->all_empty();
    REQUIRE(session.drained(ids[i]) == !repo_has_data);
    REQUIRE(session.pull(ids[i]).has_value() == repo_has_data);
  }
}

// ============================================================================
// SESS-6: an unknown id is a defined error on every verb
// ============================================================================

TEST_CASE("stream_session SESS-6: unknown ids are rejected", "[stream_session]")
{
  auto in  = make_source();
  auto out = make_sink();

  stream_session session;
  session.add_source(1, *in.source);
  session.add_sink({2}, *out.sink);

  REQUIRE_THROWS_AS(session.push(99, nullptr), sirius::invalid_input_exception);
  REQUIRE_THROWS_AS(session.close_input(99, 0), sirius::invalid_input_exception);
  REQUIRE_THROWS_AS(session.pull(99), sirius::invalid_input_exception);
  REQUIRE_THROWS_AS(session.drained(99), sirius::invalid_input_exception);

  // An output id is not usable as an input id, and vice versa — the namespaces do not leak.
  REQUIRE_THROWS_AS(session.push(2, nullptr), sirius::invalid_input_exception);
  REQUIRE_THROWS_AS(session.pull(1), sirius::invalid_input_exception);
}

// ============================================================================
// SESS-7: registration contracts
// ============================================================================

TEST_CASE("stream_session SESS-7: registration is validated", "[stream_session]")
{
  auto in    = make_source();
  auto other = make_source();
  auto out   = make_partitioned_sink(2);
  auto spare = make_sink();

  stream_session session;
  session.add_source(1, *in.source);

  // A duplicate id must be rejected rather than silently rebinding the stream to another
  // operator. (Registration takes a reference, so a null operator is not expressible.)
  REQUIRE_THROWS_AS(session.add_source(1, *other.source), sirius::invalid_input_exception);

  // One id per destination — an id list that does not match the sink's fan-out would silently
  // leave a partition unaddressable.
  REQUIRE_THROWS_AS(session.add_sink({10}, *out.sink), sirius::invalid_input_exception);
  REQUIRE_THROWS_AS(session.add_sink({10, 11, 12}, *out.sink), sirius::invalid_input_exception);

  session.add_sink({10, 11}, *out.sink);
  REQUIRE_THROWS_AS(session.add_sink({11}, *spare.sink), sirius::invalid_input_exception);
}

// ============================================================================
// SESS-8: a source → sink fragment driven entirely by stream id
// ============================================================================

TEST_CASE("stream_session SESS-8: a fragment round-trips native batches by id", "[stream_session]")
{
  auto mem_mgr    = sirius::test::operator_utils::initialize_memory_manager();
  auto* gpu_space = mem_mgr->get_memory_space(Tier::GPU, 0);
  REQUIRE(gpu_space != nullptr);

  auto stream = default_stream();
  auto in     = make_source();
  auto out    = make_sink();

  stream_session session;
  session.add_source(1, *in.source);
  session.add_sink({2}, *out.sink);

  constexpr int K = 3;
  std::vector<uint64_t> pushed_ids;
  for (int i = 0; i < K; ++i) {
    auto batch = make_numeric_batch<int32_t>(*gpu_space, {i}, cudf::type_id::INT32);
    pushed_ids.push_back(batch->get_batch_id());
    REQUIRE(session.push(1, batch));
  }
  session.close_input(1, 0);

  // Drive the fragment: source → sink, one task per batch, exactly as the executor would.
  while (auto hint = in.source->get_next_task_hint()) {
    REQUIRE(hint->hint == TaskCreationHint::READY);
    auto input = in.source->get_next_task_input_data();
    REQUIRE(input != nullptr);
    auto produced = in.source->execute(*input, stream);
    out.sink->sink(*produced, stream);
  }
  out.sink->finalize_operator();

  std::vector<uint64_t> pulled_ids;
  while (auto batch = session.pull(2)) {
    pulled_ids.push_back((*batch)->get_batch_id());
  }
  REQUIRE(pulled_ids == pushed_ids);
  REQUIRE(session.drained(2));
}

// ============================================================================
// SESS-9: leaf-fragment shape — a partitioned sink and NO source.
//
// The leaf of a distributed GROUP BY: scan → partial aggregate → partitioned
// STREAMING_SINK. Its EOS is driven by the scan finishing, not by any
// close_input, so a session with zero input streams must be legitimate.
// ============================================================================

TEST_CASE("stream_session SESS-9: a leaf fragment registers only output streams",
          "[stream_session]")
{
  auto mem_mgr    = sirius::test::operator_utils::initialize_memory_manager();
  auto* gpu_space = mem_mgr->get_memory_space(Tier::GPU, 0);
  REQUIRE(gpu_space != nullptr);

  constexpr std::size_t N = 2;
  auto out                = make_partitioned_sink(N);

  stream_session session;
  const std::vector<stream_id_t> ids{500, 501};
  session.add_sink(ids, *out.sink);

  REQUIRE(session.input_streams().empty());
  REQUIRE(session.output_streams() == ids);

  std::vector<int64_t> keys;
  std::vector<int32_t> vals;
  for (int i = 0; i < 16; ++i) {
    keys.push_back(i);
    vals.push_back(i * 2);
  }
  sink_one(*out.sink,
           make_two_column_batch<int64_t, int32_t>(*gpu_space, keys, vals, cudf::type_id::INT32));

  // Still producing: no destination may report EOS yet.
  for (auto id : ids) {
    REQUIRE_FALSE(session.drained(id));
  }

  out.sink->finalize_operator();

  int total_rows = 0;
  for (auto id : ids) {
    while (auto batch = session.pull(id)) {
      total_rows += sirius::get_cudf_table_view(**batch).num_rows();
    }
    REQUIRE(session.drained(id));
  }
  REQUIRE(total_rows == static_cast<int>(keys.size()));
}

// ============================================================================
// SESS-10: root-fragment shape — one fan-in source fed by N remote senders.
//
// The root of a distributed GROUP BY: STREAMING_SOURCE → final merge →
// STREAMING_SINK. EOS must wait for all N *distinct* senders; a duplicate
// close from one sender must not advance it.
// ============================================================================

TEST_CASE("stream_session SESS-10: a root fragment ends only after every sender closes",
          "[stream_session]")
{
  auto mem_mgr    = sirius::test::operator_utils::initialize_memory_manager();
  auto* gpu_space = mem_mgr->get_memory_space(Tier::GPU, 0);
  REQUIRE(gpu_space != nullptr);

  auto stream = default_stream();
  auto in     = make_source({0, 1});
  auto out    = make_sink();

  stream_session session;
  session.add_source(600, *in.source);
  session.add_sink({601}, *out.sink);

  // Sender 0 delivers and closes; sender 1 is still going.
  REQUIRE(session.push(600, make_numeric_batch<int32_t>(*gpu_space, {1}, cudf::type_id::INT32)));
  session.close_input(600, 0);

  // A duplicate close from sender 0 must not stand in for sender 1 — the failure mode a bare
  // counter would have, and one that would truncate the root's input.
  session.close_input(600, 0);
  REQUIRE_FALSE(in.source->all_ports_empty());

  REQUIRE(session.push(600, make_numeric_batch<int32_t>(*gpu_space, {2}, cudf::type_id::INT32)));
  session.close_input(600, 1);

  int received = 0;
  while (auto hint = in.source->get_next_task_hint()) {
    REQUIRE(hint->hint == TaskCreationHint::READY);
    auto input = in.source->get_next_task_input_data();
    REQUIRE(input != nullptr);
    auto produced = in.source->execute(*input, stream);
    out.sink->sink(*produced, stream);
    ++received;
  }
  out.sink->finalize_operator();

  REQUIRE(received == 2);
  REQUIRE(in.source->all_ports_empty());

  int pulled = 0;
  while (session.pull(601)) {
    ++pulled;
  }
  REQUIRE(pulled == 2);
  REQUIRE(session.drained(601));
}

// ============================================================================
// SESS-11: the session is move-only, and moves keep its routing intact
// ============================================================================

TEST_CASE("stream_session SESS-11: a moved session still routes", "[stream_session]")
{
  auto mem_mgr    = sirius::test::operator_utils::initialize_memory_manager();
  auto* gpu_space = mem_mgr->get_memory_space(Tier::GPU, 0);
  REQUIRE(gpu_space != nullptr);

  static_assert(!std::is_copy_constructible_v<stream_session>);
  static_assert(std::is_move_constructible_v<stream_session>);

  auto in = make_source();

  stream_session original;
  original.add_source(1, *in.source);

  stream_session moved{std::move(original)};
  REQUIRE(moved.input_streams() == std::vector<stream_id_t>{1});
  REQUIRE(moved.push(1, make_numeric_batch<int32_t>(*gpu_space, {1}, cudf::type_id::INT32)));
  REQUIRE(in.repo->total_size() == 1);
}
