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

// #819 PR2-lite: hardening gates for the coalescer-direct cached serving path.
//
// Two failure modes on this path used to be silent:
//   - a throwing databatch_provider escaped into the dispatcher (which
//     swallows task exceptions), leaving the operator's split_connector never
//     closed — the consumer blocked in get_next_split() forever (silent query
//     hang);
//   - a malformed pinned entry (per-column chunk counts disagreeing, short
//     chunk_memory_spaces, null chunks) made the cached provider return
//     nullptr mid-stream, which the drain loop reads as end-of-stream — the
//     query completed on FEWER rows than requested (silent truncation).
//
// Gates: load_balancing_scan_batch_coalescer::drain_cached_provider
// (forward-then-close; provider throw -> close(exception) -> consumer
// rethrows; pre-stopped token -> close without draining) and
// validate_pinned_entry_for_serving (malformed entries throw so
// try_assign_cached_entries falls back to the disk read; well-formed and
// zero-chunk entries pass).

#include "operator/operator_test_utils.hpp"

#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>

#include <rmm/cuda_stream.hpp>

#include <cuda_runtime.h>

#include <catch.hpp>
#include <cucascade/cudf/gpu_data_representation.hpp>
#include <cucascade/cudf/host_data_representation.hpp>
#include <cucascade/data/data_batch.hpp>
#include <cucascade/memory/memory_space.hpp>
#include <data/data_batch_utils.hpp>
#include <data/sirius_converter_registry.hpp>
#include <op/scan/sirius_gpu_scan_operator_data.hpp>
#include <scan_manager/load_balancing_scan_batch_coalescer.hpp>
#include <scan_manager/sirius_scan_manager.hpp>
#include <scan_manager/split_connector.hpp>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <stop_token>
#include <string>
#include <vector>

using sirius::scan_manager::databatch_provider;
using sirius::scan_manager::load_balancing_scan_batch_coalescer;
using sirius::scan_manager::pinned_entry;
using sirius::scan_manager::split_connector;
using sirius::scan_manager::validate_pinned_entry_for_serving;

namespace {

// Shared test environment: memory manager (+ converter registry) initialized
// once for every gate in this file. The converter needs a non-default stream
// (same constraint test_convertible_data_batch documents).
struct test_env {
  std::unique_ptr<sirius::memory::sirius_memory_reservation_manager> mgr;
  cucascade::memory::memory_space* gpu_space;
  cucascade::memory::memory_space* host_space;
  rmm::cuda_stream conv_stream;

  test_env()
    : mgr(sirius::test::operator_utils::initialize_memory_manager()),
      gpu_space(mgr->get_memory_space(cucascade::memory::Tier::GPU, 0)),
      host_space(mgr->get_memory_space(cucascade::memory::Tier::HOST, 0)),
      conv_stream()
  {
  }

  rmm::cuda_stream_view stream() { return conv_stream.view(); }
};

test_env& env()
{
  static test_env e;
  return e;
}

/// Deterministic per-cell value so a chunk/column/row mixup fails loudly.
int32_t cell(std::size_t chunk, std::size_t col, std::size_t row)
{
  return static_cast<int32_t>(1000 * chunk + 100 * col + row);
}

std::shared_ptr<cudf::column> make_gpu_column(cucascade::memory::memory_space& space,
                                              std::vector<int32_t> const& values)
{
  auto mr     = sirius::test::operator_utils::get_resource_ref(space);
  auto stream = sirius::test::operator_utils::default_stream();
  auto col    = cudf::make_numeric_column(cudf::data_type{cudf::type_id::INT32},
                                       static_cast<cudf::size_type>(values.size()),
                                       cudf::mask_state::UNALLOCATED,
                                       stream,
                                       mr);
  cudaMemcpy(col->mutable_view().data<int32_t>(),
             values.data(),
             sizeof(int32_t) * values.size(),
             cudaMemcpyHostToDevice);
  return std::shared_ptr<cudf::column>(std::move(col));
}

/// GPU-tier pinned entry with columns {k, v, w} x @p n_chunks chunks of
/// @p rows rows, every chunk placed in @p space.
pinned_entry make_gpu_entry(cucascade::memory::memory_space& space,
                            std::size_t n_chunks,
                            std::size_t rows)
{
  pinned_entry entry;
  entry.cache_info.names = {"k", "v", "w"};
  entry.tier             = cucascade::memory::Tier::GPU;
  entry.memory_space     = &space;
  for (std::size_t c = 0; c < n_chunks; ++c) {
    entry.chunk_memory_spaces.push_back(&space);
    for (std::size_t col = 0; col < entry.cache_info.names.size(); ++col) {
      std::vector<int32_t> values(rows);
      for (std::size_t r = 0; r < rows; ++r) {
        values[r] = cell(c, col, r);
      }
      entry.data_batches_by_column[entry.cache_info.names[col]].push_back(
        make_gpu_column(space, values));
    }
  }
  entry.num_rows = n_chunks * rows;
  return entry;
}

/// HOST-tier pinned entry with columns {k, v} x @p n_chunks chunks, built the
/// way the pin path builds them (GPU table -> converter -> host chunk).
pinned_entry make_host_entry(test_env& e, std::size_t n_chunks, std::size_t rows)
{
  pinned_entry entry;
  entry.cache_info.names = {"k", "v"};
  entry.tier             = cucascade::memory::Tier::HOST;
  entry.memory_space     = e.host_space;

  auto& registry = sirius::converter_registry::get();
  for (std::size_t c = 0; c < n_chunks; ++c) {
    std::vector<std::unique_ptr<cudf::column>> cols;
    for (std::size_t col = 0; col < entry.cache_info.names.size(); ++col) {
      std::vector<int32_t> values(rows);
      for (std::size_t r = 0; r < rows; ++r) {
        values[r] = cell(c, col, r);
      }
      auto shared = make_gpu_column(*e.gpu_space, values);
      cols.push_back(std::make_unique<cudf::column>(
        shared->view(), e.stream(), e.gpu_space->get_default_allocator()));
    }
    cucascade::gpu_table_representation gpu_repr(
      std::make_unique<cudf::table>(std::move(cols)), *e.gpu_space, e.stream());
    auto host_repr =
      registry.convert<cucascade::host_data_representation>(gpu_repr, e.host_space, e.stream());
    e.stream().synchronize();
    entry.host_chunks.emplace_back(std::move(host_repr));
  }
  entry.num_rows = n_chunks * rows;
  return entry;
}

/// One-column resident GPU batch — payload for the fake providers below.
std::shared_ptr<cucascade::data_batch> make_test_batch(test_env& e, std::size_t rows)
{
  auto col = make_gpu_column(*e.gpu_space, std::vector<int32_t>(rows, 7));
  std::vector<std::shared_ptr<cudf::column>> columns{col};
  std::vector<cudf::column_view> views{col->view()};
  auto const alloc_size = col->alloc_size();
  auto repr             = std::make_unique<cucascade::gpu_table_representation>(
    cudf::table_view(views), std::move(columns), alloc_size, *e.gpu_space, rmm::cuda_stream_view{});
  return cucascade::data_batch::make(sirius::get_next_batch_id(), std::move(repr));
}

/// Serves its scripted batches in order, then either ends the stream
/// (nullptr) or throws — the two provider behaviors the drain must handle.
struct scripted_provider final : databatch_provider {
  std::vector<std::shared_ptr<cucascade::data_batch>> batches;
  std::size_t served{0};
  bool throw_when_exhausted{false};

  std::shared_ptr<cucascade::data_batch> get_next_batch() override
  {
    if (served < batches.size()) { return batches[served++]; }
    if (throw_when_exhausted) { throw std::runtime_error("provider blew up mid-stream"); }
    return nullptr;
  }
};

}  // namespace

//===----------------------------------------------------------------------===//
// drain_cached_provider gates
//===----------------------------------------------------------------------===//

TEST_CASE("drain_cached_provider forwards every batch then closes",
          "[cached_serving][scan_manager]")
{
  auto& e = env();
  scripted_provider provider;
  provider.batches = {make_test_batch(e, 4), make_test_batch(e, 4), make_test_batch(e, 4)};

  split_connector connector;
  std::stop_source stop;
  load_balancing_scan_batch_coalescer::drain_cached_provider(provider, connector, stop.get_token());

  for (int i = 0; i < 3; ++i) {
    auto split = connector.get_next_split();
    REQUIRE(split.has_value());
    auto* input = dynamic_cast<sirius::op::scan::scan_operator_input*>(split->get());
    REQUIRE(input != nullptr);
    REQUIRE(input->is_resident());
  }
  REQUIRE_FALSE(connector.get_next_split().has_value());  // closed and drained
}

TEST_CASE("drain_cached_provider surfaces a provider exception instead of hanging",
          "[cached_serving][scan_manager]")
{
  auto& e = env();
  scripted_provider provider;
  provider.batches              = {make_test_batch(e, 4)};
  provider.throw_when_exhausted = true;

  split_connector connector;
  std::stop_source stop;
  // Must not propagate: the dispatcher would swallow it and leave the
  // connector open forever (the old silent-hang bug).
  REQUIRE_NOTHROW(load_balancing_scan_batch_coalescer::drain_cached_provider(
    provider, connector, stop.get_token()));

  // The stored error takes precedence over queued splits: every consumer
  // pull now rethrows the producer failure (instead of a partial stream
  // followed by an eternal block — the old silent-hang bug).
  REQUIRE_THROWS_AS(connector.get_next_split(), std::runtime_error);
  REQUIRE_THROWS_AS(connector.get_next_split(), std::runtime_error);
}

TEST_CASE("drain_cached_provider honors a pre-stopped token", "[cached_serving][scan_manager]")
{
  auto& e = env();
  scripted_provider provider;
  provider.batches = {make_test_batch(e, 4), make_test_batch(e, 4)};

  split_connector connector;
  std::stop_source stop;
  stop.request_stop();
  load_balancing_scan_batch_coalescer::drain_cached_provider(provider, connector, stop.get_token());

  REQUIRE(provider.served == 0);                          // nothing pulled after stop
  REQUIRE_FALSE(connector.get_next_split().has_value());  // still closed: consumer unblocked
}

//===----------------------------------------------------------------------===//
// validate_pinned_entry_for_serving gates
//===----------------------------------------------------------------------===//

TEST_CASE("validate_pinned_entry_for_serving accepts well-formed and zero-chunk entries",
          "[cached_serving][scan_manager]")
{
  auto& e = env();

  SECTION("well-formed GPU entry")
  {
    auto entry = make_gpu_entry(*e.gpu_space, 3, 4);
    REQUIRE_NOTHROW(validate_pinned_entry_for_serving(entry, std::vector<std::size_t>{0, 1, 2}));
  }

  SECTION("zero-chunk GPU entry")
  {
    pinned_entry entry;
    entry.cache_info.names = {"k"};
    entry.tier             = cucascade::memory::Tier::GPU;
    entry.memory_space     = e.gpu_space;
    REQUIRE_NOTHROW(validate_pinned_entry_for_serving(entry, std::vector<std::size_t>{0}));
  }

  SECTION("well-formed HOST entry")
  {
    auto entry = make_host_entry(e, 2, 4);
    REQUIRE_NOTHROW(validate_pinned_entry_for_serving(entry, std::vector<std::size_t>{0, 1}));
  }
}

TEST_CASE("validate_pinned_entry_for_serving refuses malformed entries",
          "[cached_serving][scan_manager]")
{
  auto& e = env();

  SECTION("per-column chunk counts disagree")
  {
    auto entry = make_gpu_entry(*e.gpu_space, 3, 4);
    entry.data_batches_by_column["v"].pop_back();  // v now has 2 chunks, k/w have 3
    REQUIRE_THROWS_AS(validate_pinned_entry_for_serving(entry, std::vector<std::size_t>{0, 1, 2}),
                      std::runtime_error);
  }

  SECTION("selected column missing from the entry's storage")
  {
    auto entry = make_gpu_entry(*e.gpu_space, 2, 4);
    entry.data_batches_by_column.erase("w");
    REQUIRE_THROWS_AS(validate_pinned_entry_for_serving(entry, std::vector<std::size_t>{2}),
                      std::runtime_error);
  }

  SECTION("chunk_memory_spaces does not cover every chunk")
  {
    auto entry = make_gpu_entry(*e.gpu_space, 3, 4);
    entry.chunk_memory_spaces.resize(1);
    REQUIRE_THROWS_AS(validate_pinned_entry_for_serving(entry, std::vector<std::size_t>{0}),
                      std::runtime_error);
  }

  SECTION("null chunk memory_space")
  {
    auto entry                   = make_gpu_entry(*e.gpu_space, 2, 4);
    entry.chunk_memory_spaces[1] = nullptr;
    REQUIRE_THROWS_AS(validate_pinned_entry_for_serving(entry, std::vector<std::size_t>{0}),
                      std::runtime_error);
  }

  SECTION("null GPU chunk")
  {
    auto entry                           = make_gpu_entry(*e.gpu_space, 2, 4);
    entry.data_batches_by_column["k"][1] = nullptr;
    REQUIRE_THROWS_AS(validate_pinned_entry_for_serving(entry, std::vector<std::size_t>{0}),
                      std::runtime_error);
  }

  SECTION("null host chunk")
  {
    auto entry           = make_host_entry(e, 2, 4);
    entry.host_chunks[1] = nullptr;
    REQUIRE_THROWS_AS(validate_pinned_entry_for_serving(entry, std::vector<std::size_t>{0}),
                      std::runtime_error);
  }
}
