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
// try_match_cached_entry falls back to the disk read; well-formed and
// zero-chunk entries pass). The file also gates the pinned-entry narrowing-metadata helpers — the
// pinned_column_narrowed_in_all_chunks marker fold and the pinned_column_narrow_carrier carrier
// derivation that composes it for the plan-time residency gate.

#include "operator/operator_test_utils.hpp"

#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/traits.hpp>

#include <rmm/cuda_stream.hpp>

#include <cuda_runtime.h>

#include <catch.hpp>
#include <compression/compressed_representation.hpp>
#include <cucascade/cudf/gpu_data_representation.hpp>
#include <cucascade/cudf/host_data_representation.hpp>
#include <cucascade/data/data_batch.hpp>
#include <cucascade/memory/memory_space.hpp>
#include <data/data_batch_utils.hpp>
#include <data/sirius_converter_registry.hpp>
#include <op/scan/sirius_gpu_scan_operator_data.hpp>
#include <scan_manager/load_balancing_scan_batch_coalescer.hpp>
#include <scan_manager/mvcc_chunk_mask.hpp>
#include <scan_manager/sirius_scan_manager.hpp>
#include <scan_manager/split_connector.hpp>
#include <telemetry/data_batch_probe.hpp>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <stop_token>
#include <string>
#include <vector>

using sirius::scan_manager::build_cached_scan_plan;
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

/// Zero-initialized numeric GPU column of an arbitrary carrier type — the serve-site
/// restore-destination computation reads only types, row counts, and mask presence.
std::shared_ptr<cudf::column> make_typed_gpu_column(cucascade::memory::memory_space& space,
                                                    cudf::data_type type,
                                                    std::size_t rows,
                                                    cudf::mask_state mask_state)
{
  auto mr     = sirius::test::operator_utils::get_resource_ref(space);
  auto stream = sirius::test::operator_utils::default_stream();
  auto col =
    cudf::make_numeric_column(type, static_cast<cudf::size_type>(rows), mask_state, stream, mr);
  return std::shared_ptr<cudf::column>(std::move(col));
}

/// Zone-map sidecar carrying pin-time types only (every stats cell null) — the shape the
/// restore-destination computation needs from the entry.
sirius::scan_manager::pinned_zone_maps make_type_only_zone_maps(
  duckdb::vector<duckdb::LogicalType> types, std::size_t n_chunks)
{
  auto const n_columns = types.size();
  std::vector<std::vector<duckdb::unique_ptr<duckdb::BaseStatistics>>> stats(n_chunks);
  for (auto& chunk : stats) {
    chunk.resize(n_columns);
  }
  return sirius::scan_manager::pinned_zone_maps::from_capture(
    std::move(types), std::move(stats), n_columns, n_chunks);
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

/// GPU-tier compression-enabled pinned entry with columns {k, v, w} x
/// @p n_chunks chunks, stored as UNCOMPRESSED device_pin_chunks (per-column
/// device columns) — the interleave-capable serving path a mixed pin uses.
pinned_entry make_device_chunks_entry(cucascade::memory::memory_space& space,
                                      std::size_t n_chunks,
                                      std::size_t rows)
{
  pinned_entry entry;
  entry.cache_info.names = {"k", "v", "w"};
  entry.tier             = cucascade::memory::Tier::GPU;
  entry.memory_space     = &space;
  for (std::size_t c = 0; c < n_chunks; ++c) {
    sirius::device_pin_chunk chunk;
    chunk.memory_space = &space;
    for (std::size_t col = 0; col < entry.cache_info.names.size(); ++col) {
      std::vector<int32_t> values(rows);
      for (std::size_t r = 0; r < rows; ++r) {
        values[r] = cell(c, col, r);
      }
      chunk.columns.push_back(make_gpu_column(space, values));
    }
    entry.device_chunks.push_back(std::move(chunk));
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

/// GPU column of an arbitrary fixed-width carrier (numeric or fixed-point),
/// zero rows of data read by any gate here — only type metadata matters.
std::unique_ptr<cudf::column> make_carrier_gpu_column(cucascade::memory::memory_space& space,
                                                      cudf::data_type type,
                                                      std::size_t rows)
{
  auto mr     = sirius::test::operator_utils::get_resource_ref(space);
  auto stream = sirius::test::operator_utils::default_stream();
  if (cudf::is_fixed_point(type)) {
    return cudf::make_fixed_point_column(
      type, static_cast<cudf::size_type>(rows), cudf::mask_state::UNALLOCATED, stream, mr);
  }
  return cudf::make_numeric_column(
    type, static_cast<cudf::size_type>(rows), cudf::mask_state::UNALLOCATED, stream, mr);
}

/// HOST-tier pinned entry whose chunk c stores the carriers in
/// @p chunk_carriers[c] (every chunk must list the same column count), built
/// the way the pin path builds them (GPU table -> converter -> host chunk).
/// Columns are named "c0", "c1", ...
pinned_entry make_host_entry_with_carriers(
  test_env& e, std::vector<std::vector<cudf::data_type>> const& chunk_carriers, std::size_t rows)
{
  pinned_entry entry;
  entry.tier         = cucascade::memory::Tier::HOST;
  entry.memory_space = e.host_space;
  auto const n_columns{chunk_carriers.empty() ? std::size_t{0} : chunk_carriers.front().size()};
  for (std::size_t col = 0; col < n_columns; ++col) {
    entry.cache_info.names.push_back("c" + std::to_string(col));
  }

  auto& registry = sirius::converter_registry::get();
  for (auto const& carriers : chunk_carriers) {
    std::vector<std::unique_ptr<cudf::column>> cols;
    cols.reserve(carriers.size());
    for (auto const type : carriers) {
      cols.push_back(make_carrier_gpu_column(*e.gpu_space, type, rows));
    }
    cucascade::gpu_table_representation gpu_repr(
      std::make_unique<cudf::table>(std::move(cols)), *e.gpu_space, e.stream());
    auto host_repr =
      registry.convert<cucascade::host_data_representation>(gpu_repr, e.host_space, e.stream());
    e.stream().synchronize();
    entry.host_chunks.emplace_back(std::move(host_repr));
  }
  entry.num_rows = chunk_carriers.size() * rows;
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

/// All-ones keep-mask over @p rows rows, its words aliasing a plain vector —
/// the unit-test stand-in for the mask job's pinned {reservation, blocks}
/// bundle.
sirius::scan_manager::mvcc_chunk_mask make_test_mask(std::size_t rows)
{
  auto storage = std::make_shared<std::vector<std::uint32_t>>((rows + 31) / 32, 0xFFFFFFFFu);
  return {std::shared_ptr<std::uint32_t[]>(storage, storage->data()), rows};
}

/// Serves its scripted batches (each optionally paired with a keep-mask) in
/// order, then either ends the stream or throws — the provider behaviors the
/// drain must handle.
struct scripted_provider final : databatch_provider {
  std::vector<std::shared_ptr<cucascade::data_batch>> batches;
  std::vector<sirius::scan_manager::mvcc_chunk_mask> masks;
  std::vector<bool> narrowed;
  std::size_t served{0};
  bool throw_when_exhausted{false};

  databatch_provider::batch get_next_batch() override
  {
    if (served < batches.size()) {
      auto idx = served++;
      return {batches[idx],
              idx < masks.size() ? masks[idx] : sirius::scan_manager::mvcc_chunk_mask{},
              idx < narrowed.size() && narrowed[idx]};
    }
    if (throw_when_exhausted) { throw std::runtime_error("provider blew up mid-stream"); }
    return {};
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
  load_balancing_scan_batch_coalescer::drain_cached_provider(
    provider, connector, stop.get_token(), /*row_filter_pending=*/false);

  for (int i = 0; i < 3; ++i) {
    auto split = connector.get_next_split();
    REQUIRE(split.has_value());
    auto* input = dynamic_cast<sirius::op::scan::scan_operator_input*>(split->get());
    REQUIRE(input != nullptr);
    REQUIRE(input->is_resident());
    REQUIRE_FALSE(input->row_filter_pending);  // filter-less op: nothing stamped
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
    provider, connector, stop.get_token(), /*row_filter_pending=*/false));

  // The stored error takes precedence over queued splits: every consumer
  // pull now rethrows the producer failure (instead of a partial stream
  // followed by an eternal block — the old silent-hang bug).
  REQUIRE_THROWS_AS(connector.get_next_split(), std::runtime_error);
  REQUIRE_THROWS_AS(connector.get_next_split(), std::runtime_error);
}

TEST_CASE("drain_cached_provider forwards the mvcc keep-mask and filter flag onto each split",
          "[cached_serving][scan_manager]")
{
  auto& e    = env();
  auto mask0 = make_test_mask(4);
  scripted_provider provider;
  provider.batches  = {make_test_batch(e, 4), make_test_batch(e, 4)};
  provider.masks    = {mask0};  // second batch deliberately mask-less
  provider.narrowed = {true, false};

  split_connector connector;
  std::stop_source stop;
  load_balancing_scan_batch_coalescer::drain_cached_provider(
    provider, connector, stop.get_token(), /*row_filter_pending=*/true);

  auto first = connector.get_next_split();
  REQUIRE(first.has_value());
  auto* in0 = dynamic_cast<sirius::op::scan::scan_operator_input*>(first->get());
  REQUIRE(in0 != nullptr);
  // Same word storage, same extent: the words were forwarded, not rebuilt.
  REQUIRE(in0->mvcc_keep_mask.words == mask0.words);
  REQUIRE(in0->mvcc_keep_mask.row_count == mask0.row_count);
  REQUIRE(in0->row_filter_pending);
  REQUIRE(in0->contains_narrowed_columns);

  auto second = connector.get_next_split();
  REQUIRE(second.has_value());
  auto* in1 = dynamic_cast<sirius::op::scan::scan_operator_input*>(second->get());
  REQUIRE(in1 != nullptr);
  REQUIRE_FALSE(in1->mvcc_keep_mask.has_mask());
  REQUIRE(in1->row_filter_pending);  // per-op flag: stamped on every split
  REQUIRE_FALSE(in1->contains_narrowed_columns);

  REQUIRE_FALSE(connector.get_next_split().has_value());
}

TEST_CASE("cached provider pairs chunk i with mask-set slot i", "[cached_serving][scan_manager]")
{
  auto& e    = env();
  auto entry = make_gpu_entry(*e.gpu_space, 3, 4);
  std::vector<std::size_t> cols{0, 1, 2};

  SECTION("with a mask set")
  {
    auto m0 = make_test_mask(4);
    auto m2 = make_test_mask(4);
    sirius::scan_manager::mvcc_chunk_mask_set set;
    set.push_back(m0);
    set.push_back({});  // all-visible chunk: served unmasked
    set.push_back(m2);

    // The provider takes its own copy of the set (the post-mask-run handoff
    // shape); the words themselves are shared, never duplicated. Identity
    // plan: all three chunks survive.
    sirius::scan_manager::cached_scan_plan plan{.survivor_chunk_indices = {0, 1, 2}};
    auto provider = sirius::scan_manager::make_provider_for_pinned_entry(
      entry, cols, std::move(plan), sirius::telemetry::batch_telemetry_info{}, set);
    auto b0 = provider->get_next_batch();
    REQUIRE(b0.data);
    REQUIRE(b0.mvcc_keep_mask.words == m0.words);
    auto b1 = provider->get_next_batch();
    REQUIRE(b1.data);
    REQUIRE_FALSE(b1.mvcc_keep_mask.has_mask());
    auto b2 = provider->get_next_batch();
    REQUIRE(b2.data);
    REQUIRE(b2.mvcc_keep_mask.words == m2.words);
    REQUIRE_FALSE(provider->get_next_batch().data);  // end of stream
  }

  SECTION("without a mask set every chunk serves unmasked")
  {
    sirius::scan_manager::cached_scan_plan plan{.survivor_chunk_indices = {0, 1, 2}};
    auto provider = sirius::scan_manager::make_provider_for_pinned_entry(
      entry, cols, std::move(plan), sirius::telemetry::batch_telemetry_info{});
    for (int i = 0; i < 3; ++i) {
      auto b = provider->get_next_batch();
      REQUIRE(b.data);
      REQUIRE_FALSE(b.mvcc_keep_mask.has_mask());
    }
    REQUIRE_FALSE(provider->get_next_batch().data);
  }
}

TEST_CASE("cached provider marks only selected narrow columns", "[cached_serving][scan_manager]")
{
  auto& e                = env();
  auto entry             = make_gpu_entry(*e.gpu_space, 3, 4);
  entry.narrowed_columns = {{false, true, false}, {true, false, false}, {false, false, false}};

  SECTION("a selected narrow column marks only its chunk")
  {
    std::vector<std::size_t> selected{1};
    sirius::scan_manager::cached_scan_plan plan{.survivor_chunk_indices = {0, 1, 2}};
    auto provider = sirius::scan_manager::make_provider_for_pinned_entry(
      entry, selected, std::move(plan), sirius::telemetry::batch_telemetry_info{});

    REQUIRE(provider->get_next_batch().contains_narrowed_columns);
    REQUIRE_FALSE(provider->get_next_batch().contains_narrowed_columns);
    REQUIRE_FALSE(provider->get_next_batch().contains_narrowed_columns);
  }

  SECTION("narrow unselected columns do not mark the served batch")
  {
    std::vector<std::size_t> selected{2};
    sirius::scan_manager::cached_scan_plan plan{.survivor_chunk_indices = {0, 1, 2}};
    auto provider = sirius::scan_manager::make_provider_for_pinned_entry(
      entry, selected, std::move(plan), sirius::telemetry::batch_telemetry_info{});

    for (int i = 0; i < 3; ++i) {
      auto batch = provider->get_next_batch();
      REQUIRE(batch.data);
      REQUIRE_FALSE(batch.contains_narrowed_columns);
    }
  }
}

TEST_CASE("cached provider computes exact restore-destination bytes per chunk",
          "[cached_serving][scan_manager]")
{
  auto& e = env();
  constexpr std::size_t rows{100};

  // Columns {k: INT8-stored BIGINT, v: INT32-stored BIGINT, w: native INT32}, one chunk;
  // k and v are marked narrowed, w is native.
  auto make_mixed_entry = [&](cudf::mask_state k_mask_state) {
    pinned_entry entry;
    entry.cache_info.names = {"k", "v", "w"};
    entry.tier             = cucascade::memory::Tier::GPU;
    entry.memory_space     = e.gpu_space;
    entry.chunk_memory_spaces.push_back(e.gpu_space);
    entry.data_batches_by_column["k"].push_back(make_typed_gpu_column(
      *e.gpu_space, cudf::data_type{cudf::type_id::INT8}, rows, k_mask_state));
    entry.data_batches_by_column["v"].push_back(make_typed_gpu_column(
      *e.gpu_space, cudf::data_type{cudf::type_id::INT32}, rows, cudf::mask_state::UNALLOCATED));
    entry.data_batches_by_column["w"].push_back(make_typed_gpu_column(
      *e.gpu_space, cudf::data_type{cudf::type_id::INT32}, rows, cudf::mask_state::UNALLOCATED));
    entry.narrowed_columns = {{true, true, false}};
    entry.num_rows         = rows;
    entry.zone_maps        = make_type_only_zone_maps(
      {duckdb::LogicalType::BIGINT, duckdb::LogicalType::BIGINT, duckdb::LogicalType::INTEGER},
      /*n_chunks=*/1);
    return entry;
  };

  SECTION("per-column exact destination for INT8->INT64 / INT32->INT64 mixed batches")
  {
    auto entry = make_mixed_entry(cudf::mask_state::UNALLOCATED);
    std::vector<std::size_t> selected{0, 1, 2};
    sirius::scan_manager::cached_scan_plan plan{.survivor_chunk_indices = {0}};
    auto provider = sirius::scan_manager::make_provider_for_pinned_entry(
      entry, selected, std::move(plan), sirius::telemetry::batch_telemetry_info{});

    auto batch = provider->get_next_batch();
    REQUIRE(batch.data);
    REQUIRE(batch.contains_narrowed_columns);
    // Both narrowed columns restore to INT64: 2 x rows x 8 bytes; native w contributes nothing.
    REQUIRE(batch.restore_destination_bytes == 2 * rows * sizeof(std::int64_t));
  }

  SECTION("a stored validity mask adds the destination mask bytes")
  {
    auto entry = make_mixed_entry(cudf::mask_state::ALL_VALID);
    std::vector<std::size_t> selected{0};
    sirius::scan_manager::cached_scan_plan plan{.survivor_chunk_indices = {0}};
    auto provider = sirius::scan_manager::make_provider_for_pinned_entry(
      entry, selected, std::move(plan), sirius::telemetry::batch_telemetry_info{});

    auto batch = provider->get_next_batch();
    REQUIRE(batch.data);
    REQUIRE(batch.restore_destination_bytes ==
            rows * sizeof(std::int64_t) +
              cudf::bitmask_allocation_size_bytes(static_cast<cudf::size_type>(rows)));
  }

  SECTION("non-converting selections report zero destination bytes")
  {
    auto entry = make_mixed_entry(cudf::mask_state::UNALLOCATED);
    std::vector<std::size_t> selected{2};
    sirius::scan_manager::cached_scan_plan plan{.survivor_chunk_indices = {0}};
    auto provider = sirius::scan_manager::make_provider_for_pinned_entry(
      entry, selected, std::move(plan), sirius::telemetry::batch_telemetry_info{});

    auto batch = provider->get_next_batch();
    REQUIRE(batch.data);
    REQUIRE_FALSE(batch.contains_narrowed_columns);
    REQUIRE(batch.restore_destination_bytes == 0);
  }

  SECTION("missing pin-time types leave the destination unknown")
  {
    auto entry      = make_mixed_entry(cudf::mask_state::UNALLOCATED);
    entry.zone_maps = {};  // statless pin: no type sidecar to derive destinations from
    std::vector<std::size_t> selected{0, 1, 2};
    sirius::scan_manager::cached_scan_plan plan{.survivor_chunk_indices = {0}};
    auto provider = sirius::scan_manager::make_provider_for_pinned_entry(
      entry, selected, std::move(plan), sirius::telemetry::batch_telemetry_info{});

    auto batch = provider->get_next_batch();
    REQUIRE(batch.data);
    REQUIRE(batch.contains_narrowed_columns);
    REQUIRE(batch.restore_destination_bytes == 0);  // caller keeps the conservative bound
  }

  SECTION("host-tier chunks compute from the host column metadata")
  {
    auto entry             = make_host_entry(e, /*n_chunks=*/2, /*rows=*/4);
    entry.narrowed_columns = {{true, false}, {false, false}};
    entry.zone_maps        = make_type_only_zone_maps(
      {duckdb::LogicalType::BIGINT, duckdb::LogicalType::INTEGER}, /*n_chunks=*/2);
    std::vector<std::size_t> selected{0, 1};
    sirius::scan_manager::cached_scan_plan plan{.survivor_chunk_indices = {0, 1}};
    auto provider = sirius::scan_manager::make_provider_for_pinned_entry(
      entry, selected, std::move(plan), sirius::telemetry::batch_telemetry_info{});

    auto first = provider->get_next_batch();
    REQUIRE(first.data);
    REQUIRE(first.contains_narrowed_columns);
    REQUIRE(first.restore_destination_bytes == 4 * sizeof(std::int64_t));

    auto second = provider->get_next_batch();
    REQUIRE(second.data);
    REQUIRE_FALSE(second.contains_narrowed_columns);
    REQUIRE(second.restore_destination_bytes == 0);
  }
}

TEST_CASE("pinned_column_narrowed_in_all_chunks folds the marker matrix per column",
          "[cached_serving][scan_manager]")
{
  auto& e    = env();
  auto entry = make_gpu_entry(*e.gpu_space, 3, 4);

  SECTION("empty (all-native canonical) matrix folds false")
  {
    entry.narrowed_columns = {};
    REQUIRE_FALSE(sirius::scan_manager::pinned_column_narrowed_in_all_chunks(entry, 0));
  }

  SECTION("columns fold independently")
  {
    // Column 0 narrowed in every chunk; column 1 mixed (true in one chunk,
    // false in another); column 2 never narrowed.
    entry.narrowed_columns = {{true, true, false}, {true, false, false}, {true, true, false}};
    REQUIRE(sirius::scan_manager::pinned_column_narrowed_in_all_chunks(entry, 0));
    REQUIRE_FALSE(sirius::scan_manager::pinned_column_narrowed_in_all_chunks(entry, 1));
    REQUIRE_FALSE(sirius::scan_manager::pinned_column_narrowed_in_all_chunks(entry, 2));
  }

  SECTION("out-of-range position folds false")
  {
    entry.narrowed_columns = {{true, true, true}, {true, true, true}, {true, true, true}};
    REQUIRE_FALSE(sirius::scan_manager::pinned_column_narrowed_in_all_chunks(entry, 3));
  }
}

TEST_CASE("pinned_column_narrow_carrier derives the widest stored carrier per column",
          "[cached_serving][scan_manager]")
{
  auto& e = env();
  auto const int64{cudf::data_type{cudf::type_id::INT64}};
  auto const int32{cudf::data_type{cudf::type_id::INT32}};
  auto const int16{cudf::data_type{cudf::type_id::INT16}};
  auto const int8{cudf::data_type{cudf::type_id::INT8}};

  /// Single-column GPU entry whose chunk c stores @p carriers[c]; markers all
  /// true so the derivation (not the marker fold) is what each section probes.
  auto make_single_column_gpu_entry = [&](std::vector<cudf::data_type> const& carriers) {
    pinned_entry entry;
    entry.cache_info.names = {"k"};
    entry.tier             = cucascade::memory::Tier::GPU;
    entry.memory_space     = e.gpu_space;
    for (auto const type : carriers) {
      entry.data_batches_by_column["k"].push_back(
        std::shared_ptr<cudf::column>(make_carrier_gpu_column(*e.gpu_space, type, 4)));
      entry.narrowed_columns.push_back({true});
    }
    entry.num_rows = carriers.size() * 4;
    return entry;
  };

  SECTION("uniform carrier")
  {
    auto entry             = make_gpu_entry(*e.gpu_space, 3, 4);  // INT32 stored everywhere
    entry.narrowed_columns = {{true, true, true}, {true, true, true}, {true, true, true}};
    auto const target      = sirius::scan_manager::pinned_column_narrow_carrier(entry, 0, int64);
    REQUIRE(target.has_value());
    REQUIRE(*target == int32);
  }

  SECTION("mixed widths derive the widest")
  {
    auto entry        = make_single_column_gpu_entry({int8, int16, int32});
    auto const target = sirius::scan_manager::pinned_column_narrow_carrier(entry, 0, int64);
    REQUIRE(target.has_value());
    REQUIRE(*target == int32);
  }

  SECTION("a false marker yields nullopt")
  {
    auto entry             = make_single_column_gpu_entry({int8, int16, int32});
    entry.narrowed_columns = {{true}, {false}, {true}};
    REQUIRE_FALSE(sirius::scan_manager::pinned_column_narrow_carrier(entry, 0, int64).has_value());
    entry.narrowed_columns = {};
    REQUIRE_FALSE(sirius::scan_manager::pinned_column_narrow_carrier(entry, 0, int64).has_value());
  }

  SECTION("a zero-chunk entry yields nullopt")
  {
    auto entry = make_single_column_gpu_entry({});
    REQUIRE_FALSE(sirius::scan_manager::pinned_column_narrow_carrier(entry, 0, int64).has_value());
  }

  SECTION("a carrier outside a strict same-family narrowing yields nullopt")
  {
    // Cross-family: unsigned storage against a signed native carrier.
    auto cross_family =
      make_single_column_gpu_entry({int8, cudf::data_type{cudf::type_id::UINT16}, int32});
    REQUIRE_FALSE(
      sirius::scan_manager::pinned_column_narrow_carrier(cross_family, 0, int64).has_value());

    // Not a strict narrowing: a chunk stored at the native width.
    auto native_width = make_single_column_gpu_entry({int8, int64});
    REQUIRE_FALSE(
      sirius::scan_manager::pinned_column_narrow_carrier(native_width, 0, int64).has_value());
  }

  SECTION("decimal carriers preserve the stored scale")
  {
    auto const decimal32_s2{cudf::data_type{cudf::type_id::DECIMAL32, -2}};
    auto const decimal64_s2{cudf::data_type{cudf::type_id::DECIMAL64, -2}};

    auto entry        = make_single_column_gpu_entry({decimal32_s2, decimal32_s2});
    auto const target = sirius::scan_manager::pinned_column_narrow_carrier(entry, 0, decimal64_s2);
    REQUIRE(target.has_value());
    REQUIRE(*target == decimal32_s2);

    // A chunk with a different cuDF scale is outside the carrier family.
    auto mixed_scale =
      make_single_column_gpu_entry({decimal32_s2, cudf::data_type{cudf::type_id::DECIMAL32, -1}});
    REQUIRE_FALSE(
      sirius::scan_manager::pinned_column_narrow_carrier(mixed_scale, 0, decimal64_s2).has_value());
  }

  SECTION("host-tier carriers reconstruct from the chunk column metadata")
  {
    // Column c0: INT8 then INT16 chunks; column c1: a DECIMAL(p,0)-shaped
    // carrier (cudf scale 0) in both chunks, pinning the type-id-keyed
    // two-argument fixed-point reconstruction.
    auto const decimal64_s0{cudf::data_type{cudf::type_id::DECIMAL64, 0}};
    auto const decimal128_s0{cudf::data_type{cudf::type_id::DECIMAL128, 0}};
    auto entry =
      make_host_entry_with_carriers(e, {{int8, decimal64_s0}, {int16, decimal64_s0}}, /*rows=*/4);
    entry.narrowed_columns = {{true, true}, {true, true}};

    auto const integer_target = sirius::scan_manager::pinned_column_narrow_carrier(entry, 0, int64);
    REQUIRE(integer_target.has_value());
    REQUIRE(*integer_target == int16);

    auto const decimal_target =
      sirius::scan_manager::pinned_column_narrow_carrier(entry, 1, decimal128_s0);
    REQUIRE(decimal_target.has_value());
    REQUIRE(*decimal_target == decimal64_s0);

    // Markers wider than the stored table: the position guard yields nullopt.
    entry.narrowed_columns = {{true, true, true}, {true, true, true}};
    REQUIRE_FALSE(sirius::scan_manager::pinned_column_narrow_carrier(entry, 2, int64).has_value());

    // A null host chunk yields nullopt.
    entry.narrowed_columns = {{true, true}, {true, true}};
    entry.host_chunks[1]   = nullptr;
    REQUIRE_FALSE(sirius::scan_manager::pinned_column_narrow_carrier(entry, 0, int64).has_value());
  }

  SECTION("out-of-range entry_position yields nullopt")
  {
    // Markers wider than the entry's column names: the name guard yields
    // nullopt after the fold passes.
    auto entry             = make_gpu_entry(*e.gpu_space, 2, 4);  // names {k, v, w}
    entry.narrowed_columns = {{true, true, true, true}, {true, true, true, true}};
    REQUIRE_FALSE(sirius::scan_manager::pinned_column_narrow_carrier(entry, 3, int64).has_value());
  }
}

TEST_CASE("masked and filtered resident splits report the filter-copy working-set peak",
          "[cached_serving][scan_manager]")
{
  auto& e    = env();
  auto batch = make_test_batch(e, 64);

  // Unmasked, unfiltered resident chunks serve a zero-copy view: working set
  // == data.
  sirius::op::scan::scan_operator_input plain(batch);
  REQUIRE(plain.get_estimated_working_set_size_in_bytes() == plain.get_estimated_size_in_bytes());

  // Masked chunks are filtered by copy: input + output coexist at peak, plus
  // the BOOL8 expansion (1 B/row) and the uploaded bitmask words.
  sirius::op::scan::scan_operator_input masked(batch);
  masked.mvcc_keep_mask  = make_test_mask(64);
  auto const batch_bytes = masked.get_estimated_size_in_bytes();
  REQUIRE(batch_bytes > 0);
  auto const masked_peak = 2 * batch_bytes + 64 + masked.mvcc_keep_mask.view().size_bytes();
  REQUIRE(masked.get_estimated_working_set_size_in_bytes() == masked_peak);

  // A pending row filter is also a filter-by-copy: input + compacted output.
  sirius::op::scan::scan_operator_input filtered(batch);
  filtered.row_filter_pending = true;
  REQUIRE(filtered.get_estimated_working_set_size_in_bytes() == 2 * batch_bytes);

  // Masked + filtered runs its phases sequentially and stays inside the
  // masked envelope.
  masked.row_filter_pending = true;
  REQUIRE(masked.get_estimated_working_set_size_in_bytes() == masked_peak);
}

TEST_CASE("drain_cached_provider honors a pre-stopped token", "[cached_serving][scan_manager]")
{
  auto& e = env();
  scripted_provider provider;
  provider.batches = {make_test_batch(e, 4), make_test_batch(e, 4)};

  split_connector connector;
  std::stop_source stop;
  stop.request_stop();
  load_balancing_scan_batch_coalescer::drain_cached_provider(
    provider, connector, stop.get_token(), /*row_filter_pending=*/false);

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

  SECTION("well-formed compression-enabled GPU entry (device_chunks)")
  {
    auto entry = make_device_chunks_entry(*e.gpu_space, 3, 4);
    REQUIRE_NOTHROW(validate_pinned_entry_for_serving(entry, std::vector<std::size_t>{0, 1, 2}));
  }
}

// A compression-enabled GPU pin serves from device_chunks, not the column-major
// data_batches_by_column. build_cached_scan_plan must count device_chunks or the
// scan serves zero chunks and the pipeline hangs (regression guard).
TEST_CASE("build_cached_scan_plan counts device_chunks for a compression-enabled GPU pin",
          "[cached_serving][scan_manager]")
{
  auto& e    = env();
  auto entry = make_device_chunks_entry(*e.gpu_space, 5, 4);
  auto plan  = build_cached_scan_plan(entry, /*table_filters=*/nullptr, /*column_ids=*/nullptr);
  REQUIRE(plan.survivor_chunk_indices.size() == 5);
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

  SECTION("narrowing matrix chunk count disagrees")
  {
    auto entry             = make_gpu_entry(*e.gpu_space, 2, 4);
    entry.narrowed_columns = {{true, false, false}};
    REQUIRE_THROWS_AS(validate_pinned_entry_for_serving(entry, std::vector<std::size_t>{0}),
                      std::invalid_argument);
  }

  SECTION("narrowing matrix column count disagrees")
  {
    auto entry             = make_gpu_entry(*e.gpu_space, 2, 4);
    entry.narrowed_columns = {{true, false}, {false, false}};
    REQUIRE_THROWS_AS(validate_pinned_entry_for_serving(entry, std::vector<std::size_t>{0}),
                      std::invalid_argument);
  }

  SECTION("host narrowing matrix shape disagrees")
  {
    auto entry             = make_host_entry(e, 2, 4);
    entry.narrowed_columns = {{true, false}};
    REQUIRE_THROWS_AS(validate_pinned_entry_for_serving(entry, std::vector<std::size_t>{0}),
                      std::invalid_argument);
  }

  SECTION("null host chunk")
  {
    auto entry           = make_host_entry(e, 2, 4);
    entry.host_chunks[1] = nullptr;
    REQUIRE_THROWS_AS(validate_pinned_entry_for_serving(entry, std::vector<std::size_t>{0}),
                      std::runtime_error);
  }

  SECTION("device_chunk missing a selected uncompressed column")
  {
    auto entry = make_device_chunks_entry(*e.gpu_space, 2, 4);
    entry.device_chunks[1].columns.pop_back();  // chunk 1 now lacks column w (index 2)
    REQUIRE_THROWS_AS(validate_pinned_entry_for_serving(entry, std::vector<std::size_t>{2}),
                      std::runtime_error);
  }

  SECTION("device_chunk has a null uncompressed column")
  {
    auto entry                        = make_device_chunks_entry(*e.gpu_space, 2, 4);
    entry.device_chunks[0].columns[0] = nullptr;
    REQUIRE_THROWS_AS(validate_pinned_entry_for_serving(entry, std::vector<std::size_t>{0}),
                      std::runtime_error);
  }
}

// The plan-time carrier fold and the serve-time restore sizing introspect stored chunk
// carriers, which only the uncompressed storage forms expose. Compression-enabled pins
// decline narrowing (their marker matrices stay empty), so a marker matrix that
// contradicts that invariant must degrade to "column stays native" / "conservative
// estimator bound" -- never a crash or a misread carrier.
TEST_CASE("pinned_column_narrow_carrier declines on a compression-enabled device entry",
          "[cached_serving][scan_manager][compression]")
{
  auto& e    = env();
  auto entry = make_device_chunks_entry(*e.gpu_space, /*n_chunks=*/2, /*rows=*/4);
  // Hand-set markers claiming every column narrowed: device_pin_chunk carriers are
  // opaque to introspection, so the fold must still bail to native.
  entry.narrowed_columns.assign(2, std::vector<bool>(3, true));
  REQUIRE_FALSE(sirius::scan_manager::pinned_column_narrow_carrier(
                  entry, /*entry_position=*/0, cudf::data_type{cudf::type_id::INT64})
                  .has_value());
}

TEST_CASE("host carrier introspection is total over non-uncompressed chunk forms",
          "[cached_serving][scan_manager][compression]")
{
  auto& e = env();
  // HOST entry whose single chunk is a compressed representation (empty blob --
  // nothing here decompresses); markers hand-set to contradict the decline.
  pinned_entry entry;
  entry.tier             = cucascade::memory::Tier::HOST;
  entry.memory_space     = e.host_space;
  entry.cache_info.names = {"k"};
  entry.host_chunks.emplace_back(std::make_shared<sirius::compressed_host_representation>(
    *e.host_space,
    std::make_shared<sirius::pinned_compressed_blob>(),
    std::vector<std::string>{"k"},
    /*compressed_bytes=*/64,
    /*uncompressed_bytes=*/256,
    /*num_rows=*/4));
  entry.num_rows         = 4;
  entry.narrowed_columns = {{true}};
  entry.zone_maps        = make_type_only_zone_maps({duckdb::LogicalType::BIGINT}, /*n_chunks=*/1);

  SECTION("carrier fold declines (column stays native)")
  {
    REQUIRE_FALSE(sirius::scan_manager::pinned_column_narrow_carrier(
                    entry, /*entry_position=*/0, cudf::data_type{cudf::type_id::INT64})
                    .has_value());
  }

  SECTION("restore sizing keeps the conservative bound")
  {
    std::vector<std::size_t> selected{0};
    sirius::scan_manager::cached_scan_plan plan{.survivor_chunk_indices = {0}};
    auto provider = sirius::scan_manager::make_provider_for_pinned_entry(
      entry, selected, std::move(plan), sirius::telemetry::batch_telemetry_info{});
    auto batch = provider->get_next_batch();
    REQUIRE(batch.data);
    REQUIRE(batch.contains_narrowed_columns);
    REQUIRE(batch.restore_destination_bytes == 0);
  }
}

TEST_CASE("validate_pinned_entry_for_serving checks the device-entry matrix shape",
          "[cached_serving][scan_manager][compression]")
{
  auto& e    = env();
  auto entry = make_device_chunks_entry(*e.gpu_space, /*n_chunks=*/2, /*rows=*/4);

  SECTION("empty matrix (canonical all-native) passes")
  {
    REQUIRE_NOTHROW(validate_pinned_entry_for_serving(entry, std::vector<std::size_t>{0, 1, 2}));
  }

  SECTION("matrix with the wrong chunk count throws")
  {
    entry.narrowed_columns.assign(1, std::vector<bool>(3, false));
    REQUIRE_THROWS_AS(validate_pinned_entry_for_serving(entry, std::vector<std::size_t>{0, 1, 2}),
                      std::invalid_argument);
  }

  SECTION("matrix with the wrong column count throws")
  {
    entry.narrowed_columns.assign(2, std::vector<bool>(2, false));
    REQUIRE_THROWS_AS(validate_pinned_entry_for_serving(entry, std::vector<std::size_t>{0, 1, 2}),
                      std::invalid_argument);
  }
}
