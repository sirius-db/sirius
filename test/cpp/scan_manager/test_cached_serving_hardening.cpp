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
// zero-chunk entries pass). The file also gates the pinned-entry storage-metadata helpers — the
// pinned_column_narrowed_in_all_chunks marker fold and the pinned_column_narrow_carrier carrier
// derivation that composes it for the plan-time residency gate. Both are pure folds over the
// entry's recorded column_storage matrix: compressed and uncompressed chunks, both tiers, answer
// through the same metadata, so several fixtures here carry no storage at all.

#include "operator/operator_test_utils.hpp"

#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/unary.hpp>
#include <cudf/utilities/traits.hpp>

#include <rmm/cuda_stream.hpp>
#include <rmm/error.hpp>

#include <cuda_runtime.h>

#include <catch.hpp>
#include <compression/compressed_representation.hpp>
#include <cucascade/cudf/gpu_data_representation.hpp>
#include <cucascade/cudf/host_data_representation.hpp>
#include <cucascade/data/data_batch.hpp>
#include <cucascade/memory/memory_space.hpp>
#include <data/data_batch_utils.hpp>
#include <data/sirius_converter_registry.hpp>
#include <op/scan/gpu_ingestible.hpp>
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

using sirius::pinned_column_storage_meta;

// The serve-site conversion fixtures never read the recorded pin-time native, so it defaults to
// the EMPTY sentinel; the plan-gate fixtures (pinned_column_narrow_carrier) pass it explicitly.
pinned_column_storage_meta narrow_meta(cudf::data_type carrier,
                                       cudf::data_type native = cudf::data_type{
                                         cudf::type_id::EMPTY})
{
  return {carrier, true, native};
}
pinned_column_storage_meta native_meta(cudf::data_type carrier) { return {carrier, false}; }

// Storage-metadata row of same-carrier cells with the given narrowed flags — the common shape of
// hand-built fixtures whose stored columns share one type.
std::vector<pinned_column_storage_meta> meta_row(cudf::data_type carrier,
                                                 std::initializer_list<bool> narrowed,
                                                 cudf::data_type native = cudf::data_type{
                                                   cudf::type_id::EMPTY})
{
  std::vector<pinned_column_storage_meta> row;
  row.reserve(narrowed.size());
  for (bool const flag : narrowed) {
    row.push_back({carrier, flag, native});
  }
  return row;
}

// Storage-metadata row of per-column carriers, all marked narrowed. The serve-site conversion
// decision reads carriers against the scan's plan targets, never the markers, so these fixtures
// vary the carrier and leave the flag constant.
std::vector<pinned_column_storage_meta> carrier_row(std::initializer_list<cudf::data_type> carriers)
{
  std::vector<pinned_column_storage_meta> row;
  row.reserve(carriers.size());
  for (auto const carrier : carriers) {
    row.push_back(narrow_meta(carrier));
  }
  return row;
}

// The scan's carrier targets in output order. Served slot k is output column k, so a slot at or
// past the end of this list is a trailing pure-filter column that reaches no target.
std::vector<cudf::data_type> targets(std::initializer_list<cudf::data_type> in_output_order)
{
  return {in_output_order};
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

// Zero-initialized numeric GPU column of an arbitrary carrier type — the serve-site
// restore-destination computation reads only types, row counts, and mask presence.
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

// Name the entry's cached columns, filling the column_ids and names lists insertion keeps
// lock-step so a fixture is indexable in either space.
void set_cached_columns(pinned_entry& entry, std::vector<std::string> names)
{
  for (std::size_t i = 0; i < names.size(); ++i) {
    entry.cache_info.column_ids.emplace_back(i);
  }
  entry.cache_info.names = std::move(names);
}

/// GPU-tier pinned entry with columns {k, v, w} x @p n_chunks chunks of
/// @p rows rows, every chunk placed in @p space.
pinned_entry make_gpu_entry(cucascade::memory::memory_space& space,
                            std::size_t n_chunks,
                            std::size_t rows)
{
  pinned_entry entry;
  set_cached_columns(entry, {"k", "v", "w"});
  entry.tier         = cucascade::memory::Tier::GPU;
  entry.memory_space = &space;
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
  set_cached_columns(entry, {"k", "v", "w"});
  entry.tier         = cucascade::memory::Tier::GPU;
  entry.memory_space = &space;
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
  set_cached_columns(entry, {"k", "v"});
  entry.tier         = cucascade::memory::Tier::HOST;
  entry.memory_space = e.host_space;

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

/// HOST-resident wrapper batch — the shape a host-pinned chunk's per-query slice arrives in, which
/// prepare_for_processing converts to a fresh owned GPU table.
std::shared_ptr<cucascade::data_batch> make_host_batch(
  test_env& e, const std::vector<std::vector<int32_t>>& data)
{
  std::vector<std::unique_ptr<cudf::column>> cols;
  cols.reserve(data.size());
  for (auto const& values : data) {
    auto shared = make_gpu_column(*e.gpu_space, values);
    cols.push_back(std::make_unique<cudf::column>(
      shared->view(), e.stream(), e.gpu_space->get_default_allocator()));
  }
  cucascade::gpu_table_representation gpu_repr(
    std::make_unique<cudf::table>(std::move(cols)), *e.gpu_space, e.stream());
  auto host_repr = sirius::converter_registry::get().convert<cucascade::host_data_representation>(
    gpu_repr, e.host_space, e.stream());
  e.stream().synchronize();
  return cucascade::data_batch::make(sirius::get_next_batch_id(), std::move(host_repr));
}

std::shared_ptr<cucascade::data_batch> make_host_batch(test_env& e,
                                                       std::vector<int32_t> const& values)
{
  return make_host_batch(e, std::vector<std::vector<int32_t>>{values});
}

/// Minimal concrete gpu_ingestible: materialize_table's resident branch calls
/// no virtuals, so every override is an unreachable stub.
struct stub_table_info final : sirius::op::scan::ingestible_table_info {
  [[nodiscard]] std::span<std::string const> column_names() const override { return {}; }
  [[nodiscard]] std::span<std::string const> file_paths() const override { return {}; }
};

struct stub_ingestible final : sirius::op::scan::gpu_ingestible {
  [[nodiscard]] std::unique_ptr<sirius::op::scan::batch_coalescer> create_batch_coalescer()
    const override
  {
    return nullptr;
  }
  [[nodiscard]] bool has_processed_all_metadata() const override { return true; }
  metadata_scan_task_t next_split_provider(sirius::io::ioctx_resolver /*resolve*/) override
  {
    return {};
  }
  sirius::op::scan::filtered_table materialize_metadata_to_table(
    const sirius::op::scan::scan_info& /*info*/,
    const cucascade::memory::memory_space& /*mem_space*/,
    rmm::cuda_stream_view /*stream*/) override
  {
    throw std::logic_error("stub_ingestible::materialize_metadata_to_table is unreachable");
  }
  std::unique_ptr<cudf::table> post_filter_and_project(
    sirius::op::scan::filtered_table&& /*input*/,
    const cucascade::memory::memory_space& /*mem_space*/,
    rmm::cuda_stream_view /*stream*/) override
  {
    throw std::logic_error("stub_ingestible::post_filter_and_project is unreachable");
  }
  [[nodiscard]] const sirius::op::scan::ingestible_table_info& table_info() const noexcept override
  {
    return _info;
  }
  [[nodiscard]] std::vector<std::size_t> materialized_column_order() const override { return {}; }

  stub_table_info _info;
};

template <typename T>
std::vector<T> to_host_column(cudf::table_view const& view, std::size_t column_idx)
{
  std::vector<T> out(static_cast<std::size_t>(view.num_rows()));
  cudaMemcpy(out.data(),
             view.column(static_cast<cudf::size_type>(column_idx)).data<T>(),
             sizeof(T) * out.size(),
             cudaMemcpyDeviceToHost);
  return out;
}

/// Copy the first INT32 column of @p table back to host for content checks.
std::vector<int32_t> to_host(cudf::table_view const& view)
{
  return to_host_column<int32_t>(view, 0);
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
  std::vector<bool> converting;
  std::size_t served{0};
  bool throw_when_exhausted{false};

  databatch_provider::batch get_next_batch() override
  {
    if (served < batches.size()) {
      auto idx = served++;
      return {batches[idx],
              idx < masks.size() ? masks[idx] : sirius::scan_manager::mvcc_chunk_mask{},
              idx < converting.size() && converting[idx]};
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
  provider.batches    = {make_test_batch(e, 4), make_test_batch(e, 4)};
  provider.masks      = {mask0};  // second batch deliberately mask-less
  provider.converting = {true, false};

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
  REQUIRE(in0->needs_carrier_conversion);

  auto second = connector.get_next_split();
  REQUIRE(second.has_value());
  auto* in1 = dynamic_cast<sirius::op::scan::scan_operator_input*>(second->get());
  REQUIRE(in1 != nullptr);
  REQUIRE_FALSE(in1->mvcc_keep_mask.has_mask());
  REQUIRE(in1->row_filter_pending);  // per-op flag: stamped on every split
  REQUIRE_FALSE(in1->needs_carrier_conversion);

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

TEST_CASE("cached provider marks only selected converting columns",
          "[cached_serving][scan_manager]")
{
  auto& e    = env();
  auto entry = make_gpu_entry(*e.gpu_space, 3, 4);
  auto const int64{cudf::data_type{cudf::type_id::INT64}};
  auto const int32{cudf::data_type{cudf::type_id::INT32}};
  // Every column targets INT64, so a chunk's column converts exactly when it recorded INT32.
  entry.column_storage = {carrier_row({int64, int32, int64}),
                          carrier_row({int32, int64, int64}),
                          carrier_row({int64, int64, int64})};

  SECTION("a selected converting column marks only its chunk")
  {
    std::vector<std::size_t> selected{1};
    sirius::scan_manager::cached_scan_plan plan{.survivor_chunk_indices = {0, 1, 2}};
    auto provider = sirius::scan_manager::make_provider_for_pinned_entry(
      entry,
      selected,
      std::move(plan),
      sirius::telemetry::batch_telemetry_info{},
      {},
      {},
      targets({int64}));

    REQUIRE(provider->get_next_batch().needs_carrier_conversion);
    REQUIRE_FALSE(provider->get_next_batch().needs_carrier_conversion);
    REQUIRE_FALSE(provider->get_next_batch().needs_carrier_conversion);
  }

  SECTION("converting unselected columns do not mark the served batch")
  {
    std::vector<std::size_t> selected{2};
    sirius::scan_manager::cached_scan_plan plan{.survivor_chunk_indices = {0, 1, 2}};
    auto provider = sirius::scan_manager::make_provider_for_pinned_entry(
      entry,
      selected,
      std::move(plan),
      sirius::telemetry::batch_telemetry_info{},
      {},
      {},
      targets({int64}));

    for (int i = 0; i < 3; ++i) {
      auto batch = provider->get_next_batch();
      REQUIRE(batch.data);
      REQUIRE_FALSE(batch.needs_carrier_conversion);
    }
  }
}

TEST_CASE("cached provider computes exact conversion-destination bytes per chunk",
          "[cached_serving][scan_manager]")
{
  auto& e = env();
  constexpr std::size_t rows{100};
  auto const int64{cudf::data_type{cudf::type_id::INT64}};
  auto const int32{cudf::data_type{cudf::type_id::INT32}};
  auto const int8{cudf::data_type{cudf::type_id::INT8}};
  auto const mask_bytes = cudf::bitmask_allocation_size_bytes(static_cast<cudf::size_type>(rows));

  // Columns {k, v, w} of one GPU chunk, each stored at the carrier the metadata records.
  auto make_entry = [&](std::vector<cudf::data_type> const& carriers, cudf::mask_state k_mask) {
    pinned_entry entry;
    set_cached_columns(entry, {"k", "v", "w"});
    entry.tier         = cucascade::memory::Tier::GPU;
    entry.memory_space = e.gpu_space;
    entry.chunk_memory_spaces.push_back(e.gpu_space);
    std::vector<pinned_column_storage_meta> row;
    for (std::size_t col = 0; col < entry.cache_info.names.size(); ++col) {
      entry.data_batches_by_column[entry.cache_info.names[col]].push_back(make_typed_gpu_column(
        *e.gpu_space, carriers[col], rows, col == 0 ? k_mask : cudf::mask_state::UNALLOCATED));
      row.push_back(narrow_meta(carriers[col]));
    }
    entry.column_storage = {std::move(row)};
    entry.num_rows       = rows;
    return entry;
  };

  auto serve_one = [](pinned_entry const& entry,
                      std::vector<std::size_t> const& selected,
                      std::vector<cudf::data_type> normalization_targets,
                      bool has_overrides = false) {
    sirius::scan_manager::cached_scan_plan plan{.survivor_chunk_indices = {0}};
    auto provider = sirius::scan_manager::make_provider_for_pinned_entry(
      entry,
      selected,
      std::move(plan),
      sirius::telemetry::batch_telemetry_info{},
      {},
      {},
      std::move(normalization_targets),
      has_overrides);
    return provider->get_next_batch();
  };

  SECTION("a stored carrier that already equals the plan target is charge-free")
  {
    // The stacking happy path: pinned narrow, planned narrow, so normalization issues no cast.
    auto entry = make_entry({int32, int32, int32}, cudf::mask_state::UNALLOCATED);
    auto batch = serve_one(entry, {0, 1, 2}, targets({int32, int32, int32}));
    REQUIRE(batch.data);
    REQUIRE_FALSE(batch.needs_carrier_conversion);
    REQUIRE(batch.conversion_destination_bytes == 0);
  }

  SECTION("a compressed chunk whose recorded carrier equals the plan target is charge-free")
  {
    // The GPU-tier and host-tier compressed cells: nothing about the blob is readable, and
    // nothing needs to be -- the recorded carrier already answers.
    pinned_entry device_entry;
    set_cached_columns(device_entry, {"k"});
    device_entry.tier         = cucascade::memory::Tier::GPU;
    device_entry.memory_space = e.gpu_space;
    sirius::device_pin_chunk chunk;
    chunk.memory_space = e.gpu_space;
    chunk.compressed   = std::make_shared<sirius::compressed_device_representation>(
      *e.gpu_space,
      /*blob=*/nullptr,  // select_columns/num_rows never touch the blob
      std::vector<std::string>{"k"},
      /*compressed_bytes=*/64,
      /*uncompressed_bytes=*/256,
      /*num_rows=*/static_cast<std::int64_t>(rows));
    device_entry.device_chunks.push_back(std::move(chunk));
    device_entry.num_rows       = rows;
    device_entry.column_storage = {{narrow_meta(int32)}};

    auto device_batch = serve_one(device_entry, {0}, targets({int32}));
    REQUIRE(device_batch.data);
    REQUIRE_FALSE(device_batch.needs_carrier_conversion);
    REQUIRE(device_batch.conversion_destination_bytes == 0);

    pinned_entry host_entry;
    set_cached_columns(host_entry, {"k"});
    host_entry.tier         = cucascade::memory::Tier::HOST;
    host_entry.memory_space = e.host_space;
    host_entry.host_chunks.emplace_back(std::make_shared<sirius::compressed_host_representation>(
      *e.host_space,
      std::make_shared<sirius::pinned_compressed_blob>(),
      std::vector<std::string>{"k"},
      /*compressed_bytes=*/64,
      /*uncompressed_bytes=*/256,
      /*num_rows=*/static_cast<std::int64_t>(rows)));
    host_entry.num_rows       = rows;
    host_entry.column_storage = {{narrow_meta(int32)}};

    auto host_batch = serve_one(host_entry, {0}, targets({int32}));
    REQUIRE(host_batch.data);
    REQUIRE_FALSE(host_batch.needs_carrier_conversion);
    REQUIRE(host_batch.conversion_destination_bytes == 0);
  }

  SECTION("per-column exact destination in the target's width")
  {
    auto entry = make_entry({int8, int32, int32}, cudf::mask_state::UNALLOCATED);
    auto batch = serve_one(entry, {0, 1, 2}, targets({int64, int64, int32}));
    REQUIRE(batch.data);
    REQUIRE(batch.needs_carrier_conversion);
    // k and v restore to INT64; w already sits at its target and contributes nothing.
    REQUIRE(batch.conversion_destination_bytes == 2 * rows * sizeof(std::int64_t));
  }

  SECTION("a stored validity mask adds the destination mask bytes")
  {
    auto entry = make_entry({int8, int32, int32}, cudf::mask_state::ALL_VALID);
    auto batch = serve_one(entry, {0}, targets({int64}));
    REQUIRE(batch.data);
    REQUIRE(batch.conversion_destination_bytes == rows * sizeof(std::int64_t) + mask_bytes);
  }

  SECTION("no sidecar over narrow storage still charges the native restore")
  {
    // Pin-on / query-off: the target is the native mapping, so every narrow carrier widens.
    auto entry = make_entry({int8, int32, int32}, cudf::mask_state::UNALLOCATED);
    auto batch = serve_one(entry, {0}, targets({int64}));
    REQUIRE(batch.needs_carrier_conversion);
    REQUIRE(batch.conversion_destination_bytes == rows * sizeof(std::int64_t));
  }

  SECTION("a sidecar narrowing a native cached carrier charges the narrow destination")
  {
    auto entry = make_entry({int64, int32, int32}, cudf::mask_state::UNALLOCATED);

    auto narrowing = serve_one(entry, {0}, targets({int32}), /*has_overrides=*/true);
    REQUIRE(narrowing.needs_carrier_conversion);
    REQUIRE(narrowing.conversion_destination_bytes == rows * sizeof(std::int32_t));

    // Without a sidecar normalization never narrows, so the same pair converts nothing.
    auto unplanned = serve_one(entry, {0}, targets({int32}), /*has_overrides=*/false);
    REQUIRE_FALSE(unplanned.needs_carrier_conversion);
    REQUIRE(unplanned.conversion_destination_bytes == 0);
  }

  SECTION("a trailing pure-filter column is never charged")
  {
    // Both ingestibles materialize pure-filter columns after every output column, so such a
    // column occupies a served slot past the end of the target list. post_filter_and_project
    // drops it before normalization, and it must not be charged even though its INT8 carrier
    // would convert if it had a target.
    auto entry = make_entry({int8, int8, int32}, cudf::mask_state::UNALLOCATED);

    // One output column plus one filter-only column: only the output column is charged.
    auto mixed = serve_one(entry, {0, 1}, targets({int64}));
    REQUIRE(mixed.needs_carrier_conversion);
    REQUIRE(mixed.conversion_destination_bytes == rows * sizeof(std::int64_t));

    // A scan that emits nothing at all charges nothing at all.
    auto filter_only = serve_one(entry, {0}, targets({}));
    REQUIRE(filter_only.data);
    REQUIRE_FALSE(filter_only.needs_carrier_conversion);
    REQUIRE(filter_only.conversion_destination_bytes == 0);
  }

  SECTION("non-converting selections report zero destination bytes")
  {
    auto entry = make_entry({int8, int32, int32}, cudf::mask_state::UNALLOCATED);
    auto batch = serve_one(entry, {2}, targets({int32}));
    REQUIRE(batch.data);
    REQUIRE_FALSE(batch.needs_carrier_conversion);
    REQUIRE(batch.conversion_destination_bytes == 0);
  }

  SECTION("host-tier chunks compute from the host column metadata")
  {
    auto entry           = make_host_entry(e, /*n_chunks=*/2, /*rows=*/4);
    entry.column_storage = {carrier_row({int32, int64}), carrier_row({int64, int64})};
    std::vector<std::size_t> selected{0, 1};
    sirius::scan_manager::cached_scan_plan plan{.survivor_chunk_indices = {0, 1}};
    auto provider = sirius::scan_manager::make_provider_for_pinned_entry(
      entry,
      selected,
      std::move(plan),
      sirius::telemetry::batch_telemetry_info{},
      {},
      {},
      targets({int64, int64}));

    auto first = provider->get_next_batch();
    REQUIRE(first.data);
    REQUIRE(first.needs_carrier_conversion);
    REQUIRE(first.conversion_destination_bytes == 4 * sizeof(std::int64_t));

    auto second = provider->get_next_batch();
    REQUIRE(second.data);
    REQUIRE_FALSE(second.needs_carrier_conversion);
    REQUIRE(second.conversion_destination_bytes == 0);
  }

  SECTION("compressed chunks take rows from the representation and always add the mask term")
  {
    // Empty-blob compressed chunk: metadata only, nothing here decompresses. A compressed
    // chunk's per-column nullability is opaque, so its validity-mask term is always added
    // (over-approximation, never under).
    pinned_entry host_entry;
    set_cached_columns(host_entry, {"k"});
    host_entry.tier         = cucascade::memory::Tier::HOST;
    host_entry.memory_space = e.host_space;
    host_entry.host_chunks.emplace_back(std::make_shared<sirius::compressed_host_representation>(
      *e.host_space,
      std::make_shared<sirius::pinned_compressed_blob>(),
      std::vector<std::string>{"k"},
      /*compressed_bytes=*/64,
      /*uncompressed_bytes=*/256,
      /*num_rows=*/static_cast<std::int64_t>(rows)));
    host_entry.num_rows       = rows;
    host_entry.column_storage = {{narrow_meta(int8)}};

    auto host_batch = serve_one(host_entry, {0}, targets({int64}));
    REQUIRE(host_batch.data);
    REQUIRE(host_batch.needs_carrier_conversion);
    REQUIRE(host_batch.conversion_destination_bytes == rows * sizeof(std::int64_t) + mask_bytes);

    pinned_entry device_entry;
    set_cached_columns(device_entry, {"k"});
    device_entry.tier         = cucascade::memory::Tier::GPU;
    device_entry.memory_space = e.gpu_space;
    sirius::device_pin_chunk chunk;
    chunk.memory_space = e.gpu_space;
    chunk.compressed   = std::make_shared<sirius::compressed_device_representation>(
      *e.gpu_space,
      /*blob=*/nullptr,
      std::vector<std::string>{"k"},
      /*compressed_bytes=*/64,
      /*uncompressed_bytes=*/256,
      /*num_rows=*/static_cast<std::int64_t>(rows));
    device_entry.device_chunks.push_back(std::move(chunk));
    device_entry.num_rows       = rows;
    device_entry.column_storage = {{narrow_meta(int8)}};

    auto device_batch = serve_one(device_entry, {0}, targets({int64}));
    REQUIRE(device_batch.data);
    REQUIRE(device_batch.needs_carrier_conversion);
    REQUIRE(device_batch.conversion_destination_bytes == rows * sizeof(std::int64_t) + mask_bytes);
  }
}

TEST_CASE("pinned_column_narrowed_in_all_chunks folds the marker matrix per column",
          "[cached_serving][scan_manager]")
{
  auto& e    = env();
  auto entry = make_gpu_entry(*e.gpu_space, 3, 4);
  auto const int32{cudf::data_type{cudf::type_id::INT32}};

  SECTION("empty (all-native compat) matrix folds false")
  {
    entry.column_storage = {};
    REQUIRE_FALSE(sirius::scan_manager::pinned_column_narrowed_in_all_chunks(entry, 0));
  }

  SECTION("columns fold independently")
  {
    // Column 0 narrowed in every chunk; column 1 mixed (true in one chunk,
    // false in another); column 2 never narrowed.
    entry.column_storage = {meta_row(int32, {true, true, false}),
                            meta_row(int32, {true, false, false}),
                            meta_row(int32, {true, true, false})};
    REQUIRE(sirius::scan_manager::pinned_column_narrowed_in_all_chunks(entry, 0));
    REQUIRE_FALSE(sirius::scan_manager::pinned_column_narrowed_in_all_chunks(entry, 1));
    REQUIRE_FALSE(sirius::scan_manager::pinned_column_narrowed_in_all_chunks(entry, 2));
  }

  SECTION("out-of-range position folds false")
  {
    entry.column_storage = {meta_row(int32, {true, true, true}),
                            meta_row(int32, {true, true, true}),
                            meta_row(int32, {true, true, true})};
    REQUIRE_FALSE(sirius::scan_manager::pinned_column_narrowed_in_all_chunks(entry, 3));
  }
}

TEST_CASE("pinned_column_narrow_carrier derives the widest recorded carrier per column",
          "[cached_serving][scan_manager]")
{
  auto const int64{cudf::data_type{cudf::type_id::INT64}};
  auto const int32{cudf::data_type{cudf::type_id::INT32}};
  auto const int16{cudf::data_type{cudf::type_id::INT16}};
  auto const int8{cudf::data_type{cudf::type_id::INT8}};

  // Single-column entry whose chunk c records `carriers[c]` (with `native` as the pin-time
  // native type), all marked narrowed so the derivation (not the marker fold) is what each
  // section probes. The fold reads only the recorded metadata, so the fixture carries no storage
  // at all.
  auto make_single_column_meta_entry = [](std::vector<cudf::data_type> const& carriers,
                                          cudf::data_type native) {
    pinned_entry entry;
    set_cached_columns(entry, {"k"});
    entry.tier = cucascade::memory::Tier::GPU;
    for (auto const type : carriers) {
      entry.column_storage.push_back({narrow_meta(type, native)});
    }
    entry.num_rows = carriers.size() * 4;
    return entry;
  };

  SECTION("uniform carrier")
  {
    auto entry        = make_single_column_meta_entry({int32, int32, int32}, int64);
    auto const target = sirius::scan_manager::pinned_column_narrow_carrier(entry, 0, int64);
    REQUIRE(target.has_value());
    REQUIRE(*target == int32);
  }

  SECTION("mixed widths derive the widest")
  {
    auto entry        = make_single_column_meta_entry({int8, int16, int32}, int64);
    auto const target = sirius::scan_manager::pinned_column_narrow_carrier(entry, 0, int64);
    REQUIRE(target.has_value());
    REQUIRE(*target == int32);
  }

  SECTION("a drifted or unrecorded pin-time native yields nullopt")
  {
    // Same carriers, same widths -- only the recorded native differs. TIMESTAMP_DAYS and INT32
    // share the int32 representation, so this drop/recreate drift is exactly the case the
    // same-family width checks alone cannot catch.
    auto const timestamp_days = cudf::data_type{cudf::type_id::TIMESTAMP_DAYS};
    auto date_pin             = make_single_column_meta_entry({int16, int16}, timestamp_days);
    REQUIRE(
      sirius::scan_manager::pinned_column_narrow_carrier(date_pin, 0, timestamp_days).has_value());
    REQUIRE_FALSE(
      sirius::scan_manager::pinned_column_narrow_carrier(date_pin, 0, int32).has_value());

    auto integer_pin = make_single_column_meta_entry({int16, int16}, int32);
    REQUIRE_FALSE(sirius::scan_manager::pinned_column_narrow_carrier(integer_pin, 0, timestamp_days)
                    .has_value());

    // An EMPTY (never recorded) native reads as a mismatch defensively.
    auto unrecorded =
      make_single_column_meta_entry({int16, int16}, cudf::data_type{cudf::type_id::EMPTY});
    REQUIRE_FALSE(
      sirius::scan_manager::pinned_column_narrow_carrier(unrecorded, 0, int64).has_value());
  }

  SECTION("a false marker yields nullopt")
  {
    auto entry           = make_single_column_meta_entry({int8, int16, int32}, int64);
    entry.column_storage = {
      {narrow_meta(int8, int64)}, {native_meta(int16)}, {narrow_meta(int32, int64)}};
    REQUIRE_FALSE(sirius::scan_manager::pinned_column_narrow_carrier(entry, 0, int64).has_value());
    entry.column_storage = {};
    REQUIRE_FALSE(sirius::scan_manager::pinned_column_narrow_carrier(entry, 0, int64).has_value());
  }

  SECTION("a zero-chunk entry yields nullopt")
  {
    auto entry = make_single_column_meta_entry({}, int64);
    REQUIRE_FALSE(sirius::scan_manager::pinned_column_narrow_carrier(entry, 0, int64).has_value());
  }

  SECTION("a carrier outside a strict same-family narrowing yields nullopt")
  {
    // Cross-family: an unsigned recorded carrier against a signed native carrier.
    auto cross_family =
      make_single_column_meta_entry({int8, cudf::data_type{cudf::type_id::UINT16}, int32}, int64);
    REQUIRE_FALSE(
      sirius::scan_manager::pinned_column_narrow_carrier(cross_family, 0, int64).has_value());

    // Not a strict narrowing: a chunk recorded at the native width.
    auto native_width = make_single_column_meta_entry({int8, int64}, int64);
    REQUIRE_FALSE(
      sirius::scan_manager::pinned_column_narrow_carrier(native_width, 0, int64).has_value());
  }

  SECTION("decimal carriers preserve the recorded scale")
  {
    auto const decimal32_s2{cudf::data_type{cudf::type_id::DECIMAL32, -2}};
    auto const decimal64_s2{cudf::data_type{cudf::type_id::DECIMAL64, -2}};

    auto entry        = make_single_column_meta_entry({decimal32_s2, decimal32_s2}, decimal64_s2);
    auto const target = sirius::scan_manager::pinned_column_narrow_carrier(entry, 0, decimal64_s2);
    REQUIRE(target.has_value());
    REQUIRE(*target == decimal32_s2);

    // A chunk with a different cuDF scale is outside the carrier family.
    auto mixed_scale = make_single_column_meta_entry(
      {decimal32_s2, cudf::data_type{cudf::type_id::DECIMAL32, -1}}, decimal64_s2);
    REQUIRE_FALSE(
      sirius::scan_manager::pinned_column_narrow_carrier(mixed_scale, 0, decimal64_s2).has_value());
  }

  SECTION("a DECIMAL(p,0)-shaped carrier keeps its zero scale distinct from the integer family")
  {
    auto const decimal64_s0{cudf::data_type{cudf::type_id::DECIMAL64, 0}};
    auto const decimal128_s0{cudf::data_type{cudf::type_id::DECIMAL128, 0}};
    auto entry = make_single_column_meta_entry({decimal64_s0, decimal64_s0}, decimal128_s0);
    auto const decimal_target =
      sirius::scan_manager::pinned_column_narrow_carrier(entry, 0, decimal128_s0);
    REQUIRE(decimal_target.has_value());
    REQUIRE(*decimal_target == decimal64_s0);
    REQUIRE_FALSE(sirius::scan_manager::pinned_column_narrow_carrier(entry, 0, int64).has_value());
  }

  SECTION("out-of-range entry_position yields nullopt")
  {
    // A position beyond the recorded rows fails the marker fold.
    auto entry           = make_single_column_meta_entry({int8, int16}, int64);
    entry.column_storage = {meta_row(int8, {true, true}), meta_row(int16, {true, true})};
    REQUIRE_FALSE(sirius::scan_manager::pinned_column_narrow_carrier(entry, 2, int64).has_value());
  }

  SECTION("a matrix wider than the entry's columns yields nullopt")
  {
    // The matrix and the entry's column list disagree, which insertion would have rejected. This
    // runs at plan time, on an entry no serving validator has inspected, so the narrower of the
    // two authorities wins rather than the fold answering from an unvalidated cell.
    auto entry           = make_single_column_meta_entry({int8, int16}, int64);
    entry.column_storage = {meta_row(int8, {true, true, true}),
                            meta_row(int16, {true, true, true})};
    REQUIRE_FALSE(sirius::scan_manager::pinned_column_narrow_carrier(entry, 2, int64).has_value());
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
    set_cached_columns(entry, {"k"});
    entry.tier         = cucascade::memory::Tier::GPU;
    entry.memory_space = e.gpu_space;
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

  SECTION("storage-metadata matrix chunk count disagrees")
  {
    auto entry           = make_gpu_entry(*e.gpu_space, 2, 4);
    entry.column_storage = {meta_row(cudf::data_type{cudf::type_id::INT32}, {true, false, false})};
    REQUIRE_THROWS_AS(validate_pinned_entry_for_serving(entry, std::vector<std::size_t>{0}),
                      std::invalid_argument);
  }

  SECTION("storage-metadata matrix column count disagrees")
  {
    auto entry           = make_gpu_entry(*e.gpu_space, 2, 4);
    auto const int32     = cudf::data_type{cudf::type_id::INT32};
    entry.column_storage = {meta_row(int32, {true, false}), meta_row(int32, {false, false})};
    REQUIRE_THROWS_AS(validate_pinned_entry_for_serving(entry, std::vector<std::size_t>{0}),
                      std::invalid_argument);
  }

  SECTION("host storage-metadata matrix shape disagrees")
  {
    auto entry           = make_host_entry(e, 2, 4);
    entry.column_storage = {meta_row(cudf::data_type{cudf::type_id::INT32}, {true, false})};
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

//===----------------------------------------------------------------------===//
// prepare_for_processing steal (zero-copy scan materialize)
//===----------------------------------------------------------------------===//

TEST_CASE("prepare_for_processing steals the converted table from a per-query wrapper batch",
          "[cached_serving][scan_manager]")
{
  auto& e = env();
  std::vector<int32_t> const values{10, 11, 12, 13};
  auto batch = make_host_batch(e, values);

  sirius::op::scan::scan_operator_input split{batch};
  split.prepare_for_processing(e.gpu_space, e.stream());

  // The uploaded table was taken out of the wrapper; the batch is left holding
  // a valid empty placeholder, and size estimates answer from the stolen table.
  REQUIRE(split.stolen_table != nullptr);
  REQUIRE(split.stolen_table_bytes > 0);
  REQUIRE(split.get_estimated_size_in_bytes() == split.stolen_table_bytes);
  {
    auto ro = batch->to_read_only();
    REQUIRE(ro.get_current_tier() == cucascade::memory::Tier::GPU);
    REQUIRE(ro.get_data() != nullptr);
    REQUIRE(ro.get_data()->get_size_in_bytes() == 0);
  }

  stub_ingestible ingestible;
  auto result = ingestible.materialize_table(split, e.stream());
  REQUIRE(result.state == sirius::op::scan::filter_state::UNFILTERED);
  auto out = result.table.release(e.stream(), e.gpu_space->get_default_allocator());
  REQUIRE(out != nullptr);
  e.stream().synchronize();
  REQUIRE(to_host(out->view()) == values);
  REQUIRE(split.stolen_table == nullptr);
  REQUIRE(split.stolen_table_consumed);

  // Re-entry after consumption (scan-internal OOM retry) fails loudly instead
  // of serving the emptied wrapper as zero rows...
  REQUIRE_THROWS_AS(ingestible.materialize_table(split, e.stream()), std::runtime_error);
  // ...and a re-prepare is a no-op: no second conversion, no second steal.
  split.prepare_for_processing(e.gpu_space, e.stream());
  REQUIRE(split.stolen_table == nullptr);
  REQUIRE(split.get_estimated_size_in_bytes() == split.stolen_table_bytes);
}

TEST_CASE("prepare_for_processing never steals from a GPU-resident (pin-shaped) batch",
          "[cached_serving][scan_manager]")
{
  auto& e = env();
  // View-backed shared columns already on the GPU tier — the exact shape a raw
  // GPU pin serves; no conversion happens, so nothing may be stolen.
  auto batch             = make_test_batch(e, 4);
  auto const size_before = [&] {
    auto ro = batch->to_read_only();
    return ro.get_data()->get_size_in_bytes();
  }();
  REQUIRE(size_before > 0);

  sirius::op::scan::scan_operator_input split{batch};
  split.needs_carrier_conversion = true;
  split.prepare_for_processing(e.gpu_space, e.stream());
  REQUIRE(split.stolen_table == nullptr);
  REQUIRE(split.stolen_table_bytes == 0);
  REQUIRE_FALSE(split.converted_table_steal_pending);

  stub_ingestible ingestible;
  auto result = ingestible.materialize_table(split, e.stream());
  REQUIRE(result.state == sirius::op::scan::filter_state::UNFILTERED);
  auto out = result.table.release(e.stream(), e.gpu_space->get_default_allocator());
  REQUIRE(out != nullptr);
  e.stream().synchronize();
  REQUIRE(to_host(out->view()) == std::vector<int32_t>(4, 7));

  // Pin-shaped storage untouched: same representation, same bytes.
  auto ro = batch->to_read_only();
  REQUIRE(ro.get_data() != nullptr);
  REQUIRE(ro.get_data()->get_size_in_bytes() == size_before);
}

TEST_CASE("prepare_for_processing gates immediate and transactional converted-table steals",
          "[cached_serving][scan_manager]")
{
  auto& e = env();
  std::vector<int32_t> const values{20, 21, 22, 23};

  // Each section cross-checks converted_table_transferable() against the observed arming outcome:
  // the shared predicate is the eligibility policy both steal forms consult, so its answer and the
  // arming decision must agree row by row (mask -> false, pending row filter -> false,
  // decode-applied row filter -> true, unflagged -> true).

  SECTION("mvcc keep-mask pending")
  {
    auto batch = make_host_batch(e, values);
    sirius::op::scan::scan_operator_input split{batch};
    split.mvcc_keep_mask = make_test_mask(values.size());
    split.prepare_for_processing(e.gpu_space, e.stream());
    REQUIRE(split.stolen_table == nullptr);
    REQUIRE_FALSE(split.converted_table_steal_pending);
    REQUIRE_FALSE(split.converted_table_transferable());
    // Converted in place but not stolen: the masked materialize path filters
    // by copy from the batch's view.
    auto ro = batch->to_read_only();
    REQUIRE(ro.get_current_tier() == cucascade::memory::Tier::GPU);
    REQUIRE(ro.get_data()->get_size_in_bytes() > 0);
  }

  SECTION("row filter pending")
  {
    auto batch = make_host_batch(e, values);
    sirius::op::scan::scan_operator_input split{batch};
    split.row_filter_pending = true;
    split.prepare_for_processing(e.gpu_space, e.stream());
    REQUIRE(split.stolen_table == nullptr);
    REQUIRE_FALSE(split.converted_table_steal_pending);
    REQUIRE_FALSE(split.converted_table_transferable());
    auto ro = batch->to_read_only();
    REQUIRE(ro.get_current_tier() == cucascade::memory::Tier::GPU);
    REQUIRE(ro.get_data()->get_size_in_bytes() > 0);
  }

  SECTION("carrier conversion pending")
  {
    auto batch = make_host_batch(e, values);
    sirius::op::scan::scan_operator_input split{batch};
    split.needs_carrier_conversion = true;
    split.prepare_for_processing(e.gpu_space, e.stream());
    REQUIRE(split.stolen_table == nullptr);
    REQUIRE(split.converted_table_steal_pending);
    REQUIRE(split.converted_table_transferable());
    REQUIRE(split.stolen_table_bytes > 0);
    // Converted in place and retained transactionally: scan normalization will allocate every
    // replacement before committing the source columns.
    auto ro = batch->to_read_only();
    REQUIRE(ro.get_current_tier() == cucascade::memory::Tier::GPU);
    REQUIRE(ro.get_data()->get_size_in_bytes() > 0);
  }

  SECTION("row filter pending with carrier conversion")
  {
    // Filtering is still ahead of this split, so neither steal form may take the source.
    auto batch = make_host_batch(e, values);
    sirius::op::scan::scan_operator_input split{batch};
    split.row_filter_pending       = true;
    split.needs_carrier_conversion = true;
    split.prepare_for_processing(e.gpu_space, e.stream());
    REQUIRE(split.stolen_table == nullptr);
    REQUIRE_FALSE(split.converted_table_steal_pending);
    REQUIRE_FALSE(split.converted_table_transferable());
  }

  SECTION("decode-row-filtered carrier conversion regains the transactional path")
  {
    // The decode already applied the whole filter conjunction (pre-stamped here; a plain host
    // conversion leaves the outcome untouched), so only the carrier cast remains and prepare
    // arms the transactional steal.
    auto batch = make_host_batch(e, values);
    sirius::op::scan::scan_operator_input split{batch};
    split.row_filter_pending       = true;
    split.pushdown_row_filtered    = true;
    split.needs_carrier_conversion = true;
    split.prepare_for_processing(e.gpu_space, e.stream());
    REQUIRE(split.stolen_table == nullptr);
    REQUIRE(split.converted_table_steal_pending);
    REQUIRE(split.converted_table_transferable());
    REQUIRE(split.stolen_table_bytes > 0);
    auto ro = batch->to_read_only();
    REQUIRE(ro.get_current_tier() == cucascade::memory::Tier::GPU);
    REQUIRE(ro.get_data()->get_size_in_bytes() > 0);
  }
}

TEST_CASE("transactional converted-table steal rolls back and commits on retry",
          "[cached_serving][scan_manager]")
{
  auto& e = env();
  std::vector<int32_t> const first_values{30, 31, 32, 33};
  std::vector<int32_t> const second_values{130, 131, 132, 133};
  auto batch = make_host_batch(e, std::vector<std::vector<int32_t>>{first_values, second_values});
  using replacements = sirius::op::scan::scan_operator_input::converted_column_replacements;

  sirius::op::scan::scan_operator_input split{batch};
  split.needs_carrier_conversion = true;
  split.prepare_for_processing(e.gpu_space, e.stream());
  REQUIRE(split.stolen_table == nullptr);
  REQUIRE(split.converted_table_steal_pending);
  REQUIRE(split.stolen_table_bytes > 0);
  REQUIRE(split.get_estimated_size_in_bytes() == split.stolen_table_bytes);

  const int32_t* first_source_data;
  const int32_t* second_source_data;
  {
    auto ro            = batch->to_read_only();
    auto source_view   = sirius::get_cudf_table_view(ro);
    first_source_data  = source_view.column(0).data<int32_t>();
    second_source_data = source_view.column(1).data<int32_t>();
  }
  auto const source_bytes = split.stolen_table_bytes;

  // An output-width mismatch is a non-mutating fallback; the caller may use the generic
  // materialize/filter/project path for a trailing filter-only column.
  bool mismatch_builder_called = false;
  auto mismatch                = split.transactionally_steal_converted_table(
    /*output_width=*/3,
    [&](cudf::table_view) {
      mismatch_builder_called = true;
      return replacements(3);
    },
    e.stream());
  REQUIRE(mismatch == nullptr);
  REQUIRE_FALSE(mismatch_builder_called);
  REQUIRE(split.converted_table_steal_pending);
  REQUIRE_FALSE(split.stolen_table_consumed);

  // Build a real first replacement, then inject failure before the second. Unwinding must destroy
  // that D allocation and release the lock while leaving both columns of S exactly intact.
  bool first_replacement_built = false;
  REQUIRE_THROWS_AS(split.transactionally_steal_converted_table(
                      /*output_width=*/2,
                      [&](cudf::table_view source) -> replacements {
                        replacements built(2);
                        built[0]                = cudf::cast(source.column(0),
                                              cudf::data_type{cudf::type_id::INT64},
                                              e.stream(),
                                              e.gpu_space->get_default_allocator());
                        first_replacement_built = true;
                        throw rmm::out_of_memory("injected second cast OOM");
                      },
                      e.stream()),
                    rmm::out_of_memory);
  e.stream().synchronize();
  REQUIRE(first_replacement_built);
  REQUIRE(split.converted_table_steal_pending);
  REQUIRE_FALSE(split.stolen_table_consumed);
  REQUIRE(split.stolen_table_bytes == source_bytes);
  {
    auto ro          = batch->to_read_only();
    auto source_view = sirius::get_cudf_table_view(ro);
    REQUIRE(source_view.column(0).data<int32_t>() == first_source_data);
    REQUIRE(source_view.column(1).data<int32_t>() == second_source_data);
    REQUIRE(ro.get_data()->get_size_in_bytes() == source_bytes);
    REQUIRE(to_host_column<int32_t>(source_view, 0) == first_values);
    REQUIRE(to_host_column<int32_t>(source_view, 1) == second_values);
  }

  // The scheduler may retry on another stream. Re-prepare preserves the pending transaction; the
  // transaction rebinds the already-plain wrapper before casting only the first column.
  rmm::cuda_stream retry_stream;
  auto const retry = retry_stream.view();
  split.prepare_for_processing(e.gpu_space, retry);
  REQUIRE(split.stolen_table == nullptr);
  REQUIRE(split.converted_table_steal_pending);

  // A successful retry commits once. The null second replacement moves its source column verbatim.
  auto out = split.transactionally_steal_converted_table(
    /*output_width=*/2,
    [&](cudf::table_view source) {
      replacements built(2);
      built[0] = cudf::cast(source.column(0),
                            cudf::data_type{cudf::type_id::INT64},
                            retry,
                            e.gpu_space->get_default_allocator());
      return built;
    },
    retry);
  REQUIRE(out != nullptr);
  retry.synchronize();
  auto const output_view = out->view();
  REQUIRE(output_view.column(0).type().id() == cudf::type_id::INT64);
  REQUIRE(output_view.column(1).type().id() == cudf::type_id::INT32);
  REQUIRE(static_cast<const void*>(output_view.column(0).data<int64_t>()) != first_source_data);
  REQUIRE(output_view.column(1).data<int32_t>() == second_source_data);
  REQUIRE(to_host_column<int64_t>(output_view, 0) ==
          std::vector<int64_t>(first_values.begin(), first_values.end()));
  REQUIRE(to_host_column<int32_t>(output_view, 1) == second_values);
  REQUIRE_FALSE(split.converted_table_steal_pending);
  REQUIRE(split.stolen_table_consumed);
  {
    auto ro = batch->to_read_only();
    REQUIRE(ro.get_data()->get_size_in_bytes() == 0);
  }

  bool consumed_builder_called = false;
  auto consumed                = split.transactionally_steal_converted_table(
    /*output_width=*/2,
    [&](cudf::table_view) {
      consumed_builder_called = true;
      return replacements(2);
    },
    retry);
  REQUIRE(consumed == nullptr);
  REQUIRE_FALSE(consumed_builder_called);
}

TEST_CASE("transactional steal refuses a downgraded wrapper without mutating it",
          "[cached_serving][scan_manager]")
{
  auto& e = env();
  std::vector<int32_t> const values{50, 51, 52, 53};
  using replacements = sirius::op::scan::scan_operator_input::converted_column_replacements;

  auto batch = make_host_batch(e, values);
  sirius::op::scan::scan_operator_input split{batch};
  split.needs_carrier_conversion = true;
  split.prepare_for_processing(e.gpu_space, e.stream());
  REQUIRE(split.converted_table_steal_pending);

  // A mid-query tier downgrade moves the armed wrapper's data off the GPU; the steal must refuse
  // it before any mutation (the refusal precedes the dealloc-stream rebind).
  {
    auto mut = batch->to_mutable();
    mut.convert_to<cucascade::host_data_representation>(
      sirius::converter_registry::get(), e.host_space, e.stream());
  }
  e.stream().synchronize();

  bool builder_called = false;
  auto refused        = split.transactionally_steal_converted_table(
    /*output_width=*/1,
    [&](cudf::table_view) {
      builder_called = true;
      return replacements(1);
    },
    e.stream());
  REQUIRE(refused == nullptr);
  REQUIRE_FALSE(builder_called);
  REQUIRE(split.converted_table_steal_pending);
  REQUIRE_FALSE(split.stolen_table_consumed);
}

TEST_CASE("transactional steal serves decode-row-filtered splits but refuses predicate columns",
          "[cached_serving][scan_manager]")
{
  auto& e = env();
  std::vector<int32_t> const values{40, 41, 42, 43};
  using replacements = sirius::op::scan::scan_operator_input::converted_column_replacements;

  auto batch = make_host_batch(e, values);
  sirius::op::scan::scan_operator_input split{batch};
  split.row_filter_pending       = true;
  split.pushdown_row_filtered    = true;
  split.needs_carrier_conversion = true;
  split.prepare_for_processing(e.gpu_space, e.stream());
  REQUIRE(split.converted_table_steal_pending);

  // A predicate-substituted column means the source is not values-only; the gate must refuse
  // without invoking the builder.
  split.pushdown_predicate_columns = {0};
  bool builder_called              = false;
  auto refused                     = split.transactionally_steal_converted_table(
    /*output_width=*/1,
    [&](cudf::table_view) {
      builder_called = true;
      return replacements(1);
    },
    e.stream());
  REQUIRE(refused == nullptr);
  REQUIRE_FALSE(builder_called);
  REQUIRE(split.converted_table_steal_pending);

  // A values-only decode-filtered split commits like the unfiltered case.
  split.pushdown_predicate_columns.clear();
  auto out = split.transactionally_steal_converted_table(
    /*output_width=*/1,
    [&](cudf::table_view source) {
      replacements built(1);
      built[0] = cudf::cast(source.column(0),
                            cudf::data_type{cudf::type_id::INT64},
                            e.stream(),
                            e.gpu_space->get_default_allocator());
      return built;
    },
    e.stream());
  REQUIRE(out != nullptr);
  e.stream().synchronize();
  REQUIRE(out->view().column(0).type().id() == cudf::type_id::INT64);
  REQUIRE(to_host_column<int64_t>(out->view(), 0) ==
          std::vector<int64_t>(values.begin(), values.end()));
  REQUIRE(split.stolen_table_consumed);
  REQUIRE_FALSE(split.converted_table_steal_pending);
}

// Compressed chunks are opaque blobs, but the carrier fold never needs to open them: the pin
// driver recorded each chunk's compress-input types in column_storage, and by Simpatico's
// round-trip contract decompression reproduces exactly those types. These tests pin that the
// fold answers identically for compressed storage forms on both tiers.
TEST_CASE("pinned_column_narrow_carrier derives from recorded metadata on compressed entries",
          "[cached_serving][scan_manager][compression]")
{
  auto& e = env();
  auto const int64{cudf::data_type{cudf::type_id::INT64}};
  auto const int32{cudf::data_type{cudf::type_id::INT32}};

  SECTION("compression-enabled device entry (device_chunks form)")
  {
    auto entry           = make_device_chunks_entry(*e.gpu_space, /*n_chunks=*/2, /*rows=*/4);
    entry.column_storage = {meta_row(int32, {true, true, true}, int64),
                            meta_row(int32, {true, true, true}, int64)};
    auto const target =
      sirius::scan_manager::pinned_column_narrow_carrier(entry, /*entry_position=*/1, int64);
    REQUIRE(target.has_value());
    REQUIRE(*target == int32);
  }

  SECTION("compressed host chunk (empty blob -- nothing here decompresses)")
  {
    pinned_entry entry;
    entry.tier         = cucascade::memory::Tier::HOST;
    entry.memory_space = e.host_space;
    set_cached_columns(entry, {"k"});
    entry.host_chunks.emplace_back(std::make_shared<sirius::compressed_host_representation>(
      *e.host_space,
      std::make_shared<sirius::pinned_compressed_blob>(),
      std::vector<std::string>{"k"},
      /*compressed_bytes=*/64,
      /*uncompressed_bytes=*/256,
      /*num_rows=*/4));
    entry.num_rows       = 4;
    entry.column_storage = {{narrow_meta(int32, int64)}};

    auto const target =
      sirius::scan_manager::pinned_column_narrow_carrier(entry, /*entry_position=*/0, int64);
    REQUIRE(target.has_value());
    REQUIRE(*target == int32);

    // An empty matrix (nothing recorded) keeps the column native.
    entry.column_storage = {};
    REQUIRE_FALSE(
      sirius::scan_manager::pinned_column_narrow_carrier(entry, /*entry_position=*/0, int64)
        .has_value());
  }
}

TEST_CASE("validate_pinned_entry_for_serving checks the device-entry matrix shape",
          "[cached_serving][scan_manager][compression]")
{
  auto& e    = env();
  auto entry = make_device_chunks_entry(*e.gpu_space, /*n_chunks=*/2, /*rows=*/4);
  auto const int32{cudf::data_type{cudf::type_id::INT32}};

  SECTION("empty matrix (all-native compat) passes")
  {
    REQUIRE_NOTHROW(validate_pinned_entry_for_serving(entry, std::vector<std::size_t>{0, 1, 2}));
  }

  SECTION("matrix with the wrong chunk count throws")
  {
    entry.column_storage.assign(1, meta_row(int32, {false, false, false}));
    REQUIRE_THROWS_AS(validate_pinned_entry_for_serving(entry, std::vector<std::size_t>{0, 1, 2}),
                      std::invalid_argument);
  }

  SECTION("matrix with the wrong column count throws")
  {
    entry.column_storage.assign(2, meta_row(int32, {false, false}));
    REQUIRE_THROWS_AS(validate_pinned_entry_for_serving(entry, std::vector<std::size_t>{0, 1, 2}),
                      std::invalid_argument);
  }
}

// The insert-time cross-check every pin path shares. Each insert supplies a callable that reads
// its own storage form; the three callables here mirror those, so the rule they collapse to --
// a recorded carrier must equal the stored type wherever storage can report one -- is pinned
// once for all three rather than once per insert.
TEST_CASE("validate_recorded_column_storage cross-checks recorded carriers against storage",
          "[cached_serving][scan_manager][compression]")
{
  auto const int64{cudf::data_type{cudf::type_id::INT64}};
  auto const int32{cudf::data_type{cudf::type_id::INT32}};
  constexpr std::string_view kContext = "[test]";

  // Two chunks x two columns, everything recorded INT32.
  sirius::pinned_column_storage_matrix matrix{carrier_row({int32, int32}),
                                              carrier_row({int32, int32})};

  SECTION("a matrix matching the stored types passes")
  {
    auto stored = [&](std::size_t, std::size_t) { return std::optional<cudf::data_type>{int32}; };
    REQUIRE_NOTHROW(
      sirius::scan_manager::validate_recorded_column_storage(matrix, 2, 2, kContext, stored));
  }

  SECTION("GPU tables: a contradictory recorded carrier throws")
  {
    // insert_pinned_entry reads cudf::table::get_column(i).type(); a null chunk table reports
    // nothing.
    auto stored = [&](std::size_t chunk, std::size_t column) {
      return chunk == 1 && column == 0 ? std::optional<cudf::data_type>{int64} : std::nullopt;
    };
    REQUIRE_THROWS_AS(
      sirius::scan_manager::validate_recorded_column_storage(matrix, 2, 2, kContext, stored),
      std::invalid_argument);
  }

  SECTION("host chunks: a contradictory recorded carrier throws")
  {
    // insert_pinned_entry_host reads the host column metadata; a compressed chunk reports
    // nothing, and its recorded cells are trusted.
    auto stored = [&](std::size_t chunk, std::size_t column) {
      if (chunk == 0) { return std::optional<cudf::data_type>{}; }  // compressed chunk
      return column == 1 ? std::optional<cudf::data_type>{int64} : std::optional<cudf::data_type>{};
    };
    REQUIRE_THROWS_AS(
      sirius::scan_manager::validate_recorded_column_storage(matrix, 2, 2, kContext, stored),
      std::invalid_argument);
  }

  SECTION("device chunks: a contradictory recorded carrier throws")
  {
    // insert_pinned_entry_device reads cudf::column::type() on the uncompressed form.
    auto stored = [&](std::size_t, std::size_t column) {
      return column == 0 ? std::optional<cudf::data_type>{int32}
                         : std::optional<cudf::data_type>{int64};
    };
    REQUIRE_THROWS_AS(
      sirius::scan_manager::validate_recorded_column_storage(matrix, 2, 2, kContext, stored),
      std::invalid_argument);
  }

  SECTION("an opaque chunk is trusted, not cross-checked")
  {
    // Every cell unreadable -- an all-compressed pin. The recorded carriers stand.
    auto stored = [](std::size_t, std::size_t) { return std::optional<cudf::data_type>{}; };
    REQUIRE_NOTHROW(
      sirius::scan_manager::validate_recorded_column_storage(matrix, 2, 2, kContext, stored));
  }

  SECTION("insertion requires coverage while serving tolerates an empty matrix")
  {
    auto stored = [&](std::size_t, std::size_t) { return std::optional<cudf::data_type>{int32}; };
    REQUIRE_THROWS_AS(sirius::scan_manager::validate_recorded_column_storage(
                        sirius::pinned_column_storage_matrix{}, 2, 2, kContext, stored),
                      std::invalid_argument);
    REQUIRE_NOTHROW(sirius::scan_manager::validate_column_storage_shape(
      sirius::pinned_column_storage_matrix{}, 2, 2, kContext, /*allow_empty=*/true));

    // A zero-chunk pin records nothing and is not an error.
    REQUIRE_NOTHROW(sirius::scan_manager::validate_recorded_column_storage(
      sirius::pinned_column_storage_matrix{}, 0, 2, kContext, stored));
  }
}
