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

/// HOST-resident wrapper batch (single INT32 column) — the shape a host-pinned
/// chunk's per-query slice arrives in, which prepare_for_processing converts
/// to a fresh owned GPU table.
std::shared_ptr<cucascade::data_batch> make_host_batch(test_env& e,
                                                       std::vector<int32_t> const& values)
{
  auto shared = make_gpu_column(*e.gpu_space, values);
  std::vector<std::unique_ptr<cudf::column>> cols;
  cols.push_back(std::make_unique<cudf::column>(
    shared->view(), e.stream(), e.gpu_space->get_default_allocator()));
  cucascade::gpu_table_representation gpu_repr(
    std::make_unique<cudf::table>(std::move(cols)), *e.gpu_space, e.stream());
  auto host_repr = sirius::converter_registry::get().convert<cucascade::host_data_representation>(
    gpu_repr, e.host_space, e.stream());
  e.stream().synchronize();
  return cucascade::data_batch::make(sirius::get_next_batch_id(), std::move(host_repr));
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

/// Copy the single INT32 column of @p table back to host for content checks.
std::vector<int32_t> to_host(cudf::table_view const& view)
{
  std::vector<int32_t> out(static_cast<std::size_t>(view.num_rows()));
  cudaMemcpy(out.data(),
             view.column(0).data<int32_t>(),
             sizeof(int32_t) * out.size(),
             cudaMemcpyDeviceToHost);
  return out;
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
  std::size_t served{0};
  bool throw_when_exhausted{false};

  databatch_provider::batch get_next_batch() override
  {
    if (served < batches.size()) {
      auto idx = served++;
      return {batches[idx],
              idx < masks.size() ? masks[idx] : sirius::scan_manager::mvcc_chunk_mask{}};
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
  provider.batches = {make_test_batch(e, 4), make_test_batch(e, 4)};
  provider.masks   = {mask0};  // second batch deliberately mask-less

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

  auto second = connector.get_next_split();
  REQUIRE(second.has_value());
  auto* in1 = dynamic_cast<sirius::op::scan::scan_operator_input*>(second->get());
  REQUIRE(in1 != nullptr);
  REQUIRE_FALSE(in1->mvcc_keep_mask.has_mask());
  REQUIRE(in1->row_filter_pending);  // per-op flag: stamped on every split

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
  split.prepare_for_processing(e.gpu_space, e.stream());
  REQUIRE(split.stolen_table == nullptr);
  REQUIRE(split.stolen_table_bytes == 0);

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

TEST_CASE("prepare_for_processing skips the steal for masked or row-filtered splits",
          "[cached_serving][scan_manager]")
{
  auto& e = env();
  std::vector<int32_t> const values{20, 21, 22, 23};

  SECTION("mvcc keep-mask pending")
  {
    auto batch = make_host_batch(e, values);
    sirius::op::scan::scan_operator_input split{batch};
    split.mvcc_keep_mask = make_test_mask(values.size());
    split.prepare_for_processing(e.gpu_space, e.stream());
    REQUIRE(split.stolen_table == nullptr);
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
    auto ro = batch->to_read_only();
    REQUIRE(ro.get_current_tier() == cucascade::memory::Tier::GPU);
    REQUIRE(ro.get_data()->get_size_in_bytes() > 0);
  }
}
