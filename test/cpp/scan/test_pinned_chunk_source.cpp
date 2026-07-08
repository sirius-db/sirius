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

// #819 PR2: pinned-entry serving rides the split_provider / metadata-task
// machinery as a pinned-serving MODE of the format ingestibles, powered by
// pinned_chunk_source.
//
// Unit gates cover the pieces bottom-up: build_pinned_chunk_source extraction
// + validation (malformed entries throw -> disk fallback, replacing the old
// silent truncation), pinned_chunk_source chunk claim + resident-batch
// assembly (column subset/permutation, per-chunk memory_space, HOST slice),
// the pass-through cached_batch_coalescer, and — through a minimal
// pinned-only gpu_ingestible — the UNMODIFIED split_provider::run driving
// pinned work items on real pool threads with the end-of-stream sentinel.
//
// The integration gate pins a parquet surface and SELECTs through it on a
// single-GPU env, asserting the scan-manager log marker that only the
// pinned serving path emits.

#include "operator/mgpu_test_utils.hpp"
#include "operator/operator_test_utils.hpp"

#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/table/table.hpp>

#include <rmm/cuda_stream.hpp>

#include <cuda_runtime.h>

#include <catch.hpp>
#include <cucascade/cudf/gpu_data_representation.hpp>
#include <cucascade/cudf/host_data_representation.hpp>
#include <cucascade/data/data_batch.hpp>
#include <cucascade/memory/memory_space.hpp>
#include <data/sirius_converter_registry.hpp>
#include <duckdb.hpp>
#include <exec/scoped_dispatcher.hpp>
#include <exec/thread_pool.hpp>
#include <op/scan/gpu_ingestible.hpp>
#include <op/scan/pinned_chunk_source.hpp>
#include <scan_manager/sirius_scan_manager.hpp>
#include <scan_manager/split_provider.hpp>
#include <spdlog/spdlog.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <memory>
#include <mutex>
#include <set>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace fs = std::filesystem;

using sirius::op::scan::cached_batch_coalescer;
using sirius::op::scan::cached_scan_info;
using sirius::op::scan::pinned_chunk_source;
using sirius::scan_manager::build_pinned_chunk_source;
using sirius::scan_manager::pinned_entry;

namespace {

// Shared test environment: memory manager (+ converter registry) initialized
// once for every unit gate in this file. The converter needs a non-default
// stream (same constraint test_convertible_data_batch documents).
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

/// Run every remaining work item of @p source on the calling thread and
/// return the produced cached_scan_infos.
std::vector<std::unique_ptr<sirius::op::scan::scan_info>> drain(pinned_chunk_source& source)
{
  std::vector<std::unique_ptr<sirius::op::scan::scan_info>> infos;
  while (auto work = source.next_work_item()) {
    infos.push_back(work());
  }
  return infos;
}

/// Sort emitted infos by chunk index so assertions are order-independent.
std::vector<cached_scan_info*> as_cached_sorted(
  std::vector<std::unique_ptr<sirius::op::scan::scan_info>>& infos)
{
  std::vector<cached_scan_info*> cached;
  for (auto& info : infos) {
    auto* c = dynamic_cast<cached_scan_info*>(info.get());
    REQUIRE(c != nullptr);
    cached.push_back(c);
  }
  std::sort(cached.begin(), cached.end(), [](auto* a, auto* b) {
    return a->chunk_index() < b->chunk_index();
  });
  return cached;
}

struct dummy_scan_info final : sirius::op::scan::scan_info {};

struct dummy_table_info final : sirius::op::scan::ingestible_table_info {
  [[nodiscard]] std::span<std::string const> column_names() const override { return {}; }
  [[nodiscard]] std::span<std::string const> file_paths() const override { return {}; }
};

/// Minimal ingestible that ONLY serves pinned chunks — the shape every format
/// ingestible takes in pinned mode. Lets the composition gate drive the
/// UNMODIFIED split_provider over pinned work items without a real
/// parquet/duckdb bind.
struct pinned_only_ingestible final : sirius::op::scan::gpu_ingestible {
  bool serve_from_pinned_chunks(std::unique_ptr<pinned_chunk_source> source) override
  {
    _pinned = std::move(source);
    return true;
  }

  [[nodiscard]] bool has_processed_all_metadata() const override
  {
    return !_pinned || !_pinned->has_more();
  }

  metadata_scan_task_t next_split_provider(sirius::io::ioctx_resolver /*resolve*/) override
  {
    return _pinned ? _pinned->next_work_item() : nullptr;
  }

  std::unique_ptr<sirius::op::scan::batch_coalescer> create_batch_coalescer() const override
  {
    return std::make_unique<cached_batch_coalescer>();
  }

  sirius::op::scan::filtered_table materialize_metadata_to_table(
    sirius::op::scan::scan_info const&,
    cucascade::memory::memory_space const&,
    rmm::cuda_stream_view) override
  {
    throw std::logic_error("unreachable: resident batches never materialize metadata");
  }

  std::unique_ptr<cudf::table> post_filter_and_project(sirius::op::scan::filtered_table&&,
                                                       cucascade::memory::memory_space const&,
                                                       rmm::cuda_stream_view) override
  {
    throw std::logic_error("unreachable in this test");
  }

  [[nodiscard]] sirius::op::scan::ingestible_table_info const& table_info() const noexcept override
  {
    return _tinfo;
  }

  [[nodiscard]] std::vector<std::size_t> materialized_column_order() const override { return {}; }

 private:
  dummy_table_info _tinfo;
  std::unique_ptr<pinned_chunk_source> _pinned;
};

/// Drive provider.run() on a real pool the way start_metadata_processing
/// does, collecting everything the provider pushes plus the pushing threads.
struct run_result {
  std::vector<std::unique_ptr<sirius::op::scan::scan_info>> infos;
  int empties    = 0;
  int exceptions = 0;
  std::set<std::thread::id> value_threads;
};

run_result drive(sirius::scan_manager::split_provider& provider)
{
  sirius::exec::static_thread_pool pool(4, "pinned_chunk_test");
  run_result out;
  {
    sirius::exec::scoped_dispatcher dispatcher(pool, 4);
    std::mutex mtx;
    provider.run(dispatcher, [&](sirius::scan_manager::split_provider::value_type&& entry) {
      std::lock_guard lk(mtx);
      if (entry.has_exception()) {
        ++out.exceptions;
      } else if (entry.is_empty()) {
        ++out.empties;
      } else {
        out.value_threads.insert(std::this_thread::get_id());
        out.infos.push_back(std::move(entry).value());
      }
    });
    dispatcher.wait_for_all();
    // The end-of-stream sentinel fires when the last work item drops its
    // completion token, which can trail wait_for_all by an instant.
    for (int spin = 0; spin < 5000; ++spin) {
      {
        std::lock_guard lk(mtx);
        if (out.empties + out.exceptions > 0) { break; }
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
  }
  return out;
}

}  // namespace

//===----------------------------------------------------------------------===//
// build_pinned_chunk_source + pinned_chunk_source unit gates
//===----------------------------------------------------------------------===//

TEST_CASE("pinned_chunk_source serves GPU chunks with a permuted column subset",
          "[pinned_chunk_source][scan_manager]")
{
  auto& e                       = env();
  constexpr std::size_t kChunks = 3;
  constexpr std::size_t kRows   = 8;
  auto entry                    = make_gpu_entry(*e.gpu_space, kChunks, kRows);

  // Permuted subset: serve {w, k} out of {k, v, w}.
  std::vector<std::size_t> selection{2, 0};
  auto source = build_pinned_chunk_source(entry, selection);
  REQUIRE(source->num_chunks() == kChunks);
  REQUIRE(source->has_more());

  auto infos = drain(*source);
  REQUIRE_FALSE(source->has_more());
  REQUIRE(source->next_work_item() == nullptr);  // exhausted claims stay null
  REQUIRE(infos.size() == kChunks);

  auto cached = as_cached_sorted(infos);
  for (std::size_t c = 0; c < kChunks; ++c) {
    REQUIRE(cached[c]->chunk_index() == c);
    auto batch = cached[c]->take_batch();
    REQUIRE(batch != nullptr);
    {
      auto ro = batch->to_read_only();
      REQUIRE(ro.get_memory_space() == entry.chunk_memory_spaces.at(c));
    }
    auto view = sirius::get_cudf_table_view(*batch);
    REQUIRE(view.num_columns() == static_cast<cudf::size_type>(selection.size()));
    REQUIRE(view.num_rows() == static_cast<cudf::size_type>(kRows));
    auto w = sirius::test::operator_utils::copy_column_to_host<int32_t>(view.column(0));
    auto k = sirius::test::operator_utils::copy_column_to_host<int32_t>(view.column(1));
    for (std::size_t r = 0; r < kRows; ++r) {
      REQUIRE(w[r] == cell(c, 2, r));
      REQUIRE(k[r] == cell(c, 0, r));
    }
  }
}

TEST_CASE("pinned_chunk_source serves HOST chunks as sliced resident batches",
          "[pinned_chunk_source][scan_manager]")
{
  auto& e                       = env();
  constexpr std::size_t kChunks = 2;
  constexpr std::size_t kRows   = 16;
  auto entry                    = make_host_entry(e, kChunks, kRows);

  std::vector<std::size_t> selection{1};  // just {v}
  auto source = build_pinned_chunk_source(entry, selection);
  REQUIRE(source->num_chunks() == kChunks);

  auto infos = drain(*source);
  REQUIRE(infos.size() == kChunks);

  auto cached = as_cached_sorted(infos);
  for (std::size_t c = 0; c < kChunks; ++c) {
    REQUIRE(cached[c]->chunk_index() == c);
    auto batch = cached[c]->take_batch();
    REQUIRE(batch != nullptr);
    auto ro = batch->to_read_only();
    REQUIRE(ro.get_memory_space()->get_tier() == cucascade::memory::Tier::HOST);
    REQUIRE(ro.get_data()->get_size_in_bytes() > 0);
  }
}

TEST_CASE("build_pinned_chunk_source: zero-chunk entry yields an empty source",
          "[pinned_chunk_source][scan_manager]")
{
  auto& e = env();
  pinned_entry entry;
  entry.cache_info.names = {"k"};
  entry.tier             = cucascade::memory::Tier::GPU;
  entry.memory_space     = e.gpu_space;

  auto source = build_pinned_chunk_source(entry, std::vector<std::size_t>{0});
  REQUIRE(source->num_chunks() == 0);
  REQUIRE_FALSE(source->has_more());
  REQUIRE(source->next_work_item() == nullptr);
}

TEST_CASE("build_pinned_chunk_source refuses malformed entries",
          "[pinned_chunk_source][scan_manager]")
{
  auto& e = env();

  SECTION("per-column chunk counts disagree")
  {
    auto entry = make_gpu_entry(*e.gpu_space, 3, 4);
    entry.data_batches_by_column["v"].pop_back();  // v now has 2 chunks, k/w have 3
    REQUIRE_THROWS_AS(build_pinned_chunk_source(entry, std::vector<std::size_t>{0, 1, 2}),
                      std::runtime_error);
  }

  SECTION("chunk_memory_spaces does not cover every chunk")
  {
    auto entry = make_gpu_entry(*e.gpu_space, 3, 4);
    entry.chunk_memory_spaces.resize(1);
    REQUIRE_THROWS_AS(build_pinned_chunk_source(entry, std::vector<std::size_t>{0}),
                      std::runtime_error);
  }

  SECTION("null host chunk")
  {
    auto entry           = make_host_entry(e, 2, 4);
    entry.host_chunks[1] = nullptr;
    REQUIRE_THROWS_AS(build_pinned_chunk_source(entry, std::vector<std::size_t>{0}),
                      std::runtime_error);
  }
}

//===----------------------------------------------------------------------===//
// cached_batch_coalescer unit gates
//===----------------------------------------------------------------------===//

TEST_CASE("cached_batch_coalescer passes cached splits through unchanged",
          "[pinned_chunk_source][scan_manager]")
{
  cached_batch_coalescer c;
  auto out = c.push(std::make_unique<cached_scan_info>(nullptr, /*chunk_index=*/5));
  REQUIRE(out.size() == 1);
  auto* cached = dynamic_cast<cached_scan_info*>(out[0].get());
  REQUIRE(cached != nullptr);
  REQUIRE(cached->chunk_index() == 5);
  REQUIRE(c.flush().empty());
}

TEST_CASE("cached_batch_coalescer drops foreign splits and never emits on flush",
          "[pinned_chunk_source][scan_manager]")
{
  cached_batch_coalescer c;
  REQUIRE(c.push(std::make_unique<dummy_scan_info>()).empty());
  // flush must emit nothing at all — a zero-chunk pin relies on closing with
  // zero splits (no template-split fallback like the disk coalescers).
  REQUIRE(c.flush().empty());
}

//===----------------------------------------------------------------------===//
// Plain-provider composition gate: the UNMODIFIED split_provider drives a
// pinned-mode ingestible's work items on dispatcher threads.
//===----------------------------------------------------------------------===//

TEST_CASE("split_provider drives pinned-mode work items on pool threads",
          "[pinned_chunk_source][scan_manager]")
{
  auto& e                       = env();
  constexpr std::size_t kChunks = 3;
  auto entry                    = make_gpu_entry(*e.gpu_space, kChunks, /*rows=*/4);

  pinned_only_ingestible ingestible;
  REQUIRE(ingestible.serve_from_pinned_chunks(
    build_pinned_chunk_source(entry, std::vector<std::size_t>{0, 1, 2})));
  REQUIRE_FALSE(ingestible.has_processed_all_metadata());

  sirius::scan_manager::split_provider provider(ingestible, /*resolve=*/{});
  auto out = drive(provider);

  REQUIRE(out.exceptions == 0);
  REQUIRE(out.empties >= 1);  // end-of-stream sentinel
  REQUIRE(out.infos.size() == kChunks);
  REQUIRE(ingestible.has_processed_all_metadata());
  // Work items ran on pool threads, not the driving thread.
  REQUIRE_FALSE(out.value_threads.empty());
  REQUIRE(out.value_threads.find(std::this_thread::get_id()) == out.value_threads.end());

  auto cached = as_cached_sorted(out.infos);
  for (std::size_t c = 0; c < kChunks; ++c) {
    REQUIRE(cached[c]->chunk_index() == c);
    REQUIRE(cached[c]->take_batch() != nullptr);
  }
}

TEST_CASE("split_provider emits only the sentinel for a zero-chunk pinned source",
          "[pinned_chunk_source][scan_manager]")
{
  auto& e = env();
  pinned_entry entry;
  entry.cache_info.names = {"k"};
  entry.tier             = cucascade::memory::Tier::GPU;
  entry.memory_space     = e.gpu_space;

  pinned_only_ingestible ingestible;
  REQUIRE(ingestible.serve_from_pinned_chunks(
    build_pinned_chunk_source(entry, std::vector<std::size_t>{0})));

  sirius::scan_manager::split_provider provider(ingestible, /*resolve=*/{});
  auto out = drive(provider);

  REQUIRE(out.infos.empty());
  REQUIRE(out.exceptions == 0);
  REQUIRE(out.empties >= 1);  // the connector still gets closed downstream
}

//===----------------------------------------------------------------------===//
// Integration gate: a pinned scan is served through the pinned-mode
// ingestible + plain provider (the log marker below is only emitted on that
// path in prepare_for_query).
//===----------------------------------------------------------------------===//

TEST_CASE("pinned serving handles pin_table scans end to end",
          "[pinned_chunk_source][scan_manager][pin_table]")
{
  namespace mgpu = sirius::test::mgpu;

  auto tmp = fs::temp_directory_path() / ("sirius-pinned-source-" + std::to_string(::getpid()));
  std::error_code ec;
  fs::remove_all(tmp, ec);
  fs::create_directories(tmp);

  mgpu::generate_parquet_surface(
    tmp, "SELECT range AS k, range * 2 AS v FROM range(100000)", /*num_files=*/2);

  mgpu::mgpu_env_params params;
  params.num_gpus = 1;
  auto yaml_path  = tmp / "pinned_source.yaml";
  mgpu::write_mgpu_yaml(yaml_path, params);

  mgpu::scoped_log_dir logs(tmp / "logs");
  mgpu::scoped_mgpu_env env(yaml_path);
  auto con = env.make_connection();

  auto glob = mgpu::parquet_glob(tmp);
  auto pin  = con.Query("CALL pin_table('" + glob + "', tier='gpu', name='pinned_source_gate');");
  REQUIRE(pin);
  if (pin->HasError()) { UNSCOPED_INFO("pin_table error: " << pin->GetError()); }
  REQUIRE_FALSE(pin->HasError());

  auto res =
    con.Query("CALL gpu_execution(\"SELECT MAX(k), SUM(v) FROM read_parquet('" + glob + "')\");");
  REQUIRE(res);
  if (res->HasError()) { UNSCOPED_INFO("gpu_execution error: " << res->GetError()); }
  REQUIRE_FALSE(res->HasError());
  REQUIRE(res->GetValue(0, 0).GetValue<int64_t>() == 99999);

  auto unpin = con.Query("CALL unpin_table('pinned_source_gate');");
  REQUIRE(unpin);
  REQUIRE_FALSE(unpin->HasError());

  // Flush spdlog's file sink before reading the log dir (default flush cadence
  // is seconds; the query completes well under it).
  if (auto logger = spdlog::default_logger()) { logger->flush(); }

  bool marker_found = false;
  for (auto const& file : fs::directory_iterator(logs.path())) {
    std::ifstream in(file.path());
    std::string line;
    while (std::getline(in, line)) {
      if (line.find("served via pinned chunks") != std::string::npos) {
        marker_found = true;
        break;
      }
    }
    if (marker_found) { break; }
  }
  INFO("expected the prepare_for_query pinned-serving marker in " << logs.path());
  REQUIRE(marker_found);

  fs::remove_all(tmp, ec);
}
