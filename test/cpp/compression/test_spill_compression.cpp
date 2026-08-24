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

// Tests for spill-path compression (GPU->HOST / GPU->DISK downgrades).
//
//  [compression][spill]  — the roundtrip cases require a GPU (Simpatico JIT + cuDF)
//
//  Spill plans are keyed by the source shared_data_repository* (one plan per
//  query-graph edge), so every case that exercises convert() supplies a repo
//  pointer and pre-seeds a plan for it — except the explore case, which asserts
//  that the first spill from an unseen edge discovers and caches one.

#include "operator/operator_test_utils.hpp"

#include <cudf/reduction.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/table/table.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <rmm/cuda_stream.hpp>

#include <cuda_runtime.h>

#include <catch.hpp>
#include <compression/compressed_disk_representation.hpp>
#include <compression/compressed_representation.hpp>
#include <compression/compression_converters.hpp>
#include <compression/plan_register.hpp>
#include <compression/spill_context.hpp>
#include <cucascade/cudf/gpu_data_representation.hpp>
#include <cucascade/cudf/host_data_representation.hpp>
#include <cucascade/data/data_batch.hpp>
#include <cucascade/data/data_repository.hpp>
#include <cucascade/data/disk_data_representation.hpp>
#include <cucascade/memory/reservation_manager_configurator.hpp>
#include <data/convertible_data_batch.hpp>
#include <data/sirius_converter_registry.hpp>

#include <algorithm>
#include <array>
#include <filesystem>
#include <memory>
#include <numeric>
#include <vector>

namespace fs = std::filesystem;

namespace {

// ── Utilities ────────────────────────────────────────────────────────────────

bool has_gpu()
{
  int count = 0;
  cudaGetDeviceCount(&count);
  return count >= 1;
}

/// A single-column INT32 plan that genuinely shrinks the data: bitpacking a
/// [0, n) run needs ~13 of 32 bits. A plain "delta -> differences" would NOT
/// work here — it emits full-width differences, so the ratio gate rejects it
/// (that is exactly what test_compression.cpp uses as its "saves too little"
/// fixture).
constexpr auto kOneColDsl = "input -> bitpack -> chunk_min, chunk_count, chunk_bits, packed\n";

/// A two-column plan DSL — paired with a 1-column table to force a column-count
/// mismatch inside Simpatico.
constexpr auto kTwoColDsl =
  "input -> bitpack -> chunk_min, chunk_count, chunk_bits, packed\n---\n"
  "input -> bitpack -> chunk_min, chunk_count, chunk_bits, packed\n";

/// A plan that compresses nothing (full-width output), for the ratio-gate test.
constexpr auto kNonCompressingDsl = "input -> delta -> differences\n";

// ── Shared test environment ──────────────────────────────────────────────────

struct spill_test_env {
  fs::path tmp_dir;
  std::unique_ptr<sirius::memory::sirius_memory_reservation_manager> mgr;
  cucascade::memory::memory_space* gpu_space  = nullptr;
  cucascade::memory::memory_space* host_space = nullptr;
  cucascade::memory::memory_space* disk_space = nullptr;
  rmm::cuda_stream conv_stream;

  spill_test_env() : tmp_dir(fs::temp_directory_path() / "sirius_spill_comp_test")
  {
    fs::create_directories(tmp_dir);

    sirius::converter_registry::reset_for_testing();

    cucascade::memory::reservation_manager_configurator builder;
    builder.set_number_of_gpus(1)
      .set_gpu_usage_limit(512ull << 20)
      .set_reservation_fraction_per_gpu(0.75)
      .set_per_numa_region_capacity(512ull << 20)
      .use_gpu_id_as_host_id()
      .set_reservation_fraction_per_numa_region(0.75)
      .set_disk_mounting_point(0, 2ull << 30, tmp_dir.string());

    auto space_configs = builder.build();
    mgr =
      std::make_unique<sirius::memory::sirius_memory_reservation_manager>(std::move(space_configs));

    sirius::converter_registry::initialize();

    gpu_space        = mgr->get_memory_space(cucascade::memory::Tier::GPU, 0);
    host_space       = mgr->get_memory_space(cucascade::memory::Tier::HOST, 0);
    auto disk_spaces = mgr->get_memory_spaces_for_tier(cucascade::memory::Tier::DISK);
    if (!disk_spaces.empty()) {
      disk_space = const_cast<cucascade::memory::memory_space*>(disk_spaces.front());
    }
  }

  ~spill_test_env() { fs::remove_all(tmp_dir); }

  rmm::cuda_stream_view stream() { return conv_stream.view(); }
};

spill_test_env& env()
{
  static spill_test_env e;
  return e;
}

/// Distinct repository instances stand in for distinct query-graph edges. The
/// plan register only ever uses the pointer as a key, never dereferences it.
cucascade::shared_data_repository& repo_a()
{
  static cucascade::shared_data_repository r;
  return r;
}

cucascade::shared_data_repository& repo_b()
{
  static cucascade::shared_data_repository r;
  return r;
}

// ── Single-column shorthands ─────────────────────────────────────────────────
//
// The register tracks plans and verdicts per column; every batch these tests
// build has exactly one column, so these wrap the vector/span API.

/// Change threshold used throughout these tests (20%, the production default).
constexpr double kTestChangeThreshold = 0.20;

/// Install a single-column plan as if freshly explored.
///
/// @p ratio and the throughputs are what the explorer reported. They matter on a
/// re-install: the register keeps the cached plan unless a candidate performs
/// materially differently, so a test meaning to replace a plan must offer
/// distinct measurements, not merely a different DSL string.
void set_plan_1col(const cucascade::shared_data_repository* repo,
                   std::string dsl,
                   double ratio           = 2.0,
                   double compress_gbps   = 10.0,
                   double decompress_gbps = 20.0)
{
  std::vector<sirius::compression::plan_register::column_plan_candidate> candidates;
  candidates.push_back({std::move(dsl), ratio, compress_gbps, decompress_gbps});
  sirius::compression::plan_register::global().set_spill_plan(
    repo, std::move(candidates), kTestChangeThreshold);
}

/// Install per-column plans, all reporting the same measurements.
void set_plans(const cucascade::shared_data_repository* repo,
               std::vector<std::string> dsls,
               double ratio = 2.0)
{
  std::vector<sirius::compression::plan_register::column_plan_candidate> candidates;
  candidates.reserve(dsls.size());
  for (auto& dsl : dsls) {
    candidates.push_back({std::move(dsl), ratio, 10.0, 20.0});
  }
  sirius::compression::plan_register::global().set_spill_plan(
    repo, std::move(candidates), kTestChangeThreshold);
}

/// @param achieved_ratio what the plan in use measured on the batch; 0 leaves the
///        cached ratio untouched (the pre-existing behaviour of this helper).
void conclude_1col(const cucascade::shared_data_repository* repo,
                   sirius::compression::plan_register::spill_attempt_outcome outcome,
                   std::uint64_t base_interval,
                   std::uint32_t error_tolerance,
                   double achieved_ratio = 0.0)
{
  const std::array outcomes{
    sirius::compression::plan_register::spill_column_result{outcome, achieved_ratio}};
  sirius::compression::plan_register::global().conclude_spill_attempt(
    repo, outcomes, base_interval, error_tolerance);
}

/// The first (only) column's cached state.
sirius::compression::plan_register::column_plan_state col0_of(
  const cucascade::shared_data_repository* repo)
{
  auto state = sirius::compression::plan_register::global().resolve_spill_plan(repo);
  REQUIRE(state.has_value());
  REQUIRE_FALSE(state->columns.empty());
  return state->columns[0];
}

/// Reset spill state: clear every cached plan and enable spill compression with
/// a small explorer budget (the explore case is the only one that runs it).
///
/// min_batch_bytes is 0 so the size gate never fires: these batches are a few KB,
/// far under the production default, and the cases here exercise the compression
/// logic rather than the heuristic that decides a batch is too small to bother
/// with. The gate itself is covered separately.
void reset_spill_state(bool enabled = true, std::uint64_t replan_after_uses = 0)
{
  sirius::compression::plan_register::global().clear_all();
  sirius::compression::set_spill_compression_settings(enabled,
                                                      /*explore_beam_width=*/4,
                                                      /*explore_max_bytes=*/8ull << 20,
                                                      /*max_compressed_fraction=*/0.95,
                                                      replan_after_uses,
                                                      /*error_tolerance=*/1,
                                                      /*replan_change_threshold=*/0.20,
                                                      /*explore_sample_rows=*/0,
                                                      /*min_batch_bytes=*/0,
                                                      /*release_columns_early=*/false,
                                                      /*encode_reserve_fraction=*/0.5,
                                                      /*encode_min_headroom_fraction=*/0.0);
}

/// Create a 1-column INT32 GPU batch with a known pattern [0, 1, 2, ..., n-1].
/// A run of consecutive integers compresses well under delta, so the ratio gate
/// does not reject it.
std::shared_ptr<cucascade::data_batch> make_int32_gpu_batch(std::size_t n = 1000)
{
  std::vector<int32_t> vals(n);
  std::iota(vals.begin(), vals.end(), 0);
  return sirius::test::operator_utils::make_numeric_batch(
    *env().gpu_space, vals, cudf::type_id::INT32);
}

/// Create a 2-column INT32 GPU batch, both columns holding [0, 1, ..., n-1].
/// Used to check that per-column verdicts are independent: the columns carry the
/// same data, so any difference in outcome comes from their differing plans.
std::shared_ptr<cucascade::data_batch> make_int32_gpu_batch_2col(std::size_t n)
{
  auto& space     = *env().gpu_space;
  auto mr         = sirius::test::operator_utils::get_resource_ref(space);
  auto stream     = sirius::test::operator_utils::default_stream();
  const auto size = static_cast<cudf::size_type>(n);

  std::vector<int32_t> vals(n);
  std::iota(vals.begin(), vals.end(), 0);

  std::vector<std::unique_ptr<cudf::column>> cols;
  for (int c = 0; c < 2; ++c) {
    auto col = cudf::make_numeric_column(
      cudf::data_type{cudf::type_id::INT32}, size, cudf::mask_state::UNALLOCATED, stream, mr);
    cudaMemcpy(col->mutable_view().data<int32_t>(),
               vals.data(),
               sizeof(int32_t) * vals.size(),
               cudaMemcpyHostToDevice);
    cols.push_back(std::move(col));
  }

  auto table = std::make_unique<cudf::table>(std::move(cols));
  auto repr =
    std::make_unique<cucascade::gpu_table_representation>(std::move(table), space, stream);
  return cucascade::data_batch::make(sirius::get_next_batch_id(), std::move(repr));
}

/// Sum column @p idx of a gpu_table_representation as int64.
int64_t gpu_col_sum_at(const cucascade::gpu_table_representation& rep, cudf::size_type idx)
{
  auto result = cudf::reduce(rep.get_table_view().column(idx),
                             *cudf::make_sum_aggregation<cudf::reduce_aggregation>(),
                             cudf::data_type{cudf::type_id::INT64},
                             cudf::get_default_stream(),
                             rmm::mr::get_current_device_resource_ref());
  return static_cast<cudf::numeric_scalar<int64_t>*>(result.get())
    ->value(cudf::get_default_stream());
}

/// Sum the first column of a gpu_table_representation as int64.
int64_t gpu_col_sum(const cucascade::gpu_table_representation& rep)
{
  return gpu_col_sum_at(rep, 0);
}

/// Expected SUM of [0, n).
int64_t expected_sum_of(std::size_t n)
{
  return static_cast<int64_t>(n) * (static_cast<int64_t>(n) - 1) / 2;
}

cucascade::memory::Tier get_tier(cucascade::data_batch& b)
{
  auto ro = b.to_read_only();
  return ro.get_memory_space()->get_tier();
}

bool is_compressed_host(cucascade::data_batch& b)
{
  auto ro = b.to_read_only();
  return dynamic_cast<const sirius::compressed_host_representation*>(ro.get_data()) != nullptr;
}

bool is_compressed_disk(cucascade::data_batch& b)
{
  auto ro = b.to_read_only();
  return dynamic_cast<const sirius::compressed_disk_representation*>(ro.get_data()) != nullptr;
}

/// Move `batch` into `space`, asserting the conversion happened.
void spill_to(std::shared_ptr<cucascade::data_batch> const& batch,
              cucascade::memory::memory_space* space,
              const cucascade::shared_data_repository* repo)
{
  sirius::convertible_data_batch w(batch, repo);
  REQUIRE(w.convert({space}, env().stream(), *env().mgr, true).has_value());
}

/// Restore `batch` to the GPU and check its column still sums to `expected`.
void require_restores_to(std::shared_ptr<cucascade::data_batch> const& batch,
                         const cucascade::shared_data_repository* repo,
                         int64_t expected)
{
  spill_to(batch, env().gpu_space, repo);
  REQUIRE(get_tier(*batch) == cucascade::memory::Tier::GPU);

  auto ro   = batch->to_read_only();
  auto& gpu = ro.get_data()->cast<cucascade::gpu_table_representation>();
  REQUIRE(gpu_col_sum(gpu) == expected);
}

}  // anonymous namespace

// ── Roundtrips ───────────────────────────────────────────────────────────────

TEST_CASE("spill compression: GPU->compressed_host roundtrip",
          "[compression][spill][isolated_context]")
{
  if (!has_gpu()) {
    SUCCEED("No GPU available — skipping spill compression tests");
    return;
  }

  auto& e = env();
  reset_spill_state();
  set_plan_1col(&repo_a(), kOneColDsl);

  const std::size_t n = 5000;
  auto batch          = make_int32_gpu_batch(n);

  spill_to(batch, e.host_space, &repo_a());
  REQUIRE(is_compressed_host(*batch));
  REQUIRE(get_tier(*batch) == cucascade::memory::Tier::HOST);

  require_restores_to(batch, &repo_a(), expected_sum_of(n));

  reset_spill_state(false);
}

TEST_CASE("spill compression: releasing columns early still round-trips",
          "[compression][spill][isolated_context]")
{
  if (!has_gpu()) {
    SUCCEED("No GPU available — skipping spill compression tests");
    return;
  }

  // release_columns_early frees each source column the moment it has been
  // encoded, so the encoder is reading from memory it is also destroying. If the
  // ordering were wrong the symptom would be silent corruption rather than a
  // failure, hence a value check rather than just a tier check.
  auto& e = env();
  reset_spill_state();
  sirius::compression::set_spill_compression_settings(/*enabled=*/true,
                                                      /*explore_beam_width=*/4,
                                                      /*explore_max_bytes=*/8ull << 20,
                                                      /*max_compressed_fraction=*/0.95,
                                                      /*replan_after_uses=*/0,
                                                      /*error_tolerance=*/1,
                                                      /*replan_change_threshold=*/0.20,
                                                      /*explore_sample_rows=*/0,
                                                      /*min_batch_bytes=*/0,
                                                      /*release_columns_early=*/true,
                                                      /*encode_reserve_fraction=*/0.5,
                                                      /*encode_min_headroom_fraction=*/0.0);
  set_plan_1col(&repo_a(), kOneColDsl);

  const std::size_t n = 5000;
  auto batch          = make_int32_gpu_batch(n);

  spill_to(batch, e.host_space, &repo_a());
  REQUIRE(is_compressed_host(*batch));
  REQUIRE(get_tier(*batch) == cucascade::memory::Tier::HOST);

  require_restores_to(batch, &repo_a(), expected_sum_of(n));

  reset_spill_state(false);
}

TEST_CASE("spill compression: GPU->compressed_disk roundtrip",
          "[compression][spill][isolated_context]")
{
  if (!has_gpu()) {
    SUCCEED("No GPU available — skipping spill compression tests");
    return;
  }

  auto& e = env();
  if (e.disk_space == nullptr) {
    SUCCEED("No DISK space configured — skipping");
    return;
  }

  reset_spill_state();
  set_plan_1col(&repo_a(), kOneColDsl);

  const std::size_t n = 5000;
  auto batch          = make_int32_gpu_batch(n);

  spill_to(batch, e.disk_space, &repo_a());
  REQUIRE(is_compressed_disk(*batch));
  REQUIRE(get_tier(*batch) == cucascade::memory::Tier::DISK);

  // The .hpln file must exist on the configured mount and be non-empty.
  {
    auto ro    = batch->to_read_only();
    auto& disk = ro.get_data()->cast<sirius::compressed_disk_representation>();
    REQUIRE(fs::exists(disk.path()));
    REQUIRE(fs::file_size(disk.path()) > 0);
  }

  require_restores_to(batch, &repo_a(), expected_sum_of(n));

  reset_spill_state(false);
}

TEST_CASE("spill compression: compressed_host->compressed_disk blob flush",
          "[compression][spill][isolated_context]")
{
  if (!has_gpu()) {
    SUCCEED("No GPU available — skipping spill compression tests");
    return;
  }

  auto& e = env();
  if (e.disk_space == nullptr) {
    SUCCEED("No DISK space configured — skipping");
    return;
  }

  reset_spill_state();
  set_plan_1col(&repo_a(), kOneColDsl);

  const std::size_t n = 5000;
  auto batch          = make_int32_gpu_batch(n);

  // GPU -> compressed_host.
  spill_to(batch, e.host_space, &repo_a());
  REQUIRE(is_compressed_host(*batch));

  // The pinned blob is already in .hpln layout, so record its size to check the
  // flush below writes exactly header + payload with no re-compression.
  std::size_t expected_file_size = 0;
  {
    auto ro            = batch->to_read_only();
    auto& host         = ro.get_data()->cast<sirius::compressed_host_representation>();
    expected_file_size = host.header().size() + static_cast<std::size_t>(host.payload_bytes());
  }

  // compressed_host -> compressed_disk (straight file write, no recompress).
  spill_to(batch, e.disk_space, &repo_a());
  REQUIRE(is_compressed_disk(*batch));
  REQUIRE(get_tier(*batch) == cucascade::memory::Tier::DISK);

  {
    auto ro    = batch->to_read_only();
    auto& disk = ro.get_data()->cast<sirius::compressed_disk_representation>();
    REQUIRE(fs::exists(disk.path()));
    REQUIRE(fs::file_size(disk.path()) == expected_file_size);
  }

  require_restores_to(batch, &repo_a(), expected_sum_of(n));

  reset_spill_state(false);
}

TEST_CASE("spill compression: per-column host->disk flush writes one file per column",
          "[compression][spill][isolated_context]")
{
  if (!has_gpu()) {
    SUCCEED("No GPU available — skipping spill compression tests");
    return;
  }

  auto& e = env();
  if (e.disk_space == nullptr) {
    SUCCEED("No DISK space configured — skipping");
    return;
  }

  // With release_columns_early the host batch holds one 1-column .hpln per column
  // instead of a single whole-table blob, so the cascade to disk has to write and
  // reassemble N artifacts. Before this was supported the flush refused and the
  // batch was stuck on the host.
  reset_spill_state();
  sirius::compression::set_spill_compression_settings(/*enabled=*/true,
                                                      /*explore_beam_width=*/4,
                                                      /*explore_max_bytes=*/8ull << 20,
                                                      /*max_compressed_fraction=*/0.95,
                                                      /*replan_after_uses=*/0,
                                                      /*error_tolerance=*/1,
                                                      /*replan_change_threshold=*/0.20,
                                                      /*explore_sample_rows=*/0,
                                                      /*min_batch_bytes=*/0,
                                                      /*release_columns_early=*/true,
                                                      /*encode_reserve_fraction=*/0.5,
                                                      /*encode_min_headroom_fraction=*/0.0);
  set_plans(&repo_a(), {kOneColDsl, kOneColDsl});

  const std::size_t n = 5000;
  auto batch          = make_int32_gpu_batch_2col(n);

  spill_to(batch, e.host_space, &repo_a());
  REQUIRE(is_compressed_host(*batch));

  std::vector<std::size_t> artifact_sizes;
  {
    auto ro    = batch->to_read_only();
    auto& host = ro.get_data()->cast<sirius::compressed_host_representation>();
    REQUIRE(host.column_blobs().size() == 2);
    for (auto const& blob : host.column_blobs()) {
      artifact_sizes.push_back(blob->header.size() +
                               static_cast<std::size_t>(blob->payload_bytes));
    }
    // One artifact is staged on the device at a time, so the decode transient is
    // the largest column rather than the sum of both.
    REQUIRE(host.decode_transient_bytes() ==
            *std::max_element(artifact_sizes.begin(), artifact_sizes.end()));
  }

  spill_to(batch, e.disk_space, &repo_a());
  REQUIRE(is_compressed_disk(*batch));
  REQUIRE(get_tier(*batch) == cucascade::memory::Tier::DISK);

  std::vector<std::string> paths;
  {
    auto ro    = batch->to_read_only();
    auto& disk = ro.get_data()->cast<sirius::compressed_disk_representation>();
    REQUIRE(disk.is_per_column());
    REQUIRE(disk.column_paths().size() == 2);
    paths = disk.column_paths();
    // Each file is the blob flushed verbatim — header + payload, no recompression.
    for (std::size_t i = 0; i < paths.size(); ++i) {
      REQUIRE(fs::exists(paths[i]));
      REQUIRE(fs::file_size(paths[i]) == artifact_sizes[i]);
    }
    REQUIRE(disk.get_size_in_bytes() ==
            std::accumulate(artifact_sizes.begin(), artifact_sizes.end(), std::size_t{0}));
    REQUIRE(disk.decode_transient_bytes() ==
            *std::max_element(artifact_sizes.begin(), artifact_sizes.end()));
  }

  // Both columns must come back, in order: a per-column decode assembles the batch
  // from independent files, so a mix-up would show as a wrong or missing column
  // rather than a failure.
  spill_to(batch, e.gpu_space, &repo_a());
  REQUIRE(get_tier(*batch) == cucascade::memory::Tier::GPU);
  {
    auto ro   = batch->to_read_only();
    auto& gpu = ro.get_data()->cast<cucascade::gpu_table_representation>();
    REQUIRE(gpu.get_table_view().num_columns() == 2);
    REQUIRE(gpu_col_sum_at(gpu, 0) == expected_sum_of(n));
    REQUIRE(gpu_col_sum_at(gpu, 1) == expected_sum_of(n));
  }

  // Every artifact is unlinked with the batch, not just the first.
  for (const auto& p : paths) {
    REQUIRE_FALSE(fs::exists(p));
  }

  reset_spill_state(false);
}

TEST_CASE("spill compression: materialization estimate covers the decode transient",
          "[compression][spill][isolated_context]")
{
  if (!has_gpu()) {
    SUCCEED("No GPU available — skipping spill compression tests");
    return;
  }

  auto& e = env();
  reset_spill_state();
  set_plan_1col(&repo_a(), kOneColDsl);

  const std::size_t n = 5000;
  auto batch          = make_int32_gpu_batch(n);

  // A GPU-resident table decodes nothing, so its estimate is just its footprint.
  {
    auto ro = batch->to_read_only();
    REQUIRE(sirius::estimated_materialization_bytes(*ro.get_data()) ==
            ro.get_data()->get_uncompressed_data_size_in_bytes());
  }

  spill_to(batch, e.host_space, &repo_a());
  REQUIRE(is_compressed_host(*batch));

  // Decoding stages the compressed payload on device and builds the table beside
  // it, so the peak is both at once — strictly more than the decompressed table
  // alone, which is what the reservation used to be sized to.
  {
    auto ro                 = batch->to_read_only();
    auto const* data        = ro.get_data();
    const auto uncompressed = data->get_uncompressed_data_size_in_bytes();
    const auto compressed   = data->get_size_in_bytes();

    REQUIRE(compressed > 0);
    REQUIRE(sirius::estimated_materialization_bytes(*data) == uncompressed + compressed);
    REQUIRE(sirius::estimated_materialization_bytes(*data) > uncompressed);
  }

  require_restores_to(batch, &repo_a(), expected_sum_of(n));

  reset_spill_state(false);
}

// ── First-spill plan discovery ───────────────────────────────────────────────

TEST_CASE("spill compression: first spill from an edge explores and caches a plan",
          "[compression][spill][isolated_context]")
{
  if (!has_gpu()) {
    SUCCEED("No GPU available — skipping spill compression tests");
    return;
  }

  auto& e   = env();
  auto& reg = sirius::compression::plan_register::global();
  reset_spill_state();

  // No plan for this edge yet — the converter must run the explorer.
  REQUIRE_FALSE(reg.resolve_spill_plan(&repo_b()).has_value());

  const std::size_t n = 5000;
  auto batch          = make_int32_gpu_batch(n);

  spill_to(batch, e.host_space, &repo_b());
  REQUIRE(is_compressed_host(*batch));

  // The explored plan is now cached for this edge, one entry per column.
  auto discovered = reg.resolve_spill_plan(&repo_b());
  REQUIRE(discovered.has_value());
  REQUIRE(discovered->columns.size() == 1);
  REQUIRE_FALSE(discovered->columns[0].dsl.empty());
  REQUIRE(discovered->columns[0].viable);

  // A different edge is unaffected — plans are per-edge, not global.
  REQUIRE_FALSE(reg.resolve_spill_plan(&repo_a()).has_value());

  // The data survives the roundtrip under the discovered plan.
  require_restores_to(batch, &repo_b(), expected_sum_of(n));

  // A second batch from the same edge reuses the cached plan (unchanged DSL).
  auto batch2 = make_int32_gpu_batch(n);
  spill_to(batch2, e.host_space, &repo_b());
  REQUIRE(is_compressed_host(*batch2));

  auto after = reg.resolve_spill_plan(&repo_b());
  REQUIRE(after.has_value());
  REQUIRE(after->columns[0].dsl == discovered->columns[0].dsl);
  // ...and each spill attempt is counted against the entry.
  REQUIRE(after->uses > discovered->uses);

  reset_spill_state(false);
}

// ── Fallbacks ────────────────────────────────────────────────────────────────

TEST_CASE("spill compression: disabled setting spills uncompressed",
          "[compression][spill][isolated_context]")
{
  if (!has_gpu()) {
    SUCCEED("No GPU available — skipping spill compression tests");
    return;
  }

  auto& e = env();
  reset_spill_state(/*enabled=*/false);
  // A plan exists for the edge, but the feature is off.
  set_plan_1col(&repo_a(), kOneColDsl);

  auto batch = make_int32_gpu_batch(1000);

  spill_to(batch, e.host_space, &repo_a());
  REQUIRE_FALSE(is_compressed_host(*batch));
  REQUIRE(get_tier(*batch) == cucascade::memory::Tier::HOST);

  reset_spill_state(false);
}

TEST_CASE("spill compression: a batch under min_batch_bytes spills uncompressed",
          "[compression][spill][isolated_context]")
{
  if (!has_gpu()) {
    SUCCEED("No GPU available — skipping spill compression tests");
    return;
  }

  auto& e = env();
  // Everything else is favourable: the feature is on and the edge has a plan that
  // compresses this data well. Only the size gate should stop it.
  reset_spill_state();
  set_plan_1col(&repo_a(), kOneColDsl);
  sirius::compression::set_spill_compression_settings(/*enabled=*/true,
                                                      /*explore_beam_width=*/4,
                                                      /*explore_max_bytes=*/8ull << 20,
                                                      /*max_compressed_fraction=*/0.95,
                                                      /*replan_after_uses=*/0,
                                                      /*error_tolerance=*/1,
                                                      /*replan_change_threshold=*/0.20,
                                                      /*explore_sample_rows=*/0,
                                                      /*min_batch_bytes=*/1ull << 30,
                                                      /*release_columns_early=*/false,
                                                      /*encode_reserve_fraction=*/0.5,
                                                      /*encode_min_headroom_fraction=*/0.0);

  auto batch = make_int32_gpu_batch(1000);  // ~4 KB, far under the 1 GiB gate

  spill_to(batch, e.host_space, &repo_a());
  REQUIRE_FALSE(is_compressed_host(*batch));
  REQUIRE(get_tier(*batch) == cucascade::memory::Tier::HOST);

  // The gate must not be mistaken for a verdict about the edge: an undersized
  // batch says nothing about whether this edge's data compresses, so the cached
  // plan has to survive for the larger batches that follow.
  REQUIRE(sirius::compression::plan_register::global()
            .decide_spill_plan(&repo_a(), /*replan_after_uses=*/0)
            .verdict != sirius::compression::plan_register::spill_plan_verdict::skip);

  reset_spill_state(false);
}

TEST_CASE("spill compression: batch with no source edge compresses with dtype defaults",
          "[compression][spill][isolated_context]")
{
  if (!has_gpu()) {
    SUCCEED("No GPU available — skipping spill compression tests");
    return;
  }

  auto& e = env();
  reset_spill_state();

  auto batch = make_int32_gpu_batch(1000);

  // No repo pointer (a batch the downgrade executor reached without a producing
  // repository). There is no plan key and no lineage, but compression needs only
  // a carrier: each column falls back to default_plan_for its dtype — bitpack
  // here, which shrinks an int32 column. These batches are the bulk of spill
  // traffic on q5/SF1000, so skipping them outright left most of the win unused.
  sirius::convertible_data_batch w(batch, nullptr);
  REQUIRE(w.convert({e.host_space}, e.stream(), *e.mgr, true).has_value());

  REQUIRE(is_compressed_host(*batch));
  REQUIRE(get_tier(*batch) == cucascade::memory::Tier::HOST);
  require_restores_to(batch, nullptr, expected_sum_of(1000));

  // Nothing is recorded against the register: there is no edge to record it
  // against, so an unkeyed batch neither consumes plan uses nor steers a verdict.
  REQUIRE_FALSE(
    sirius::compression::plan_register::global().resolve_spill_plan(nullptr).has_value());

  reset_spill_state(false);
}

TEST_CASE("spill compression: a cached plan for a different column count is re-explored",
          "[compression][spill][isolated_context]")
{
  if (!has_gpu()) {
    SUCCEED("No GPU available — skipping spill compression tests");
    return;
  }

  auto& e   = env();
  auto& reg = sirius::compression::plan_register::global();
  reset_spill_state();
  // Two per-column plans cached against a 1-column batch: the entry describes a
  // different schema, so it must be discarded and the edge explored afresh
  // rather than applied blindly (which would throw inside Simpatico).
  set_plans(&repo_a(), {kOneColDsl, kOneColDsl});

  const std::size_t n = 5000;
  auto batch          = make_int32_gpu_batch(n);

  sirius::convertible_data_batch w(batch, &repo_a());
  REQUIRE_NOTHROW(w.convert({e.host_space}, e.stream(), *e.mgr, true));

  // The stale entry was replaced by one matching this schema, and the batch
  // compressed under the fresh plan.
  REQUIRE(is_compressed_host(*batch));
  REQUIRE(reg.resolve_spill_plan(&repo_a())->columns.size() == 1);

  require_restores_to(batch, &repo_a(), expected_sum_of(n));

  reset_spill_state(false);
}

TEST_CASE("spill compression: a plan that fails to compress falls back without throwing",
          "[compression][spill][isolated_context]")
{
  if (!has_gpu()) {
    SUCCEED("No GPU available — skipping spill compression tests");
    return;
  }

  auto& e = env();
  reset_spill_state();
  // Right column count, but the plan names an operator that does not exist, so
  // compress_column fails. The exception must not escape convert().
  set_plan_1col(&repo_a(), "input -> no_such_operator\n");

  auto batch = make_int32_gpu_batch(1000);

  sirius::convertible_data_batch w(batch, &repo_a());
  REQUIRE_NOTHROW(w.convert({e.host_space}, e.stream(), *e.mgr, true));

  // The batch reaches the host tier intact either way. Where the failed column
  // lands depends on its dtype: the fallback is the dtype's default carrier
  // (bitpack for INT32 here), not plain identity, because identity leaves are
  // reconstructed with cudf::make_numeric_column and so cannot round-trip a
  // decimal or timestamp. Bitpack does shrink an int32 column, so this batch
  // legitimately ends up compressed; what matters is that nothing threw and the
  // values survive.
  REQUIRE(get_tier(*batch) == cucascade::memory::Tier::HOST);
  require_restores_to(batch, &repo_a(), expected_sum_of(1000));

  // With error_tolerance 1 the failure is durable, so the column is written off
  // and the next batch skips compression instead of failing again.
  REQUIRE_FALSE(col0_of(&repo_a()).viable);

  reset_spill_state(false);
}

TEST_CASE("spill compression: poor compression ratio falls back to uncompressed",
          "[compression][spill][isolated_context]")
{
  if (!has_gpu()) {
    SUCCEED("No GPU available — skipping spill compression tests");
    return;
  }

  auto& e = env();
  sirius::compression::plan_register::global().clear_all();
  sirius::compression::set_spill_compression_settings(true,
                                                      /*explore_beam_width=*/4,
                                                      /*explore_max_bytes=*/8ull << 20,
                                                      /*max_compressed_fraction=*/0.75,
                                                      /*replan_after_uses=*/0,
                                                      /*error_tolerance=*/1,
                                                      /*replan_change_threshold=*/0.20,
                                                      /*explore_sample_rows=*/0,
                                                      /*min_batch_bytes=*/0,
                                                      /*release_columns_early=*/false,
                                                      /*encode_reserve_fraction=*/0.5,
                                                      /*encode_min_headroom_fraction=*/0.0);
  // A plan whose output is the same width as its input cannot reach 0.75.
  set_plan_1col(&repo_a(), kNonCompressingDsl);

  auto batch = make_int32_gpu_batch(1000);

  spill_to(batch, e.host_space, &repo_a());
  REQUIRE_FALSE(is_compressed_host(*batch));
  REQUIRE(get_tier(*batch) == cucascade::memory::Tier::HOST);

  reset_spill_state(false);
}

// ── Per-column verdicts ──────────────────────────────────────────────────────

TEST_CASE("spill compression: one incompressible column does not disable its neighbours",
          "[compression][spill][isolated_context]")
{
  if (!has_gpu()) {
    SUCCEED("No GPU available — skipping spill compression tests");
    return;
  }

  auto& e   = env();
  auto& reg = sirius::compression::plan_register::global();
  reg.clear_all();
  sirius::compression::set_spill_compression_settings(true,
                                                      /*explore_beam_width=*/4,
                                                      /*explore_max_bytes=*/8ull << 20,
                                                      /*max_compressed_fraction=*/0.75,
                                                      /*replan_after_uses=*/0,
                                                      /*error_tolerance=*/1,
                                                      /*replan_change_threshold=*/0.20,
                                                      /*explore_sample_rows=*/0,
                                                      /*min_batch_bytes=*/0,
                                                      /*release_columns_early=*/false,
                                                      /*encode_reserve_fraction=*/0.5,
                                                      /*encode_min_headroom_fraction=*/0.0);

  // Two identical columns: one bitpacks well, the other is given a plan that
  // cannot shrink it. Same data, so the outcome difference is purely the plan.
  const std::size_t n = 5000;
  auto batch          = make_int32_gpu_batch_2col(n);
  set_plans(&repo_a(), {kOneColDsl, kNonCompressingDsl});

  spill_to(batch, e.host_space, &repo_a());
  REQUIRE(is_compressed_host(*batch));

  // Each column is judged on its own bytes: the bitpacked one stays viable, the
  // other is written off. Previously either verdict would have applied to both.
  auto state = reg.resolve_spill_plan(&repo_a());
  REQUIRE(state->columns.size() == 2);
  REQUIRE(state->columns[0].viable);
  REQUIRE_FALSE(state->columns[1].viable);

  // The edge is still worth compressing overall, so it is not skipped.
  REQUIRE(reg.decide_spill_plan(&repo_a(), 0).verdict ==
          sirius::compression::plan_register::spill_plan_verdict::use);

  // The next batch compresses column 0 and stores column 1 raw; the data still
  // round-trips intact.
  auto batch2 = make_int32_gpu_batch_2col(n);
  spill_to(batch2, e.host_space, &repo_a());
  REQUIRE(is_compressed_host(*batch2));
  require_restores_to(batch2, &repo_a(), expected_sum_of(n));

  reset_spill_state(false);
}

// ── Unviable-edge memoization and re-explore ─────────────────────────────────

TEST_CASE("spill compression: a rejected edge is marked and later batches skip compression",
          "[compression][spill][isolated_context]")
{
  if (!has_gpu()) {
    SUCCEED("No GPU available — skipping spill compression tests");
    return;
  }

  auto& e   = env();
  auto& reg = sirius::compression::plan_register::global();
  reg.clear_all();
  sirius::compression::set_spill_compression_settings(true,
                                                      /*explore_beam_width=*/4,
                                                      /*explore_max_bytes=*/8ull << 20,
                                                      /*max_compressed_fraction=*/0.75,
                                                      /*replan_after_uses=*/0,
                                                      /*error_tolerance=*/1,
                                                      /*replan_change_threshold=*/0.20,
                                                      /*explore_sample_rows=*/0,
                                                      /*min_batch_bytes=*/0,
                                                      /*release_columns_early=*/false,
                                                      /*encode_reserve_fraction=*/0.5,
                                                      /*encode_min_headroom_fraction=*/0.0);  // never expire
  set_plan_1col(&repo_a(), kNonCompressingDsl);

  // First batch: compression runs, misses the threshold, and the edge is marked.
  auto batch = make_int32_gpu_batch(1000);
  spill_to(batch, e.host_space, &repo_a());
  REQUIRE_FALSE(is_compressed_host(*batch));

  REQUIRE_FALSE(col0_of(&repo_a()).viable);

  // With every column marked unviable, the spill path must now decide to skip
  // rather than compress again.
  const auto decision = reg.decide_spill_plan(&repo_a(), /*replan_after_uses=*/0);
  REQUIRE(decision.verdict == sirius::compression::plan_register::spill_plan_verdict::skip);

  // A second batch still spills, uncompressed, and stays skipped.
  auto batch2 = make_int32_gpu_batch(1000);
  spill_to(batch2, e.host_space, &repo_a());
  REQUIRE_FALSE(is_compressed_host(*batch2));
  REQUIRE_FALSE(col0_of(&repo_a()).viable);

  reset_spill_state(false);
}

TEST_CASE("spill compression: an unviable edge is re-explored once its entry expires",
          "[compression][spill][isolated_context]")
{
  if (!has_gpu()) {
    SUCCEED("No GPU available — skipping spill compression tests");
    return;
  }

  auto& e   = env();
  auto& reg = sirius::compression::plan_register::global();
  reg.clear_all();
  // Expire after a single use so the retry is observable in one extra spill.
  sirius::compression::set_spill_compression_settings(true,
                                                      /*explore_beam_width=*/4,
                                                      /*explore_max_bytes=*/8ull << 20,
                                                      /*max_compressed_fraction=*/0.75,
                                                      /*replan_after_uses=*/1,
                                                      /*error_tolerance=*/1,
                                                      /*replan_change_threshold=*/0.20,
                                                      /*explore_sample_rows=*/0,
                                                      /*min_batch_bytes=*/0,
                                                      /*release_columns_early=*/false,
                                                      /*encode_reserve_fraction=*/0.5,
                                                      /*encode_min_headroom_fraction=*/0.0);
  set_plan_1col(&repo_a(), kNonCompressingDsl);

  // First batch: rejected, edge marked unviable, one use recorded.
  auto batch = make_int32_gpu_batch(5000);
  spill_to(batch, e.host_space, &repo_a());
  REQUIRE_FALSE(is_compressed_host(*batch));
  REQUIRE_FALSE(col0_of(&repo_a()).viable);

  // The entry has now expired, so the verdict is explore rather than skip —
  // a previously written-off edge gets another chance.
  REQUIRE(reg.decide_spill_plan(&repo_a(), /*replan_after_uses=*/1).verdict ==
          sirius::compression::plan_register::spill_plan_verdict::explore);

  // The next batch re-explores and finds a plan that does compress, so the edge
  // recovers instead of staying disabled for the rest of the query.
  auto batch2 = make_int32_gpu_batch(5000);
  spill_to(batch2, e.host_space, &repo_a());
  REQUIRE(is_compressed_host(*batch2));

  const auto recovered = col0_of(&repo_a());
  REQUIRE(recovered.viable);
  REQUIRE(recovered.dsl != kNonCompressingDsl);

  require_restores_to(batch2, &repo_a(), expected_sum_of(5000));

  reset_spill_state(false);
}

// ── plan_register spill-plan API ─────────────────────────────────────────────

TEST_CASE("plan_register: decide_spill_plan verdicts", "[compression][plan_register]")
{
  using verdict = sirius::compression::plan_register::spill_plan_verdict;
  auto& reg     = sirius::compression::plan_register::global();
  reg.clear_all();

  // No entry at all.
  REQUIRE(reg.decide_spill_plan(&repo_a(), 0).verdict == verdict::explore);

  set_plan_1col(&repo_a(), "some dsl");
  auto d = reg.decide_spill_plan(&repo_a(), 0);
  REQUIRE(d.verdict == verdict::use);
  REQUIRE(d.columns.size() == 1);
  REQUIRE(d.columns[0].dsl == "some dsl");

  // The only column measured as not worth it -> skip, but its DSL and the use
  // count are retained so the entry still ages toward expiry.
  conclude_1col(&repo_a(),
                sirius::compression::plan_register::spill_attempt_outcome::not_worth_it,
                /*base_interval=*/0,
                /*error_tolerance=*/1);
  REQUIRE(reg.decide_spill_plan(&repo_a(), 0).verdict == verdict::skip);
  REQUIRE(col0_of(&repo_a()).dsl == "some dsl");

  // Expiry overrides both use and skip.
  reg.note_spill_plan_use(&repo_a());
  REQUIRE(reg.resolve_spill_plan(&repo_a())->uses == 1);
  REQUIRE(reg.decide_spill_plan(&repo_a(), /*replan_after_uses=*/1).verdict == verdict::explore);
  // ...but 0 means "never expire", so the skip verdict stands.
  REQUIRE(reg.decide_spill_plan(&repo_a(), /*replan_after_uses=*/0).verdict == verdict::skip);

  // Adopting a materially better plan clears the verdicts and resets the counter.
  set_plan_1col(&repo_a(), "fresh dsl", /*ratio=*/8.0);
  auto fresh = reg.resolve_spill_plan(&repo_a());
  REQUIRE(fresh->columns[0].viable);
  REQUIRE(fresh->uses == 0);

  reg.clear_all();
}

TEST_CASE("plan_register: replan interval backs off when a re-explore learns nothing",
          "[compression][plan_register]")
{
  using outcome = sirius::compression::plan_register::spill_attempt_outcome;

  auto& reg                     = sirius::compression::plan_register::global();
  constexpr std::uint64_t kBase = 8;
  constexpr std::uint32_t kTol  = 1;
  constexpr auto kCompressed    = outcome::compressed;
  constexpr auto kNotWorthIt    = outcome::not_worth_it;
  const std::string kPlan       = "plan one";
  const std::string kOtherPlan  = "plan two";
  // A plan is only adopted when it measures materially differently, so the
  // "different plan" cases must report a different ratio, not just other text.
  constexpr double kOtherRatio = 5.0;

  auto interval_of = [&](const cucascade::shared_data_repository* r) {
    return reg.resolve_spill_plan(r)->replan_interval;
  };

  SECTION("same plan, still working -> double")
  {
    reg.clear_all();
    set_plan_1col(&repo_a(), kPlan);
    conclude_1col(&repo_a(), kCompressed, kBase, kTol);  // first install: no backoff
    REQUIRE(interval_of(&repo_a()) == 0);                // still on the configured schedule

    set_plan_1col(&repo_a(), kPlan);  // re-explore returned the same plan
    conclude_1col(&repo_a(), kCompressed, kBase, kTol);
    REQUIRE(interval_of(&repo_a()) == kBase * 2);

    set_plan_1col(&repo_a(), kPlan);
    conclude_1col(&repo_a(), kCompressed, kBase, kTol);
    REQUIRE(interval_of(&repo_a()) == kBase * 4);
  }

  SECTION("different plan that still fails the threshold -> double")
  {
    reg.clear_all();
    set_plan_1col(&repo_a(), kPlan);
    conclude_1col(&repo_a(), kNotWorthIt, kBase, kTol);

    set_plan_1col(&repo_a(), kOtherPlan, kOtherRatio);   // explorer found something else...
    conclude_1col(&repo_a(), kNotWorthIt, kBase, kTol);  // ...that still does not compress
    REQUIRE(interval_of(&repo_a()) == kBase * 2);
  }

  SECTION("different plan that works -> reset to the configured interval")
  {
    reg.clear_all();
    set_plan_1col(&repo_a(), kPlan);
    conclude_1col(&repo_a(), kCompressed, kBase, kTol);
    // Stretch the interval out first.
    set_plan_1col(&repo_a(), kPlan);
    conclude_1col(&repo_a(), kCompressed, kBase, kTol);
    REQUIRE(interval_of(&repo_a()) == kBase * 2);

    set_plan_1col(&repo_a(), kOtherPlan, kOtherRatio);
    conclude_1col(&repo_a(), kCompressed, kBase, kTol);
    REQUIRE(interval_of(&repo_a()) == kBase);
  }

  SECTION("viability recovering on the same plan -> reset")
  {
    reg.clear_all();
    set_plan_1col(&repo_a(), kPlan);
    conclude_1col(&repo_a(), kNotWorthIt, kBase, kTol);  // not worth compressing
    set_plan_1col(&repo_a(), kPlan);
    conclude_1col(&repo_a(), kNotWorthIt, kBase, kTol);  // still not — back off
    REQUIRE(interval_of(&repo_a()) == kBase * 2);

    // Same plan, but it now compresses: the edge recovered, so resume checking
    // on schedule.
    set_plan_1col(&repo_a(), kPlan);
    conclude_1col(&repo_a(), kCompressed, kBase, kTol);
    REQUIRE(interval_of(&repo_a()) == kBase);
    REQUIRE(col0_of(&repo_a()).viable);
  }

  SECTION("a stretched interval overrides the configured one when deciding")
  {
    using verdict = sirius::compression::plan_register::spill_plan_verdict;
    reg.clear_all();
    set_plan_1col(&repo_a(), kPlan);
    conclude_1col(&repo_a(), kCompressed, kBase, kTol);
    set_plan_1col(&repo_a(), kPlan);
    conclude_1col(&repo_a(), kCompressed, kBase, kTol);
    REQUIRE(interval_of(&repo_a()) == kBase * 2);

    for (std::uint64_t i = 0; i < kBase; ++i) {
      reg.note_spill_plan_use(&repo_a());
    }
    // At the configured interval the entry would expire, but its own stretched
    // interval has not elapsed yet.
    REQUIRE(reg.decide_spill_plan(&repo_a(), kBase).verdict == verdict::use);

    for (std::uint64_t i = 0; i < kBase; ++i) {
      reg.note_spill_plan_use(&repo_a());
    }
    REQUIRE(reg.decide_spill_plan(&repo_a(), kBase).verdict == verdict::explore);
  }

  reg.clear_all();
}

TEST_CASE("plan_register: transient compression errors do not write off an edge",
          "[compression][plan_register]")
{
  using outcome = sirius::compression::plan_register::spill_attempt_outcome;
  using verdict = sirius::compression::plan_register::spill_plan_verdict;

  auto& reg                     = sirius::compression::plan_register::global();
  constexpr std::uint64_t kBase = 8;
  constexpr std::uint32_t kTol  = 3;

  SECTION("errors below the tolerance leave the edge usable")
  {
    reg.clear_all();
    set_plan_1col(&repo_a(), "plan");

    // Two failures with a tolerance of three: the edge stays viable, so spilling
    // keeps compressing rather than being disabled by a passing blip.
    conclude_1col(&repo_a(), outcome::failed, kBase, kTol);
    REQUIRE(reg.decide_spill_plan(&repo_a(), kBase).verdict == verdict::use);
    conclude_1col(&repo_a(), outcome::failed, kBase, kTol);
    REQUIRE(reg.decide_spill_plan(&repo_a(), kBase).verdict == verdict::use);
    REQUIRE(col0_of(&repo_a()).viable);

    // The third makes it durable.
    conclude_1col(&repo_a(), outcome::failed, kBase, kTol);
    REQUIRE(reg.decide_spill_plan(&repo_a(), kBase).verdict == verdict::skip);
  }

  SECTION("a success resets the error streak")
  {
    reg.clear_all();
    set_plan_1col(&repo_a(), "plan");

    conclude_1col(&repo_a(), outcome::failed, kBase, kTol);
    conclude_1col(&repo_a(), outcome::failed, kBase, kTol);
    conclude_1col(&repo_a(), outcome::compressed, kBase, kTol);
    REQUIRE(col0_of(&repo_a()).consecutive_errors == 0);

    // Two more failures must not tip it over — the streak restarted.
    conclude_1col(&repo_a(), outcome::failed, kBase, kTol);
    conclude_1col(&repo_a(), outcome::failed, kBase, kTol);
    REQUIRE(reg.decide_spill_plan(&repo_a(), kBase).verdict == verdict::use);
  }

  SECTION("a tolerated error does not disturb a pending replan comparison")
  {
    reg.clear_all();
    set_plan_1col(&repo_a(), "plan");
    conclude_1col(&repo_a(), outcome::compressed, kBase, kTol);

    // Re-explore returns the same plan, but the attempt errors transiently. The
    // interval must not move: the cycle has not been judged yet.
    set_plan_1col(&repo_a(), "plan");
    conclude_1col(&repo_a(), outcome::failed, kBase, kTol);
    REQUIRE(reg.resolve_spill_plan(&repo_a())->replan_interval == 0);

    // The next real outcome concludes it — same plan, so back off.
    conclude_1col(&repo_a(), outcome::compressed, kBase, kTol);
    REQUIRE(reg.resolve_spill_plan(&repo_a())->replan_interval == kBase * 2);
  }

  SECTION("a tolerance of one writes off on the first error")
  {
    reg.clear_all();
    set_plan_1col(&repo_a(), "plan");
    conclude_1col(&repo_a(), outcome::failed, kBase, /*error_tolerance=*/1);
    REQUIRE(reg.decide_spill_plan(&repo_a(), kBase).verdict == verdict::skip);
  }

  reg.clear_all();
}

TEST_CASE("plan_register: a re-explored plan is only adopted when it performs differently",
          "[compression][plan_register]")
{
  using outcome = sirius::compression::plan_register::spill_attempt_outcome;
  auto& reg     = sirius::compression::plan_register::global();

  constexpr double kEps         = 0.20;
  constexpr std::uint64_t kBase = 8;
  constexpr std::uint32_t kTol  = 1;

  SECTION("a differently spelled plan with the same measurements is not adopted")
  {
    reg.clear_all();
    set_plan_1col(&repo_a(), "cached", /*ratio=*/2.0, /*c=*/10.0, /*d=*/20.0);
    conclude_1col(&repo_a(), outcome::compressed, kBase, kTol);

    // Same numbers, different text: the explorer found an equivalent cascade.
    set_plan_1col(&repo_a(), "rewritten", /*ratio=*/2.0, /*c=*/10.0, /*d=*/20.0);
    REQUIRE(col0_of(&repo_a()).dsl == "cached");

    // ...and because nothing was adopted, the cycle counts as no change, so the
    // replan interval backs off rather than resetting.
    conclude_1col(&repo_a(), outcome::compressed, kBase, kTol);
    REQUIRE(reg.resolve_spill_plan(&repo_a())->replan_interval == kBase * 2);
  }

  SECTION("within the threshold on every metric is not adopted")
  {
    reg.clear_all();
    set_plan_1col(&repo_a(), "cached", 2.0, 10.0, 20.0);
    // +19% ratio, -15% compress, +10% decompress: all inside 20%.
    set_plan_1col(&repo_a(), "marginal", 2.38, 8.5, 22.0);
    REQUIRE(col0_of(&repo_a()).dsl == "cached");
    REQUIRE(col0_of(&repo_a()).compression_ratio == 2.0);
  }

  SECTION("a materially better ratio is adopted")
  {
    reg.clear_all();
    set_plan_1col(&repo_a(), "cached", 2.0, 10.0, 20.0);
    set_plan_1col(&repo_a(), "better", /*ratio=*/4.0, 10.0, 20.0);
    REQUIRE(col0_of(&repo_a()).dsl == "better");
    REQUIRE(col0_of(&repo_a()).compression_ratio == 4.0);
  }

  SECTION("a materially different throughput alone is enough")
  {
    reg.clear_all();
    set_plan_1col(&repo_a(), "cached", 2.0, 10.0, 20.0);
    // Same ratio and compress speed, but decompresses far faster.
    set_plan_1col(&repo_a(), "faster_decode", 2.0, 10.0, /*d=*/40.0);
    REQUIRE(col0_of(&repo_a()).dsl == "faster_decode");
  }

  SECTION("keeping a cached plan keeps its verdict, so a written-off column stays skipped")
  {
    reg.clear_all();
    set_plan_1col(&repo_a(), "cached", 2.0, 10.0, 20.0);
    conclude_1col(&repo_a(), outcome::not_worth_it, kBase, kTol);
    REQUIRE_FALSE(col0_of(&repo_a()).viable);

    // An equivalent re-explore must not resurrect the column: it will not
    // compress any better than the plan already judged.
    set_plan_1col(&repo_a(), "rewritten", 2.0, 10.0, 20.0);
    REQUIRE_FALSE(col0_of(&repo_a()).viable);

    // A materially different plan does earn a fresh chance.
    set_plan_1col(&repo_a(), "better", 4.0, 10.0, 20.0);
    REQUIRE(col0_of(&repo_a()).viable);
  }

  SECTION("adoption is decided per column")
  {
    reg.clear_all();
    std::vector<sirius::compression::plan_register::column_plan_candidate> first{
      {"a0", 2.0, 10.0, 20.0}, {"b0", 2.0, 10.0, 20.0}};
    reg.set_spill_plan(&repo_a(), first, kEps);

    // Column 0 is equivalent; column 1 is materially better.
    std::vector<sirius::compression::plan_register::column_plan_candidate> second{
      {"a1", 2.0, 10.0, 20.0}, {"b1", 5.0, 10.0, 20.0}};
    reg.set_spill_plan(&repo_a(), second, kEps);

    auto state = reg.resolve_spill_plan(&repo_a());
    REQUIRE(state->columns[0].dsl == "a0");
    REQUIRE(state->columns[1].dsl == "b1");
  }

  SECTION("a threshold of zero adopts anything that measured differently at all")
  {
    reg.clear_all();
    std::vector<sirius::compression::plan_register::column_plan_candidate> a{
      {"cached", 2.0, 10.0, 20.0}};
    reg.set_spill_plan(&repo_a(), a, /*change_threshold=*/0.0);
    std::vector<sirius::compression::plan_register::column_plan_candidate> b{
      {"tiny_delta", 2.0001, 10.0, 20.0}};
    reg.set_spill_plan(&repo_a(), b, /*change_threshold=*/0.0);
    REQUIRE(col0_of(&repo_a()).dsl == "tiny_delta");
  }

  reg.clear_all();
}

TEST_CASE("plan_register: note_spill_plan_use on an absent edge is a no-op",
          "[compression][plan_register]")
{
  auto& reg = sirius::compression::plan_register::global();
  reg.clear_all();

  // Must not create an entry — a use recorded before any plan exists would
  // otherwise age a plan that has not been installed yet.
  reg.note_spill_plan_use(&repo_a());
  REQUIRE_FALSE(reg.resolve_spill_plan(&repo_a()).has_value());
  conclude_1col(&repo_a(),
                sirius::compression::plan_register::spill_attempt_outcome::not_worth_it,
                /*base_interval=*/8,
                /*error_tolerance=*/1);
  REQUIRE_FALSE(reg.resolve_spill_plan(&repo_a()).has_value());

  reg.clear_all();
}

TEST_CASE("plan_register: spill plan round-trips per repository", "[compression][plan_register]")
{
  auto& reg = sirius::compression::plan_register::global();
  reg.clear_all();

  REQUIRE_FALSE(reg.resolve_spill_plan(&repo_a()).has_value());

  const std::string dsl = "input -> delta -> differences\n";
  set_plan_1col(&repo_a(), dsl);

  auto result = reg.resolve_spill_plan(&repo_a());
  REQUIRE(result.has_value());
  REQUIRE(result->columns.size() == 1);
  REQUIRE(result->columns[0].dsl == dsl);
  REQUIRE(result->columns[0].viable);
  REQUIRE(result->uses == 0);

  // Plans are per-edge: another repository is unaffected.
  REQUIRE_FALSE(reg.resolve_spill_plan(&repo_b()).has_value());

  // Storing no columns reads back as "no plan".
  set_plans(&repo_a(), {});
  REQUIRE_FALSE(reg.resolve_spill_plan(&repo_a()).has_value());

  // A multi-column edge keeps one entry per column, in schema order.
  set_plans(&repo_a(), {"plan a", "plan b", "plan c"});
  auto multi = reg.resolve_spill_plan(&repo_a());
  REQUIRE(multi->columns.size() == 3);
  REQUIRE(multi->columns[1].dsl == "plan b");
  REQUIRE(multi->viable_count() == 3);

  reg.clear_all();
}

TEST_CASE("plan_register: clear_spill_plan removes only the named edge",
          "[compression][plan_register]")
{
  auto& reg = sirius::compression::plan_register::global();
  reg.clear_all();

  set_plan_1col(&repo_a(), "plan a");
  set_plan_1col(&repo_b(), "plan b");

  reg.clear_spill_plan(&repo_a());
  REQUIRE_FALSE(reg.resolve_spill_plan(&repo_a()).has_value());
  REQUIRE(reg.resolve_spill_plan(&repo_b()).has_value());

  reg.clear_all();
}

namespace {

// A two-column plan in the exact shape the offline generator writes, including
// the `#` measurement comments that split_plan_dsl strips.
constexpr auto kMeasuredTablePlan = R"(# column: l_orderkey  dtype: i64  ratio: 20.711x  depth: 2
# comp: 531.34 GB/s  decomp: 626.76 GB/s
input -> delta -> differences
delta.differences -> ans

---
# column: l_partkey  dtype: i32  ratio: 1.140x  depth: 1
# comp: 681.79 GB/s  decomp: 1496.30 GB/s
input -> bitpack -> chunk_min, chunk_count, chunk_bits, packed
)";

}  // namespace

TEST_CASE("plan_register: plan metrics are parsed and index-aligned with plan blocks",
          "[compression][plan_register]")
{
  auto& reg = sirius::compression::plan_register::global();
  reg.clear_all();

  reg.set_table_plan("lineitem", kMeasuredTablePlan);

  auto m0 = reg.resolve_plan_metrics("lineitem", 0);
  REQUIRE(m0.has_value());
  REQUIRE(m0->compression_ratio == Approx(20.711));
  // "comp:" must not match inside "decomp:" — both live on one line.
  REQUIRE(m0->compress_gbps == Approx(531.34));
  REQUIRE(m0->decompress_gbps == Approx(626.76));

  auto m1 = reg.resolve_plan_metrics("lineitem", 1);
  REQUIRE(m1.has_value());
  REQUIRE(m1->compression_ratio == Approx(1.140));
  REQUIRE(m1->compress_gbps == Approx(681.79));

  // Indices line up with the blocks select_plan_blocks hands out.
  auto block0 = sirius::compression::select_plan_blocks(kMeasuredTablePlan, {0});
  REQUIRE(block0.has_value());
  REQUIRE(block0->find("delta") != std::string::npos);
  auto block1 = sirius::compression::select_plan_blocks(kMeasuredTablePlan, {1});
  REQUIRE(block1.has_value());
  REQUIRE(block1->find("bitpack") != std::string::npos);

  REQUIRE_FALSE(reg.resolve_plan_metrics("lineitem", 2).has_value());
  REQUIRE_FALSE(reg.resolve_plan_metrics("nosuchtable", 0).has_value());

  // A plan with no measurement comments parses to a present-but-empty entry, so
  // it neither shifts its neighbours nor claims measurements it does not have.
  reg.set_table_plan("plain", "input -> identity\n");
  REQUIRE_FALSE(reg.resolve_plan_metrics("plain", 0).has_value());

  reg.clear_all();
}

TEST_CASE("plan_register: output plan selection admits only columns clearing the gate",
          "[compression][plan_register]")
{
  using reg_t = sirius::compression::plan_register;
  auto& reg   = reg_t::global();
  reg.clear_all();

  reg.set_table_plan("lineitem", kMeasuredTablePlan);

  // col0 -> l_orderkey (20.7x, fast) qualifies; col1 -> l_partkey (1.14x) does
  // not clear the ratio bar; col2 is computed and has no origin at all.
  reg.set_spill_column_origins(&repo_a(),
                               {reg_t::spill_column_origin{"lineitem", 0},
                                reg_t::spill_column_origin{"lineitem", 1},
                                std::nullopt});

  const reg_t::plan_quality_gate gate{/*min_ratio=*/3.0,
                                      /*min_compress_gbps=*/250.0,
                                      /*min_decompress_gbps=*/250.0};

  auto picked = reg.select_output_plans(&repo_a(), /*expected_columns=*/3, gate);
  REQUIRE(picked.size() == 1);
  REQUIRE(picked[0].column_index == 0);
  REQUIRE(picked[0].metrics.compression_ratio == Approx(20.711));
  REQUIRE(picked[0].dsl.find("delta") != std::string::npos);
  // delta's ratio came from the sorted base table, so it must be verified on
  // first use rather than trusted on a shuffled task output.
  REQUIRE(picked[0].order_dependent);

  // Raising the compress bar above l_orderkey's 531 GB/s drops it.
  auto strict = reg.select_output_plans(
    &repo_a(), 3, reg_t::plan_quality_gate{3.0, /*min_compress_gbps=*/600.0, 250.0});
  REQUIRE(strict.empty());

  // A column-count mismatch describes a different schema — select nothing rather
  // than misattribute origins to the wrong columns.
  REQUIRE(reg.select_output_plans(&repo_a(), /*expected_columns=*/2, gate).empty());

  reg.clear_all();
}

TEST_CASE("plan_register: an output plan that misses its ratio on real data is dropped",
          "[compression][plan_register]")
{
  using reg_t = sirius::compression::plan_register;
  auto& reg   = reg_t::global();
  reg.clear_all();

  reg.set_table_plan("lineitem", kMeasuredTablePlan);
  reg.set_spill_column_origins(
    &repo_a(),
    {reg_t::spill_column_origin{"lineitem", 0}, reg_t::spill_column_origin{"lineitem", 1}});

  const reg_t::plan_quality_gate gate{3.0, 250.0, 250.0};

  auto first = reg.decide_output_plan(&repo_a(), 2, gate);
  REQUIRE(first.has_value());
  REQUIRE((*first)[0].has_value());  // l_orderkey: delta, admitted on 20.7x
  REQUIRE_FALSE((*first)[1].has_value());

  // The decision is cached: a second call does not re-select.
  REQUIRE(reg.decide_output_plan(&repo_a(), 2, gate).has_value());

  // First batch reality check: the base-table 20.7x came from sorted storage, and
  // this operator output is shuffled — delta only managed 1.2x. Drop it.
  const std::array<double, 2> achieved{1.2, 0.0};
  reg.conclude_output_attempt(&repo_a(), achieved, gate);

  // Nothing viable left, so the edge stops being offered plans at all.
  REQUIRE_FALSE(reg.decide_output_plan(&repo_a(), 2, gate).has_value());

  // A ratio that clears the bar is kept instead.
  reg.clear_all();
  reg.set_table_plan("lineitem", kMeasuredTablePlan);
  reg.set_spill_column_origins(&repo_b(), {reg_t::spill_column_origin{"lineitem", 0}});
  REQUIRE(reg.decide_output_plan(&repo_b(), 1, gate).has_value());
  const std::array<double, 1> good{9.5};
  reg.conclude_output_attempt(&repo_b(), good, gate);
  auto kept = reg.decide_output_plan(&repo_b(), 1, gate);
  REQUIRE(kept.has_value());
  REQUIRE((*kept)[0].has_value());

  reg.clear_all();
}

TEST_CASE("plan_register: output plans are dropped with the rest of the per-query state",
          "[compression][plan_register]")
{
  using reg_t = sirius::compression::plan_register;
  auto& reg   = reg_t::global();
  reg.clear_all();

  reg.set_table_plan("lineitem", kMeasuredTablePlan);
  reg.set_spill_column_origins(&repo_a(), {reg_t::spill_column_origin{"lineitem", 0}});
  REQUIRE(
    reg.decide_output_plan(&repo_a(), 1, reg_t::plan_quality_gate{3.0, 250.0, 250.0}).has_value());
  REQUIRE(reg.resolve_output_plan(&repo_a()).has_value());

  // Keyed by a repository pointer that QueryEnd frees — a recycled address must
  // not inherit this edge's decision.
  reg.clear_spill_state();
  REQUIRE_FALSE(reg.resolve_output_plan(&repo_a()).has_value());
  REQUIRE(reg.resolve_table_plan("lineitem").has_value());

  reg.clear_all();
}

TEST_CASE("plan_register: clear_spill_state drops per-query state but keeps table plans",
          "[compression][plan_register]")
{
  auto& reg = sirius::compression::plan_register::global();
  reg.clear_all();

  // Offline plans come from input_plan_dir at startup, not from a query.
  reg.set_table_plan("lineitem", "input -> bitpack -> chunk_min, chunk_count, chunk_bits, packed");
  set_plan_1col(&repo_a(), "explored dsl");
  reg.set_spill_column_origins(
    &repo_a(), {sirius::compression::plan_register::spill_column_origin{"lineitem", 0}});

  REQUIRE(reg.resolve_spill_plan(&repo_a()).has_value());
  REQUIRE(reg.resolve_spill_column_origins(&repo_a()).has_value());

  // Query end: everything keyed by a repository pointer must go, because those
  // repositories are destroyed and their addresses can be recycled by the next
  // query's repositories.
  reg.clear_spill_state();

  REQUIRE_FALSE(reg.resolve_spill_plan(&repo_a()).has_value());
  REQUIRE_FALSE(reg.resolve_spill_column_origins(&repo_a()).has_value());
  REQUIRE(reg.decide_spill_plan(&repo_a(), /*replan_after_uses=*/0).verdict ==
          sirius::compression::plan_register::spill_plan_verdict::explore);

  // ...but the offline table plans survive: the next query re-seeds from them.
  REQUIRE(reg.resolve_table_plan("lineitem").has_value());

  reg.clear_all();
}

TEST_CASE("plan_register: clear_all removes spill plans", "[compression][plan_register]")
{
  auto& reg = sirius::compression::plan_register::global();
  reg.clear_all();

  set_plan_1col(&repo_a(), "some dsl");
  REQUIRE(reg.resolve_spill_plan(&repo_a()).has_value());

  reg.clear_all();
  REQUIRE_FALSE(reg.resolve_spill_plan(&repo_a()).has_value());
}
