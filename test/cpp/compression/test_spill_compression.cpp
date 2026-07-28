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
      .set_per_host_capacity(512ull << 20)
      .use_host_per_gpu()
      .set_reservation_fraction_per_host(0.75)
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

/// Reset spill state: clear every cached plan and enable spill compression with
/// a small explorer budget (the explore case is the only one that runs it).
void reset_spill_state(bool enabled = true, std::uint64_t replan_after_uses = 0)
{
  sirius::compression::plan_register::global().clear_all();
  sirius::compression::set_spill_compression_settings(enabled,
                                                      /*explore_beam_width=*/4,
                                                      /*explore_max_bytes=*/8ull << 20,
                                                      /*max_compressed_fraction=*/0.95,
                                                      replan_after_uses);
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

/// Sum the first column of a gpu_table_representation as int64.
int64_t gpu_col_sum(const cucascade::gpu_table_representation& rep)
{
  auto result = cudf::reduce(rep.get_table_view().column(0),
                             *cudf::make_sum_aggregation<cudf::reduce_aggregation>(),
                             cudf::data_type{cudf::type_id::INT64},
                             cudf::get_default_stream(),
                             rmm::mr::get_current_device_resource_ref());
  return static_cast<cudf::numeric_scalar<int64_t>*>(result.get())
    ->value(cudf::get_default_stream());
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
  sirius::compression::plan_register::global().set_spill_plan(&repo_a(), kOneColDsl);

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
  sirius::compression::plan_register::global().set_spill_plan(&repo_a(), kOneColDsl);

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
  sirius::compression::plan_register::global().set_spill_plan(&repo_a(), kOneColDsl);

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

  // The explored plan is now cached for this edge and non-empty.
  auto discovered = reg.resolve_spill_plan(&repo_b());
  REQUIRE(discovered.has_value());
  REQUIRE_FALSE(discovered->dsl.empty());
  REQUIRE(discovered->viable);

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
  REQUIRE(after->dsl == discovered->dsl);
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
  sirius::compression::plan_register::global().set_spill_plan(&repo_a(), kOneColDsl);

  auto batch = make_int32_gpu_batch(1000);

  spill_to(batch, e.host_space, &repo_a());
  REQUIRE_FALSE(is_compressed_host(*batch));
  REQUIRE(get_tier(*batch) == cucascade::memory::Tier::HOST);

  reset_spill_state(false);
}

TEST_CASE("spill compression: batch with no source edge spills uncompressed",
          "[compression][spill][isolated_context]")
{
  if (!has_gpu()) {
    SUCCEED("No GPU available — skipping spill compression tests");
    return;
  }

  auto& e = env();
  reset_spill_state();

  auto batch = make_int32_gpu_batch(1000);

  // No repo pointer (e.g. a batch not sourced from a repository): the spill path
  // has no plan key, so it must fall through to uncompressed rather than throw.
  sirius::convertible_data_batch w(batch, nullptr);
  REQUIRE(w.convert({e.host_space}, e.stream(), *e.mgr, true).has_value());

  REQUIRE_FALSE(is_compressed_host(*batch));
  REQUIRE(get_tier(*batch) == cucascade::memory::Tier::HOST);

  reset_spill_state(false);
}

TEST_CASE("spill compression: column-count mismatch falls back without throwing",
          "[compression][spill][isolated_context]")
{
  if (!has_gpu()) {
    SUCCEED("No GPU available — skipping spill compression tests");
    return;
  }

  auto& e = env();
  reset_spill_state();
  // 2-column plan against a 1-column batch: Simpatico throws inside the
  // converter, which must not escape convert().
  sirius::compression::plan_register::global().set_spill_plan(&repo_a(), kTwoColDsl);

  auto batch = make_int32_gpu_batch(1000);

  sirius::convertible_data_batch w(batch, &repo_a());
  REQUIRE_NOTHROW(w.convert({e.host_space}, e.stream(), *e.mgr, true));

  REQUIRE_FALSE(is_compressed_host(*batch));
  REQUIRE(get_tier(*batch) == cucascade::memory::Tier::HOST);

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
                                                      /*replan_after_uses=*/0);
  // A plan whose output is the same width as its input cannot reach 0.75.
  sirius::compression::plan_register::global().set_spill_plan(&repo_a(), kNonCompressingDsl);

  auto batch = make_int32_gpu_batch(1000);

  spill_to(batch, e.host_space, &repo_a());
  REQUIRE_FALSE(is_compressed_host(*batch));
  REQUIRE(get_tier(*batch) == cucascade::memory::Tier::HOST);

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
                                                      /*replan_after_uses=*/0);  // never expire
  reg.set_spill_plan(&repo_a(), kNonCompressingDsl);

  // First batch: compression runs, misses the threshold, and the edge is marked.
  auto batch = make_int32_gpu_batch(1000);
  spill_to(batch, e.host_space, &repo_a());
  REQUIRE_FALSE(is_compressed_host(*batch));

  auto state = reg.resolve_spill_plan(&repo_a());
  REQUIRE(state.has_value());
  REQUIRE_FALSE(state->viable);

  // With the entry marked unviable, the spill path must now decide to skip
  // rather than compress again.
  const auto decision = reg.decide_spill_plan(&repo_a(), /*replan_after_uses=*/0);
  REQUIRE(decision.verdict == sirius::compression::plan_register::spill_plan_verdict::skip);

  // A second batch still spills, uncompressed, and stays skipped.
  auto batch2 = make_int32_gpu_batch(1000);
  spill_to(batch2, e.host_space, &repo_a());
  REQUIRE_FALSE(is_compressed_host(*batch2));
  REQUIRE_FALSE(reg.resolve_spill_plan(&repo_a())->viable);

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
                                                      /*replan_after_uses=*/1);
  reg.set_spill_plan(&repo_a(), kNonCompressingDsl);

  // First batch: rejected, edge marked unviable, one use recorded.
  auto batch = make_int32_gpu_batch(5000);
  spill_to(batch, e.host_space, &repo_a());
  REQUIRE_FALSE(is_compressed_host(*batch));
  REQUIRE_FALSE(reg.resolve_spill_plan(&repo_a())->viable);

  // The entry has now expired, so the verdict is explore rather than skip —
  // a previously written-off edge gets another chance.
  REQUIRE(reg.decide_spill_plan(&repo_a(), /*replan_after_uses=*/1).verdict ==
          sirius::compression::plan_register::spill_plan_verdict::explore);

  // The next batch re-explores and finds a plan that does compress, so the edge
  // recovers instead of staying disabled for the rest of the query.
  auto batch2 = make_int32_gpu_batch(5000);
  spill_to(batch2, e.host_space, &repo_a());
  REQUIRE(is_compressed_host(*batch2));

  auto recovered = reg.resolve_spill_plan(&repo_a());
  REQUIRE(recovered.has_value());
  REQUIRE(recovered->viable);
  REQUIRE(recovered->dsl != kNonCompressingDsl);

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

  reg.set_spill_plan(&repo_a(), "some dsl");
  auto d = reg.decide_spill_plan(&repo_a(), 0);
  REQUIRE(d.verdict == verdict::use);
  REQUIRE(d.dsl == "some dsl");

  // A failed attempt -> skip, but the DSL and use count are retained so the
  // entry still ages toward expiry.
  reg.conclude_spill_attempt(&repo_a(), /*compressed_ok=*/false, /*base_interval=*/0);
  REQUIRE(reg.decide_spill_plan(&repo_a(), 0).verdict == verdict::skip);
  REQUIRE(reg.resolve_spill_plan(&repo_a())->dsl == "some dsl");

  // Expiry overrides both use and skip.
  reg.note_spill_plan_use(&repo_a());
  REQUIRE(reg.resolve_spill_plan(&repo_a())->uses == 1);
  REQUIRE(reg.decide_spill_plan(&repo_a(), /*replan_after_uses=*/1).verdict == verdict::explore);
  // ...but 0 means "never expire", so the skip verdict stands.
  REQUIRE(reg.decide_spill_plan(&repo_a(), /*replan_after_uses=*/0).verdict == verdict::skip);

  // Re-installing a plan clears the verdict and resets the counter.
  reg.set_spill_plan(&repo_a(), "fresh dsl");
  auto fresh = reg.resolve_spill_plan(&repo_a());
  REQUIRE(fresh->viable);
  REQUIRE(fresh->uses == 0);

  reg.clear_all();
}

TEST_CASE("plan_register: replan interval backs off when a re-explore learns nothing",
          "[compression][plan_register]")
{
  auto& reg                     = sirius::compression::plan_register::global();
  constexpr std::uint64_t kBase = 8;
  const std::string kPlan       = "plan one";
  const std::string kOtherPlan  = "plan two";

  auto interval_of = [&](const cucascade::shared_data_repository* r) {
    return reg.resolve_spill_plan(r)->replan_interval;
  };

  SECTION("same plan, still working -> double")
  {
    reg.clear_all();
    reg.set_spill_plan(&repo_a(), kPlan);
    reg.conclude_spill_attempt(&repo_a(), true, kBase);  // first install: no backoff
    REQUIRE(interval_of(&repo_a()) == 0);                // still on the configured schedule

    reg.set_spill_plan(&repo_a(), kPlan);  // re-explore returned the same plan
    reg.conclude_spill_attempt(&repo_a(), true, kBase);
    REQUIRE(interval_of(&repo_a()) == kBase * 2);

    reg.set_spill_plan(&repo_a(), kPlan);
    reg.conclude_spill_attempt(&repo_a(), true, kBase);
    REQUIRE(interval_of(&repo_a()) == kBase * 4);
  }

  SECTION("different plan that still fails the threshold -> double")
  {
    reg.clear_all();
    reg.set_spill_plan(&repo_a(), kPlan);
    reg.conclude_spill_attempt(&repo_a(), false, kBase);

    reg.set_spill_plan(&repo_a(), kOtherPlan);            // explorer found something else...
    reg.conclude_spill_attempt(&repo_a(), false, kBase);  // ...that still does not compress
    REQUIRE(interval_of(&repo_a()) == kBase * 2);
  }

  SECTION("different plan that works -> reset to the configured interval")
  {
    reg.clear_all();
    reg.set_spill_plan(&repo_a(), kPlan);
    reg.conclude_spill_attempt(&repo_a(), true, kBase);
    // Stretch the interval out first.
    reg.set_spill_plan(&repo_a(), kPlan);
    reg.conclude_spill_attempt(&repo_a(), true, kBase);
    REQUIRE(interval_of(&repo_a()) == kBase * 2);

    reg.set_spill_plan(&repo_a(), kOtherPlan);
    reg.conclude_spill_attempt(&repo_a(), true, kBase);
    REQUIRE(interval_of(&repo_a()) == kBase);
  }

  SECTION("viability recovering on the same plan -> reset")
  {
    reg.clear_all();
    reg.set_spill_plan(&repo_a(), kPlan);
    reg.conclude_spill_attempt(&repo_a(), false, kBase);  // not worth compressing
    reg.set_spill_plan(&repo_a(), kPlan);
    reg.conclude_spill_attempt(&repo_a(), false, kBase);  // still not — back off
    REQUIRE(interval_of(&repo_a()) == kBase * 2);

    // Same plan, but it now compresses: the edge recovered, so resume checking
    // on schedule.
    reg.set_spill_plan(&repo_a(), kPlan);
    reg.conclude_spill_attempt(&repo_a(), true, kBase);
    REQUIRE(interval_of(&repo_a()) == kBase);
    REQUIRE(reg.resolve_spill_plan(&repo_a())->viable);
  }

  SECTION("a stretched interval overrides the configured one when deciding")
  {
    using verdict = sirius::compression::plan_register::spill_plan_verdict;
    reg.clear_all();
    reg.set_spill_plan(&repo_a(), kPlan);
    reg.conclude_spill_attempt(&repo_a(), true, kBase);
    reg.set_spill_plan(&repo_a(), kPlan);
    reg.conclude_spill_attempt(&repo_a(), true, kBase);
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

TEST_CASE("plan_register: note_spill_plan_use on an absent edge is a no-op",
          "[compression][plan_register]")
{
  auto& reg = sirius::compression::plan_register::global();
  reg.clear_all();

  // Must not create an entry — a use recorded before any plan exists would
  // otherwise age a plan that has not been installed yet.
  reg.note_spill_plan_use(&repo_a());
  REQUIRE_FALSE(reg.resolve_spill_plan(&repo_a()).has_value());
  reg.conclude_spill_attempt(&repo_a(), /*compressed_ok=*/false, /*base_interval=*/8);
  REQUIRE_FALSE(reg.resolve_spill_plan(&repo_a()).has_value());

  reg.clear_all();
}

TEST_CASE("plan_register: spill plan round-trips per repository", "[compression][plan_register]")
{
  auto& reg = sirius::compression::plan_register::global();
  reg.clear_all();

  REQUIRE_FALSE(reg.resolve_spill_plan(&repo_a()).has_value());

  const std::string dsl = "input -> delta -> differences\n";
  reg.set_spill_plan(&repo_a(), dsl);

  auto result = reg.resolve_spill_plan(&repo_a());
  REQUIRE(result.has_value());
  REQUIRE(result->dsl == dsl);
  REQUIRE(result->viable);
  REQUIRE(result->uses == 0);

  // Plans are per-edge: another repository is unaffected.
  REQUIRE_FALSE(reg.resolve_spill_plan(&repo_b()).has_value());

  // An empty plan reads back as "no plan".
  reg.set_spill_plan(&repo_a(), {});
  REQUIRE_FALSE(reg.resolve_spill_plan(&repo_a()).has_value());

  reg.clear_all();
}

TEST_CASE("plan_register: clear_spill_plan removes only the named edge",
          "[compression][plan_register]")
{
  auto& reg = sirius::compression::plan_register::global();
  reg.clear_all();

  reg.set_spill_plan(&repo_a(), "plan a");
  reg.set_spill_plan(&repo_b(), "plan b");

  reg.clear_spill_plan(&repo_a());
  REQUIRE_FALSE(reg.resolve_spill_plan(&repo_a()).has_value());
  REQUIRE(reg.resolve_spill_plan(&repo_b()).has_value());

  reg.clear_all();
}

TEST_CASE("plan_register: clear_all removes spill plans", "[compression][plan_register]")
{
  auto& reg = sirius::compression::plan_register::global();
  reg.clear_all();

  reg.set_spill_plan(&repo_a(), "some dsl");
  REQUIRE(reg.resolve_spill_plan(&repo_a()).has_value());

  reg.clear_all();
  REQUIRE_FALSE(reg.resolve_spill_plan(&repo_a()).has_value());
}
