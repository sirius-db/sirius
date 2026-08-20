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

// A hash join whose BUILD SIDE is spilled to HOST mid-build and read back, still returning the
// right answer.
//
// A join's build side sits in its build-port repository, GPU-resident and idle, until the join task
// claims it — the state the downgrade executor picks spill candidates from. Nothing else covers a
// real operator there: test/cpp/downgrade/* spills synthetic repositories with no operator
// attached, and test_oom_reschedule.cpp uses stub tasks.
//
// Three details are load-bearing:
//
//   * The eviction threshold forces the spill, not a tight budget. downgrade_trigger drops to 0.05
//     of a 2 GiB budget (~102 MiB) so the ~384 MB build side is evicted while still accumulating,
//     while usage_limit_bytes stays at 2 GiB so the join can upgrade it back.
//   * The LEFT join pins the build side. Table shape does NOT control it — the planner builds the
//     side smaller IN BYTES, so an INNER join here builds the NARROW table. An outer join's inner
//     side cannot be swapped. Every fact key exists in dim, so nothing is NULL-padded and the
//     result matches the inner join.
//   * The round trip is asserted per BATCH ID, not in bytes. Spill volume has no operator, port,
//     batch or direction attribution and counts re-spills repeatedly, so no byte threshold can show
//     that build-side data specifically made the trip. The engine traces which batches reach the
//     build port and which cross tiers in each direction; the test intersects those.

#include "config.hpp"
#include "downgrade/downgrade_executor.hpp"
#include "log/logging.hpp"
#include "sirius_context.hpp"

#include <catch.hpp>
#include <cucascade/memory/common.hpp>
#include <duckdb.hpp>
#include <unistd.h>
#include <utils/parquet_fixture_utils.hpp>
#include <utils/sirius_test_env.hpp>

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <map>
#include <regex>
#include <set>
#include <sstream>
#include <string>

namespace fs = std::filesystem;

namespace {

// Build side (pinned there by the LEFT join): 2M rows x 24 int64 -> ~384 MB decoded, well above
// the eviction threshold below, so the build side alone creates the pressure.
constexpr std::int64_t kDimRows = 2'000'000;
constexpr int kDimPayloadCols   = 23;  // plus the key column
constexpr std::size_t kDimBytes =
  static_cast<std::size_t>(kDimRows) * (kDimPayloadCols + 1) * sizeof(std::int64_t);

// Probe side: one column, so the entire side is small -- ~24 MB decoded.
constexpr std::int64_t kFactRows = 3'000'000;
constexpr std::size_t kFactBytes = static_cast<std::size_t>(kFactRows) * sizeof(std::int64_t);

// GPU budget is generous; only the eviction threshold is tight (see the file header).
constexpr std::size_t kGpuBudgetBytes = 2048ull << 20;  // 2 GiB
constexpr double kDowngradeTrigger    = 0.05;           // ~102 MiB
constexpr double kDowngradeStop       = 0.025;          // ~51 MiB
constexpr std::size_t kScanBatchBytes = 8ull << 20;  // small batches -> granular spill candidates

/// Ceiling on the probe side's real footprint: kFactBytes is logical row arithmetic, but the engine
/// sizes that table at 24,375,040. The bounds below compare against MEASURED bytes, so they have to
/// clear the measured value — a margin thinner than the layout overhead proves nothing.
constexpr std::size_t kProbeSideCeilingBytes = 32'000'000;

/// The counter sums bytes per conversion, so re-spills count repeatedly. Four probe sides' worth
/// allows four full re-spill cycles and still cannot be reached without build-side data.
constexpr std::size_t kBuildSideEvidenceBytes = 4 * kProbeSideCeilingBytes;

/// Half of `dim` is ~8x anything `fact` can measure, so a flipped plan cannot satisfy it.
constexpr std::size_t kBuildSideFloorBytes = kDimBytes / 2;

/// The DuckDB CPU answer for the join, computed before Sirius is in the picture.
struct cpu_reference {
  std::string count_star;
  std::string sum_dim_payload;
};

std::string dim_projection()
{
  std::string cols = "range AS k";
  for (int i = 1; i <= kDimPayloadCols; ++i) {
    cols += ", range * " + std::to_string(i) + " AS c" + std::to_string(i);
  }
  return cols;
}

/// `d.c1 + d.c2 + ... + d.cN` — every payload column, so none can be pruned away.
std::string dim_payload_sum()
{
  std::string expr = "d.c1";
  for (int i = 2; i <= kDimPayloadCols; ++i) {
    expr += " + d.c" + std::to_string(i);
  }
  return expr;
}

std::string join_query(fs::path const& dim_path, fs::path const& fact_path)
{
  // Summing EVERY payload column is deliberate: reference one and column pruning reads a
  // two-column build side, so the ~384 MB that creates the pressure never materializes.
  return "SELECT count(*) AS n, sum(" + dim_payload_sum() +
         ") AS sc "
         "FROM read_parquet(" +
         sirius::test::sql_literal(fact_path.string()) +
         ") AS f "
         "LEFT JOIN read_parquet(" +
         sirius::test::sql_literal(dim_path.string()) + ") AS d ON f.k = d.k;";
}

/// Sirius is disabled here so the extension callback never builds a SiriusContext on this throwaway
/// instance — the tight-trigger one comes later.
cpu_reference generate_inputs_and_reference(fs::path const& dim_path, fs::path const& fact_path)
{
  sirius::test::scoped_sirius_disable disable_sirius;
  duckdb::DuckDB gen_db(nullptr);
  duckdb::Connection gen(gen_db);

  auto dim =
    gen.Query("COPY (SELECT " + dim_projection() + " FROM range(" + std::to_string(kDimRows) +
              ")) TO " + sirius::test::sql_literal(dim_path.string()) + " (FORMAT PARQUET);");
  REQUIRE(dim);
  REQUIRE_FALSE(dim->HasError());

  auto fact = gen.Query("COPY (SELECT range % " + std::to_string(kDimRows) + " AS k FROM range(" +
                        std::to_string(kFactRows) + ")) TO " +
                        sirius::test::sql_literal(fact_path.string()) + " (FORMAT PARQUET);");
  REQUIRE(fact);
  REQUIRE_FALSE(fact->HasError());

  auto ref = gen.Query(join_query(dim_path, fact_path));
  REQUIRE(ref);
  if (ref->HasError()) { UNSCOPED_INFO("CPU reference error: " << ref->GetError()); }
  REQUIRE_FALSE(ref->HasError());
  REQUIRE(ref->RowCount() == 1);

  return cpu_reference{ref->GetValue(0, 0).ToString(), ref->GetValue(1, 0).ToString()};
}

/// Mirrors integration.yaml, but with an eviction threshold far below the build side so the
/// downgrade monitor fires during the build, and small scan batches so it has candidates.
void write_config(fs::path const& yaml_path)
{
  std::ofstream f(yaml_path);
  f << "sirius:\n"
       "  topology:\n"
       "    num_gpus: 1\n"
       "  memory:\n"
       "    gpu:\n"
       "      usage_limit_bytes: "
    << kGpuBudgetBytes
    << "\n"
       "      reservation_limit_fraction: 1.0\n"
       "      downgrade_trigger_fraction: "
    << kDowngradeTrigger
    << "\n"
       "      downgrade_stop_fraction: "
    << kDowngradeStop
    << "\n"
       "    host:\n"
       "      capacity_bytes: 32000000000\n"
       "      initial_number_pools: 10\n"
       "      pool_size: 512\n"
       "      block_size: 1048576\n"
       "  executor:\n"
       "    pipeline:\n"
       "      num_threads: 4\n"
       "    task_creator:\n"
       "      num_threads: 2\n"
       "    downgrade:\n"
       "      num_threads: 1\n"
       "      monitor_period: 10ms\n"
       "  operator_params:\n"
       "    scan_task_batch_size: "
    << kScanBatchBytes
    << "\n"
       "    max_sort_partition_bytes: 0\n"
       "    hash_partition_bytes: 1000000000\n"
       "    concat_batch_bytes: 100000000\n"
       // Above the ~384 MB build side, so the join keeps one hash table rather than splitting into
       // partitions and the build side parks as one repository's worth of batches.
       "    max_build_hash_table_bytes: 800000000\n";
}

/// Points Sirius at @p dir for debug logging, restoring what was set before. The build-side
/// assertion reads the join's own sizing line, which is DEBUG level.
class scoped_debug_log_dir {
 public:
  explicit scoped_debug_log_dir(fs::path const& dir) : _dir(dir)
  {
    fs::create_directories(dir);
    save_env("SIRIUS_LOG_DIR", _prev_dir, _had_prev_dir);
    save_env("SIRIUS_LOG_LEVEL", _prev_level, _had_prev_level);
    // The extension callback overwrites Config::LOG_* only when the env var is set, so restoring
    // the environment alone would leave this directory and level installed for every later test.
    _prev_cfg_dir   = duckdb::Config::LOG_DIR;
    _prev_cfg_level = duckdb::Config::LOG_LEVEL;
    setenv("SIRIUS_LOG_DIR", dir.c_str(), 1);
    setenv("SIRIUS_LOG_LEVEL", "debug", 1);
  }

  /// Restore the environment, the process-global config and the sink, then drop the directory.
  /// Call once the local SiriusContext is gone; the destructor repeats it if a test throws first.
  void restore()
  {
    if (_restored) { return; }
    _restored = true;
    restore_env("SIRIUS_LOG_DIR", _prev_dir, _had_prev_dir);
    restore_env("SIRIUS_LOG_LEVEL", _prev_level, _had_prev_level);
    duckdb::Config::LOG_DIR   = _prev_cfg_dir;
    duckdb::Config::LOG_LEVEL = _prev_cfg_level;
    duckdb::install_configured_log_sink(nullptr);
    std::error_code ec;
    fs::remove_all(_dir, ec);
  }

  ~scoped_debug_log_dir() { restore(); }

  scoped_debug_log_dir(scoped_debug_log_dir const&)            = delete;
  scoped_debug_log_dir& operator=(scoped_debug_log_dir const&) = delete;

 private:
  static void save_env(char const* name, std::string& out, bool& had)
  {
    if (char const* prev = std::getenv(name)) {
      out = prev;
      had = true;
    }
  }

  static void restore_env(char const* name, std::string const& prev, bool had)
  {
    if (had) {
      setenv(name, prev.c_str(), 1);
    } else {
      unsetenv(name);
    }
  }

  fs::path _dir;
  std::string _prev_dir;
  std::string _prev_level;
  std::string _prev_cfg_dir;
  std::string _prev_cfg_level;
  bool _had_prev_dir{false};
  bool _had_prev_level{false};
  bool _restored{false};
};

/// Everything the test reads back out of the debug log.
struct log_facts {
  std::size_t build_side_bytes = 0;       ///< largest "build side N bytes" the join reported
  std::set<std::uint64_t> build_batches;  ///< batches published to the join's build port
  std::set<std::uint64_t> spilled;        ///< batches the downgrade executor moved off GPU
  std::map<std::uint64_t, int> upgraded;  ///< batches restored to GPU -> their column count
};

/// Which side is built is a planner decision and a batch's port is not visible at the tier layer,
/// so both are read back from the engine's own traces rather than assumed.
log_facts read_log_facts(fs::path const& log_dir)
{
  (void)sirius::log::get_sink()->flush();
  std::regex const build_side_re(R"(build side ([0-9]+) bytes)");
  std::regex const build_batch_re(R"(\[hash_join\] id [0-9]+ build-port batch ([0-9]+))");
  std::regex const spill_re(R"(\[downgrade\] batch ([0-9]+) spilled)");
  std::regex const upgrade_re(
    R"(\[upgrade\] batch ([0-9]+) restored to GPU for processing, ([0-9]+) columns)");

  log_facts facts;
  std::error_code ec;
  for (auto const& entry : fs::recursive_directory_iterator(log_dir, ec)) {
    if (!entry.is_regular_file()) { continue; }
    std::ifstream in(entry.path());
    std::stringstream ss;
    ss << in.rdbuf();
    auto const text = ss.str();

    for (auto it = std::sregex_iterator(text.begin(), text.end(), build_side_re);
         it != std::sregex_iterator();
         ++it) {
      facts.build_side_bytes =
        std::max(facts.build_side_bytes, static_cast<std::size_t>(std::stoull((*it)[1].str())));
    }
    auto collect = [&text](std::regex const& re, std::set<std::uint64_t>& out) {
      for (auto it = std::sregex_iterator(text.begin(), text.end(), re);
           it != std::sregex_iterator();
           ++it) {
        out.insert(std::stoull((*it)[1].str()));
      }
    };
    collect(build_batch_re, facts.build_batches);
    collect(spill_re, facts.spilled);
    for (auto it = std::sregex_iterator(text.begin(), text.end(), upgrade_re);
         it != std::sregex_iterator();
         ++it) {
      facts.upgraded[std::stoull((*it)[1].str())] = std::stoi((*it)[2].str());
    }
  }
  return facts;
}

/// Build-side batches that went to a lower tier and came back.
///
/// Attribution is by SCHEMA, not by port. The join's build port only ever holds one batch — the
/// concat-folded build side — and that batch is created late and consumed promptly, so it is not
/// what spills; its inputs are. Those carry dim's full width, and dim is the only relation here
/// with more than one column, so a round-tripped batch of that width is build-side data.
std::vector<std::uint64_t> build_side_batches_round_tripped(log_facts const& facts)
{
  constexpr int kDimColumns = kDimPayloadCols + 1;
  std::vector<std::uint64_t> ids;
  for (auto const& [id, columns] : facts.upgraded) {
    if (columns == kDimColumns && facts.spilled.count(id) != 0) { ids.push_back(id); }
  }
  return ids;
}

/// Repository-sourced bytes spilled out of every GPU space, summed across downgrade executors.
std::size_t gpu_repository_bytes_downgraded(duckdb::SiriusContext const& ctx)
{
  std::size_t total = 0;
  for (auto const& executor : ctx.get_downgrade_executors()) {
    if (executor->get_space_id().tier != cucascade::memory::Tier::GPU) { continue; }
    total += executor->repository_bytes_downgraded();
  }
  return total;
}

}  // namespace

// NB: no [integration]/[shared_context] tag — those bind a shared env, which would fight this
// test's own tight-trigger local env. Like the other isolated-context integration tests, this
// TEST_CASE pauses the shared envs itself.
TEST_CASE("gpu_execution - hash join build side spills to HOST and is read back mid-build",
          "[gpu_execution][join][memory_pressure][downgrade]")
{
  if (sirius::test::g_shared_env && sirius::test::g_shared_env->is_active()) {
    sirius::test::g_shared_env->pause();
  }
  if (sirius::test::g_integration_env && sirius::test::g_integration_env->is_active()) {
    sirius::test::g_integration_env->pause();
  }
  if (sirius::test::g_integration_env_2gpu && sirius::test::g_integration_env_2gpu->is_active()) {
    sirius::test::g_integration_env_2gpu->pause();
  }

  sirius::test::scratch_dir scratch{"join_mem_pressure"};
  auto const& tmp = scratch.path();

  auto dim_path  = tmp / "dim.parquet";
  auto fact_path = tmp / "fact.parquet";
  auto reference = generate_inputs_and_reference(dim_path, fact_path);

  auto yaml_path = tmp / "join_mem_pressure.yaml";
  write_config(yaml_path);
  REQUIRE(fs::exists(yaml_path));

  // Outside `scratch`: the guard removes it in restore(), once the sink has been pointed away.
  auto log_dir = fs::temp_directory_path() /
                 ("sirius_test_join_mem_pressure_logs_" + std::to_string(::getpid()));
  std::error_code log_ec;
  fs::remove_all(log_dir, log_ec);  // clear a previous run's files rather than read them back
  scoped_debug_log_dir log_guard{log_dir};

  {
    sirius::test::shared_test_env local_env(yaml_path);
    auto con = local_env.make_connection();

    // Force GPU execution so a spill-path failure surfaces instead of silently falling back.
    auto fb = con.Query("SET enable_duckdb_fallback = false;");
    REQUIRE(fb);
    REQUIRE_FALSE(fb->HasError());

    auto sirius_ctx = con.context->registered_state->Get<duckdb::SiriusContext>("sirius_state");
    REQUIRE(sirius_ctx);

    auto const before = gpu_repository_bytes_downgraded(*sirius_ctx);

    auto result = con.Query(join_query(dim_path, fact_path));
    REQUIRE(result);
    if (result->HasError()) { UNSCOPED_INFO("GPU join error: " << result->GetError()); }
    REQUIRE_FALSE(result->HasError());
    REQUIRE(result->RowCount() == 1);

    // (1) Correctness: the join survived the spill with the right answer.
    REQUIRE(result->GetValue(0, 0).ToString() == reference.count_star);
    REQUIRE(result->GetValue(1, 0).ToString() == reference.sum_dim_payload);

    auto const facts = read_log_facts(log_dir);

    // (2) `dim` really is the build side. A planner change that swaps the sides would otherwise
    // leave the test green while it exercised the probe side.
    UNSCOPED_INFO("hash join reported build side = "
                  << facts.build_side_bytes << " bytes; must exceed " << kBuildSideFloorBytes);
    REQUIRE(facts.build_side_bytes > kBuildSideFloorBytes);

    // (3) The claim itself: a build-side batch was spilled off GPU and restored before the join
    // consumed it. Bytes cannot show this — the aggregate counter has no operator, batch or
    // direction attribution — so it is asserted per batch id, with the side identified by width.
    auto const round_tripped = build_side_batches_round_tripped(facts);
    UNSCOPED_INFO("spilled=" << facts.spilled.size() << " upgraded=" << facts.upgraded.size()
                             << " build-side round-tripped=" << round_tripped.size()
                             << "; the join's folded build-port batch itself stays resident ("
                             << facts.build_batches.size() << " such batch(es))");
    REQUIRE_FALSE(round_tripped.empty());

    // (4) And the spill was substantial, not a single stray batch.
    auto const spilled = gpu_repository_bytes_downgraded(*sirius_ctx) - before;
    UNSCOPED_INFO("repository bytes downgraded during the join = " << spilled);
    REQUIRE(spilled > kBuildSideEvidenceBytes);
  }

  // The local context is gone; put the global log config and sink back before the directory goes.
  log_guard.restore();
}
