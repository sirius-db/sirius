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

// End-to-end file-subset serving of a pinned parquet glob (HOST + GPU tiers).
// A pin over three files must serve a scan that names only some of them: the
// pin path coalesces within file boundaries and records which file each chunk
// came from, and try_match_cached_entry serves exactly the chunks the scan's
// files cover. The test asserts the recorded provenance (one file per chunk,
// the pinned set exactly), that two-file and one-file scans return the same
// rows as the unpinned read and log the subset hit, that the exact set is still
// served without the subset path, that a scan naming a file outside the pin is
// a miss that stays correct, and that zone-map pruning and the all-pruned
// sentinel respect the allowed-chunk restriction.

#include "sirius_context.hpp"

#include <catch.hpp>
#include <cucascade/memory/memory_space.hpp>
#include <duckdb.hpp>
#include <op/scan/parquet_gpu_ingestible.hpp>
#include <scan_manager/sirius_scan_manager.hpp>
#include <utils/log_test_utils.hpp>
#include <utils/parquet_fixture_utils.hpp>
#include <utils/sirius_test_env.hpp>

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <set>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace fs = std::filesystem;

namespace {

// File i holds k in [i*N, (i+1)*N) with v = 2k: 200k rows x 2 int64 -> ~3 MiB per
// file, far under the scan batch, so every file is exactly one chunk.
constexpr std::int64_t kRowsPerFile = 200'000;
constexpr int kFiles                = 3;

void require_ok(duckdb::unique_ptr<duckdb::MaterializedQueryResult> const& r, char const* what)
{
  REQUIRE(r);
  if (r->HasError()) { UNSCOPED_INFO(what << " error: " << r->GetError()); }
  REQUIRE_FALSE(r->HasError());
}

void generate_parquet(fs::path const& path, int file_index)
{
  sirius::test::scoped_sirius_disable disable_sirius;
  duckdb::DuckDB gen_db(nullptr);
  duckdb::Connection gen(gen_db);
  auto const lo = static_cast<std::int64_t>(file_index) * kRowsPerFile;
  auto r = gen.Query("COPY (SELECT range AS k, range * 2 AS v FROM range(" + std::to_string(lo) +
                     ", " + std::to_string(lo + kRowsPerFile) + ") ORDER BY k) TO " +
                     sirius::test::sql_literal(path.string()) + " (FORMAT PARQUET);");
  REQUIRE(r);
  REQUIRE_FALSE(r->HasError());
}

void write_config(fs::path const& yaml_path)
{
  std::ofstream f(yaml_path);
  f << "sirius:\n"
       "  topology:\n"
       "    num_gpus: 1\n"
       "  memory:\n"
       "    gpu:\n"
       "      usage_limit_bytes: "
    << (2ull << 30)
    << "\n"
       "      reservation_limit_fraction: 1.0\n"
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
       "    scan_task_batch_size: 100000000\n"
       "    max_sort_partition_bytes: 0\n"
       "    hash_partition_bytes: 100000000\n"
       "    concat_batch_bytes: 100000000\n"
       "    max_build_hash_table_bytes: 90000000\n";
}

void pause_shared_envs()
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
}

sirius::scan_manager::pinned_entry const* find_entry(duckdb::SiriusContext& sirius_ctx,
                                                     std::string_view entry_name)
{
  sirius::scan_manager::pinned_entry const* entry_ptr = nullptr;
  sirius_ctx.get_scan_manager().visit_pinned_entries(
    [&entry_ptr, entry_name](std::string_view name, auto const& e) {
      if (name == entry_name) {
        entry_ptr = &e;
        return true;
      }
      return false;
    });
  return entry_ptr;
}

std::size_t entry_chunk_count(sirius::scan_manager::pinned_entry const& e)
{
  if (e.tier == cucascade::memory::Tier::HOST) { return e.host_chunks.size(); }
  if (!e.device_chunks.empty()) { return e.device_chunks.size(); }
  return e.data_batches_by_column.empty() ? 0 : e.data_batches_by_column.begin()->second.size();
}

/// `read_parquet([...])` over @p paths, quote-safe.
std::string read_parquet_list(std::vector<std::string> const& paths)
{
  std::string sql = "read_parquet([";
  for (std::size_t i = 0; i < paths.size(); ++i) {
    if (i > 0) { sql += ", "; }
    sql += sirius::test::sql_literal(paths[i]);
  }
  sql += "])";
  return sql;
}

/// (count(*), sum(v)) of @p relation under @p where, as printed strings.
std::pair<std::string, std::string> query_agg(duckdb::Connection& con,
                                              std::string const& relation,
                                              std::string const& where = "")
{
  auto r = con.Query("SELECT count(*), sum(v) FROM " + relation + " " + where + ";");
  require_ok(r, "aggregate query");
  return {r->GetValue(0, 0).ToString(), r->GetValue(1, 0).ToString()};
}

/// True when the serve path logged that 'subset_t' served a scan as a file
/// subset with the given "N/M files, K/L chunks" tail.
bool logged_subset_hit(sirius::test::recording_log_sink const& sink, std::string const& shape)
{
  for (auto const& record : sink.records()) {
    if (record.message.find("pinned entry 'subset_t' serves operator") != std::string::npos &&
        record.message.find("as a file subset: " + shape) != std::string::npos) {
      return true;
    }
  }
  return false;
}

bool logged_any_serve(sirius::test::recording_log_sink const& sink)
{
  for (auto const& record : sink.records()) {
    if (record.message.find("pinned entry 'subset_t' serves operator") != std::string::npos) {
      return true;
    }
  }
  return false;
}

}  // namespace

// NB: no [integration]/[shared_context] tag — this TEST_CASE builds its own
// SiriusContext and manages (pauses) the shared envs itself, mirroring the other
// isolated-context pin tests.
TEST_CASE("gpu_execution - a pinned parquet glob serves scans over a subset of its files",
          "[gpu_execution][parquet][pin_table_file_subset]")
{
  pause_shared_envs();

  sirius::test::scratch_dir scratch{"pin_file_subset"};
  auto const& tmp = scratch.path();

  std::vector<std::string> files;
  for (int i = 0; i < kFiles; ++i) {
    auto const path = tmp / ("part_" + std::to_string(i) + ".parquet");
    generate_parquet(path, i);
    files.push_back(path.string());
  }
  // Never pinned: a scan that adds it is a superset of the pin and must miss.
  auto const extra = tmp / "extra.parquet";
  generate_parquet(extra, kFiles);

  auto yaml_path = tmp / "pin_file_subset.yaml";
  write_config(yaml_path);
  REQUIRE(fs::exists(yaml_path));

  auto const glob        = (tmp / "part_*.parquet").string();
  auto const rel_two     = read_parquet_list({files[0], files[2]});
  auto const rel_one     = read_parquet_list({files[1]});
  auto const rel_all     = "read_parquet(" + sirius::test::sql_literal(glob) + ")";
  auto const rel_super   = read_parquet_list({files[0], files[1], files[2], extra.string()});
  auto const where_file2 = "WHERE k >= " + std::to_string(2 * kRowsPerFile);

  {
    sirius::test::shared_test_env local_env(yaml_path);
    auto con = local_env.make_connection();

    require_ok(con.Query("SET enable_duckdb_fallback = false;"), "disable fallback");

    auto sirius_ctx = con.context->registered_state->Get<duckdb::SiriusContext>("sirius_state");
    REQUIRE(sirius_ctx);

    // Unpinned oracle from the same engine, before any pin exists.
    auto const two_disk       = query_agg(con, rel_two);
    auto const one_disk       = query_agg(con, rel_one);
    auto const all_disk       = query_agg(con, rel_all);
    auto const super_disk     = query_agg(con, rel_super);
    auto const two_file2_disk = query_agg(con, rel_two, where_file2);
    REQUIRE(two_disk.first == std::to_string(2 * kRowsPerFile));
    REQUIRE(one_disk.first == std::to_string(kRowsPerFile));
    REQUIRE(all_disk.first == std::to_string(kFiles * kRowsPerFile));
    REQUIRE(super_disk.first == std::to_string((kFiles + 1) * kRowsPerFile));
    REQUIRE(two_file2_disk.first == std::to_string(kRowsPerFile));

    for (auto const* tier : {"gpu", "host"}) {
      DYNAMIC_SECTION("tier = " << tier)
      {
        require_ok(con.Query("CALL pin_table(" + sirius::test::sql_literal(glob) + ", tier='" +
                             std::string(tier) + "', name='subset_t');"),
                   "pin_table");

        // Provenance: one chunk per file, each naming exactly that file.
        auto const* entry = find_entry(*sirius_ctx, "subset_t");
        REQUIRE(entry != nullptr);
        REQUIRE(entry->cache_info.resolved_file_paths.size() == static_cast<std::size_t>(kFiles));
        auto const n_chunks = entry_chunk_count(*entry);
        REQUIRE(n_chunks == static_cast<std::size_t>(kFiles));
        REQUIRE(entry->chunk_file_paths.size() == n_chunks);
        std::set<std::string> recorded;
        for (auto const& chunk_files : entry->chunk_file_paths) {
          REQUIRE(chunk_files.size() == 1);
          recorded.insert(chunk_files.front());
        }
        std::set<std::string> expected;
        for (auto const& f : files) {
          expected.insert(sirius::op::scan::canonical_scan_file_path(f));
        }
        REQUIRE(recorded == expected);

        {
          sirius::test::scoped_recording_log_sink logs{"info"};
          REQUIRE(query_agg(con, rel_two) == two_disk);
          REQUIRE(logged_subset_hit(logs.sink(), "2/3 files, 2/3 chunks"));
        }
        {
          sirius::test::scoped_recording_log_sink logs{"info"};
          REQUIRE(query_agg(con, rel_one) == one_disk);
          REQUIRE(logged_subset_hit(logs.sink(), "1/3 files, 1/3 chunks"));
        }
        {
          // The exact set is served as before, not through the subset path.
          sirius::test::scoped_recording_log_sink logs{"info"};
          REQUIRE(query_agg(con, rel_all) == all_disk);
          REQUIRE_FALSE(logged_any_serve(logs.sink()));
        }
        {
          // A superset names a file the pin cannot provide: a miss, still correct.
          sirius::test::scoped_recording_log_sink logs{"info"};
          REQUIRE(query_agg(con, rel_super) == super_disk);
          REQUIRE_FALSE(logged_any_serve(logs.sink()));
        }
        {
          // Zone maps prune file 0's chunk out of the two-file subset; the
          // selects-nothing filter leaves only the sentinel, which must be one of
          // the allowed chunks (the GPU filter then empties it).
          sirius::test::scoped_recording_log_sink logs{"info"};
          REQUIRE(query_agg(con, rel_two, where_file2) == two_file2_disk);
          REQUIRE(logged_subset_hit(logs.sink(), "2/3 files, 2/3 chunks"));
          auto none = con.Query("SELECT count(*) FROM " + rel_two + " WHERE k < 0;");
          require_ok(none, "selects-nothing subset query");
          REQUIRE(none->GetValue(0, 0).ToString() == "0");
        }

        require_ok(con.Query("CALL unpin_table('subset_t');"), "unpin");
      }
    }
  }
}
