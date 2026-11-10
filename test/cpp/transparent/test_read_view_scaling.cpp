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

#include "op/scan/table_scan/bound_read_view.hpp"
#include "transparent/read_view_registry.hpp"
#include "utils/dynamic_filter_test_utils.hpp"
#include "utils/gpu_execution_fixture.hpp"

#include <duckdb/execution/physical_plan_generator.hpp>

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <set>
#include <sstream>
#include <string>
#include <vector>

namespace {
using namespace sirius::op::scan;
using namespace sirius::transparent;
using clock_type = std::chrono::steady_clock;

struct temporary_files {
  std::filesystem::path root;
  ~temporary_files()
  {
    std::error_code error;
    std::filesystem::remove_all(root, error);
  }
};

bound_read_view independent_view(bound_read_view const& source)
{
  auto copy     = source;
  copy.identity = std::make_shared<bound_read_identity const>(*source.identity);
  return copy;
}

double elapsed_us(clock_type::time_point start, clock_type::time_point end)
{
  return std::chrono::duration<double, std::micro>(end - start).count();
}

double percentile(std::vector<double> samples, unsigned percent)
{
  std::sort(samples.begin(), samples.end());
  return samples[(samples.size() * percent + 99) / 100 - 1];
}

std::string plan_sql(std::vector<std::string> const& paths, unsigned scans, bool join)
{
  std::string sql;
  for (unsigned index = 0; index < scans; ++index) {
    auto source = "read_parquet('" + paths[index] + "') t" + std::to_string(index);
    if (join) {
      sql += index == 0 ? "SELECT t0.id FROM " + source
                        : " JOIN " + source + " ON t0.id=t" + std::to_string(index) + ".id";
    } else {
      if (index) sql += " UNION ALL ";
      sql += "SELECT id FROM " + source;
    }
  }
  return sql;
}
}  // namespace

// An opt-in phase microbenchmark, not a query latency or optimizer benchmark.
// Real logical/physical plans supply the views; registry allocation and fresh
// identity copies are outside the timers. No query execution is timed.
TEST_CASE_METHOD(sirius::test::GpuExecutionFixture,
                 "Read-view matching scales with scan count in join and union plans",
                 "[.][integration][read_view_scaling]")
{
  namespace fs           = std::filesystem;
  auto const join        = GENERATE(false, true);
  auto const long_prefix = GENERATE(false, true);
  auto const reversed    = GENERATE(false, true);
  temporary_files files{temp_db_path + "_read_view_scaling"};
  auto directory = files.root;
  // Equal-length paths differ only in their final file number. The second
  // series increases the common prefix without increasing files per scan.
  if (long_prefix) {
    for (unsigned index = 0; index < 6; ++index)
      directory /= std::string(160, 'p');
  }
  fs::create_directories(directory);
  run_ok("SET gpu_execution=false");
  // Keep join enumeration out of this scan-matching measurement.
  sirius::test::disabled_optimizers_guard optimizer_guard{*con, "join_order,build_side_probe_side"};
  std::vector<std::string> paths;
  for (unsigned index = 0; index < 128; ++index) {
    std::ostringstream name;
    name << "part-" << std::setw(3) << std::setfill('0') << index << ".parquet";
    paths.push_back((directory / name.str()).string());
    if (index == 0) {
      run_ok("COPY (SELECT 1::BIGINT AS id) TO '" + paths.back() + "' (FORMAT PARQUET)");
    } else {
      fs::create_hard_link(paths.front(), paths.back());
    }
  }

  for (unsigned scans : {1u, 4u, 8u, 16u, 32u, 64u, 128u}) {
    CAPTURE(join, long_prefix, reversed, scans);
    run_ok("BEGIN TRANSACTION READ ONLY");
    auto plan          = con->ExtractPlan(plan_sql(paths, scans, join));
    auto logical_views = capture_bound_read_views(*plan, *con->context);
    duckdb::PhysicalPlanGenerator generator(*con->context);
    auto physical_plan  = generator.Plan(std::move(plan));
    auto physical_views = capture_bound_read_views(physical_plan->Root(), *con->context);
    REQUIRE(logical_views.size() == scans);
    REQUIRE(physical_views.size() == scans);
    std::set<std::string> identities;
    auto const canonical_bytes = logical_views.front().view.identity->fingerprint.canonical.size();
    // Normalize the input order outside the timed phases, so natural/reverse
    // have a defined meaning even if DuckDB changes physical traversal order.
    std::vector<bound_read_view> ordered_physical;
    for (auto const& binding : logical_views) {
      REQUIRE(binding.view.identity);
      auto const& identity = *binding.view.identity;
      REQUIRE(std::get<file_inventory>(identity.data_view).count == 1);
      REQUIRE(identity.fingerprint.canonical.size() == canonical_bytes);
      identities.insert(identity.fingerprint.canonical);
      auto match =
        std::find_if(physical_views.begin(), physical_views.end(), [&](auto const& view) {
          return view.identity->fingerprint.canonical == identity.fingerprint.canonical;
        });
      REQUIRE(match != physical_views.end());
      ordered_physical.push_back(*match);
    }
    REQUIRE(identities.size() == scans);

    std::vector<double> comparison_times, sharing_times, publication_times, total_times;
    constexpr int warmup  = 5;
    constexpr int samples = 40;
    for (int attempt = -warmup; attempt < samples; ++attempt) {
      logical_bound_read_view_capture logical;
      std::vector<bound_read_view> physical;
      read_view_registry candidate;
      for (unsigned index = 0; index < scans; ++index) {
        auto const& original = logical_views[index];
        logical.views.push_back({original.table_index, independent_view(original.view)});
        physical.push_back(independent_view(ordered_physical[index]));
        auto const candidate_index = reversed ? scans - index - 1 : index;
        auto const& binding        = logical_views[candidate_index];
        allocate_scan_contract(
          candidate,
          std::nullopt,
          1,
          candidate_index + 1,
          std::make_shared<bound_read_view const>(independent_view(binding.view)),
          {},
          {},
          {},
          {},
          binding.table_index);
      }
      auto start      = clock_type::now();
      auto comparison = compare_read_views(candidate_origin::copy, &logical, physical, candidate);
      auto compared   = clock_type::now();
      REQUIRE(comparison.equal);
      REQUIRE(comparison.correspondence == "table_index");
      auto share_start = clock_type::now();
      share_equal_read_view_identities(&logical, physical, candidate);
      auto shared = clock_type::now();
      candidate.publish_correspondence(
        certificate_evidence_scope::binding_correspondence, comparison.correspondence, physical);
      auto published = clock_type::now();
      // Verify results outside all timers, including preservation of physical evidence.
      for (auto const& entry : candidate.entries()) {
        auto match =
          std::find_if(logical.views.begin(), logical.views.end(), [&](auto const& binding) {
            return binding.view.identity == entry.contract.view->identity;
          });
        REQUIRE(match != logical.views.end());
        auto index = static_cast<std::size_t>(match - logical.views.begin());
        REQUIRE(physical[index].identity == match->view.identity);
        REQUIRE(entry.physical_evidence == ordered_physical[index].evidence);
        REQUIRE(entry.eligibility.verdict == eligibility_verdict::supported);
      }
      if (attempt < 0) continue;
      auto const comparison_us  = elapsed_us(start, compared);
      auto const sharing_us     = elapsed_us(share_start, shared);
      auto const publication_us = elapsed_us(shared, published);
      comparison_times.push_back(comparison_us);
      sharing_times.push_back(sharing_us);
      publication_times.push_back(publication_us);
      total_times.push_back(sharing_us + publication_us);
      std::cout << "READ_VIEW_SCALING_SAMPLE shape=" << (join ? "join" : "union_all")
                << " prefix=" << (long_prefix ? "long" : "short")
                << " order=" << (reversed ? "reverse" : "natural") << " scans=" << scans
                << " canonical_bytes_per_scan=" << canonical_bytes << " attempt=" << attempt
                << " compare_us=" << comparison_us << " share_us=" << sharing_us
                << " publish_us=" << publication_us << '\n';
    }
    std::cout << "READ_VIEW_SCALING_SUMMARY shape=" << (join ? "join" : "union_all")
              << " prefix=" << (long_prefix ? "long" : "short")
              << " order=" << (reversed ? "reverse" : "natural") << " scans=" << scans
              << " files_per_scan=1 canonical_bytes_per_scan=" << canonical_bytes
              << " n=" << samples << " compare_p50_us=" << percentile(comparison_times, 50)
              << " compare_p95_us=" << percentile(comparison_times, 95)
              << " share_p50_us=" << percentile(sharing_times, 50)
              << " share_p95_us=" << percentile(sharing_times, 95)
              << " publish_p50_us=" << percentile(publication_times, 50)
              << " publish_p95_us=" << percentile(publication_times, 95)
              << " post_compare_p50_us=" << percentile(total_times, 50)
              << " post_compare_p95_us=" << percentile(total_times, 95) << '\n';
    physical_plan.reset();
    run_ok("ROLLBACK");
  }
}
