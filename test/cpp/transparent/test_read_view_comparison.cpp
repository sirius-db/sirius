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
#include "op/sirius_physical_table_scan.hpp"
#include "planner/connector_registry.hpp"
#include "transparent/read_view_registry.hpp"

#include <catch.hpp>

#include <iomanip>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using namespace sirius::op::scan;
using namespace sirius::transparent;

std::shared_ptr<bound_read_view const> view(std::string path,
                                            std::optional<std::string> selector = std::nullopt,
                                            bool selector_required              = false)
{
  bound_read_identity identity;
  identity.source      = {"read_parquet", source_kind::parquet_local, "test.parquet.v1"};
  identity.data_view   = file_inventory{1};
  identity.bound_types = {duckdb::LogicalType::INTEGER};
  identity.bound_names = {"id"};
  bound_read_view result;
  std::vector<std::string> paths{std::move(path)};
  result.identity                   = make_bound_read_identity(std::move(identity), paths);
  result.selector_evidence_required = selector_required;
  result.logical_selector_evidence  = std::move(selector);
  return std::make_shared<bound_read_view const>(std::move(result));
}

read_view_fingerprint fingerprint(std::shared_ptr<bound_read_view const> const& value)
{
  return value->identity->fingerprint;
}
}  // namespace

TEST_CASE("Copy-origin read views require table-index correspondence", "[transparent][read_view]")
{
  auto a = view("a.parquet");
  auto b = view("b.parquet");
  std::vector<original_binding> logical{{10, a}, {20, b}};
  std::vector<read_view_fingerprint> physical{fingerprint(a), fingerprint(b)};
  std::vector<candidate_binding> swapped{{10, b}, {20, a}};

  auto comparison = compare_read_views(candidate_origin::copy, logical, physical, swapped);

  CHECK_FALSE(comparison.equal);
  CHECK(comparison.correspondence == "table_index");
  CHECK(comparison.reason == "binding_mismatch");
  CHECK(comparison.only_original.empty());
  CHECK(comparison.only_candidate.empty());
}

TEST_CASE("Copy-origin correspondence is one-to-one", "[transparent][read_view]")
{
  auto a = view("a.parquet");
  std::vector<original_binding> logical{{10, a}, {20, a}};
  std::vector<read_view_fingerprint> physical{fingerprint(a), fingerprint(a)};
  std::vector<candidate_binding> duplicate{{10, a}, {10, a}};

  auto comparison = compare_read_views(candidate_origin::copy, logical, physical, duplicate);

  CHECK_FALSE(comparison.equal);
  CHECK(comparison.correspondence == "table_index");
  CHECK(comparison.reason == "binding_mismatch");
}

TEST_CASE("Replan-origin read views admit only a single corresponding scan",
          "[transparent][read_view]")
{
  auto a = view("a.parquet");
  auto b = view("b.parquet");
  std::vector<read_view_fingerprint> physical_two{fingerprint(a), fingerprint(b)};
  std::vector<candidate_binding> candidate_two{{31, a}, {32, b}};

  auto multi = compare_read_views(candidate_origin::replan, {}, physical_two, candidate_two);
  CHECK_FALSE(multi.equal);
  CHECK(multi.correspondence == "none");
  CHECK(multi.reason == "no_correspondence");

  std::vector<read_view_fingerprint> physical_one{fingerprint(a)};
  std::vector<candidate_binding> candidate_one{{91, a}};
  auto single = compare_read_views(candidate_origin::replan, {}, physical_one, candidate_one);
  CHECK(single.equal);
  CHECK(single.correspondence == "single");
  CHECK(single.reason.empty());
}

TEST_CASE("Read-view mismatch summaries are bounded", "[transparent][read_view]")
{
  std::vector<read_view_fingerprint> physical;
  std::vector<candidate_binding> candidate;
  for (int index = 0; index < 5; ++index) {
    physical.push_back(fingerprint(view("original-" + std::to_string(index) + ".parquet")));
    candidate.push_back(
      {static_cast<duckdb::idx_t>(index), view("candidate-" + std::to_string(index) + ".parquet")});
  }

  auto comparison = compare_read_views(candidate_origin::replan, {}, physical, candidate);

  CHECK_FALSE(comparison.equal);
  CHECK(comparison.reason == "no_correspondence");
  CHECK(comparison.only_original.empty());
  CHECK(comparison.only_candidate.empty());

  candidate.resize(1);
  physical.resize(1);
  for (int index = 1; index < 5; ++index) {
    physical.push_back(fingerprint(view("original-" + std::to_string(index) + ".parquet")));
  }
  comparison = compare_read_views(candidate_origin::copy, {}, physical, candidate);
  CHECK_FALSE(comparison.equal);
  CHECK(comparison.reason == "fingerprint_mismatch");
  CHECK(comparison.original_count == 5);
  CHECK(comparison.candidate_count == 1);
  CHECK(comparison.different_original_count == 5);
  CHECK(comparison.different_candidate_count == 1);
  CHECK(comparison.original_hash.has_value());
  CHECK(comparison.candidate_hash.has_value());
  CHECK(comparison.only_original.size() == 3);
  CHECK(comparison.only_candidate.size() == 1);
  auto const description = describe_read_view_mismatch(comparison);
  CHECK(description.find("different_total=6") != std::string::npos);
  CHECK(description.find("different_original=5") != std::string::npos);
  CHECK(description.find("different_candidate=1") != std::string::npos);
  CHECK(description.find("original_hash=none") == std::string::npos);
  CHECK(description.find("candidate_hash=none") == std::string::npos);
}

TEST_CASE("Read-view multiset comparison verifies canonical text after a hash collision",
          "[transparent][read_view]")
{
  auto original             = view("original.parquet");
  auto collided             = std::make_shared<bound_read_view>(*view("candidate.parquet"));
  auto identity             = *collided->identity;
  identity.fingerprint.hash = original->identity->fingerprint.hash;
  collided->identity        = std::make_shared<bound_read_identity const>(std::move(identity));

  std::vector<read_view_fingerprint> physical{fingerprint(original)};
  std::vector<candidate_binding> candidate{{10, collided}};
  auto comparison = compare_read_views(candidate_origin::replan, {}, physical, candidate);

  CHECK_FALSE(comparison.equal);
  CHECK(comparison.reason == "fingerprint_mismatch");
  CHECK(comparison.different_original_count == 1);
  CHECK(comparison.different_candidate_count == 1);
}

TEST_CASE("Selector evidence requires a same-generation logical original",
          "[transparent][read_view]")
{
  auto selected = view("iceberg.parquet", "snapshot=2", true);
  std::vector<read_view_fingerprint> physical{fingerprint(selected)};
  std::vector<candidate_binding> candidate{{44, selected}};

  auto missing = compare_read_views(candidate_origin::replan, {}, physical, candidate);
  CHECK_FALSE(missing.equal);
  CHECK(missing.correspondence == "single");
  CHECK(missing.reason == "selector_unproven");

  std::vector<original_binding> original{{7, view("iceberg.parquet", "snapshot=2", true)}};
  auto proven = compare_read_views(candidate_origin::replan, original, physical, candidate);
  CHECK(proven.equal);
  CHECK(proven.reason.empty());

  auto missing_evidence = view("iceberg.parquet", std::nullopt, true);
  std::vector<candidate_binding> missing_candidate{{44, missing_evidence}};
  auto absent = compare_read_views(candidate_origin::replan, original, physical, missing_candidate);
  CHECK_FALSE(absent.equal);
  CHECK(absent.reason == "selector_unproven");
}

TEST_CASE("Equal read-view sides share candidate identity storage", "[transparent][read_view]")
{
  logical_bound_read_view_capture logical;
  logical.views.push_back({10, *view("shared.parquet")});
  std::vector<bound_read_view> physical{*view("shared.parquet")};
  logical.views.front().view.evidence = std::make_shared<file_evidence_arrays>();
  physical.front().evidence           = std::make_shared<file_evidence_arrays>();
  auto const logical_evidence         = logical.views.front().view.evidence;
  auto const physical_evidence        = physical.front().evidence;
  read_view_registry candidate;
  auto candidate_view = view("shared.parquet");
  auto const contract =
    allocate_scan_contract(candidate, std::nullopt, 1, 101, candidate_view, {}, {}, {}, {}, 10);

  auto comparison = compare_read_views(candidate_origin::copy, &logical, physical, candidate);
  REQUIRE(comparison.equal);
  CHECK(candidate.entry(contract).eligibility.verdict == eligibility_verdict::not_evaluated);
  CHECK(candidate.entry(contract).eligibility.evidence_scope == certificate_evidence_scope::none);
  share_equal_read_view_identities(&logical, physical, candidate);
  candidate.publish_correspondence(
    certificate_evidence_scope::binding_correspondence, comparison.correspondence, physical);

  auto const& shared = candidate.entry(contract).contract.view->identity;
  CHECK(logical.views.front().view.identity == shared);
  CHECK(physical.front().identity == shared);
  CHECK(logical.views.front().view.evidence == logical_evidence);
  CHECK(physical.front().evidence == physical_evidence);
  CHECK(logical_evidence != physical_evidence);
  CHECK(candidate.entry(contract).eligibility.verdict == eligibility_verdict::supported);
  CHECK(candidate.entry(contract).eligibility.evidence_scope ==
        certificate_evidence_scope::binding_correspondence);
  std::ostringstream expected_identity;
  expected_identity << std::hex << std::setw(16) << std::setfill('0') << shared->fingerprint.hash;
  CHECK(candidate.entry(contract).eligibility.cpu_gpu_view_identity == expected_identity.str());
  CHECK(candidate.entry(contract).eligibility.correspondence == "table_index");
}

TEST_CASE("Correspondence preserves a recorded scan refusal", "[transparent][read_view]")
{
  read_view_registry registry;
  auto candidate_view = view("refused.parquet");
  auto const contract =
    allocate_scan_contract(registry, std::nullopt, 1, 102, candidate_view, {}, {}, {}, {}, 10);
  certification_result result;
  result.verdict          = eligibility_verdict::unsupported;
  result.reason           = verdict_reason::iceberg_equality_deletes;
  result.reason_text      = "equality deletes";
  result.later_checks     = check_bit(later_check::footer_per_file);
  result.semantic_columns = {true, false};
  auto invalid            = result;
  invalid.verdict         = eligibility_verdict::incomplete;
  CHECK_THROWS_AS(registry.record_verdict(contract, invalid), std::logic_error);
  registry.record_verdict(contract, result);
  CHECK_THROWS_AS(registry.record_verdict(contract, result), std::logic_error);

  std::vector<bound_read_view> physical{*candidate_view};
  registry.publish_correspondence(
    certificate_evidence_scope::binding_correspondence, "table_index", physical);
  auto const& certificate = registry.entry(contract).eligibility;
  CHECK(certificate.verdict == eligibility_verdict::unsupported);
  CHECK(certificate.reason == verdict_reason::iceberg_equality_deletes);
  CHECK(certificate.reason_text == "equality deletes");
  CHECK(certificate.later_checks == result.later_checks);
  CHECK(certificate.semantic_columns == result.semantic_columns);
  CHECK(certificate.evidence_scope == certificate_evidence_scope::binding_correspondence);
}

TEST_CASE("Pre-declined scan receives a contract without a candidate view",
          "[transparent][read_view]")
{
  auto registry = std::make_shared<read_view_registry>();
  duckdb::TableFunction function("iceberg_scan", {}, nullptr, nullptr);
  sirius::op::sirius_physical_table_scan scan({},
                                              std::move(function),
                                              nullptr,
                                              {},
                                              {},
                                              {},
                                              {},
                                              duckdb::make_uniq<duckdb::TableFilterSet>(),
                                              1,
                                              duckdb::ExtraOperatorInfo(),
                                              {},
                                              duckdb::virtual_column_map_t{});
  scan.read_views                   = registry;
  scan.scan_node_id                 = 77;
  scan.table_index                  = 12;
  scan.contract_finalize_generation = 9;
  sirius::planner::connector connector{};
  connector.function_name    = "iceberg_scan";
  connector.kind             = source_kind::parquet_local;
  connector.registry_profile = "iceberg.v1";

  auto id            = registry->allocate_declined_scan(scan, connector);
  auto const& record = registry->entry(id);
  CHECK(record.contract.view == nullptr);
  CHECK(record.contract.scan_node_id == 77);
  CHECK(record.contract.table_index == 12);
  CHECK(record.finalize_generation == 9);
  CHECK(record.eligibility.materializer.kind == materializer_kind::iceberg);
  CHECK_THROWS_AS(registry->allocate_declined_scan(scan, connector), std::runtime_error);
}
