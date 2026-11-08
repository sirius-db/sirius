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

#include <algorithm>
#include <array>
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

std::shared_ptr<bound_read_view const> iceberg_view(
  iceberg_bound_selection selection,
  std::string path                    = "shared.parquet",
  std::optional<std::string> selector = "stable-options")
{
  bound_read_identity identity;
  identity.source      = {"iceberg_scan", source_kind::parquet_local, "test.iceberg.v1"};
  identity.data_view   = file_inventory{1};
  identity.bound_types = {duckdb::LogicalType::INTEGER};
  identity.bound_names = {"id"};
  identity.selection   = std::move(selection);
  bound_read_view result;
  std::vector<std::string> paths{std::move(path)};
  result.identity                   = make_bound_read_identity(std::move(identity), paths);
  result.selector_evidence_required = true;
  result.logical_selector_evidence  = std::move(selector);
  return std::make_shared<bound_read_view const>(std::move(result));
}
}  // namespace

TEST_CASE("Iceberg drift requires a changed same-table selection and preserved other pairs",
          "[transparent][read_view][iceberg][iceberg_bound_selection]")
{
  auto const origin        = GENERATE(candidate_origin::copy, candidate_origin::replan);
  auto const mixed         = GENERATE(false, true);
  std::string const change = GENERATE(
    "snapshot", "schema", "metadata", "uuid", "inventory", "other_identity", "other_selector");
  CAPTURE(static_cast<int>(origin), mixed, change);
  iceberg_bound_selection selection{"U", 101, 7, "m1.json"};
  auto original = iceberg_view(selection);
  auto other    = mixed ? view("other.parquet", "other-options", true)
                        : iceberg_view({"V", 201, 8, "v.json"}, "other.parquet", "other-options");
  auto changed  = selection;
  if (change == "schema") {
    ++changed.bound_schema_id;
  } else if (change == "metadata") {
    changed.metadata_file_path = "m2.json";
  } else if (change == "uuid") {
    changed.table_uuid = "replacement-U";
  } else if (change != "inventory") {
    ++changed.snapshot_id;
  }
  auto candidate = iceberg_view(changed, change == "inventory" ? "new.parquet" : "shared.parquet");
  auto other_candidate = std::make_shared<bound_read_view>(*other);
  if (change == "other_identity") {
    other_candidate = std::make_shared<bound_read_view>(*view("unrelated.parquet"));
  } else if (change == "other_selector") {
    other_candidate->logical_selector_evidence.reset();
  }
  std::vector<original_binding> logical{{10, original}, {20, other}};
  std::vector<read_view_fingerprint> physical{fingerprint(original), fingerprint(other)};
  std::vector<candidate_binding> planned{{10, candidate}, {20, other_candidate}};
  auto const comparison = compare_read_views(origin, logical, physical, planned);
  CHECK_FALSE(comparison.equal);
  bool const drift = change == "snapshot" || change == "schema" || change == "metadata";
  CHECK(comparison.reason == (drift ? "iceberg_snapshot_drift" : "fingerprint_mismatch"));
  CHECK(comparison.correspondence ==
        (origin == candidate_origin::copy ? "table_index" : "table_index_replan"));
  CHECK(comparison.original_count == 2);
  CHECK(comparison.candidate_count == 2);
}

TEST_CASE("Iceberg drift refuses incomplete occurrence and physical count proofs",
          "[transparent][read_view][iceberg][iceberg_bound_selection]")
{
  auto const origin        = GENERATE(candidate_origin::copy, candidate_origin::replan);
  std::string const change = GENERATE(
    "duplicate_index", "missing_index", "missing_original", "physical_count", "candidate_count");
  CAPTURE(static_cast<int>(origin), change);
  auto a = iceberg_view({"U", 101, 7, "m1.json"});
  auto b = iceberg_view({"U", 101, 7, "m1.json"});
  REQUIRE(a->identity != b->identity);
  auto next = iceberg_view({"U", 102, 7, "m2.json"});
  std::vector<original_binding> logical{{10, a}, {20, b}};
  std::vector<read_view_fingerprint> physical{fingerprint(a), fingerprint(b)};
  std::vector<candidate_binding> candidate{{10, next}, {20, b}};
  auto const before = compare_read_views(origin, logical, physical, candidate);
  REQUIRE_FALSE(before.equal);
  REQUIRE(before.reason == "iceberg_snapshot_drift");
  if (change == "duplicate_index") candidate[1].table_index = 10;
  if (change == "missing_index") candidate[1].table_index = 30;
  if (change == "missing_original") logical.pop_back();
  if (change == "physical_count") physical.push_back(fingerprint(a));
  if (change == "candidate_count") candidate.push_back({30, b});
  auto const comparison = compare_read_views(origin, logical, physical, candidate);
  CHECK_FALSE(comparison.equal);
  CHECK(comparison.reason == "fingerprint_mismatch");
  CHECK(comparison.correspondence ==
        (origin == candidate_origin::copy ? "table_index" : "table_index_replan"));
}

TEST_CASE("Iceberg snapshot swaps refine per-index failure after equal multisets",
          "[transparent][read_view][iceberg][iceberg_bound_selection]")
{
  auto const origin = GENERATE(candidate_origin::copy, candidate_origin::replan);
  auto first        = iceberg_view({"U", 101, 7, "m1.json"});
  auto second       = iceberg_view({"U", 102, 7, "m2.json"});
  std::vector<original_binding> logical{{10, first}, {20, second}};
  std::vector<read_view_fingerprint> physical{fingerprint(first), fingerprint(second)};
  std::vector<candidate_binding> candidate{{10, second}, {20, first}};
  auto const comparison = compare_read_views(origin, logical, physical, candidate);
  CHECK_FALSE(comparison.equal);
  CHECK(comparison.reason == "iceberg_snapshot_drift");
  // Pairing failed after equal multisets. Drift refines only the reason:
  // copy keeps "table_index"; multi-replan has already changed to "none".
  CHECK(comparison.correspondence == (origin == candidate_origin::copy ? "table_index" : "none"));
  CHECK(comparison.only_original.empty());
  CHECK(comparison.only_candidate.empty());
}

TEST_CASE("Single Iceberg replan drift cannot hide a changed skipped definition",
          "[transparent][read_view][iceberg][iceberg_bound_selection]")
{
  auto original = iceberg_view({"U", 101, 7, "m1.json"});
  auto next     = iceberg_view({"U", 102, 7, "m2.json"});
  std::vector<original_binding> logical{{10, original}};
  std::vector<read_view_fingerprint> physical{fingerprint(original)};
  std::vector<candidate_binding> planned{{20, next}};
  auto comparison = compare_read_views(candidate_origin::replan, logical, physical, planned);
  REQUIRE(comparison.reason == "iceberg_snapshot_drift");
  REQUIRE_FALSE(comparison.equal);
  REQUIRE(comparison.correspondence == "single");

  physical.push_back(fingerprint(view("P.parquet")));
  std::vector<candidate_binding> skipped{{30, view("Q.parquet")}};
  comparison = compare_read_views(candidate_origin::replan, logical, physical, planned, skipped);
  CHECK_FALSE(comparison.equal);
  CHECK(comparison.reason == "fingerprint_mismatch");
  CHECK(comparison.correspondence == "single");
  CHECK(comparison.original_count == comparison.candidate_count);

  comparison = compare_read_views(candidate_origin::replan, logical, physical, planned);
  CHECK_FALSE(comparison.equal);
  CHECK(comparison.reason == "fingerprint_mismatch");
  CHECK(comparison.correspondence == "single");
  CHECK(comparison.original_count == 2);
  CHECK(comparison.candidate_count == 1);
}

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

TEST_CASE("Replan without logical originals preserves the single-scan shortcut",
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

TEST_CASE("Multi-scan correspondence pairs independent identities by table index",
          "[transparent][read_view][iceberg_multiscan]")
{
  auto const origin   = GENERATE(candidate_origin::copy, candidate_origin::replan);
  auto const reversed = GENERATE(false, true);
  std::vector<original_binding> logical;
  std::vector<read_view_fingerprint> physical;
  std::vector<candidate_binding> candidate;
  for (duckdb::idx_t index = 0; index < 3; ++index) {
    auto const path     = "data-" + std::to_string(index) + ".parquet";
    auto const selector = "snapshot=" + std::to_string(index + 1);
    auto original       = view(path, selector, true);
    auto rebound        = view(path, selector, true);
    REQUIRE(original->identity != rebound->identity);
    logical.push_back({10 + index, original});
    physical.push_back(fingerprint(original));
    candidate.push_back({10 + index, rebound});
  }
  if (reversed) std::reverse(candidate.begin(), candidate.end());

  auto const comparison = compare_read_views(origin, logical, physical, candidate);
  REQUIRE(comparison.equal);
  CHECK(comparison.correspondence ==
        (origin == candidate_origin::copy ? "table_index" : "table_index_replan"));
  CHECK(comparison.reason.empty());
  CHECK(comparison.original_count == 3);
  CHECK(comparison.candidate_count == 3);
  CHECK(comparison.only_original.empty());
  CHECK(comparison.only_candidate.empty());
}

TEST_CASE("Multi-scan correspondence rejects incomplete and permuted index maps",
          "[transparent][read_view][iceberg_multiscan]")
{
  auto const origin = GENERATE(candidate_origin::copy, candidate_origin::replan);
  auto const defect = GENERATE("permutation",
                               "duplicate_candidate",
                               "duplicate_original",
                               "missing_index",
                               "missing_original",
                               "extra_original",
                               "paired_identity");
  CAPTURE(defect);
  auto a = view("a.parquet", "snapshot=1", true);
  auto b = view("b.parquet", "snapshot=2", true);
  auto c = view("c.parquet", "snapshot=3", true);
  std::vector<original_binding> logical{{10, a}, {20, b}, {30, c}};
  std::vector<read_view_fingerprint> physical{fingerprint(a), fingerprint(b), fingerprint(c)};
  std::vector<candidate_binding> candidate{{10, a}, {20, b}, {30, c}};
  std::string const choice = defect;
  if (choice == "permutation") {
    std::swap(candidate[0].table_index, candidate[1].table_index);
  } else if (choice == "duplicate_candidate") {
    logical[1].view   = view("a.parquet", "snapshot=1", true);
    candidate[1].view = view("a.parquet", "snapshot=1", true);
    physical[1]       = fingerprint(a);
    REQUIRE(logical[1].view->identity != candidate[1].view->identity);
    REQUIRE(compare_read_views(origin, logical, physical, candidate).equal);
    candidate[1].table_index = candidate[0].table_index;
  } else if (choice == "duplicate_original") {
    logical[1].table_index = logical[0].table_index;
  } else if (choice == "missing_index") {
    candidate[0].table_index = 99;
  } else if (choice == "missing_original") {
    logical.pop_back();
  } else if (choice == "extra_original") {
    logical.push_back({40, a});
  } else {
    logical[0].view = view("different.parquet", "snapshot=1", true);
  }

  auto const comparison = compare_read_views(origin, logical, physical, candidate);
  CHECK_FALSE(comparison.equal);
  CHECK(comparison.reason ==
        (origin == candidate_origin::copy ? "binding_mismatch" : "no_correspondence"));
  CHECK(comparison.correspondence == (origin == candidate_origin::copy ? "table_index" : "none"));
  CHECK(comparison.only_original.empty());
  CHECK(comparison.only_candidate.empty());
}

TEST_CASE("Multi-scan correspondence proves each required snapshot selector",
          "[transparent][read_view][iceberg_multiscan]")
{
  auto const origin = GENERATE(candidate_origin::copy, candidate_origin::replan);
  auto const defect = GENERATE(
    "changed", "permuted", "missing_candidate", "missing_original", "original_not_required");
  CAPTURE(defect);
  // Different delete sets can share the same data inventory and schema.
  auto a = view("shared.parquet", "snapshot=1", true);
  auto b = view("shared.parquet", "snapshot=2", true);
  auto c = view("other.parquet", "snapshot=3", true);
  REQUIRE(fingerprint(a).canonical == fingerprint(b).canonical);
  std::vector<original_binding> logical{{10, a}, {20, b}, {30, c}};
  std::vector<read_view_fingerprint> physical{fingerprint(a), fingerprint(b), fingerprint(c)};
  std::vector<candidate_binding> candidate{{10, a}, {20, b}, {30, c}};
  std::string const choice = defect;
  if (choice == "changed") {
    candidate[1].view = view("shared.parquet", "snapshot=99", true);
  } else if (choice == "permuted") {
    std::swap(candidate[0].view, candidate[1].view);
  } else if (choice == "missing_candidate") {
    candidate[1].view = view("shared.parquet", std::nullopt, true);
  } else if (choice == "missing_original") {
    logical[1].view = view("shared.parquet", std::nullopt, true);
  } else {
    logical[1].view = view("shared.parquet", "snapshot=2", false);
  }

  auto const comparison = compare_read_views(origin, logical, physical, candidate);
  CHECK_FALSE(comparison.equal);
  CHECK(comparison.reason == "selector_unproven");
  CHECK(comparison.correspondence ==
        (origin == candidate_origin::copy ? "table_index" : "table_index_replan"));
  CHECK(comparison.only_original.empty());
  CHECK(comparison.only_candidate.empty());
}

TEST_CASE("Multi-scan replans include skipped definitions and preserve physical proof",
          "[transparent][read_view][iceberg_multiscan]")
{
  auto const defect = GENERATE("none",
                               "skipped_index",
                               "skipped_identity",
                               "skipped_selector",
                               "omitted",
                               "missing_identity",
                               "hash_collision",
                               "multiplicity");
  CAPTURE(defect);
  auto a = view("a.parquet", "snapshot=1", true);
  auto b = view("b.parquet", "snapshot=2", true);
  auto c = view("c.parquet", "snapshot=3", true);
  std::vector<original_binding> logical{{10, a}, {20, b}, {30, c}};
  std::vector<read_view_fingerprint> physical{fingerprint(a), fingerprint(b), fingerprint(c)};
  std::vector<candidate_binding> planned{{10, a}, {20, b}};
  std::vector<candidate_binding> skipped{{30, c}};
  std::string const choice = defect;
  std::string reason;
  std::string token = "table_index_replan";
  if (choice == "skipped_index") {
    skipped[0].table_index = 99;
    reason                 = "no_correspondence";
    token                  = "none";
  } else if (choice == "skipped_identity") {
    skipped[0].view = view("different.parquet", "snapshot=3", true);
    reason          = "fingerprint_mismatch";
  } else if (choice == "skipped_selector") {
    skipped[0].view = view("c.parquet", "snapshot=99", true);
    reason          = "selector_unproven";
  } else if (choice == "omitted") {
    skipped.clear();
    reason = "fingerprint_mismatch";
  } else if (choice == "missing_identity") {
    planned[1].view.reset();
    reason = "missing_identity";
  } else if (choice == "hash_collision") {
    auto collision            = std::make_shared<bound_read_view>(*view("collision.parquet"));
    auto identity             = *collision->identity;
    identity.fingerprint.hash = fingerprint(c).hash;
    collision->identity       = std::make_shared<bound_read_identity const>(std::move(identity));
    skipped[0].view           = std::move(collision);
    reason                    = "fingerprint_mismatch";
  } else if (choice == "multiplicity") {
    physical        = {fingerprint(a), fingerprint(a), fingerprint(b)};
    logical[2].view = b;
    skipped[0].view = b;
    reason          = "fingerprint_mismatch";
  }

  auto const comparison =
    compare_read_views(candidate_origin::replan, logical, physical, planned, skipped);
  CHECK(comparison.equal == (choice == "none"));
  CHECK(comparison.reason == reason);
  CHECK(comparison.correspondence == token);
  CHECK(comparison.original_count == 3);
  CHECK(comparison.candidate_count == (choice == "omitted" ? 2 : 3));
}

TEST_CASE("Skipped CTE definition scans count toward correspondence", "[transparent][read_view]")
{
  auto a = view("a.parquet");
  auto b = view("b.parquet");
  std::vector<original_binding> logical{{10, a}, {20, b}};
  std::vector<read_view_fingerprint> physical{fingerprint(a), fingerprint(b)};
  std::vector<candidate_binding> planned{{20, b}};

  auto missing = compare_read_views(candidate_origin::copy, logical, physical, planned);
  CHECK_FALSE(missing.equal);
  CHECK(missing.reason == "fingerprint_mismatch");

  std::vector<candidate_binding> skipped{{10, a}};
  auto copy = compare_read_views(candidate_origin::copy, logical, physical, planned, skipped);
  CHECK(copy.equal);
  CHECK(copy.correspondence == "table_index");

  std::vector<candidate_binding> wrong_index{{11, a}};
  auto rebound =
    compare_read_views(candidate_origin::copy, logical, physical, planned, wrong_index);
  CHECK_FALSE(rebound.equal);
  CHECK(rebound.reason == "binding_mismatch");

  std::vector<candidate_binding> wrong_view{{10, view("c.parquet")}};
  auto changed = compare_read_views(candidate_origin::copy, logical, physical, planned, wrong_view);
  CHECK_FALSE(changed.equal);
  CHECK(changed.reason == "fingerprint_mismatch");

  // Replan names the single planned scan; the skipped one still joins the multiset check.
  std::vector<candidate_binding> replanned{{31, b}};
  std::vector<candidate_binding> replan_skipped{{30, a}};
  auto replan =
    compare_read_views(candidate_origin::replan, {}, physical, replanned, replan_skipped);
  CHECK(replan.equal);
  CHECK(replan.correspondence == "single");

  std::vector<candidate_binding> none;
  std::vector<read_view_fingerprint> physical_one{fingerprint(a)};
  auto only_skipped =
    compare_read_views(candidate_origin::replan, {}, physical_one, none, replan_skipped);
  CHECK(only_skipped.equal);
  CHECK(only_skipped.correspondence == "none");
}

TEST_CASE("Read-view mismatch summaries are bounded", "[transparent][read_view][iceberg_multiscan]")
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
  CHECK(comparison.correspondence == "table_index_replan");
  CHECK(comparison.reason == "fingerprint_mismatch");
  CHECK(comparison.different_original_count == 5);
  CHECK(comparison.different_candidate_count == 5);
  CHECK(comparison.only_original.size() == 3);
  CHECK(comparison.only_candidate.size() == 3);

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

TEST_CASE("Equal read-view sides share candidate identity storage",
          "[transparent][read_view][iceberg_multiscan]")
{
  auto const origin = GENERATE(candidate_origin::copy, candidate_origin::replan);
  logical_bound_read_view_capture logical;
  std::vector<bound_read_view> physical;
  std::vector<scan_contract_id> contracts;
  read_view_registry candidate;
  for (duckdb::idx_t index = 0; index < 3; ++index) {
    auto const path = "shared-" + std::to_string(index) + ".parquet";
    logical.views.push_back({10 + index, *view(path)});
    logical.views.back().view.evidence = std::make_shared<file_evidence_arrays>();
    physical.push_back(*view(path));
    physical.back().evidence = std::make_shared<file_evidence_arrays>();
    physical.back().depth    = evidence_depth::path_size_and_tag;
    contracts.push_back(allocate_scan_contract(
      candidate, std::nullopt, 1, 101 + index, view(path), {}, {}, {}, {}, 10 + index));
  }
  auto const logical_before  = logical;
  auto const physical_before = physical;

  auto comparison = compare_read_views(origin, &logical, physical, candidate);
  REQUIRE(comparison.equal);
  CHECK(comparison.original_depth == evidence_depth::path_size_and_tag);
  CHECK(comparison.candidate_depth == evidence_depth::path);
  for (auto contract : contracts) {
    CHECK(candidate.entry(contract).eligibility.verdict == eligibility_verdict::not_evaluated);
    CHECK(candidate.entry(contract).eligibility.evidence_scope == certificate_evidence_scope::none);
    CHECK(candidate.entry(contract).eligibility.correspondence.empty());
  }
  share_equal_read_view_identities(&logical, physical, candidate);
  candidate.publish_correspondence(
    certificate_evidence_scope::binding_correspondence, comparison.correspondence, physical);

  for (std::size_t index = 0; index < contracts.size(); ++index) {
    auto const& entry  = candidate.entry(contracts[index]);
    auto const& shared = entry.contract.view->identity;
    CHECK(logical.views[index].view.identity == shared);
    CHECK(physical[index].identity == shared);
    CHECK(logical.views[index].view.evidence == logical_before.views[index].view.evidence);
    CHECK(physical[index].evidence == physical_before[index].evidence);
    CHECK(logical.views[index].view.evidence != physical[index].evidence);
    CHECK(entry.physical_evidence == physical[index].evidence);
    CHECK(entry.eligibility.depth == evidence_depth::path_size_and_tag);
    CHECK(entry.eligibility.verdict == eligibility_verdict::supported);
    CHECK(entry.eligibility.evidence_scope == certificate_evidence_scope::binding_correspondence);
    std::ostringstream expected_identity;
    expected_identity << std::hex << std::setw(16) << std::setfill('0') << shared->fingerprint.hash;
    CHECK(entry.eligibility.cpu_gpu_view_identity == expected_identity.str());
    CHECK(entry.eligibility.correspondence ==
          (origin == candidate_origin::copy ? "table_index" : "table_index_replan"));
  }
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

TEST_CASE("Pre-declined scan preserves an available candidate view", "[transparent][read_view]")
{
  auto registry = std::make_shared<read_view_registry>();
  duckdb::TableFunction function("iceberg_scan", {}, nullptr, nullptr);
  sirius::op::sirius_physical_table_scan scan({},
                                              duckdb::BoundTableFunction(function),
                                              nullptr,
                                              {},
                                              {},
                                              {},
                                              {},
                                              duckdb::make_uniq<duckdb::TableFilterSet>(),
                                              1,
                                              duckdb::ExtraOperatorInfo(),
                                              {},
                                              duckdb::virtual_column_map_t{},
                                              /*projected_output_count=*/0);
  scan.read_views                   = registry;
  scan.scan_node_id                 = 77;
  scan.table_index                  = 12;
  scan.contract_finalize_generation = 9;
  sirius::planner::connector connector{};
  connector.function_name    = "iceberg_scan";
  connector.kind             = source_kind::parquet_local;
  connector.registry_profile = "iceberg.v1";

  SECTION("no safe pre-capture") {}
  SECTION("connector semantic capture") { scan.bound_view = view("refused.parquet"); }

  auto id            = registry->allocate_declined_scan(scan, connector);
  auto const& record = registry->entry(id);
  CHECK(record.contract.view == scan.bound_view);
  CHECK(registry->candidate_bindings().front().view == scan.bound_view);
  CHECK(record.contract.scan_node_id == 77);
  CHECK(record.contract.table_index == 12);
  CHECK(record.finalize_generation == 9);
  CHECK(record.eligibility.materializer.kind == materializer_kind::iceberg);
  CHECK_THROWS_AS(registry->allocate_declined_scan(scan, connector), std::runtime_error);
}

TEST_CASE("Provider read identities preserve semantics without enumerating files",
          "[transparent][read_view]")
{
  bound_read_identity identity;
  identity.source      = {"provider", source_kind::parquet_local, "provider.v1"};
  identity.data_view   = provider_read_identity{"table-uuid/snapshot=12/schema=3"};
  identity.bound_types = {duckdb::LogicalType::BIGINT};
  identity.bound_names = {"id"};
  auto first           = make_bound_read_identity(identity, {});
  CHECK(first->fingerprint == make_bound_read_identity(identity, {})->fingerprint);
  identity.data_view = provider_read_identity{"table-uuid/snapshot=13/schema=3"};
  CHECK_FALSE(first->fingerprint == make_bound_read_identity(identity, {})->fingerprint);
  identity.data_view = provider_read_identity{"other-uuid/snapshot=12/schema=3"};
  CHECK_FALSE(first->fingerprint == make_bound_read_identity(identity, {})->fingerprint);
  identity.data_view = provider_read_identity{"table-uuid/snapshot=12/schema=4"};
  CHECK_FALSE(first->fingerprint == make_bound_read_identity(identity, {})->fingerprint);
  std::array<std::string, 1> files{"must-not-be-invented.parquet"};
  CHECK_THROWS_AS(make_bound_read_identity(identity, files), std::invalid_argument);
  identity.data_view = provider_read_identity{};
  CHECK_THROWS_AS(make_bound_read_identity(identity, {}), std::invalid_argument);
}
