/*
 * Copyright 2025, Sirius Contributors.
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

// test
#include "operator/operator_test_utils.hpp"

#include <catch.hpp>

// sirius
#include <cuvs/distance/distance.hpp>
#include <vss/cuvs_index_cache.hpp>

#include <string>

namespace {

using sirius::vss::cuvs_index_cache;
using sirius::vss::index_kind;
using sirius::vss::index_metadata;
using Metric = cuvs::distance::DistanceType;

index_metadata make_meta(std::string table, std::string column, Metric metric)
{
  index_metadata meta;
  meta.kind           = index_kind::ivf_flat;
  meta.table_name     = std::move(table);
  meta.column_name    = std::move(column);
  meta.dim            = 3;
  meta.num_rows       = 100;
  meta.n_lists        = 4;
  meta.metric         = metric;
  meta.reserved_bytes = 1024;
  return meta;
}

// Pin a dummy (type-erased) index under `name`. The cache is index-type agnostic,
// so an int holder stands in for a real cuVS index, only the metadata and the
// reservation lifetime are under test here.
void insert_dummy(cuvs_index_cache& cache, std::string name, index_metadata meta)
{
  auto reservation = cache.reserve_index_memory(/*bytes=*/1024, /*preferred_gpu=*/0);
  cache.insert(
    std::move(name), std::move(meta), sirius::vss::make_cuvs_index(int{0}), std::move(reservation));
}

}  // namespace

TEST_CASE("cuvs_index_cache inserts and looks up by management name", "[vss]")
{
  auto manager = sirius::test::operator_utils::initialize_memory_manager();
  cuvs_index_cache cache(*manager);

  REQUIRE(cache.size() == 0);
  REQUIRE(cache.find("idx") == nullptr);
  REQUIRE_FALSE(cache.contains("idx"));

  insert_dummy(cache, "idx", make_meta("docs", "vec", Metric::L2SqrtExpanded));

  REQUIRE(cache.size() == 1);
  REQUIRE(cache.contains("idx"));
  const auto* entry = cache.find("idx");
  REQUIRE(entry != nullptr);
  REQUIRE(entry->meta.table_name == "docs");
  REQUIRE(entry->meta.column_name == "vec");
  REQUIRE(entry->meta.metric == Metric::L2SqrtExpanded);
  REQUIRE(entry->index != nullptr);
  REQUIRE(entry->reservation != nullptr);
}

TEST_CASE("cuvs_index_cache find_by_column matches the auto-routing identity", "[vss]")
{
  auto manager = sirius::test::operator_utils::initialize_memory_manager();
  cuvs_index_cache cache(*manager);

  insert_dummy(cache, "idx", make_meta("docs", "vec", Metric::L2SqrtExpanded));

  SECTION("exact (table, column, metric) match hits")
  {
    const auto* e = cache.find_by_column("docs", "vec", Metric::L2SqrtExpanded);
    REQUIRE(e != nullptr);
    REQUIRE(e->meta.table_name == "docs");
  }
  SECTION("wrong column misses")
  {
    REQUIRE(cache.find_by_column("docs", "other", Metric::L2SqrtExpanded) == nullptr);
  }
  SECTION("wrong table misses")
  {
    REQUIRE(cache.find_by_column("other", "vec", Metric::L2SqrtExpanded) == nullptr);
  }
  SECTION("right column but wrong metric misses (an l2 index can't serve a cosine query)")
  {
    REQUIRE(cache.find_by_column("docs", "vec", Metric::CosineExpanded) == nullptr);
  }
}

TEST_CASE("cuvs_index_cache find_by_column folds L2 expanded/unexpanded", "[vss]")
{
  auto manager = sirius::test::operator_utils::initialize_memory_manager();
  cuvs_index_cache cache(*manager);

  SECTION("index built L2SqrtExpanded matches an L2SqrtUnexpanded query")
  {
    insert_dummy(cache, "idx", make_meta("docs", "vec", Metric::L2SqrtExpanded));
    REQUIRE(cache.find_by_column("docs", "vec", Metric::L2SqrtUnexpanded) != nullptr);
  }
  SECTION("index built L2SqrtUnexpanded matches an L2SqrtExpanded query")
  {
    insert_dummy(cache, "idx", make_meta("docs", "vec", Metric::L2SqrtUnexpanded));
    REQUIRE(cache.find_by_column("docs", "vec", Metric::L2SqrtExpanded) != nullptr);
  }
  SECTION("the fold stays within the metric family (an L2 query misses a cosine index)")
  {
    insert_dummy(cache, "idx", make_meta("docs", "vec", Metric::CosineExpanded));
    REQUIRE(cache.find_by_column("docs", "vec", Metric::L2SqrtUnexpanded) == nullptr);
  }
}

TEST_CASE("cuvs_index_cache insert replaces an existing name in place", "[vss]")
{
  auto manager = sirius::test::operator_utils::initialize_memory_manager();
  cuvs_index_cache cache(*manager);

  insert_dummy(cache, "idx", make_meta("docs", "vec", Metric::L2SqrtExpanded));
  insert_dummy(cache, "idx", make_meta("docs", "embedding", Metric::CosineExpanded));

  REQUIRE(cache.size() == 1);  // replaced, not appended
  const auto* entry = cache.find("idx");
  REQUIRE(entry != nullptr);
  REQUIRE(entry->meta.column_name == "embedding");
  REQUIRE(entry->meta.metric == Metric::CosineExpanded);
}

TEST_CASE("cuvs_index_cache erase and clear release entries", "[vss]")
{
  auto manager = sirius::test::operator_utils::initialize_memory_manager();
  cuvs_index_cache cache(*manager);

  insert_dummy(cache, "a", make_meta("t", "v", Metric::L2SqrtExpanded));
  insert_dummy(cache, "b", make_meta("t", "w", Metric::L2SqrtExpanded));
  REQUIRE(cache.size() == 2);

  SECTION("erase removes exactly the named entry")
  {
    REQUIRE(cache.erase("a"));
    REQUIRE(cache.size() == 1);
    REQUIRE(cache.find("a") == nullptr);
    REQUIRE(cache.find("b") != nullptr);
    REQUIRE_FALSE(cache.erase("a"));  // already gone
  }
  SECTION("clear drops everything")
  {
    cache.clear();
    REQUIRE(cache.size() == 0);
    REQUIRE(cache.find("a") == nullptr);
    REQUIRE(cache.find("b") == nullptr);
  }
}

TEST_CASE("cuvs_index_cache find_by_column returns a match among several entries", "[vss]")
{
  auto manager = sirius::test::operator_utils::initialize_memory_manager();
  cuvs_index_cache cache(*manager);

  insert_dummy(cache, "idx_a", make_meta("images", "clip", Metric::CosineExpanded));
  insert_dummy(cache, "idx_b", make_meta("docs", "vec", Metric::L2SqrtExpanded));
  insert_dummy(cache, "idx_c", make_meta("docs", "title_vec", Metric::L2SqrtExpanded));

  const auto* e = cache.find_by_column("docs", "title_vec", Metric::L2SqrtExpanded);
  REQUIRE(e != nullptr);
  REQUIRE(e->meta.table_name == "docs");
  REQUIRE(e->meta.column_name == "title_vec");
}

TEST_CASE("cuvs_index_cache erase_by_column drops the matching index before a rebuild", "[vss]")
{
  auto manager = sirius::test::operator_utils::initialize_memory_manager();
  cuvs_index_cache cache(*manager);

  SECTION("removes the identity match regardless of its management name, leaving others")
  {
    insert_dummy(cache, "custom_name", make_meta("docs", "vec", Metric::L2SqrtExpanded));
    insert_dummy(cache, "other", make_meta("docs", "title_vec", Metric::L2SqrtExpanded));

    REQUIRE(cache.erase_by_column("docs", "vec", Metric::L2SqrtExpanded) == 1);
    REQUIRE(cache.find("custom_name") == nullptr);
    REQUIRE(cache.find("other") != nullptr);  // different column untouched
  }
  SECTION("matches up to metric canonicalization (unexpanded query drops an expanded index)")
  {
    insert_dummy(cache, "idx", make_meta("docs", "vec", Metric::L2SqrtExpanded));
    REQUIRE(cache.erase_by_column("docs", "vec", Metric::L2SqrtUnexpanded) == 1);
    REQUIRE(cache.size() == 0);
  }
  SECTION("no match removes nothing")
  {
    insert_dummy(cache, "idx", make_meta("docs", "vec", Metric::L2SqrtExpanded));
    REQUIRE(cache.erase_by_column("docs", "vec", Metric::CosineExpanded) == 0);
    REQUIRE(cache.erase_by_column("docs", "other", Metric::L2SqrtExpanded) == 0);
    REQUIRE(cache.size() == 1);
  }
}
