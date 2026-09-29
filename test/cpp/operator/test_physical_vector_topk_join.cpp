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

#include "helper/type_conversions.hpp"
#include "operator_test_utils.hpp"

#include <cudf/column/column_factories.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/table/table.hpp>

#include <catch.hpp>
#include <cucascade/memory/memory_space.hpp>
#include <duckdb/common/types.hpp>
#include <op/sirius_physical_vector_topk_join.hpp>
#include <op/sirius_physical_vector_topk_merge.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <map>
#include <numeric>
#include <set>
#include <tuple>
#include <vector>

using namespace sirius::op;
using namespace sirius::test::operator_utils;

namespace {

constexpr int DIM = 3;

cucascade::memory::memory_space* mem_space()
{
  static auto manager = sirius::test::operator_utils::initialize_memory_manager();
  return manager->get_memory_space(cucascade::memory::Tier::GPU, 0);
}

//! FLOAT[dim] column as Sirius stores it: a LIST with a contiguous FLOAT32 child. Rows listed in
//! @p null_rows are NULL.
std::unique_ptr<cudf::column> make_vector_column(std::vector<std::vector<float>> const& rows,
                                                 std::vector<cudf::size_type> const& null_rows = {})
{
  auto const n = static_cast<cudf::size_type>(rows.size());
  std::vector<float> flat;
  for (auto const& r : rows) {
    flat.insert(flat.end(), r.begin(), r.end());
  }
  auto child = cudf::make_numeric_column(
    cudf::data_type{cudf::type_id::FLOAT32}, n * DIM, cudf::mask_state::UNALLOCATED);
  cudaMemcpy(child->mutable_view().data<float>(),
             flat.data(),
             sizeof(float) * flat.size(),
             cudaMemcpyHostToDevice);
  std::vector<int32_t> offsets(n + 1);
  for (cudf::size_type i = 0; i <= n; ++i) {
    offsets[i] = i * DIM;
  }
  auto offsets_col = cudf::make_numeric_column(
    cudf::data_type{cudf::type_id::INT32}, n + 1, cudf::mask_state::UNALLOCATED);
  cudaMemcpy(offsets_col->mutable_view().data<int32_t>(),
             offsets.data(),
             sizeof(int32_t) * offsets.size(),
             cudaMemcpyHostToDevice);

  rmm::device_buffer mask{};
  if (!null_rows.empty()) {
    mask = cudf::create_null_mask(n, cudf::mask_state::ALL_VALID);
    for (auto r : null_rows) {
      cudf::set_null_mask(static_cast<cudf::bitmask_type*>(mask.data()), r, r + 1, false);
    }
  }
  return cudf::make_lists_column(n,
                                 std::move(offsets_col),
                                 std::move(child),
                                 static_cast<cudf::size_type>(null_rows.size()),
                                 std::move(mask));
}

std::unique_ptr<cudf::column> make_int_column(std::vector<int32_t> const& values)
{
  auto col = cudf::make_numeric_column(cudf::data_type{cudf::type_id::INT32},
                                       static_cast<cudf::size_type>(values.size()),
                                       cudf::mask_state::UNALLOCATED);
  cudaMemcpy(col->mutable_view().data<int32_t>(),
             values.data(),
             sizeof(int32_t) * values.size(),
             cudaMemcpyHostToDevice);
  return col;
}

//! Left batch is [vec, id]; right batch is [id, vec] so the two vector columns sit at different
//! indices.
std::shared_ptr<cucascade::data_batch> make_left_batch(
  std::vector<std::vector<float>> const& vecs,
  std::vector<int32_t> const& ids,
  std::vector<cudf::size_type> const& nulls = {})
{
  std::vector<std::unique_ptr<cudf::column>> cols;
  cols.push_back(make_vector_column(vecs, nulls));
  cols.push_back(make_int_column(ids));
  return sirius::make_data_batch(std::make_unique<cudf::table>(std::move(cols)),
                                 *mem_space(),
                                 default_stream(),
                                 sirius::telemetry::batch_telemetry_info{});
}

std::shared_ptr<cucascade::data_batch> make_right_batch(std::vector<std::vector<float>> const& vecs,
                                                        std::vector<int32_t> const& ids)
{
  std::vector<std::unique_ptr<cudf::column>> cols;
  cols.push_back(make_int_column(ids));
  cols.push_back(make_vector_column(vecs));
  return sirius::make_data_batch(std::make_unique<cudf::table>(std::move(cols)),
                                 *mem_space(),
                                 default_stream(),
                                 sirius::telemetry::batch_telemetry_info{});
}

struct topk_ops {
  std::unique_ptr<sirius_physical_vector_topk_join> join;
  std::unique_ptr<sirius_physical_vector_topk_merge> merge;
};

topk_ops make_ops(std::int64_t k,
                  std::string metric,
                  bool is_similarity,
                  uint64_t batch_bytes       = 1 << 20,
                  duckdb::JoinType join_type = duckdb::JoinType::INNER)
{
  auto const vec = duckdb::LogicalType::ARRAY(duckdb::LogicalType::FLOAT, DIM);
  auto left      = duckdb::make_uniq<sirius_physical_operator>(
    SiriusPhysicalOperatorType::PROJECTION,
    sirius::from_duckdb_vec(duckdb::vector<duckdb::LogicalType>{vec, duckdb::LogicalType::INTEGER}),
    0);
  auto right = duckdb::make_uniq<sirius_physical_operator>(
    SiriusPhysicalOperatorType::PROJECTION,
    sirius::from_duckdb_vec(duckdb::vector<duckdb::LogicalType>{duckdb::LogicalType::INTEGER, vec}),
    0);
  topk_ops ops;
  ops.join  = std::make_unique<sirius_physical_vector_topk_join>(std::move(left),
                                                                std::move(right),
                                                                /*left_vector_col_idx=*/0,
                                                                /*right_vector_col_idx=*/1,
                                                                k,
                                                                std::move(metric),
                                                                is_similarity,
                                                                DIM,
                                                                join_type,
                                                                0);
  ops.merge = std::make_unique<sirius_physical_vector_topk_merge>(*ops.join, batch_bytes);
  // No pipeline assigns ids here; execute() reads the operator id for batch telemetry.
  ops.join->operator_id              = 0;
  ops.join->children[0]->operator_id = 1;
  ops.join->children[1]->operator_id = 2;
  ops.merge->operator_id             = 3;
  return ops;
}

using batch_list = std::vector<std::shared_ptr<cucascade::data_batch>>;

//! Runs the select stage on every (left batch, right batch) pair, routing its outputs by merge
//! partition as the pipeline would, then the merge once per left batch. Returns the merge outputs.
batch_list run_batches(topk_ops& ops, batch_list const& lefts, batch_list const& rights)
{
  std::map<std::size_t, batch_list> by_partition;
  auto select = [&](batch_list in, std::size_t ordinal, bool emit_left) {
    auto out =
      ops.join->execute(topk_pair_input(std::move(in), ordinal, emit_left), default_stream());
    auto const& partial = dynamic_cast<const topk_partial_output&>(*out);
    for (std::size_t b = 0; b < partial.get_data_batches().size(); ++b) {
      by_partition[partial.partitions[b]].push_back(partial.get_data_batches()[b]);
    }
  };
  for (std::size_t i = 0; i < lefts.size(); ++i) {
    if (rights.empty()) { select({lefts[i]}, i, true); }
    for (std::size_t j = 0; j < rights.size(); ++j) {
      select({lefts[i], rights[j]}, i, j == 0);
    }
  }

  batch_list merged;
  for (std::size_t i = 0; i < lefts.size(); ++i) {
    auto inputs = by_partition[topk_left_partition(i)];
    REQUIRE(inputs.size() == 1);
    auto const& partials = by_partition[topk_partials_partition(i)];
    inputs.insert(inputs.end(), partials.begin(), partials.end());
    auto out =
      ops.merge->execute(partitioned_operator_data(std::move(inputs), i), default_stream());
    auto const& batches = dynamic_cast<const pipelineable_operator_data&>(*out).get_data_batches();
    REQUIRE(!batches.empty());
    merged.insert(merged.end(), batches.begin(), batches.end());
  }
  return merged;
}

//! One output row: (left id, right id, ranking value).
using result_row = std::tuple<int32_t, int32_t, float>;

//! Flattens the merge outputs, checking the output schema on the way.
std::vector<result_row> run(topk_ops& ops,
                            batch_list const& lefts,
                            batch_list const& rights,
                            std::size_t* n_batches = nullptr)
{
  auto batches = run_batches(ops, lefts, rights);
  if (n_batches) { *n_batches = batches.size(); }
  std::vector<result_row> rows;
  for (auto const& b : batches) {
    auto view = sirius::get_cudf_table_view(*b);
    // [left vec, left id, right id, right vec, ranking]
    REQUIRE(view.num_columns() == 5);
    auto lid = copy_column_to_host<int32_t>(view.column(1));
    auto rid = copy_column_to_host<int32_t>(view.column(2));
    auto val = copy_column_to_host<float>(view.column(4));
    for (std::size_t i = 0; i < lid.size(); ++i) {
      rows.emplace_back(lid[i], rid[i], val[i]);
    }
  }
  return rows;
}

//! Slice [begin, end) of a host-side list.
template <typename T>
std::vector<T> slice_of(std::vector<T> const& v, std::size_t begin, std::size_t end)
{
  return std::vector<T>(v.begin() + begin, v.begin() + end);
}

float l2(std::vector<float> const& a, std::vector<float> const& b)
{
  float s = 0;
  for (int i = 0; i < DIM; ++i) {
    s += (a[i] - b[i]) * (a[i] - b[i]);
  }
  return std::sqrt(s);
}

float cosine_similarity(std::vector<float> const& a, std::vector<float> const& b)
{
  float dot = 0, na = 0, nb = 0;
  for (int i = 0; i < DIM; ++i) {
    dot += a[i] * b[i];
    na += a[i] * a[i];
    nb += b[i] * b[i];
  }
  return dot / (std::sqrt(na) * std::sqrt(nb));
}

//! CPU reference: for each left row, the k right rows with the best score.
template <typename Score>
std::map<int32_t, std::vector<int32_t>> reference_topk(std::vector<std::vector<float>> const& lv,
                                                       std::vector<int32_t> const& lid,
                                                       std::vector<std::vector<float>> const& rv,
                                                       std::vector<int32_t> const& rid,
                                                       std::size_t k,
                                                       bool keep_largest,
                                                       Score score)
{
  std::map<int32_t, std::vector<int32_t>> out;
  for (std::size_t i = 0; i < lv.size(); ++i) {
    std::vector<std::size_t> order(rv.size());
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](auto a, auto b) {
      auto sa = score(lv[i], rv[a]), sb = score(lv[i], rv[b]);
      return keep_largest ? sa > sb : sa < sb;
    });
    for (std::size_t j = 0; j < std::min(k, rv.size()); ++j) {
      out[lid[i]].push_back(rid[order[j]]);
    }
  }
  return out;
}

std::map<int32_t, std::vector<int32_t>> group_by_left(std::vector<result_row> const& rows)
{
  std::map<int32_t, std::vector<int32_t>> out;
  for (auto const& [l, r, v] : rows) {
    out[l].push_back(r);
  }
  return out;
}

// Distinct distances from every left row, so the expected neighbor order has no ties.
std::vector<std::vector<float>> const LEFT_VECS  = {{0, 0, 0}, {10, 0, 0}, {0, 7, 1}};
std::vector<int32_t> const LEFT_IDS              = {100, 101, 102};
std::vector<std::vector<float>> const RIGHT_VECS = {
  {1, 0, 0}, {0, 2, 0}, {0.5F, 0, 3.5F}, {9, 0.5F, 0}, {-4, 1, 1}, {0, 6, 2}};
std::vector<int32_t> const RIGHT_IDS = {1, 2, 3, 4, 5, 6};

}  // namespace

TEST_CASE("vector top-k join: L2 returns each left row's k nearest, nearest first",
          "[operator][vss][vector_topk]")
{
  auto ops = make_ops(/*k=*/3, "l2", /*is_similarity=*/false);
  auto rows =
    run(ops, {make_left_batch(LEFT_VECS, LEFT_IDS)}, {make_right_batch(RIGHT_VECS, RIGHT_IDS)});

  REQUIRE(rows.size() == LEFT_IDS.size() * 3);
  REQUIRE(group_by_left(rows) ==
          reference_topk(LEFT_VECS, LEFT_IDS, RIGHT_VECS, RIGHT_IDS, 3, false, l2));

  // The ranking column is the actual Euclidean distance.
  for (auto const& [l, r, v] : rows) {
    auto li = std::find(LEFT_IDS.begin(), LEFT_IDS.end(), l) - LEFT_IDS.begin();
    auto ri = std::find(RIGHT_IDS.begin(), RIGHT_IDS.end(), r) - RIGHT_IDS.begin();
    REQUIRE(v == Approx(l2(LEFT_VECS[li], RIGHT_VECS[ri])).margin(1e-3));
  }
}

TEST_CASE("vector top-k join: several right batches merge into the overall top-k",
          "[operator][vss][vector_topk]")
{
  auto const expected = reference_topk(LEFT_VECS, LEFT_IDS, RIGHT_VECS, RIGHT_IDS, 3, false, l2);

  // Three right batches of 2 rows each: every batch is smaller than k, so every partial is padded.
  {
    auto ops  = make_ops(/*k=*/3, "l2", /*is_similarity=*/false);
    auto rows = run(ops,
                    {make_left_batch(LEFT_VECS, LEFT_IDS)},
                    {make_right_batch(slice_of(RIGHT_VECS, 0, 2), slice_of(RIGHT_IDS, 0, 2)),
                     make_right_batch(slice_of(RIGHT_VECS, 2, 4), slice_of(RIGHT_IDS, 2, 4)),
                     make_right_batch(slice_of(RIGHT_VECS, 4, 6), slice_of(RIGHT_IDS, 4, 6))});
    REQUIRE(rows.size() == LEFT_IDS.size() * 3);
    REQUIRE(group_by_left(rows) == expected);
  }
  // Uneven batches: one larger than k and one smaller.
  {
    auto ops  = make_ops(/*k=*/3, "l2", /*is_similarity=*/false);
    auto rows = run(ops,
                    {make_left_batch(LEFT_VECS, LEFT_IDS)},
                    {make_right_batch(slice_of(RIGHT_VECS, 0, 5), slice_of(RIGHT_IDS, 0, 5)),
                     make_right_batch(slice_of(RIGHT_VECS, 5, 6), slice_of(RIGHT_IDS, 5, 6))});
    REQUIRE(group_by_left(rows) == expected);
  }
}

TEST_CASE("vector top-k join: several left batches each get their own top-k",
          "[operator][vss][vector_topk]")
{
  auto ops  = make_ops(/*k=*/2, "l2", /*is_similarity=*/false);
  auto rows = run(ops,
                  {make_left_batch(slice_of(LEFT_VECS, 0, 1), slice_of(LEFT_IDS, 0, 1)),
                   make_left_batch(slice_of(LEFT_VECS, 1, 3), slice_of(LEFT_IDS, 1, 3))},
                  {make_right_batch(slice_of(RIGHT_VECS, 0, 3), slice_of(RIGHT_IDS, 0, 3)),
                   make_right_batch(slice_of(RIGHT_VECS, 3, 6), slice_of(RIGHT_IDS, 3, 6))});
  REQUIRE(group_by_left(rows) ==
          reference_topk(LEFT_VECS, LEFT_IDS, RIGHT_VECS, RIGHT_IDS, 2, false, l2));
}

TEST_CASE("vector top-k join: k larger than the right side returns every right row",
          "[operator][vss][vector_topk]")
{
  auto const expected = reference_topk(LEFT_VECS, LEFT_IDS, RIGHT_VECS, RIGHT_IDS, 50, false, l2);
  {
    auto ops = make_ops(/*k=*/50, "l2", /*is_similarity=*/false);
    auto rows =
      run(ops, {make_left_batch(LEFT_VECS, LEFT_IDS)}, {make_right_batch(RIGHT_VECS, RIGHT_IDS)});
    REQUIRE(rows.size() == LEFT_IDS.size() * RIGHT_IDS.size());
    REQUIRE(group_by_left(rows) == expected);
  }
  // Split across batches, so the padding rows from every batch must be dropped after the merge.
  {
    auto ops  = make_ops(/*k=*/50, "l2", /*is_similarity=*/false);
    auto rows = run(ops,
                    {make_left_batch(LEFT_VECS, LEFT_IDS)},
                    {make_right_batch(slice_of(RIGHT_VECS, 0, 4), slice_of(RIGHT_IDS, 0, 4)),
                     make_right_batch(slice_of(RIGHT_VECS, 4, 6), slice_of(RIGHT_IDS, 4, 6))});
    REQUIRE(rows.size() == LEFT_IDS.size() * RIGHT_IDS.size());
    REQUIRE(group_by_left(rows) == expected);
  }
}

TEST_CASE("vector top-k join: cosine similarity keeps the most similar and reports similarity",
          "[operator][vss][vector_topk]")
{
  std::vector<std::vector<float>> const lv = {{1, 0, 0}, {0, 1, 1}};
  auto ops                                 = make_ops(/*k=*/2, "cosine", /*is_similarity=*/true);
  auto rows                                = run(ops,
                                                 {make_left_batch(lv, {100, 101})},
                                                 {make_right_batch(slice_of(RIGHT_VECS, 0, 3), slice_of(RIGHT_IDS, 0, 3)),
                                                  make_right_batch(slice_of(RIGHT_VECS, 3, 6), slice_of(RIGHT_IDS, 3, 6))});

  REQUIRE(rows.size() == 4);
  REQUIRE(group_by_left(rows) ==
          reference_topk(lv, {100, 101}, RIGHT_VECS, RIGHT_IDS, 2, true, cosine_similarity));
  for (auto const& [l, r, v] : rows) {
    auto ri = std::find(RIGHT_IDS.begin(), RIGHT_IDS.end(), r) - RIGHT_IDS.begin();
    REQUIRE(v == Approx(cosine_similarity(lv[l - 100], RIGHT_VECS[ri])).margin(1e-3));
  }
}

TEST_CASE("vector top-k join: output splits into several batches under a small byte budget",
          "[operator][vss][vector_topk]")
{
  auto ops              = make_ops(/*k=*/3, "l2", /*is_similarity=*/false, /*batch_bytes=*/64);
  std::size_t n_batches = 0;
  auto rows             = run(ops,
                              {make_left_batch(LEFT_VECS, LEFT_IDS)},
                              {make_right_batch(RIGHT_VECS, RIGHT_IDS)},
                  &n_batches);

  REQUIRE(n_batches > 1);
  REQUIRE(group_by_left(rows) ==
          reference_topk(LEFT_VECS, LEFT_IDS, RIGHT_VECS, RIGHT_IDS, 3, false, l2));
}

TEST_CASE("vector top-k join: INNER with an empty right side emits one empty batch",
          "[operator][vss][vector_topk]")
{
  auto ops = make_ops(/*k=*/3, "l2", /*is_similarity=*/false);
  // An empty right table reaches the join as no batch at all.
  std::size_t n_batches = 0;
  auto rows             = run(ops, {make_left_batch(LEFT_VECS, LEFT_IDS)}, {}, &n_batches);
  REQUIRE(n_batches == 1);
  REQUIRE(rows.empty());
}

TEST_CASE("vector top-k join: LEFT with an empty right side keeps every left row, NULL-padded",
          "[operator][vss][vector_topk]")
{
  auto ops     = make_ops(/*k=*/3, "l2", /*is_similarity=*/false, 1 << 20, duckdb::JoinType::LEFT);
  auto batches = run_batches(ops, {make_left_batch(LEFT_VECS, LEFT_IDS)}, {});
  REQUIRE(batches.size() == 1);
  auto view = sirius::get_cudf_table_view(*batches[0]);
  // [left vec, left id, right id, right vec, ranking]
  REQUIRE(view.num_columns() == 5);
  REQUIRE(view.num_rows() == static_cast<cudf::size_type>(LEFT_IDS.size()));
  REQUIRE(copy_column_to_host<int32_t>(view.column(1)) == LEFT_IDS);
  for (int c : {2, 3, 4}) {
    REQUIRE(view.column(c).null_count() == view.num_rows());
  }
}

TEST_CASE("vector top-k join: LEFT with a non-empty right side matches INNER",
          "[operator][vss][vector_topk]")
{
  auto ops = make_ops(/*k=*/3, "l2", /*is_similarity=*/false, 1 << 20, duckdb::JoinType::LEFT);
  auto rows =
    run(ops, {make_left_batch(LEFT_VECS, LEFT_IDS)}, {make_right_batch(RIGHT_VECS, RIGHT_IDS)});
  REQUIRE(group_by_left(rows) ==
          reference_topk(LEFT_VECS, LEFT_IDS, RIGHT_VECS, RIGHT_IDS, 3, false, l2));
}

TEST_CASE("vector top-k join: a NULL vector throws a clear error", "[operator][vss][vector_topk]")
{
  auto ops = make_ops(/*k=*/3, "l2", /*is_similarity=*/false);
  REQUIRE_THROWS_WITH(run(ops,
                          {make_left_batch(LEFT_VECS, LEFT_IDS, /*nulls=*/{1})},
                          {make_right_batch(RIGHT_VECS, RIGHT_IDS)}),
                      Catch::Contains("NULL vectors"));
}

namespace {

std::unique_ptr<sirius_physical_vector_topk_join> make_global_join(std::int64_t k,
                                                                   std::string metric,
                                                                   bool is_similarity)
{
  auto const vec = duckdb::LogicalType::ARRAY(duckdb::LogicalType::FLOAT, DIM);
  auto left      = duckdb::make_uniq<sirius_physical_operator>(
    SiriusPhysicalOperatorType::PROJECTION,
    sirius::from_duckdb_vec(duckdb::vector<duckdb::LogicalType>{vec, duckdb::LogicalType::INTEGER}),
    0);
  auto right = duckdb::make_uniq<sirius_physical_operator>(
    SiriusPhysicalOperatorType::PROJECTION,
    sirius::from_duckdb_vec(duckdb::vector<duckdb::LogicalType>{duckdb::LogicalType::INTEGER, vec}),
    0);
  auto join         = std::make_unique<sirius_physical_vector_topk_join>(std::move(left),
                                                                 std::move(right),
                                                                 /*left_vector_col_idx=*/0,
                                                                 /*right_vector_col_idx=*/1,
                                                                 k,
                                                                 std::move(metric),
                                                                 is_similarity,
                                                                 DIM,
                                                                 duckdb::JoinType::INNER,
                                                                 0,
                                                                 topk_scope::global);
  join->operator_id = 0;
  join->children[0]->operator_id = 1;
  join->children[1]->operator_id = 2;
  return join;
}

//! Runs the global select stage on every pair, then keeps the overall k best rows on the host
//! (what TOP_N / MERGE_TOP_N do downstream). Returns (left id, right id, ranking value).
std::vector<result_row> run_global(sirius_physical_vector_topk_join& join,
                                   batch_list const& lefts,
                                   batch_list const& rights,
                                   std::size_t k,
                                   bool keep_largest)
{
  std::vector<result_row> rows;
  auto collect = [&](batch_list in) {
    auto out = join.execute(topk_pair_input(std::move(in), 0, false), default_stream());
    for (auto const& b : dynamic_cast<const pipelineable_operator_data&>(*out).get_data_batches()) {
      auto view = sirius::get_cudf_table_view(*b);
      // [left vec, left id, right id, right vec, ranking]
      REQUIRE(view.num_columns() == 5);
      auto lid = copy_column_to_host<int32_t>(view.column(1));
      auto rid = copy_column_to_host<int32_t>(view.column(2));
      auto val = copy_column_to_host<float>(view.column(4));
      // Each pair contributes at most k candidates.
      REQUIRE(lid.size() <= k);
      for (std::size_t i = 0; i < lid.size(); ++i) {
        rows.emplace_back(lid[i], rid[i], val[i]);
      }
    }
  };
  for (auto const& l : lefts) {
    if (rights.empty()) { collect({l}); }
    for (auto const& r : rights) {
      collect({l, r});
    }
  }
  std::sort(rows.begin(), rows.end(), [&](auto const& a, auto const& b) {
    return keep_largest ? std::get<2>(a) > std::get<2>(b) : std::get<2>(a) < std::get<2>(b);
  });
  if (rows.size() > k) { rows.resize(k); }
  return rows;
}

//! CPU reference: the k best (left id, right id) pairs over the full cross product.
template <typename Score>
std::set<std::pair<int32_t, int32_t>> reference_global(std::vector<std::vector<float>> const& lv,
                                                       std::vector<int32_t> const& lid,
                                                       std::vector<std::vector<float>> const& rv,
                                                       std::vector<int32_t> const& rid,
                                                       std::size_t k,
                                                       bool keep_largest,
                                                       Score score)
{
  std::vector<std::tuple<float, int32_t, int32_t>> all;
  for (std::size_t i = 0; i < lv.size(); ++i) {
    for (std::size_t j = 0; j < rv.size(); ++j) {
      all.emplace_back(score(lv[i], rv[j]), lid[i], rid[j]);
    }
  }
  std::sort(all.begin(), all.end(), [&](auto const& a, auto const& b) {
    return keep_largest ? std::get<0>(a) > std::get<0>(b) : std::get<0>(a) < std::get<0>(b);
  });
  std::set<std::pair<int32_t, int32_t>> out;
  for (std::size_t i = 0; i < std::min(k, all.size()); ++i) {
    out.emplace(std::get<1>(all[i]), std::get<2>(all[i]));
  }
  return out;
}

std::set<std::pair<int32_t, int32_t>> pairs_of(std::vector<result_row> const& rows)
{
  std::set<std::pair<int32_t, int32_t>> out;
  for (auto const& [l, r, v] : rows) {
    out.emplace(l, r);
  }
  return out;
}

}  // namespace

TEST_CASE("vector global top-k join: k best pairs overall, with their distances",
          "[operator][vss][vector_topk]")
{
  auto join = make_global_join(/*k=*/4, "l2", /*is_similarity=*/false);
  auto rows = run_global(*join,
                         {make_left_batch(LEFT_VECS, LEFT_IDS)},
                         {make_right_batch(RIGHT_VECS, RIGHT_IDS)},
                         4,
                         false);
  REQUIRE(pairs_of(rows) ==
          reference_global(LEFT_VECS, LEFT_IDS, RIGHT_VECS, RIGHT_IDS, 4, false, l2));
  for (auto const& [l, r, v] : rows) {
    auto li = std::find(LEFT_IDS.begin(), LEFT_IDS.end(), l) - LEFT_IDS.begin();
    auto ri = std::find(RIGHT_IDS.begin(), RIGHT_IDS.end(), r) - RIGHT_IDS.begin();
    REQUIRE(v == Approx(l2(LEFT_VECS[li], RIGHT_VECS[ri])).margin(1e-3));
  }
}

TEST_CASE("vector global top-k join: several left and right batches",
          "[operator][vss][vector_topk]")
{
  auto join = make_global_join(/*k=*/5, "l2", /*is_similarity=*/false);
  auto rows = run_global(*join,
                         {make_left_batch(slice_of(LEFT_VECS, 0, 1), slice_of(LEFT_IDS, 0, 1)),
                          make_left_batch(slice_of(LEFT_VECS, 1, 3), slice_of(LEFT_IDS, 1, 3))},
                         {make_right_batch(slice_of(RIGHT_VECS, 0, 2), slice_of(RIGHT_IDS, 0, 2)),
                          make_right_batch(slice_of(RIGHT_VECS, 2, 6), slice_of(RIGHT_IDS, 2, 6))},
                         5,
                         false);
  REQUIRE(pairs_of(rows) ==
          reference_global(LEFT_VECS, LEFT_IDS, RIGHT_VECS, RIGHT_IDS, 5, false, l2));
}

TEST_CASE("vector global top-k join: k larger than all pairs returns every pair",
          "[operator][vss][vector_topk]")
{
  auto join = make_global_join(/*k=*/100, "l2", /*is_similarity=*/false);
  auto rows = run_global(*join,
                         {make_left_batch(LEFT_VECS, LEFT_IDS)},
                         {make_right_batch(RIGHT_VECS, RIGHT_IDS)},
                         100,
                         false);
  REQUIRE(rows.size() == LEFT_IDS.size() * RIGHT_IDS.size());
}

TEST_CASE("vector global top-k join: cosine similarity keeps the most similar pairs",
          "[operator][vss][vector_topk]")
{
  std::vector<std::vector<float>> const lv = {{1, 0, 0}, {0, 1, 1}};
  auto join = make_global_join(/*k=*/2, "cosine", /*is_similarity=*/true);
  auto rows = run_global(
    *join, {make_left_batch(lv, {100, 101})}, {make_right_batch(RIGHT_VECS, RIGHT_IDS)}, 2, true);
  REQUIRE(pairs_of(rows) ==
          reference_global(lv, {100, 101}, RIGHT_VECS, RIGHT_IDS, 2, true, cosine_similarity));
  for (auto const& [l, r, v] : rows) {
    auto ri = std::find(RIGHT_IDS.begin(), RIGHT_IDS.end(), r) - RIGHT_IDS.begin();
    REQUIRE(v == Approx(cosine_similarity(lv[l - 100], RIGHT_VECS[ri])).margin(1e-3));
  }
}

TEST_CASE("vector global top-k join: empty right side emits one empty batch",
          "[operator][vss][vector_topk]")
{
  auto join = make_global_join(/*k=*/3, "l2", /*is_similarity=*/false);
  auto out  = join->execute(topk_pair_input({make_left_batch(LEFT_VECS, LEFT_IDS)}, 0, false),
                           default_stream());
  auto const& batches = dynamic_cast<const pipelineable_operator_data&>(*out).get_data_batches();
  REQUIRE(batches.size() == 1);
  REQUIRE(sirius::get_cudf_table_view(*batches[0]).num_rows() == 0);
}
