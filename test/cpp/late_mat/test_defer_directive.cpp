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

// [late_mat][directive] — the two halves of a deferral. No GPU.
//
// The invariants here are all about the pair being inseparable and the batch
// being identified exactly. Every failure mode is silent: a scan that
// substitutes with no consumer loses columns, a consumer that fires on the
// wrong batch reads arbitrary rows of a pinned table, and a schema that drifts
// between the two halves materializes into positions that moved. None of those
// announce themselves, so they are refused at construction instead.

#include <catch.hpp>
#include <late_mat/defer_directive.hpp>

#include <memory>
#include <vector>

using sirius::late_mat::column_origin;
using sirius::late_mat::defer_pair;
using sirius::late_mat::kPlaceholderType;
using sirius::late_mat::kRowidType;
using sirius::late_mat::make_defer_pair;
using sirius::late_mat::pin_entry_handle;

namespace {

cudf::data_type dt(cudf::type_id id) { return cudf::data_type{id}; }

/// q10's customer bundle in miniature: a key, then the wide payload.
std::vector<cudf::data_type> customer_schema()
{
  return {dt(cudf::type_id::INT64),      // c_custkey
          dt(cudf::type_id::STRING),     // c_name
          dt(cudf::type_id::DECIMAL64),  // c_acctbal
          dt(cudf::type_id::STRING),     // c_address
          dt(cudf::type_id::INT32)};     // n_nationkey
}

std::vector<column_origin> origins(std::size_t n)
{
  auto handle = std::make_shared<pin_entry_handle>("customer", 3);
  std::vector<column_origin> out;
  for (std::size_t i = 0; i < n; ++i) {
    column_origin o;
    o.handle     = handle;
    o.column_pos = static_cast<std::uint32_t>(i);
    o.generation = handle->generation();
    out.push_back(std::move(o));
  }
  return out;
}

}  // namespace

TEST_CASE("a deferral substitutes a rowid and placeholders in place", "[late_mat][directive]")
{
  auto const pair =
    make_defer_pair(customer_schema(), {1, 2, 3}, customer_schema(), {1, 2, 3}, origins(3));
  REQUIRE(pair.valid());

  // Arity is preserved and only the deferred positions change type, so nothing
  // between the two halves sees a different table shape.
  auto const& schema = pair.port.expected_schema;
  REQUIRE(schema.size() == customer_schema().size());
  REQUIRE(schema[0] == dt(cudf::type_id::INT64));
  REQUIRE(schema[4] == dt(cudf::type_id::INT32));

  REQUIRE(schema[1].id() == kRowidType);        // the ride
  REQUIRE(schema[2].id() == kPlaceholderType);  // holding a position, nothing more
  REQUIRE(schema[3].id() == kPlaceholderType);
  REQUIRE(pair.scan.rowid_position() == 1);
  REQUIRE(pair.port.rowid_position() == 1);
}

TEST_CASE("the restored types are the producer's original ones", "[late_mat][directive]")
{
  // What comes back has to be what would have been there, or the consumer's
  // downstream expressions are typed against a column that no longer matches.
  auto const pair =
    make_defer_pair(customer_schema(), {1, 3}, customer_schema(), {1, 3}, origins(2));
  REQUIRE(pair.valid());
  REQUIRE(pair.port.restored_types ==
          std::vector<cudf::data_type>{dt(cudf::type_id::STRING), dt(cudf::type_id::STRING)});
}

TEST_CASE("the two halves name the same positions", "[late_mat][directive]")
{
  auto const pair =
    make_defer_pair(customer_schema(), {2, 3}, customer_schema(), {2, 3}, origins(2));
  REQUIRE(pair.valid());
  REQUIRE(pair.scan.output_positions == pair.port.output_positions);
  REQUIRE(pair.scan.defers(2));
  REQUIRE(pair.scan.defers(3));
  REQUIRE_FALSE(pair.scan.defers(0));
}

TEST_CASE("a directive fires only on the batch it was built for", "[late_mat][directive]")
{
  auto const pair =
    make_defer_pair(customer_schema(), {1, 2}, customer_schema(), {1, 2}, origins(2));
  REQUIRE(pair.port.matches(pair.port.expected_schema));
}

TEST_CASE("a batch of another shape is declined", "[late_mat][directive]")
{
  auto const pair =
    make_defer_pair(customer_schema(), {1, 2}, customer_schema(), {1, 2}, origins(2));

  // Same arity, one type different: materializing here would write pinned rows
  // into a batch that never asked for them.
  auto other = pair.port.expected_schema;
  other[4]   = dt(cudf::type_id::INT64);
  REQUIRE_FALSE(pair.port.matches(other));

  // A batch that never had the substitution at all.
  REQUIRE_FALSE(pair.port.matches(customer_schema()));

  // Same types, fewer columns.
  auto shorter = pair.port.expected_schema;
  shorter.pop_back();
  REQUIRE_FALSE(pair.port.matches(shorter));
}

TEST_CASE("unbuildable requests produce no pair at all", "[late_mat][directive]")
{
  // Each of these would install a half-deferral, which is worse than none.
  REQUIRE_FALSE(make_defer_pair(customer_schema(), {}, customer_schema(), {}, {}).valid());
  REQUIRE_FALSE(
    make_defer_pair(customer_schema(), {1, 2}, customer_schema(), {1, 2}, origins(1)).valid());
  REQUIRE_FALSE(
    make_defer_pair(customer_schema(), {1, 99}, customer_schema(), {1, 99}, origins(2)).valid());
  REQUIRE_FALSE(make_defer_pair(customer_schema(), {3, 1}, customer_schema(), {3, 1}, origins(2))
                  .valid());  // unordered
  REQUIRE_FALSE(make_defer_pair(customer_schema(), {1, 1}, customer_schema(), {1, 1}, origins(2))
                  .valid());  // repeated
}

TEST_CASE("a pair with an empty origin is refused", "[late_mat][directive]")
{
  // An origin that cannot resolve is a deferral that cannot be undone.
  std::vector<column_origin> bad(2);
  REQUIRE_FALSE(make_defer_pair(customer_schema(), {1, 2}, customer_schema(), {1, 2}, bad).valid());
}

TEST_CASE("the port's coordinates are its own, not the scan's", "[late_mat][directive]")
{
  // The failure this exists to catch: building the port half from the SCAN's
  // schema. A join between the two ends widens the table and reorders it, so a
  // port directive in the scan's coordinates matches no batch that ever
  // arrives — and the reader is handed the rowid instead of the values, which
  // is a wrong answer rather than a refused optimization.
  //
  // Here customer columns 1 and 3 arrive at the port as columns 6 and 4 of a
  // joined table: reordered, so the rowid does NOT ride at the front.
  std::vector<cudf::data_type> const joined{dt(cudf::type_id::INT64),
                                            dt(cudf::type_id::INT64),
                                            dt(cudf::type_id::DECIMAL64),
                                            dt(cudf::type_id::INT32),
                                            dt(cudf::type_id::STRING),  // was customer col 3
                                            dt(cudf::type_id::INT32),
                                            dt(cudf::type_id::STRING)};  // was customer col 1

  auto const pair = make_defer_pair(customer_schema(), {1, 3}, joined, {6, 4}, origins(2));
  REQUIRE(pair.valid());

  // The scan half is in the scan's coordinates, rowid at the first deferred one.
  REQUIRE(pair.scan.output_positions == std::vector<std::size_t>{1, 3});
  REQUIRE(pair.scan.rowid_position() == 1);

  // The port half is in the port's, ascending — and the rowid is at 6, where
  // the column carrying it actually landed, not at the front of the bundle.
  REQUIRE(pair.port.output_positions == std::vector<std::size_t>{4, 6});
  REQUIRE(pair.port.rowid_position() == 6);
  REQUIRE(pair.port.expected_schema[6].id() == sirius::late_mat::kRowidType);
  REQUIRE(pair.port.expected_schema[4].id() == sirius::late_mat::kPlaceholderType);
  // Everything else is the joined table untouched.
  REQUIRE(pair.port.expected_schema[0] == joined[0]);
  REQUIRE(pair.port.expected_schema.size() == joined.size());

  // Origins and restored types follow the PORT's order, since that is the order
  // the splice walks.
  REQUIRE(pair.port.restored_types ==
          std::vector<cudf::data_type>{dt(cudf::type_id::STRING), dt(cudf::type_id::STRING)});
  REQUIRE(pair.port.origins[0].column_pos == 1);  // customer col 3's origin
  REQUIRE(pair.port.origins[1].column_pos == 0);  // customer col 1's origin
}

TEST_CASE("a column may not change type on the ride", "[late_mat][directive]")
{
  // The port restores what the scan gave up. If the two disagree about a
  // deferred column's type, the walk has mistracked the position — and
  // materializing would put a column of one type where another was expected.
  std::vector<cudf::data_type> wrong = customer_schema();
  wrong[1]                           = dt(cudf::type_id::INT32);  // was STRING
  REQUIRE_FALSE(make_defer_pair(customer_schema(), {1, 3}, wrong, {1, 3}, origins(2)).valid());
}
