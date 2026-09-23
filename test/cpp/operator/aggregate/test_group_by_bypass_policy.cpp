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

#include "op/aggregate/group_by_bypass_policy.hpp"

#include <catch.hpp>

#include <limits>

using namespace sirius::op::group_by_bypass;

namespace {

/// A candidate that passes every gate, with a budget large enough that the memory check is the
/// only thing under test. Individual tests break exactly one field.
candidate_input good_candidate()
{
  candidate_input in;
  in.auto_num_partitions          = 8;
  in.num_admitted_gpus            = 1;
  in.upstream_complete            = true;
  in.single_gpu_resident          = true;
  in.supported_state              = true;
  in.supported_downstream         = true;
  in.total_rows                   = 1'000'000;
  in.key_width_bytes              = 8;   // one INT64 key
  in.agg_width_bytes              = 16;  // two INT64 partial states
  in.key_columns                  = 1;
  in.agg_columns                  = 2;
  in.nullable_key_columns         = 0;
  in.nullable_agg_columns         = 0;
  in.admissible_additional_budget = 8ULL * 1024 * 1024 * 1024;
  in.headroom_fraction            = 0.25;
  return in;
}

}  // namespace

TEST_CASE("group-by bypass selects P=1 for a supported complete candidate",
          "[group_by_bypass][policy]")
{
  auto const d = decide(good_candidate());
  CHECK(d.reason == decision_reason::bypass_selected);
  CHECK(d.num_partitions == 1);
  CHECK(d.model_evaluated);
  CHECK(d.model.required_bytes > 0);
}

TEST_CASE("group-by bypass preserves the automatic count when a gate rejects",
          "[group_by_bypass][policy]")
{
  // Each case breaks one precondition; every one must leave AUTO untouched and claim no
  // activation. Rejection must preserve the automatic count.
  auto expect_auto_preserved = [](candidate_input in, decision_reason expected) {
    auto const d = decide(in);
    INFO("expected reason: " << reason_name(expected) << ", got: " << reason_name(d.reason));
    CHECK(d.reason == expected);
    CHECK(d.num_partitions == in.auto_num_partitions);
  };

  SECTION("upstream still running")
  {
    auto in              = good_candidate();
    in.upstream_complete = false;
    expect_auto_preserved(in, decision_reason::projected_input);
  }
  SECTION("more than one admitted GPU")
  {
    auto in              = good_candidate();
    in.num_admitted_gpus = 4;
    expect_auto_preserved(in, decision_reason::multi_gpu);
  }
  SECTION("input not resident on a single GPU")
  {
    auto in                = good_candidate();
    in.single_gpu_resident = false;
    expect_auto_preserved(in, decision_reason::unsupported_residency);
  }
  SECTION("unsupported key or aggregate state")
  {
    auto in            = good_candidate();
    in.supported_state = false;
    expect_auto_preserved(in, decision_reason::unsupported_state);
  }
  SECTION("unsupported downstream")
  {
    auto in                 = good_candidate();
    in.supported_downstream = false;
    expect_auto_preserved(in, decision_reason::unsupported_downstream);
  }
  SECTION("row count unknown")
  {
    auto in       = good_candidate();
    in.total_rows = std::nullopt;
    expect_auto_preserved(in, decision_reason::unknown_metadata);
  }
  SECTION("column count unknown")
  {
    auto in        = good_candidate();
    in.agg_columns = std::nullopt;
    expect_auto_preserved(in, decision_reason::unknown_metadata);
  }
  SECTION("nullability unknown is not the same as zero nullable columns")
  {
    auto in                 = good_candidate();
    in.nullable_agg_columns = std::nullopt;
    expect_auto_preserved(in, decision_reason::unknown_metadata);
  }
  SECTION("budget unknown")
  {
    auto in                         = good_candidate();
    in.admissible_additional_budget = std::nullopt;
    expect_auto_preserved(in, decision_reason::unknown_metadata);
  }
  SECTION("budget too small")
  {
    auto in                         = good_candidate();
    in.admissible_additional_budget = 1024;
    expect_auto_preserved(in, decision_reason::insufficient_budget);
  }
}

TEST_CASE("group-by bypass reports an existing one-partition choice as already_one",
          "[group_by_bypass][policy]")
{
  auto in                = good_candidate();
  in.auto_num_partitions = 1;
  auto const d           = decide(in);
  CHECK(d.reason == decision_reason::already_one);
  CHECK(d.num_partitions == 1);
  // The pre-existing automatic choice must not be treated as a bypass selection, and must
  // not acquire the new reservation floor that a real activation gets.
  CHECK_FALSE(d.model_evaluated);
}

TEST_CASE("group-by bypass rejects rather than wrapping on oversized inputs",
          "[group_by_bypass][policy]")
{
  SECTION("above cuDF's concatenated row limit")
  {
    auto in       = good_candidate();
    in.total_rows = CUDF_MAX_ROWS + 1;
    auto const d  = decide(in);
    CHECK(d.reason == decision_reason::size_overflow);
    CHECK(d.num_partitions == in.auto_num_partitions);
  }

  SECTION("exactly at the row limit is still evaluated, not rejected outright")
  {
    auto in                         = good_candidate();
    in.total_rows                   = CUDF_MAX_ROWS;
    in.admissible_additional_budget = std::numeric_limits<std::uint64_t>::max();
    auto const d                    = decide(in);
    // 2^31 rows of 24-byte partials is a real, representable requirement; the model must size it
    // rather than saturate.
    CHECK(d.model_evaluated);
    CHECK_FALSE(d.model.overflowed);
    CHECK(d.reason == decision_reason::bypass_selected);
  }

  SECTION("saturating width arithmetic rejects")
  {
    auto in                         = good_candidate();
    in.total_rows                   = CUDF_MAX_ROWS;
    in.agg_width_bytes              = std::numeric_limits<std::uint64_t>::max();
    in.admissible_additional_budget = std::numeric_limits<std::uint64_t>::max();
    auto const d                    = decide(in);
    CHECK(d.reason == decision_reason::size_overflow);
    CHECK(d.num_partitions == in.auto_num_partitions);
  }
}

TEST_CASE("group-by bypass model charges the terms it claims to", "[group_by_bypass][policy]")
{
  auto in              = good_candidate();
  in.total_rows        = 1024;
  in.headroom_fraction = 0.0;
  auto const base      = model_additional_bytes(in);

  // Worst-case cardinality: the dense output is sized for every partial row being its own group,
  // so it matches the concatenated table rather than shrinking to some assumed group count.
  CHECK(base.output_bytes == base.concat_bytes);
  CHECK(base.additional_needed == base.concat_bytes + base.hash_set_bytes + base.gather_map_bytes +
                                    base.sparse_agg_bytes + base.output_bytes +
                                    base.downstream_bytes);
  CHECK(base.headroom_bytes == 0);

  SECTION("nullable columns are charged validity masks, not ignored")
  {
    auto nulls                 = in;
    nulls.nullable_key_columns = 1;
    nulls.nullable_agg_columns = 2;
    auto const with_masks      = model_additional_bytes(nulls);
    CHECK(with_masks.concat_bytes > base.concat_bytes);
    CHECK(with_masks.sparse_agg_bytes > base.sparse_agg_bytes);
    CHECK(with_masks.additional_needed > base.additional_needed);
  }

  SECTION("downstream columns are charged by their own widths, not the merge output's")
  {
    CHECK(base.downstream_bytes == 0);
    // Ten computed INT64 columns: far wider than the 24-byte merge output row.
    auto projected                 = in;
    projected.downstream_row_bytes = 80;
    projected.downstream_columns   = 10;
    auto const wide                = model_additional_bytes(projected);
    CHECK(wide.downstream_bytes >= 80 * *in.total_rows);
    CHECK(wide.downstream_bytes > base.output_bytes);
    CHECK(wide.additional_needed == base.additional_needed + wide.downstream_bytes);
  }

  SECTION("each column is charged its own allocation padding")
  {
    // The same 24 bytes per row split across more columns needs more padded allocations.
    auto split         = in;
    split.total_rows   = 1;
    split.agg_columns  = 16;
    auto packed        = split;
    packed.agg_columns = 1;
    auto const many    = model_additional_bytes(split);
    auto const one     = model_additional_bytes(packed);
    // One row of 16 separately allocated aggregate columns occupies 16 aligned 256-byte blocks.
    CHECK(many.concat_bytes >= 256 + 16 * 256);
    CHECK(many.concat_bytes > one.concat_bytes);
  }

  SECTION("headroom scales the requirement and is reported separately")
  {
    auto margin              = in;
    margin.headroom_fraction = 0.25;
    auto const with_margin   = model_additional_bytes(margin);
    CHECK(with_margin.additional_needed == base.additional_needed);
    CHECK(with_margin.headroom_bytes >= base.additional_needed / 4);
    CHECK(with_margin.required_bytes == with_margin.additional_needed + with_margin.headroom_bytes);
  }
}

TEST_CASE("group-by bypass budget check is inclusive at the boundary", "[group_by_bypass][policy]")
{
  auto in              = good_candidate();
  in.total_rows        = 65'536;
  in.headroom_fraction = 0.0;
  auto const model     = model_additional_bytes(in);
  REQUIRE(model.required_bytes > 1);

  SECTION("a budget exactly equal to the requirement fits")
  {
    in.admissible_additional_budget = model.required_bytes;
    CHECK(decide(in).reason == decision_reason::bypass_selected);
  }
  SECTION("one byte short does not")
  {
    in.admissible_additional_budget = model.required_bytes - 1;
    CHECK(decide(in).reason == decision_reason::insufficient_budget);
  }
}
