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

// GPU-vs-CPU correctness for cast-shaped DATE predicates — the shapes DuckDB's
// constant folding produces for qgen-style date arithmetic:
//
//   d <= DATE '1998-12-01' - INTERVAL '72' DAY
//     ⇒ table filter  CAST(d AS TIMESTAMP) <= TIMESTAMP '1998-09-20 00:00:00'
//
// These arrive as EXPRESSION_FILTERs, which scan_filter_analysis.cpp lowers to
// stored-day bounds. An off-by-one bound silently changes results, so every
// comparison op runs at midnight and non-midnight constants against rows
// sitting exactly on the cutoffs — on a plain scan and on a GPU-pinned table,
// where the fused decode path actually engages.

#include <catch.hpp>
#include <duckdb.hpp>
#include <utils/gpu_execution_fixture.hpp>

#include <cstdlib>
#include <string>
#include <vector>

namespace {

// Arm the gate before any test latches its function-local static, without
// overriding an explicit setting. The fused path's contract is byte-identical
// output, so this is behavior-neutral for the rest of the binary.
struct fused_gate_armer {
  fused_gate_armer() { setenv("SIRIUS_EXP_FUSED_SCAN_FILTER", "1", /*overwrite=*/0); }
};
[[maybe_unused]] fused_gate_armer const arm_fused_gate{};

// Rows sit exactly on / around every cutoff used below: the qgen q1 cutoff
// 1998-09-20 (= DATE '1998-12-01' - 72 days), the 1994 year-range bounds, the
// epoch boundary (negative stored days), plus NULLs and duplicates.
class CastDatePredicateFixture : public sirius::test::GpuExecutionFixture {
 public:
  CastDatePredicateFixture()
  {
    run_ok("CREATE TABLE t (id INTEGER, d DATE);");
    run_ok(
      "INSERT INTO t VALUES "
      "(1,  DATE '1998-09-19'),"  // cutoff - 1
      "(2,  DATE '1998-09-20'),"  // exactly the q1 cutoff
      "(3,  DATE '1998-09-20'),"  // duplicate on the cutoff
      "(4,  DATE '1998-09-21'),"  // cutoff + 1
      "(5,  DATE '1998-12-01'),"
      "(6,  DATE '1993-12-31'),"   // year-range lower bound - 1
      "(7,  DATE '1994-01-01'),"   // year-range lower bound
      "(8,  DATE '1994-12-31'),"   // last day inside the year range
      "(9,  DATE '1995-01-01'),"   // year-range upper bound (excluded by <)
      "(10, DATE '1969-12-31'),"   // stored day -1: pre-epoch floor/ceil sides
      "(11, DATE '1970-01-01'),"   // stored day 0
      "(12, NULL),"                // must be dropped by every predicate
      "(13, DATE '2262-04-11');")  // far future, still castable to every flavor
      ;
    run_ok("CHECKPOINT;");
  }

  void compare_all(const std::vector<std::string>& predicates)
  {
    for (const auto& pred : predicates) {
      DYNAMIC_SECTION(pred)
      {
        compare_gpu_vs_cpu("SELECT id FROM t WHERE " + pred);
        // Aggregate shape too: a wrong decode-time bound that only miscounts
        // (rather than mis-selects ids) would still show here.
        compare_gpu_vs_cpu("SELECT count(*), sum(id) FROM t WHERE " + pred);
      }
    }
  }
};

// Every comparison op, midnight and non-midnight, both operand orders, plus
// the folded-arithmetic originals. Written as SQL text so DuckDB itself
// performs the constant folding that produces the cast-shaped table filters.
const std::vector<std::string> kFoldedPredicates = {
  // qgen q1 shape (folds to CAST(d AS TIMESTAMP) <= TIMESTAMP '1998-09-20 00:00:00')
  "d <= DATE '1998-12-01' - INTERVAL '72' DAY",
  "d <  DATE '1998-12-01' - INTERVAL '72' DAY",
  "d >= DATE '1998-12-01' - INTERVAL '72' DAY",
  "d >  DATE '1998-12-01' - INTERVAL '72' DAY",
  "d =  DATE '1998-12-01' - INTERVAL '72' DAY",
  // explicit midnight timestamp literal
  "d <= TIMESTAMP '1998-09-20 00:00:00'",
  "d <  TIMESTAMP '1998-09-20 00:00:00'",
  "d >= TIMESTAMP '1998-09-20 00:00:00'",
  "d >  TIMESTAMP '1998-09-20 00:00:00'",
  "d =  TIMESTAMP '1998-09-20 00:00:00'",
  // non-midnight constants: the instant falls strictly between two days
  "d <= TIMESTAMP '1998-09-20 12:00:00'",
  "d <  TIMESTAMP '1998-09-20 12:00:00'",
  "d >= TIMESTAMP '1998-09-20 12:00:00'",
  "d >  TIMESTAMP '1998-09-20 12:00:00'",
  "d =  TIMESTAMP '1998-09-20 12:00:00'",  // can never match a DATE: constant false
  "d <  TIMESTAMP '1998-09-20 00:00:00.000001'",
  "d >= TIMESTAMP '1998-09-20 00:00:00.000001'",
  "d <= TIMESTAMP '1998-09-19 23:59:59.999999'",
  "d >  TIMESTAMP '1998-09-19 23:59:59.999999'",
  // constant on the left
  "TIMESTAMP '1998-09-20 00:00:00' >= d",
  "TIMESTAMP '1998-09-20 00:00:00' <  d",
  "TIMESTAMP '1998-09-20 12:00:00' >  d",
  "TIMESTAMP '1998-09-20 12:00:00' =  d",
  // pre-epoch boundaries (stored day -1 / 0)
  "d <= TIMESTAMP '1969-12-31 00:00:00'",
  "d <  TIMESTAMP '1969-12-31 00:00:00.000001'",
  "d >= TIMESTAMP '1969-12-31 00:00:00.000001'",
  "d >= TIMESTAMP '1969-12-31 12:00:00'",
  // q6-style year range (both conjuncts fold to cast comparisons)
  "d >= DATE '1994-01-01' AND d < DATE '1994-01-01' + INTERVAL '1' YEAR",
  "d BETWEEN DATE '1994-01-01' AND DATE '1998-12-01' - INTERVAL '72' DAY",
  // interval months (q4/q10/q14/q15/q20 shapes)
  "d >= DATE '1994-01-01' AND d < DATE '1994-01-01' + INTERVAL '3' MONTH",
  // plain DATE constants for contrast (ConstantFilter path, already fused)
  "d <= DATE '1998-09-20'",
  "d =  DATE '1998-09-20'",
};

}  // namespace

TEST_CASE_METHOD(CastDatePredicateFixture,
                 "gpu_execution cast-shaped DATE predicates match CPU (plain scan)",
                 "[integration][gpu_execution][filter][fused_scan_filter][cast_date]")
{
  compare_all(kFoldedPredicates);
}

TEST_CASE_METHOD(CastDatePredicateFixture,
                 "gpu_execution cast-shaped DATE predicates match CPU (gpu-pinned table)",
                 "[integration][gpu_execution][filter][fused_scan_filter][cast_date][pin_table]")
{
  // Pinned entries route through sirius_scan_manager's range extraction and,
  // where the pin compressed the column to a bitpack plan, through the fused
  // in-decode masking whose bounds this change produces.
  run_ok("CALL pin_table(format='duckdb', name='t', tier='gpu');");
  compare_all(kFoldedPredicates);
  run_ok("CALL unpin_table('t');");
}
