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

// Unit tests for GpuExecutionFixture::cells_equal -- the cell-equality helper
// behind compare_gpu_vs_cpu / compare_gpu_vs_cpu_approx. It's a pure function on
// stringified cells, so it's tested directly here without a GPU or a database.

#include <catch.hpp>
#include <utils/gpu_execution_fixture.hpp>

namespace {
using sirius::test::GpuExecutionFixture;
constexpr double kTol = 1e-6;
}  // namespace

TEST_CASE("cells_equal: identical strings match regardless of tolerance",
          "[gpu_execution][comparator]")
{
  REQUIRE(GpuExecutionFixture::cells_equal("5", "5", 0.0));
  REQUIRE(GpuExecutionFixture::cells_equal("abc", "abc", 0.0));
  REQUIRE(GpuExecutionFixture::cells_equal("NULL", "NULL", kTol));
  REQUIRE_FALSE(GpuExecutionFixture::cells_equal("5", "6", 0.0));
}

TEST_CASE("cells_equal: tol=0 is exact string compare (no numeric coercion)",
          "[gpu_execution][comparator]")
{
  // Numerically equal but different spellings must NOT match when exact -- this is
  // the behaviour the default compare_gpu_vs_cpu relies on for keys/counts.
  REQUIRE_FALSE(GpuExecutionFixture::cells_equal("1.0", "1.00", 0.0));
  REQUIRE_FALSE(GpuExecutionFixture::cells_equal("5", "5.0", 0.0));
}

TEST_CASE("cells_equal: numeric cells match within tolerance, not beyond",
          "[gpu_execution][comparator]")
{
  // Low-bit summation-order noise -> within tolerance.
  REQUIRE(GpuExecutionFixture::cells_equal("36.770280867630696", "36.7702808676307", kTol));
  // A genuine divergence (sign + magnitude) -> beyond any meaningful tolerance.
  REQUIRE_FALSE(GpuExecutionFixture::cells_equal("726.855982214156", "-832.1158130671506", kTol));
  // An off-by-one on a modest count is still caught (relative error ~1e-2).
  REQUIRE_FALSE(GpuExecutionFixture::cells_equal("100", "101", kTol));
}

TEST_CASE("cells_equal: absolute floor handles values near zero", "[gpu_execution][comparator]")
{
  // Relative error is large near zero, but the absolute diff is within rel_tol.
  REQUIRE(GpuExecutionFixture::cells_equal("0.0000001", "0.0", kTol));
  // Beyond the absolute floor -> mismatch.
  REQUIRE_FALSE(GpuExecutionFixture::cells_equal("0.001", "0.0", kTol));
}

TEST_CASE("cells_equal: NULLs and partial-numeric strings are never approximate",
          "[gpu_execution][comparator]")
{
  REQUIRE_FALSE(GpuExecutionFixture::cells_equal("NULL", "5", kTol));
  REQUIRE_FALSE(GpuExecutionFixture::cells_equal("5", "NULL", kTol));
  // A trailing non-numeric tail must not be parsed as a number.
  REQUIRE_FALSE(GpuExecutionFixture::cells_equal("12abc", "12", kTol));
  REQUIRE_FALSE(GpuExecutionFixture::cells_equal("12", "12abc", kTol));
  // ...but identical non-numeric strings still match exactly.
  REQUIRE(GpuExecutionFixture::cells_equal("12abc", "12abc", kTol));
}

TEST_CASE("cells_equal: negatives compare approximately", "[gpu_execution][comparator]")
{
  REQUIRE(GpuExecutionFixture::cells_equal("-5.0000001", "-5.0", kTol));
  REQUIRE_FALSE(GpuExecutionFixture::cells_equal("-5.0", "5.0", kTol));
}

TEST_CASE("cells_equal: non-finite values only match exactly", "[gpu_execution][comparator]")
{
  // Without the isfinite guard, the tolerance test gives inf <= rel_tol*inf == true,
  // so inf would wrongly equal a large finite value or an opposite-sign inf.
  REQUIRE_FALSE(GpuExecutionFixture::cells_equal("inf", "1e300", kTol));
  REQUIRE_FALSE(GpuExecutionFixture::cells_equal("inf", "-inf", kTol));
  REQUIRE_FALSE(GpuExecutionFixture::cells_equal("nan", "0", kTol));
  // Identical spellings still match via the exact-string path.
  REQUIRE(GpuExecutionFixture::cells_equal("inf", "inf", kTol));
}
