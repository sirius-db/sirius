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

/**
 * @file test_gpu_execution_substring.cpp
 * @brief GPU substring matches DuckDB across offset and length boundaries.
 *
 * Covers the two-argument form, offset 0, negative offsets and lengths, and bounds past the end of
 * every string, on both an ASCII-only table and one with multi-byte characters. Bounds with no
 * single GPU slice that matches DuckDB must fall back during planning.
 */

#include <catch.hpp>
#include <duckdb.hpp>
#include <utils/gpu_execution_fixture.hpp>

#include <string>

using SubstringFixture = sirius::test::GpuExecutionFixture;

namespace {

void create_substring_tables(SubstringFixture& fx)
{
  fx.run_ok("CREATE TABLE ascii_t(id INTEGER, s VARCHAR);");
  fx.run_ok(
    "INSERT INTO ascii_t VALUES"
    " (1, 'hello'), (2, ''), (3, NULL), (4, 'ab'), (5, 'abcdefghijklmnopqrstuvwxyz'), (6, 'x');");
  fx.run_ok("CREATE TABLE unicode_t(id INTEGER, s VARCHAR);");
  fx.run_ok(
    "INSERT INTO unicode_t VALUES"
    " (1, 'héllo wörld'), (2, ''), (3, NULL), (4, '日本'), (5, 'ascii only'), (6, 'ñ');");
  fx.run_ok("CHECKPOINT;");
}

}  // namespace

TEST_CASE_METHOD(SubstringFixture,
                 "gpu_execution two-argument substring matches DuckDB",
                 "[integration][gpu_execution][projection][substring]")
{
  create_substring_tables(*this);
  for (auto const* table : {"ascii_t", "unicode_t"}) {
    CAPTURE(table);
    for (auto const* offset : {"-100", "-5", "-3", "-1", "0", "1", "4", "100", "2147483648"}) {
      CAPTURE(offset);
      compare_gpu_vs_cpu(std::string("SELECT id, substring(s, ") + offset + ") AS r FROM " + table);
    }
    compare_gpu_vs_cpu(std::string("SELECT id, substr(s, 2) AS r FROM ") + table);
    compare_gpu_vs_cpu(std::string("SELECT id FROM ") + table + " WHERE substring(s, 2) <> ''");
  }
}

TEST_CASE_METHOD(SubstringFixture,
                 "gpu_execution three-argument substring matches DuckDB at boundaries",
                 "[integration][gpu_execution][projection][substring]")
{
  create_substring_tables(*this);
  for (auto const* table : {"ascii_t", "unicode_t"}) {
    CAPTURE(table);
    for (auto const* bounds : {"0, 3",
                               "0, 1",
                               "0, -1",
                               "1, 2",
                               "2, 3",
                               "4, 1000000",
                               "5, 0",
                               "30, 2",
                               "3, 2147483648",
                               "-3, 3",
                               "-3, 10",
                               "-1, 1",
                               "-2, -2",
                               "-1, -1",
                               "-10, -2",
                               "-100, -1"}) {
      CAPTURE(bounds);
      compare_gpu_vs_cpu(std::string("SELECT id, substring(s, ") + bounds + ") AS r FROM " + table);
    }
  }
}

TEST_CASE_METHOD(SubstringFixture,
                 "substring bounds without a matching GPU slice fall back at plan time",
                 "[integration][gpu_execution][projection][substring]")
{
  create_substring_tables(*this);
  for (auto const* table : {"ascii_t", "unicode_t"}) {
    CAPTURE(table);
    // DuckDB's ASCII and Unicode kernels disagree on these windows for short strings.
    for (auto const* bounds : {"3, -2", "10, -2", "-5, 3", "-3, 2", "id", "1, id"}) {
      CAPTURE(bounds);
      expect_plan_fallback_matches_cpu(std::string("SELECT id, substring(s, ") + bounds +
                                       ") AS r FROM " + table);
    }
  }
}
