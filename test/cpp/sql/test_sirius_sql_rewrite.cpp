/*
 * Copyright 2026, Sirius Contributors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * See the LICENSE file at the repo root for the full text.
 */

#include "catch.hpp"
#include "sirius_sql_rewrite.hpp"

#include <string>

TEST_CASE("S3 SQL rewrite targets only Sirius-owned remote parquet calls", "[sql][s3][rewrite]")
{
  SECTION("R1 rewrites a single s3 read_parquet call")
  {
    auto const input  = std::string{"SELECT * FROM read_parquet('s3://x/y.parquet')"};
    auto const output = sirius::rewrite_sirius_owned_remote_parquet_calls(input);
    CHECK(output == "SELECT * FROM sirius_read_parquet('s3://x/y.parquet')");
  }

  SECTION("R1b matches the function name case-insensitively")
  {
    CHECK(sirius::rewrite_sirius_owned_remote_parquet_calls(
            "SELECT * FROM READ_PARQUET('s3://x/y.parquet')") ==
          "SELECT * FROM sirius_read_parquet('s3://x/y.parquet')");
    CHECK(sirius::rewrite_sirius_owned_remote_parquet_calls(
            "SELECT * FROM Read_Parquet('s3://x/y.parquet')") ==
          "SELECT * FROM sirius_read_parquet('s3://x/y.parquet')");
  }

  SECTION("R1c leaves string literals and comments alone")
  {
    auto const input =
      "SELECT '/* read_parquet(''s3://literal/file.parquet'') */' AS txt "
      "FROM read_parquet('s3://x/y.parquet') "
      "-- read_parquet('s3://comment/file.parquet')\n"
      "WHERE txt <> 'read_parquet(''s3://another-literal/file.parquet'')'";
    auto const expected =
      "SELECT '/* read_parquet(''s3://literal/file.parquet'') */' AS txt "
      "FROM sirius_read_parquet('s3://x/y.parquet') "
      "-- read_parquet('s3://comment/file.parquet')\n"
      "WHERE txt <> 'read_parquet(''s3://another-literal/file.parquet'')'";
    CHECK(sirius::rewrite_sirius_owned_remote_parquet_calls(input) == expected);
  }

  SECTION("R1c leaves double-quoted identifiers alone")
  {
    auto const input =
      "SELECT \"read_parquet('s3://identifier/file.parquet')\" "
      "FROM read_parquet('s3://x/y.parquet')";
    auto const expected =
      "SELECT \"read_parquet('s3://identifier/file.parquet')\" "
      "FROM sirius_read_parquet('s3://x/y.parquet')";
    CHECK(sirius::rewrite_sirius_owned_remote_parquet_calls(input) == expected);
  }

  SECTION("R1d does not rewrite adjacent parquet functions")
  {
    CHECK(sirius::rewrite_sirius_owned_remote_parquet_calls(
            "SELECT * FROM parquet_scan('s3://x/y.parquet')") ==
          "SELECT * FROM parquet_scan('s3://x/y.parquet')");
    CHECK(sirius::rewrite_sirius_owned_remote_parquet_calls(
            "SELECT * FROM read_parquet_mr('s3://x/y.parquet')") ==
          "SELECT * FROM read_parquet_mr('s3://x/y.parquet')");
    CHECK(sirius::rewrite_sirius_owned_remote_parquet_calls(
            "SELECT * FROM parquet_metadata('s3://x/y.parquet')") ==
          "SELECT * FROM parquet_metadata('s3://x/y.parquet')");
  }

  SECTION("R2 leaves local paths untouched")
  {
    CHECK(sirius::rewrite_sirius_owned_remote_parquet_calls(
            "SELECT * FROM read_parquet('/tmp/a.parquet')") ==
          "SELECT * FROM read_parquet('/tmp/a.parquet')");
    CHECK(sirius::rewrite_sirius_owned_remote_parquet_calls(
            "SELECT * FROM read_parquet('file:///tmp/a.parquet')") ==
          "SELECT * FROM read_parquet('file:///tmp/a.parquet')");
  }
}
