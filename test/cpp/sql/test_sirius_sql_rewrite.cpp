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

  SECTION("leaves schema-qualified read_parquet calls untouched")
  {
    auto const input = std::string{"SELECT * FROM main.read_parquet('s3://b/k.parquet')"};
    CHECK(sirius::rewrite_sirius_owned_remote_parquet_calls(input) == input);
  }

  SECTION("leaves catalog and schema qualified read_parquet calls untouched")
  {
    auto const schema_qualified =
      std::string{"SELECT * FROM myschema.read_parquet('s3://b/k.parquet')"};
    auto const catalog_schema_qualified =
      std::string{"SELECT * FROM cat.sch.read_parquet('s3://b/k.parquet')"};

    CHECK(sirius::rewrite_sirius_owned_remote_parquet_calls(schema_qualified) == schema_qualified);
    CHECK(sirius::rewrite_sirius_owned_remote_parquet_calls(catalog_schema_qualified) ==
          catalog_schema_qualified);
  }

  SECTION("rewrites only the bare call in mixed qualified and bare queries")
  {
    auto const input = std::string{
      "SELECT * FROM main.read_parquet('s3://a/file.parquet') "
      "UNION ALL SELECT * FROM read_parquet('s3://b/file.parquet')"};
    auto const expected = std::string{
      "SELECT * FROM main.read_parquet('s3://a/file.parquet') "
      "UNION ALL SELECT * FROM sirius_read_parquet('s3://b/file.parquet')"};

    CHECK(sirius::rewrite_sirius_owned_remote_parquet_calls(input) == expected);
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
            "SELECT * FROM xread_parquet('s3://x/y.parquet')") ==
          "SELECT * FROM xread_parquet('s3://x/y.parquet')");
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

  SECTION("rewrites s3 paths that contain dots inside the string literal")
  {
    CHECK(sirius::rewrite_sirius_owned_remote_parquet_calls(
            "SELECT * FROM read_parquet('s3://bucket.with.dots/k.parquet')") ==
          "SELECT * FROM sirius_read_parquet('s3://bucket.with.dots/k.parquet')");
  }

  SECTION("rewrites when a stray dot is separated from the function by whitespace")
  {
    auto const input =
      std::string{"SELECT * FROM t WHERE exists (SELECT 1) . read_parquet('s3://b/k.parquet')"};
    auto const expected = std::string{
      "SELECT * FROM t WHERE exists (SELECT 1) . "
      "sirius_read_parquet('s3://b/k.parquet')"};
    CHECK(sirius::rewrite_sirius_owned_remote_parquet_calls(input) == expected);
  }
}

TEST_CASE("S3 SQL rewrite predicate detects Sirius-owned remote parquet calls",
          "[s3][sql][rewrite]")
{
  SECTION("detects real s3 read_parquet calls")
  {
    CHECK(
      sirius::references_sirius_owned_s3_parquet("SELECT * FROM read_parquet('s3://b/k.parquet')"));
    CHECK(sirius::references_sirius_owned_s3_parquet("SELECT * FROM READ_PARQUET('S3://B/K')"));
    CHECK(sirius::references_sirius_owned_s3_parquet(
      "SELECT count(*) FROM read_parquet('s3://b/k.parquet', union_by_name=true)"));
    CHECK(sirius::references_sirius_owned_s3_parquet(
      "SELECT * FROM read_parquet('/tmp/a.parquet') UNION ALL "
      "SELECT * FROM read_parquet('s3://b/k.parquet')"));
  }

  SECTION("does not treat qualified read_parquet calls as Sirius-owned")
  {
    CHECK_FALSE(sirius::references_sirius_owned_s3_parquet(
      "SELECT * FROM main.read_parquet('s3://b/k.parquet')"));
    CHECK_FALSE(sirius::references_sirius_owned_s3_parquet(
      "SELECT * FROM cat.sch.read_parquet('s3://b/k.parquet')"));
    CHECK(
      sirius::references_sirius_owned_s3_parquet("SELECT * FROM read_parquet('s3://b/k.parquet')"));
  }

  SECTION("ignores local paths and unrelated queries")
  {
    CHECK_FALSE(
      sirius::references_sirius_owned_s3_parquet("SELECT * FROM read_parquet('/local/x.parquet')"));
    CHECK_FALSE(sirius::references_sirius_owned_s3_parquet(
      "SELECT * FROM read_parquet('file:///local/x.parquet')"));
    CHECK_FALSE(sirius::references_sirius_owned_s3_parquet("SELECT 42"));
  }

  SECTION("ignores comments and quoted strings")
  {
    CHECK_FALSE(sirius::references_sirius_owned_s3_parquet(
      "-- read_parquet('s3://comment/file.parquet')\nSELECT 1"));
    CHECK_FALSE(sirius::references_sirius_owned_s3_parquet(
      "/* read_parquet('s3://comment/file.parquet') */ SELECT 1"));
    CHECK_FALSE(sirius::references_sirius_owned_s3_parquet(
      "SELECT 'read_parquet(''s3://literal/file.parquet'')'"));
    CHECK_FALSE(sirius::references_sirius_owned_s3_parquet(
      "SELECT \"read_parquet('s3://identifier/file.parquet')\""));
  }

  SECTION("requires the bare read_parquet function name")
  {
    CHECK_FALSE(sirius::references_sirius_owned_s3_parquet(
      "SELECT * FROM my_read_parquet('s3://b/k.parquet')"));
    CHECK_FALSE(sirius::references_sirius_owned_s3_parquet(
      "SELECT * FROM read_parquet_mr('s3://b/k.parquet')"));
  }
}
