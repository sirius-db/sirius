/*
 * Copyright 2026, Sirius Contributors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 */

#include "catch.hpp"
#include "io/s3/duckdb_secret_config.hpp"

#include <duckdb.hpp>
#include <duckdb/common/string_util.hpp>
#include <duckdb/main/secret/secret.hpp>
#include <duckdb/main/secret/secret_manager.hpp>

#include <algorithm>
#include <utility>

namespace {

void require_sql_ok(duckdb::Connection& con, std::string const& sql)
{
  auto result = con.Query(sql);
  REQUIRE(result);
  INFO((result->HasError() ? result->GetError() : ""));
  REQUIRE_FALSE(result->HasError());
}

duckdb::unique_ptr<duckdb::BaseSecret> create_test_httpfs_s3_secret(
  duckdb::ClientContext&, duckdb::CreateSecretInput& input)
{
  auto secret =
    duckdb::make_uniq<duckdb::KeyValueSecret>(input.scope, input.type, input.provider, input.name);
  for (auto const& [key, value] : input.options) {
    secret->secret_map[key] = value;
  }
  secret->redact_keys.insert("secret");
  return duckdb::unique_ptr_cast<duckdb::KeyValueSecret, duckdb::BaseSecret>(std::move(secret));
}

void register_test_httpfs_s3_secret(duckdb::SecretManager& manager)
{
  auto const secret_types = manager.AllSecretTypes();
  if (std::any_of(secret_types.begin(), secret_types.end(), [](duckdb::SecretType const& type) {
        return duckdb::StringUtil::CIEquals(type.name, "s3");
      })) {
    return;
  }
  duckdb::SecretType type;
  type.name             = "s3";
  type.deserializer     = duckdb::KeyValueSecret::Deserialize<duckdb::KeyValueSecret>;
  type.default_provider = "config";
  type.extension        = "test_only";
  manager.RegisterSecretType(type);

  duckdb::CreateSecretFunction function;
  function.secret_type                  = "s3";
  function.provider                     = "config";
  function.function                     = create_test_httpfs_s3_secret;
  function.named_parameters["key_id"]   = duckdb::LogicalType::VARCHAR;
  function.named_parameters["secret"]   = duckdb::LogicalType::VARCHAR;
  function.named_parameters["region"]   = duckdb::LogicalType::VARCHAR;
  function.named_parameters["endpoint"] = duckdb::LogicalType::VARCHAR;
  manager.RegisterSecretFunction(std::move(function), duckdb::OnCreateConflict::ERROR_ON_CONFLICT);
}

}  // namespace

TEST_CASE("Sirius S3 secrets resolve without httpfs and rotate by path", "[s3][secret]")
{
  duckdb::DuckDB db(nullptr);
  duckdb::Connection con(db);
  // Direct resolver calls mirror bind-time catalog lookups, which run in a transaction.
  require_sql_ok(con, "BEGIN TRANSACTION");

  require_sql_ok(con,
                 "CREATE OR REPLACE SECRET unrelated (TYPE SIRIUS_S3, PROVIDER CONFIG, "
                 "SCOPE 's3://bucket/other/', KEY_ID 'wrong-key', SECRET 'wrong-secret', "
                 "REGION 'other-region')");
  require_sql_ok(con,
                 "CREATE OR REPLACE SECRET scoped (TYPE SIRIUS_S3, PROVIDER CONFIG, "
                 "SCOPE 's3://bucket/data/', KEY_ID 'first-key', SECRET 'first-secret', "
                 "REGION 'test-region-1', SESSION_TOKEN 'first-token', "
                 "ENDPOINT 'object-store:9000', USE_SSL true, VERIFY_SSL false, URL_STYLE 'PATH')");

  sirius::io::object_store_config defaults;
  defaults.endpoint      = "https://yaml-endpoint";
  defaults.region        = "yaml-region";
  defaults.access_key    = "yaml-key";
  defaults.secret_key    = "yaml-secret";
  defaults.session_token = "yaml-token";
  auto const matching    = sirius::io::s3::resolve_duckdb_s3_secret(
    *con.context, "s3://bucket/data/table.parquet", defaults);
  CHECK(matching.endpoint == "https://object-store:9000");
  CHECK(matching.region == "test-region-1");
  CHECK(matching.access_key == "first-key");
  CHECK(matching.secret_key == "first-secret");
  CHECK(matching.session_token == "first-token");
  CHECK_FALSE(matching.tls_verify);

  auto const uppercase_scheme = sirius::io::s3::resolve_duckdb_s3_secret(
    *con.context, "S3://bucket/data/CaseSensitive.parquet", defaults);
  CHECK(uppercase_scheme.access_key == "first-key");
  CHECK(uppercase_scheme.secret_key == "first-secret");

  auto const outside_scope = sirius::io::s3::resolve_duckdb_s3_secret(
    *con.context, "s3://bucket/data_elsewhere/table.parquet", defaults);
  CHECK(outside_scope.endpoint == "https://yaml-endpoint");
  CHECK(outside_scope.access_key == "yaml-key");

  require_sql_ok(con,
                 "CREATE OR REPLACE SECRET region_only (TYPE SIRIUS_S3, PROVIDER CONFIG, "
                 "SCOPE 's3://bucket/region_only/', REGION 'test-region-3')");
  CHECK_THROWS(sirius::io::s3::resolve_duckdb_s3_secret(
    *con.context, "s3://bucket/region_only/table.parquet", defaults));

  require_sql_ok(con,
                 "CREATE OR REPLACE SECRET token_only (TYPE SIRIUS_S3, PROVIDER CONFIG, "
                 "SCOPE 's3://bucket/token_only/', SESSION_TOKEN 'token-only-token')");
  CHECK_THROWS(sirius::io::s3::resolve_duckdb_s3_secret(
    *con.context, "s3://bucket/token_only/table.parquet", defaults));

  require_sql_ok(con,
                 "CREATE OR REPLACE SECRET aws_default (TYPE SIRIUS_S3, PROVIDER CONFIG, "
                 "SCOPE 's3://bucket/aws_default/', KEY_ID 'aws-key', SECRET 'aws-secret', "
                 "REGION 'test-region-4')");
  auto const aws_default = sirius::io::s3::resolve_duckdb_s3_secret(
    *con.context, "s3://bucket/aws_default/table.parquet", defaults);
  CHECK(aws_default.endpoint == "https://s3.test-region-4.amazonaws.com");
  CHECK(aws_default.region == "test-region-4");
  CHECK(aws_default.session_token.empty());
  CHECK(aws_default.access_key == "aws-key");

  require_sql_ok(con,
                 "CREATE OR REPLACE SECRET partial (TYPE SIRIUS_S3, PROVIDER CONFIG, "
                 "SCOPE 's3://bucket/partial/', KEY_ID 'partial-key')");
  CHECK_THROWS(sirius::io::s3::resolve_duckdb_s3_secret(
    *con.context, "s3://bucket/partial/table.parquet", defaults));

  require_sql_ok(con,
                 "CREATE OR REPLACE SECRET scoped (TYPE SIRIUS_S3, PROVIDER CONFIG, "
                 "SCOPE 's3://bucket/data/', KEY_ID 'rotated-key', SECRET 'rotated-secret', "
                 "REGION 'test-region-2', SESSION_TOKEN 'rotated-token', "
                 "ENDPOINT 'http://object-store:9000', USE_SSL false, URL_STYLE 'path')");
  auto const rotated = sirius::io::s3::resolve_duckdb_s3_secret(
    *con.context, "s3://bucket/data/table.parquet", defaults);
  CHECK(rotated.endpoint == "http://object-store:9000");
  CHECK(rotated.region == "test-region-2");
  CHECK(rotated.access_key == "rotated-key");
  CHECK(rotated.secret_key == "rotated-secret");
  CHECK(rotated.session_token == "rotated-token");

  require_sql_ok(con, "ROLLBACK");
}

TEST_CASE("Sirius S3 secret lookup prefers Sirius then httpfs then config", "[s3][secret]")
{
  duckdb::DuckDB db(nullptr);
  duckdb::Connection con(db);
  auto& manager = duckdb::SecretManager::Get(*db.instance);
  register_test_httpfs_s3_secret(manager);
  // Direct resolver calls mirror bind-time catalog lookups, which run in a transaction.
  require_sql_ok(con, "BEGIN TRANSACTION");

  require_sql_ok(con,
                 "CREATE SECRET legacy (TYPE S3, SCOPE 's3://bucket/data/', "
                 "KEY_ID 'legacy-key', SECRET 'legacy-secret', REGION 'us-east-1')");
  sirius::io::object_store_config defaults;
  auto const legacy = sirius::io::s3::resolve_duckdb_s3_secret(
    *con.context, "s3://bucket/data/file.parquet", defaults);
  CHECK(legacy.access_key == "legacy-key");

  require_sql_ok(con,
                 "CREATE SECRET sirius (TYPE SIRIUS_S3, SCOPE 's3://bucket/data/', "
                 "KEY_ID 'sirius-key', SECRET 'sirius-secret', REGION 'us-east-1')");
  auto const tied = sirius::io::s3::resolve_duckdb_s3_secret(
    *con.context, "s3://bucket/data/file.parquet", defaults);
  CHECK(tied.access_key == "sirius-key");

  require_sql_ok(con,
                 "CREATE SECRET legacy_nested (TYPE S3, SCOPE 's3://bucket/data/nested/', "
                 "KEY_ID 'nested-key', SECRET 'nested-secret', REGION 'us-east-1')");
  auto const nested = sirius::io::s3::resolve_duckdb_s3_secret(
    *con.context, "s3://bucket/data/nested/file.parquet", defaults);
  CHECK(nested.access_key == "sirius-key");

  require_sql_ok(con,
                 "CREATE SECRET legacy_invalid (TYPE S3, SCOPE 's3://bucket/invalid/', "
                 "KEY_ID 'valid-key', SECRET 'valid-secret', REGION 'us-east-1')");
  require_sql_ok(con,
                 "CREATE SECRET sirius_invalid (TYPE SIRIUS_S3, SCOPE 's3://bucket/invalid/', "
                 "KEY_ID 'incomplete-key')");
  CHECK_THROWS(sirius::io::s3::resolve_duckdb_s3_secret(
    *con.context, "s3://bucket/invalid/file.parquet", defaults));

  require_sql_ok(con, "ROLLBACK");
}
