/*
 * Copyright 2026, Sirius Contributors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 */

#include "catch.hpp"
#include "io/scoped_object_store_configs.hpp"

TEST_CASE("S3 config scopes isolate paths and publish immutable replacement snapshots",
          "[s3][routing]")
{
  sirius::io::scoped_object_store_configs scopes;

  cucascade::io::object_store_config alpha;
  alpha.endpoint   = "http://127.0.0.1:19001";
  alpha.region     = "us-east-1";
  alpha.access_key = "alpha-key";
  alpha.secret_key = "alpha-secret";
  alpha.tls_verify = false;
  CHECK_FALSE(scopes.install("s3://alpha-bucket/table/data.parquet", alpha).has_value());
  auto const alpha_first = scopes.resolve("s3://alpha-bucket/table/data.parquet");
  auto const alpha_again = scopes.resolve("s3://alpha-bucket/table/data.parquet");
  auto const neighbor    = scopes.resolve("s3://alpha-bucket/table/neighbor.parquet");
  REQUIRE(alpha_first.has_value());
  REQUIRE(alpha_again.has_value());
  CHECK(alpha_first->id == alpha_again->id);
  CHECK_FALSE(neighbor.has_value());
  CHECK_FALSE(scopes.install("s3://alpha-bucket/table/other.parquet", alpha).has_value());
  auto const alpha_other = scopes.resolve("s3://alpha-bucket/table/other.parquet");
  REQUIRE(alpha_other.has_value());
  CHECK(alpha_other->id == alpha_first->id);

  auto beta       = alpha;
  beta.endpoint   = "http://127.0.0.1:19002";
  beta.access_key = "beta-key";
  beta.secret_key = "beta-secret";
  CHECK_FALSE(scopes.install("s3://beta-bucket/table/*.parquet", beta).has_value());
  auto const beta_file    = scopes.resolve("s3://beta-bucket/table/part-000.parquet");
  auto const beta_sibling = scopes.resolve("s3://beta-bucket/table/part-001.parquet");
  REQUIRE(beta_file.has_value());
  REQUIRE(beta_sibling.has_value());
  CHECK(beta_file->id == beta_sibling->id);
  CHECK(beta_file->config->endpoint == beta.endpoint);
  CHECK_FALSE(scopes.install("s3://beta-bucket/table/part-000.parquet", alpha).has_value());
  auto const beta_exact = scopes.resolve("s3://beta-bucket/table/part-000.parquet");
  REQUIRE(beta_exact.has_value());
  CHECK(beta_exact->id == alpha_first->id);
  CHECK(scopes.resolve("s3://beta-bucket/table/part-001.parquet")->id == beta_file->id);

  auto bucket = alpha;
  CHECK_FALSE(scopes.install("s3://gamma-bucket", bucket).has_value());
  auto const bucket_object   = scopes.resolve("s3://gamma-bucket/table/part.parquet");
  auto const adjacent_bucket = scopes.resolve("s3://gamma-bucket-extra/table/part.parquet");
  REQUIRE(bucket_object.has_value());
  CHECK(bucket_object->config->endpoint == bucket.endpoint);
  CHECK(bucket_object->id == alpha_first->id);
  CHECK_FALSE(adjacent_bucket.has_value());

  // Replacing a scope gives future lookups a fresh immutable snapshot while a
  // previously returned snapshot remains alive for any in-flight datasource.
  alpha.endpoint        = "http://127.0.0.1:19003";
  alpha.access_key      = "rotated-key";
  alpha.secret_key      = "rotated-secret";
  auto const superseded = scopes.install("s3://alpha-bucket/table/data.parquet", alpha);
  // The old snapshot remains referenced by the gamma bucket scope.
  CHECK_FALSE(superseded.has_value());
  auto const rotated = scopes.resolve("s3://alpha-bucket/table/data.parquet");
  REQUIRE(rotated.has_value());
  CHECK(rotated->id != alpha_first->id);
  CHECK(alpha_first->config->access_key == "alpha-key");
  CHECK(rotated->config->access_key == "rotated-key");

  beta.endpoint              = "http://127.0.0.1:19005";
  auto const beta_superseded = scopes.install("s3://beta-bucket/table/*.parquet", beta);
  REQUIRE(beta_superseded.has_value());
  CHECK(*beta_superseded == beta_file->id);
}
