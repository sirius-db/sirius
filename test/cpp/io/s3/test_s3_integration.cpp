/*
 * Copyright 2025, Sirius Contributors.
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

// PR15 integration scaffolding - exercises the S3 backend against a live
// MinIO started by test/cpp/integration/s3/docker-compose.yml. Tests are skipped
// (SUCCEED) when the SIRIUS_TEST_S3_* env vars are absent so the default
// `sirius_unittest` run on a dockerless CI runner stays green.
//
// These tests complement test_s3_ioctx.cpp by focusing on:
//   - bit-equality between bytes served by S3 and the local fixture copy,
//   - multi-range reads across a large object,
//   - error surfaces (404 on missing key, 403 on bad credentials),
// rather than re-covering HEAD / small range GETs already in test_s3_ioctx.
//
// The `small.bin` / `medium.bin` fixtures are opaque deterministic blobs (not
// real parquet) - see test/cpp/integration/s3/generate_fixtures.py. The bit-equal
// checks below do not care about file format, only that local and remote
// bytes match.

#include "catch.hpp"
#include "io/datasource_factory.hpp"
#include "io/s3/s3_io_object.hpp"
#include "io/s3/s3_ioctx.hpp"
#include "io/sirius_datasource.hpp"
#include "sirius_config.hpp"
#include "utils/s3_live_test.hpp"

#include <array>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

using sirius::io::datasource_factory;
using sirius::io::datasource_registry;
using sirius::io::io_datasource;
using sirius::io::s3::s3_io_object;
using sirius::io::s3::s3_ioctx;
using sirius::io::s3::s3_ioctx_config;

namespace {

struct env_cfg {
  std::string endpoint;
  std::string region;
  std::string access_key;
  std::string secret_key;
  std::string bucket;
  std::filesystem::path local_dir;

  bool present() const
  {
    return !endpoint.empty() && !access_key.empty() && !secret_key.empty() && !bucket.empty() &&
           !local_dir.empty();
  }
};

env_cfg read_env()
{
  env_cfg c;
  c.endpoint   = sirius::test::s3::getenv_or("SIRIUS_TEST_S3_ENDPOINT");
  c.region     = sirius::test::s3::getenv_or("SIRIUS_TEST_S3_REGION", "us-east-1");
  c.access_key = sirius::test::s3::getenv_or("SIRIUS_TEST_S3_ACCESS_KEY");
  c.secret_key = sirius::test::s3::getenv_or("SIRIUS_TEST_S3_SECRET_KEY");
  c.bucket     = sirius::test::s3::getenv_or("SIRIUS_TEST_S3_BUCKET");
  c.local_dir  = sirius::test::s3::getenv_or("SIRIUS_TEST_S3_LOCAL_DIR");
  return c;
}

std::shared_ptr<s3_ioctx> make_ctx(env_cfg const& e,
                                   std::string access = {},
                                   std::string secret = {})
{
  s3_ioctx_config cfg;
  cfg.endpoint   = e.endpoint;
  cfg.region     = e.region;
  cfg.access_key = access.empty() ? e.access_key : std::move(access);
  cfg.secret_key = secret.empty() ? e.secret_key : std::move(secret);
  return std::make_shared<s3_ioctx>(std::move(cfg));
}

std::vector<std::uint8_t> read_file_bytes(std::filesystem::path const& p)
{
  std::ifstream f(p, std::ios::binary);
  REQUIRE(f.good());
  f.seekg(0, std::ios::end);
  auto n = static_cast<std::size_t>(f.tellg());
  f.seekg(0);
  std::vector<std::uint8_t> buf(n);
  f.read(reinterpret_cast<char*>(buf.data()), static_cast<std::streamsize>(n));
  REQUIRE(f.gcount() == static_cast<std::streamsize>(n));
  return buf;
}

// All integration tests share a skip-guard pattern: if env vars or fixtures
// are missing, report SUCCEED with a reason and return.
bool skip_if_env_missing(env_cfg const& e)
{
  if (!e.present()) {
    SUCCEED("Skipping: SIRIUS_TEST_S3_* not set (see test/cpp/integration/s3/README.md)");
    return true;
  }
  if (!std::filesystem::is_directory(e.local_dir)) {
    SUCCEED("Skipping: SIRIUS_TEST_S3_LOCAL_DIR not present - run `make s3-up` first");
    return true;
  }
  return false;
}

}  // namespace

TEST_CASE("s3_integration: hello.txt bytes match local fixture exactly", "[s3][integration]")
{
  auto e = read_env();
  if (skip_if_env_missing(e)) return;

  auto const local_path = e.local_dir / "hello.txt";
  auto const local      = read_file_bytes(local_path);
  REQUIRE(local.size() == 16);

  auto ctx                = make_ctx(e);
  std::size_t object_size = 0;
  try {
    object_size = ctx->head_object_size(e.bucket, "hello.txt");
  } catch (std::exception const& ex) {
    sirius::test::s3::handle_live_runtime_failure(
      "HEAD hello.txt failed", ex, "Skipping: MinIO unreachable or fixture missing");
    return;
  }
  REQUIRE(object_size == local.size());

  auto obj = std::make_unique<s3_io_object>(e.bucket, "hello.txt", object_size);
  std::vector<std::uint8_t> remote(object_size);
  auto got = ctx->host_read(*obj, 0, object_size, remote.data());
  REQUIRE(got == object_size);
  CHECK(std::memcmp(remote.data(), local.data(), object_size) == 0);
}

TEST_CASE("s3_integration: small.bin bit-equal via factory", "[s3][integration]")
{
  auto e = read_env();
  if (skip_if_env_missing(e)) return;

  auto const local_path = e.local_dir / "small.bin";
  auto const local      = read_file_bytes(local_path);
  REQUIRE(local.size() > 0);

  datasource_registry reg;
  reg.register_ioctx("s3", make_ctx(e));
  sirius::sirius_config cfg;

  std::unique_ptr<io_datasource> ds;
  try {
    ds = datasource_factory::create("s3://" + e.bucket + "/small.bin", reg, cfg);
  } catch (std::exception const& ex) {
    sirius::test::s3::handle_live_runtime_failure(
      "factory::create failed", ex, "Skipping: MinIO unreachable or small.bin missing");
    return;
  }
  REQUIRE(ds != nullptr);
  REQUIRE(ds->size() == local.size());

  // Full-object read - this is the byte sequence a downstream reader would
  // see. If it diverges from the local file the scan would yield different
  // bytes than the local fixture.
  auto buf = ds->host_read(0, local.size());
  REQUIRE(buf != nullptr);
  REQUIRE(buf->size() == local.size());
  CHECK(std::memcmp(buf->data(), local.data(), local.size()) == 0);

  // Small tail fetch: mirrors the access pattern a parquet reader would use
  // when probing the footer at the end of an object.
  constexpr std::size_t tail = 8;
  auto tail_buf              = ds->host_read(local.size() - tail, tail);
  REQUIRE(tail_buf->size() == tail);
  CHECK(std::memcmp(tail_buf->data(), local.data() + local.size() - tail, tail) == 0);
}

TEST_CASE("s3_integration: multi-range reads on medium.bin match local bytes", "[s3][integration]")
{
  auto e = read_env();
  if (skip_if_env_missing(e)) return;

  auto const local_path = e.local_dir / "medium.bin";
  auto const local      = read_file_bytes(local_path);
  REQUIRE(local.size() > 4 * 1024 * 1024);  // expect at least 4 MiB

  auto ctx                = make_ctx(e);
  std::size_t object_size = 0;
  try {
    object_size = ctx->head_object_size(e.bucket, "medium.bin");
  } catch (std::exception const& ex) {
    sirius::test::s3::handle_live_runtime_failure(
      "HEAD medium.bin failed", ex, "Skipping: MinIO unreachable or medium.bin missing");
    return;
  }
  REQUIRE(object_size == local.size());

  auto obj = std::make_unique<s3_io_object>(e.bucket, "medium.bin", object_size);

  // Four disjoint 512 KB windows spread across the file. Using odd offsets
  // ensures we are not accidentally aligned to MinIO's internal chunk size.
  struct range {
    std::size_t off, len;
  };
  std::array<range, 4> const windows{{
    {1, 512 * 1024},
    {object_size / 4 + 123, 512 * 1024},
    {object_size / 2 + 777, 512 * 1024},
    {object_size - 512 * 1024 - 17, 512 * 1024},
  }};

  for (auto const& w : windows) {
    REQUIRE(w.off + w.len <= object_size);
    std::vector<std::uint8_t> remote(w.len);
    auto got = ctx->host_read(*obj, w.off, w.len, remote.data());
    REQUIRE(got == w.len);
    CHECK(std::memcmp(remote.data(), local.data() + w.off, w.len) == 0);
  }
}

TEST_CASE("s3_integration: HEAD on missing key reports an error", "[s3][integration]")
{
  auto e = read_env();
  if (skip_if_env_missing(e)) return;

  auto ctx = make_ctx(e);
  CHECK_THROWS_AS(
    ctx->head_object_size(e.bucket, "definitely-does-not-exist-" + std::to_string(std::rand())),
    std::runtime_error);
}

TEST_CASE("s3_integration: bad credentials rejected at HEAD", "[s3][integration]")
{
  auto e = read_env();
  if (skip_if_env_missing(e)) return;

  // Construct a ctx with deliberately-wrong secret key. Server should return
  // 403 / SignatureDoesNotMatch; our ioctx surfaces that as std::runtime_error.
  auto ctx = make_ctx(e, e.access_key, "not-the-right-secret-key");
  CHECK_THROWS_AS(ctx->head_object_size(e.bucket, "hello.txt"), std::runtime_error);
}
