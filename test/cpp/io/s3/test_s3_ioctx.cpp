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

#include "catch.hpp"
#include "io/datasource_factory.hpp"
#include "io/s3/s3_io_object.hpp"
#include "io/s3/s3_ioctx.hpp"
#include "io/sirius_datasource.hpp"
#include "sirius_config.hpp"
#include "utils/s3_live_test.hpp"

#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

using sirius::io::datasource_factory;
using sirius::io::datasource_registry;
using sirius::io::s3::s3_ioctx;
using sirius::io::s3::s3_ioctx_config;

namespace {

// All integration tests here require a live S3-compatible server. We source
// the endpoint/creds from env vars so the CI runner can skip them. Format:
//   SIRIUS_TEST_S3_ENDPOINT=http://127.0.0.1:9000
//   SIRIUS_TEST_S3_REGION=us-east-1          (optional)
//   SIRIUS_TEST_S3_ACCESS_KEY=minioadmin
//   SIRIUS_TEST_S3_SECRET_KEY=minioadmin
//   SIRIUS_TEST_S3_BUCKET=sirius-test
//   SIRIUS_TEST_S3_KEY=hello.txt             (existing small object)
struct env_cfg {
  std::string endpoint;
  std::string region;
  std::string access_key;
  std::string secret_key;
  std::string bucket;
  std::string key;

  bool present() const
  {
    return !endpoint.empty() && !access_key.empty() && !secret_key.empty() && !bucket.empty() &&
           !key.empty();
  }
};

env_cfg read_env()
{
  env_cfg c;
  c.endpoint   = sirius::test::s3::getenv_or("SIRIUS_TEST_S3_ENDPOINT");
  c.region     = sirius::test::s3::getenv_or("SIRIUS_TEST_S3_REGION");
  c.access_key = sirius::test::s3::getenv_or("SIRIUS_TEST_S3_ACCESS_KEY");
  c.secret_key = sirius::test::s3::getenv_or("SIRIUS_TEST_S3_SECRET_KEY");
  c.bucket     = sirius::test::s3::getenv_or("SIRIUS_TEST_S3_BUCKET");
  c.key        = sirius::test::s3::getenv_or("SIRIUS_TEST_S3_KEY");
  return c;
}

std::shared_ptr<s3_ioctx> make_ctx(env_cfg const& e)
{
  s3_ioctx_config cfg;
  cfg.endpoint   = e.endpoint;
  cfg.region     = e.region.empty() ? "us-east-1" : e.region;
  cfg.access_key = e.access_key;
  cfg.secret_key = e.secret_key;
  return std::make_shared<s3_ioctx>(std::move(cfg));
}

}  // namespace

TEST_CASE("s3_ioctx: ctor rejects empty endpoint", "[s3][ioctx]")
{
  s3_ioctx_config cfg;
  cfg.access_key = "a";
  cfg.secret_key = "b";
  CHECK_THROWS_AS(s3_ioctx{cfg}, std::invalid_argument);
}

TEST_CASE("s3_ioctx: ctor rejects empty credentials", "[s3][ioctx]")
{
  s3_ioctx_config cfg;
  cfg.endpoint = "http://127.0.0.1:9000";
  CHECK_THROWS_AS(s3_ioctx{cfg}, std::invalid_argument);
}

TEST_CASE("s3_ioctx: ctor rejects malformed endpoint (no scheme)", "[s3][ioctx]")
{
  s3_ioctx_config cfg;
  cfg.endpoint   = "127.0.0.1:9000";
  cfg.access_key = "a";
  cfg.secret_key = "b";
  CHECK_THROWS_AS(s3_ioctx{cfg}, std::invalid_argument);
}

TEST_CASE("s3_ioctx: HEAD + range GET against live endpoint", "[s3][ioctx][integration]")
{
  auto e = read_env();
  if (!e.present()) {
    SUCCEED("Skipping: SIRIUS_TEST_S3_* not set");
    return;
  }

  auto ctx = make_ctx(e);

  std::size_t obj_size = 0;
  try {
    obj_size = ctx->head_object_size(e.bucket, e.key);
  } catch (std::exception const& ex) {
    sirius::test::s3::handle_live_runtime_failure(
      "HEAD failed", ex, "Skipping: endpoint unreachable or object missing");
    return;
  }
  REQUIRE(obj_size > 0);

  // Full-object range GET.
  std::vector<std::uint8_t> buf(obj_size);
  auto obj = std::make_unique<sirius::io::s3::s3_io_object>(e.bucket, e.key, obj_size);
  auto got = ctx->host_read(*obj, 0, obj_size, buf.data());
  CHECK(got == obj_size);

  // Partial range GET (first N bytes).
  auto partial = std::min<std::size_t>(16, obj_size);
  std::vector<std::uint8_t> head(partial);
  CHECK(ctx->host_read(*obj, 0, partial, head.data()) == partial);
  for (std::size_t i = 0; i < partial; ++i)
    CHECK(head[i] == buf[i]);
}

TEST_CASE("datasource_factory: end-to-end s3:// via live endpoint", "[s3][ioctx][integration]")
{
  auto e = read_env();
  if (!e.present()) {
    SUCCEED("Skipping: SIRIUS_TEST_S3_* not set");
    return;
  }

  datasource_registry reg;
  reg.register_ioctx("s3", make_ctx(e));
  sirius::sirius_config cfg;

  std::unique_ptr<cudf::io::datasource> ds;
  try {
    ds = datasource_factory::create("s3://" + e.bucket + "/" + e.key, reg, cfg);
  } catch (std::exception const& ex) {
    sirius::test::s3::handle_live_runtime_failure(
      "factory::create failed", ex, "Skipping: endpoint unreachable or auth misconfigured");
    return;
  }
  REQUIRE(ds != nullptr);
  CHECK(ds->size() > 0);
}
