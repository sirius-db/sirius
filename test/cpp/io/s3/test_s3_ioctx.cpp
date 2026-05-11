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
#include "io/io_errors.hpp"
#include "io/s3/credential_provider.hpp"
#include "io/s3/mock_credential_provider.hpp"
#include "io/s3/s3_io_object.hpp"
#include "io/s3/s3_ioctx.hpp"
#include "io/s3/sirius_sigv4_credential_provider.hpp"

#include <cudf/io/text/byte_range_info.hpp>
#include <cudf/utilities/span.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <future>
#include <iterator>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

using sirius::io::credential_error;
using sirius::io::s3::credential_provider;
using sirius::io::s3::mock_credential_provider;
using sirius::io::s3::presign_method;
using sirius::io::s3::s3_io_object;
using sirius::io::s3::s3_ioctx;
using sirius::io::s3::s3_ioctx_config;
using sirius::io::s3::s3_object_ref;
using sirius::io::s3::sirius_sigv4_credential_provider;
using sirius::io::s3::static_credentials;

namespace {

namespace fs = std::filesystem;
using namespace std::chrono_literals;

std::string env_or(std::string_view name, std::string fallback = {})
{
  auto const* value = std::getenv(std::string{name}.c_str());
  return value ? std::string{value} : std::move(fallback);
}

bool truthy_env(std::string_view name)
{
  auto value = env_or(name);
  return value == "1" || value == "true" || value == "TRUE" || value == "yes" || value == "YES";
}

struct s3_test_env {
  std::string endpoint;
  std::string region;
  std::string access_key;
  std::string secret_key;
  std::string bucket;
  fs::path local_dir;
  bool strict{false};
};

std::optional<s3_test_env> read_s3_test_env()
{
  auto endpoint   = env_or("SIRIUS_TEST_S3_ENDPOINT");
  auto access_key = env_or("SIRIUS_TEST_S3_ACCESS_KEY");
  auto secret_key = env_or("SIRIUS_TEST_S3_SECRET_KEY");
  auto bucket     = env_or("SIRIUS_TEST_S3_BUCKET");
  auto local_dir  = env_or("SIRIUS_TEST_S3_LOCAL_DIR");

  if (endpoint.empty() || access_key.empty() || secret_key.empty() || bucket.empty() ||
      local_dir.empty()) {
    return std::nullopt;
  }

  return s3_test_env{std::move(endpoint),
                     env_or("SIRIUS_TEST_S3_REGION", "us-east-1"),
                     std::move(access_key),
                     std::move(secret_key),
                     std::move(bucket),
                     fs::path{local_dir},
                     truthy_env("SIRIUS_TEST_S3_STRICT")};
}

std::vector<std::uint8_t> read_binary_file(fs::path const& path)
{
  std::ifstream in(path, std::ios::binary);
  REQUIRE(in.good());
  return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}

std::string s3_uri(std::string_view bucket, std::string_view key)
{
  return "s3://" + std::string{bucket} + "/" + std::string{key};
}

std::shared_ptr<s3_io_object> make_s3_object(std::string bucket, std::string key, std::size_t size)
{
  auto path = s3_uri(bucket, key);
  return std::make_shared<s3_io_object>(std::move(bucket), std::move(key), size, std::move(path));
}

std::shared_ptr<s3_ioctx> make_live_ioctx(s3_test_env const& env)
{
  static_credentials creds;
  creds.access_key_id     = env.access_key;
  creds.secret_access_key = env.secret_key;
  auto provider           = std::make_shared<sirius_sigv4_credential_provider>(
    std::move(creds), env.region, env.endpoint, 30min);
  return std::make_shared<s3_ioctx>(s3_ioctx_config{std::move(provider), 4, 20});
}

std::optional<std::size_t> try_head_or_skip(s3_ioctx& ctx,
                                            s3_test_env const& env,
                                            std::string_view key)
{
  try {
    return ctx.head_object_size(env.bucket, key);
  } catch (std::exception const& e) {
    if (env.strict) { FAIL("S3 test environment is configured but HEAD failed: " << e.what()); }
    WARN("Skipping live S3 test because HEAD failed: " << e.what());
    return std::nullopt;
  }
}

void require_bytes_equal(std::vector<std::byte> const& got,
                         std::vector<std::uint8_t> const& expected,
                         std::size_t offset)
{
  REQUIRE(offset + got.size() <= expected.size());
  for (std::size_t i = 0; i < got.size(); ++i) {
    CHECK(static_cast<std::uint8_t>(got[i]) == expected[offset + i]);
  }
}

using async_read_result = std::pair<std::size_t, std::exception_ptr>;

async_read_result read_ranges_async(s3_ioctx& ctx,
                                    s3_io_object& obj,
                                    std::vector<cudf::io::text::byte_range_info> const& ranges,
                                    std::span<cudf::host_span<std::byte>> dst)
{
  std::promise<async_read_result> done;
  auto fut = done.get_future();

  ctx.host_read_ranges_async_io(
    obj, ranges, dst, [&done](auto bytes, auto ep) { done.set_value({bytes, ep}); });

  REQUIRE(fut.wait_for(5s) == std::future_status::ready);
  return fut.get();
}

class blocking_throwing_provider final : public credential_provider {
 public:
  blocking_throwing_provider()
    : _entered(_entered_promise.get_future().share()),
      _release(_release_promise.get_future().share())
  {
  }

  std::string get_presigned_url(s3_object_ref const&, presign_method method) override
  {
    _method.store(method == presign_method::GET ? 1 : 2, std::memory_order_relaxed);
    std::call_once(_entered_once, [&] { _entered_promise.set_value(); });
    _release.wait();
    throw credential_error("blocking_throwing_provider: forced failure");
  }

  bool wait_until_entered(std::chrono::milliseconds timeout) const
  {
    return _entered.wait_for(timeout) == std::future_status::ready;
  }

  void release()
  {
    std::call_once(_release_once, [&] { _release_promise.set_value(); });
  }

  presign_method method() const
  {
    return _method.load(std::memory_order_relaxed) == 1 ? presign_method::GET
                                                        : presign_method::HEAD;
  }

 private:
  std::promise<void> _entered_promise;
  std::shared_future<void> _entered;
  std::promise<void> _release_promise;
  std::shared_future<void> _release;
  mutable std::once_flag _entered_once;
  std::once_flag _release_once;
  std::atomic<int> _method{1};
};

class blocking_first_get_provider final : public credential_provider {
 public:
  explicit blocking_first_get_provider(std::shared_ptr<credential_provider> delegate)
    : _delegate(std::move(delegate)),
      _entered(_entered_promise.get_future().share()),
      _release(_release_promise.get_future().share())
  {
  }

  std::string get_presigned_url(s3_object_ref const& obj, presign_method method) override
  {
    if (method == presign_method::GET && !_blocked_first_get.exchange(true)) {
      std::call_once(_entered_once, [&] { _entered_promise.set_value(); });
      _release.wait();
    }
    return _delegate->get_presigned_url(obj, method);
  }

  bool wait_until_first_get_entered(std::chrono::milliseconds timeout) const
  {
    return _entered.wait_for(timeout) == std::future_status::ready;
  }

  void release()
  {
    std::call_once(_release_once, [&] { _release_promise.set_value(); });
  }

 private:
  std::shared_ptr<credential_provider> _delegate;
  std::promise<void> _entered_promise;
  std::shared_future<void> _entered;
  std::promise<void> _release_promise;
  std::shared_future<void> _release;
  mutable std::once_flag _entered_once;
  std::once_flag _release_once;
  std::atomic<bool> _blocked_first_get{false};
};

}  // namespace

TEST_CASE("s3_io_object preserves S3 identity and cache id", "[s3][ioctx]")
{
  s3_io_object obj("bucket", "/path/to/object.parquet", 123, "s3://bucket//path/to/object.parquet");

  CHECK(obj.bucket() == "bucket");
  CHECK(obj.key() == "/path/to/object.parquet");
  CHECK(obj.size() == 123);
  CHECK(obj.raw_file_cache_id() == "s3://bucket//path/to/object.parquet");
}

TEST_CASE("s3_io_object object_path returns the constructor path", "[s3][io_object]")
{
  s3_io_object obj("bucket", "key", 1024, "s3://bucket/key");

  CHECK(obj.object_path() == "s3://bucket/key");
  CHECK(obj.raw_file_cache_id() == "s3://bucket/key");
}

TEST_CASE("s3_ioctx supports accepts S3 URIs without presigning", "[s3][ioctx]")
{
  auto provider = std::make_shared<mock_credential_provider>("http://127.0.0.1:1/not-used");
  s3_ioctx ctx{s3_ioctx_config{provider, 1, 1}};

  CHECK(ctx.supports("s3://bucket/key"));
  CHECK(provider->call_count() == 0);
}

TEST_CASE("s3_ioctx supports rejects non-S3 paths without throwing", "[s3][ioctx]")
{
  auto provider = std::make_shared<mock_credential_provider>("http://127.0.0.1:1/not-used");
  s3_ioctx ctx{s3_ioctx_config{provider, 1, 1}};

  CHECK_FALSE(ctx.supports("file:///tmp/foo"));
  CHECK_FALSE(ctx.supports("/abs/path"));
  CHECK_FALSE(ctx.supports("https://example.com"));
  CHECK(provider->call_count() == 0);
}

TEST_CASE("s3_ioctx supports treats the S3 scheme case-insensitively", "[s3][ioctx]")
{
  auto provider = std::make_shared<mock_credential_provider>("http://127.0.0.1:1/not-used");
  s3_ioctx ctx{s3_ioctx_config{provider, 1, 1}};

  CHECK(ctx.supports("S3://bucket/key"));
  CHECK(ctx.supports("s3://bucket/key"));
  CHECK(provider->call_count() == 0);
}

TEST_CASE("s3_ioctx create_io_object rejects non-S3 paths before presigning", "[s3][ioctx]")
{
  auto provider = std::make_shared<mock_credential_provider>("http://127.0.0.1:1/not-used");
  s3_ioctx ctx{s3_ioctx_config{provider, 1, 1}};

  try {
    (void)ctx.create_io_object("file:///foo");
    FAIL("create_io_object accepted a non-S3 path");
  } catch (std::invalid_argument const& e) {
    CHECK(std::string_view{e.what()}.find("unsupported") != std::string_view::npos);
  }
  CHECK(provider->call_count() == 0);
}

TEST_CASE("s3_ioctx validates config and clips EOF host reads before presigning", "[s3][ioctx]")
{
  CHECK_THROWS_AS(s3_ioctx(s3_ioctx_config{}), std::invalid_argument);

  auto provider = std::make_shared<mock_credential_provider>("http://127.0.0.1:1/not-used");
  auto ctx      = std::make_shared<s3_ioctx>(s3_ioctx_config{provider, 1, 1});
  auto obj      = make_s3_object("bucket", "key", 8);
  std::vector<std::uint8_t> dst(4, 0xAB);

  CHECK(ctx->host_read(*obj, 8, dst.size(), dst.data()) == 0);
  CHECK(ctx->host_read(*obj, 20, dst.size(), dst.data()) == 0);
  CHECK(provider->call_count() == 0);
  CHECK(std::all_of(dst.begin(), dst.end(), [](auto b) { return b == 0xAB; }));
}

TEST_CASE("s3_ioctx asks credential_provider for method-specific presigned URLs", "[s3][ioctx]")
{
  auto provider = std::make_shared<mock_credential_provider>("http://127.0.0.1:1/not-used");
  provider->set_throw("stop before libcurl performs a request");
  auto ctx = std::make_shared<s3_ioctx>(s3_ioctx_config{provider, 1, 1});
  auto obj = make_s3_object("bucket", "key", 8);
  std::vector<std::uint8_t> dst(4);

  CHECK_THROWS_AS(ctx->head_object_size("bucket", "key"), credential_error);
  CHECK(provider->head_count() == 1);
  CHECK(provider->last_bucket() == "bucket");
  CHECK(provider->last_key() == "key");

  CHECK_THROWS_AS(ctx->host_read(*obj, 0, dst.size(), dst.data()), credential_error);
  CHECK(provider->get_count() == 1);
  CHECK(provider->last_bucket() == "bucket");
  CHECK(provider->last_key() == "key");
}

TEST_CASE("s3_ioctx clips physical byte ranges to file size", "[s3][ioctx]")
{
  auto provider = std::make_shared<mock_credential_provider>("http://127.0.0.1:1/not-used");
  auto ctx      = std::make_shared<s3_ioctx>(s3_ioctx_config{provider, 1, 1});

  auto clipped = ctx->compute_physical_range({4, 16}, 10);
  CHECK(clipped.offset() == 4);
  CHECK(clipped.size() == 6);

  auto eof = ctx->compute_physical_range({10, 16}, 10);
  CHECK(eof.offset() == 10);
  CHECK(eof.size() == 0);
}

TEST_CASE("s3_ioctx async host reads keep context and object alive until completion", "[s3][ioctx]")
{
  auto provider = std::make_shared<blocking_throwing_provider>();
  auto ctx      = std::make_shared<s3_ioctx>(s3_ioctx_config{provider, 1, 1});
  auto obj      = make_s3_object("bucket", "key", 16);
  std::vector<std::uint8_t> dst(4);

  auto done = std::make_shared<std::promise<std::pair<std::size_t, std::exception_ptr>>>();
  auto fut  = done->get_future();
  ctx->host_read_async_io(
    *obj, 0, dst.size(), dst.data(), [done](auto bytes, auto ep) { done->set_value({bytes, ep}); });

  REQUIRE(provider->wait_until_entered(5s));
  ctx.reset();
  obj.reset();
  provider->release();

  REQUIRE(fut.wait_for(5s) == std::future_status::ready);
  auto [bytes, ep] = fut.get();
  CHECK(bytes == 0);
  REQUIRE(ep != nullptr);
  CHECK_THROWS_AS(std::rethrow_exception(ep), credential_error);
  CHECK(provider->method() == presign_method::GET);
}

TEST_CASE("s3_ioctx reads single objects from MinIO with presigned range GETs",
          "[s3][ioctx][integration]")
{
  auto env = read_s3_test_env();
  if (!env) {
    WARN("Skipping live S3 test because SIRIUS_TEST_S3_* environment is not configured");
    return;
  }

  auto key   = env_or("SIRIUS_TEST_S3_KEY", "hello.txt");
  auto local = read_binary_file(env->local_dir / key);
  auto ctx   = make_live_ioctx(*env);
  auto size  = try_head_or_skip(*ctx, *env, key);
  if (!size) return;

  REQUIRE(*size == local.size());
  auto obj = make_s3_object(env->bucket, key, *size);

  std::vector<std::uint8_t> full(local.size());
  CHECK(ctx->host_read(*obj, 0, full.size(), full.data()) == full.size());
  CHECK(full == local);

  std::vector<std::uint8_t> tail(5);
  CHECK(ctx->host_read(*obj, local.size() - tail.size(), tail.size(), tail.data()) == tail.size());
  CHECK(
    std::equal(tail.begin(), tail.end(), local.end() - static_cast<std::ptrdiff_t>(tail.size())));

  std::vector<std::uint8_t> eof(8, 0xCC);
  CHECK(ctx->host_read(*obj, local.size(), eof.size(), eof.data()) == 0);
  CHECK(std::all_of(eof.begin(), eof.end(), [](auto b) { return b == 0xCC; }));
}

TEST_CASE("s3_ioctx create_io_object populates S3 object metadata from MinIO",
          "[s3][ioctx][integration]")
{
  auto env = read_s3_test_env();
  if (!env) {
    WARN("Skipping live S3 test because SIRIUS_TEST_S3_* environment is not configured");
    return;
  }

  auto key   = env_or("SIRIUS_TEST_S3_KEY", "hello.txt");
  auto local = read_binary_file(env->local_dir / key);
  auto ctx   = make_live_ioctx(*env);
  auto path  = s3_uri(env->bucket, key);

  auto object = ctx->create_io_object(path);
  REQUIRE(object != nullptr);
  CHECK(object->size() == local.size());
  CHECK(object->object_path() == path);
  CHECK(object->raw_file_cache_id() == path);

  auto s3_object = std::dynamic_pointer_cast<s3_io_object>(object);
  REQUIRE(s3_object != nullptr);
  CHECK(s3_object->bucket() == env->bucket);
  CHECK(s3_object->key() == key);
}

TEST_CASE("s3_ioctx create_io_object propagates missing S3 key HEAD failures",
          "[s3][ioctx][integration]")
{
  auto env = read_s3_test_env();
  if (!env) {
    WARN("Skipping live S3 test because SIRIUS_TEST_S3_* environment is not configured");
    return;
  }

  auto ctx = make_live_ioctx(*env);
  CHECK_THROWS(ctx->create_io_object(s3_uri(env->bucket, "nonexistent-key-xyz")));
}

TEST_CASE("s3_ioctx reads multiple MinIO byte ranges", "[s3][ioctx][integration]")
{
  auto env = read_s3_test_env();
  if (!env) {
    WARN("Skipping live S3 test because SIRIUS_TEST_S3_* environment is not configured");
    return;
  }

  std::string const key = "medium.bin";
  auto local            = read_binary_file(env->local_dir / key);
  auto ctx              = make_live_ioctx(*env);
  auto size             = try_head_or_skip(*ctx, *env, key);
  if (!size) return;
  REQUIRE(*size == local.size());

  auto obj = make_s3_object(env->bucket, key, *size);
  std::vector<cudf::io::text::byte_range_info> ranges{
    {0, 16},
    {17, 31},
    {4093, 64},
    {static_cast<int64_t>(local.size() - 33), 33},
  };

  std::vector<std::vector<std::byte>> buffers;
  buffers.reserve(ranges.size());
  for (auto const& range : ranges) {
    buffers.emplace_back(static_cast<std::size_t>(range.size()));
  }

  std::vector<cudf::host_span<std::byte>> spans;
  spans.reserve(buffers.size());
  for (auto& buffer : buffers) {
    spans.emplace_back(buffer.data(), buffer.size());
  }

  auto const [total, ep] =
    read_ranges_async(*ctx, *obj, ranges, std::span<cudf::host_span<std::byte>>{spans});
  REQUIRE(ep == nullptr);
  CHECK(total == 16 + 31 + 64 + 33);
  for (std::size_t i = 0; i < ranges.size(); ++i) {
    require_bytes_equal(buffers[i], local, static_cast<std::size_t>(ranges[i].offset()));
  }
}

TEST_CASE("s3_ioctx async range reads copy caller span descriptors before dispatch",
          "[s3][ioctx][integration]")
{
  auto env = read_s3_test_env();
  if (!env) {
    WARN("Skipping live S3 test because SIRIUS_TEST_S3_* environment is not configured");
    return;
  }

  std::string const key = "medium.bin";
  auto local            = read_binary_file(env->local_dir / key);
  REQUIRE(local.size() >= 32);

  static_credentials creds;
  creds.access_key_id     = env->access_key;
  creds.secret_access_key = env->secret_key;
  auto delegate           = std::make_shared<sirius_sigv4_credential_provider>(
    std::move(creds), env->region, env->endpoint, 30min);
  auto provider = std::make_shared<blocking_first_get_provider>(std::move(delegate));
  auto ctx      = std::make_shared<s3_ioctx>(s3_ioctx_config{provider, 2, 20});
  auto obj      = make_s3_object(env->bucket, key, local.size());

  std::vector<cudf::io::text::byte_range_info> ranges{{0, 8}, {8, 8}};
  std::vector<std::byte> first(8);
  std::vector<std::byte> second(8);
  std::vector<std::byte> poison_second(8, std::byte{0xA5});

  auto done = std::make_shared<std::promise<std::pair<std::size_t, std::exception_ptr>>>();
  auto fut  = done->get_future();

  {
    std::vector<cudf::host_span<std::byte>> descriptors;
    descriptors.emplace_back(first.data(), first.size());
    descriptors.emplace_back(second.data(), second.size());

    ctx->host_read_ranges_async_io(*obj,
                                   ranges,
                                   std::span<cudf::host_span<std::byte>>{descriptors},
                                   [done](auto bytes, auto ep) { done->set_value({bytes, ep}); });

    auto const entered = provider->wait_until_first_get_entered(5s);
    if (!entered) { provider->release(); }
    REQUIRE(entered);
    descriptors[1] = cudf::host_span<std::byte>{poison_second.data(), poison_second.size()};
  }

  provider->release();

  REQUIRE(fut.wait_for(5s) == std::future_status::ready);
  auto [bytes, ep] = fut.get();
  REQUIRE(ep == nullptr);
  CHECK(bytes == 16);
  require_bytes_equal(first, local, 0);
  require_bytes_equal(second, local, 8);
  CHECK(std::all_of(
    poison_second.begin(), poison_second.end(), [](auto b) { return b == std::byte{0xA5}; }));
}

TEST_CASE("s3_ioctx host_read_ranges clips EOF-crossing range before dst validation",
          "[s3][ioctx][integration]")
{
  auto env = read_s3_test_env();
  if (!env) {
    WARN("Skipping live S3 test because SIRIUS_TEST_S3_* environment is not configured");
    return;
  }

  std::string const key = "medium.bin";
  auto local            = read_binary_file(env->local_dir / key);
  REQUIRE(local.size() >= 200);
  auto ctx = make_live_ioctx(*env);
  auto obj = make_s3_object(env->bucket, key, local.size());

  auto const offset = local.size() - 100;
  std::vector<cudf::io::text::byte_range_info> ranges{
    {static_cast<int64_t>(offset), 200},
  };
  std::vector<std::byte> buffer(100);
  std::vector<cudf::host_span<std::byte>> spans;
  spans.emplace_back(buffer.data(), buffer.size());

  auto const [total, ep] =
    read_ranges_async(*ctx, *obj, ranges, std::span<cudf::host_span<std::byte>>{spans});

  REQUIRE(ep == nullptr);
  CHECK(total == 100);
  require_bytes_equal(buffer, local, offset);
}

TEST_CASE("s3_ioctx host_read_ranges returns zero for ranges starting at EOF",
          "[s3][ioctx][integration]")
{
  auto env = read_s3_test_env();
  if (!env) {
    WARN("Skipping live S3 test because SIRIUS_TEST_S3_* environment is not configured");
    return;
  }

  std::string const key = "medium.bin";
  auto local            = read_binary_file(env->local_dir / key);
  auto ctx              = make_live_ioctx(*env);
  auto obj              = make_s3_object(env->bucket, key, local.size());

  std::vector<cudf::io::text::byte_range_info> ranges{
    {static_cast<int64_t>(local.size()), 50},
  };
  std::vector<std::byte> empty;
  std::vector<cudf::host_span<std::byte>> spans;
  spans.emplace_back(empty.data(), empty.size());

  auto const [total, ep] =
    read_ranges_async(*ctx, *obj, ranges, std::span<cudf::host_span<std::byte>>{spans});

  REQUIRE(ep == nullptr);
  CHECK(total == 0);
}

TEST_CASE("s3_ioctx host_read_ranges rejects dst smaller than clipped size",
          "[s3][ioctx][integration]")
{
  auto env = read_s3_test_env();
  if (!env) {
    WARN("Skipping live S3 test because SIRIUS_TEST_S3_* environment is not configured");
    return;
  }

  std::string const key = "medium.bin";
  auto local            = read_binary_file(env->local_dir / key);
  REQUIRE(local.size() >= 200);
  auto ctx = make_live_ioctx(*env);
  auto obj = make_s3_object(env->bucket, key, local.size());

  auto const offset = local.size() - 100;
  std::vector<cudf::io::text::byte_range_info> ranges{
    {static_cast<int64_t>(offset), 200},
  };
  std::vector<std::byte> buffer(50);
  std::vector<cudf::host_span<std::byte>> spans;
  spans.emplace_back(buffer.data(), buffer.size());

  auto const [bytes, ep] =
    read_ranges_async(*ctx, *obj, ranges, std::span<cudf::host_span<std::byte>>{spans});

  CHECK(bytes == 0);
  REQUIRE(ep != nullptr);
  CHECK_THROWS_WITH(std::rethrow_exception(ep), "s3_ioctx::host_read_ranges: dst span too small");
}

TEST_CASE("s3_ioctx host_read_ranges clips EOF-crossing ranges independently",
          "[s3][ioctx][integration]")
{
  auto env = read_s3_test_env();
  if (!env) {
    WARN("Skipping live S3 test because SIRIUS_TEST_S3_* environment is not configured");
    return;
  }

  std::string const key = "medium.bin";
  auto local            = read_binary_file(env->local_dir / key);
  REQUIRE(local.size() >= 150);
  auto ctx = make_live_ioctx(*env);
  auto obj = make_s3_object(env->bucket, key, local.size());

  auto const eof_crossing_offset = local.size() - 50;
  std::vector<cudf::io::text::byte_range_info> ranges{
    {0, 100},
    {static_cast<int64_t>(eof_crossing_offset), 100},
  };
  std::vector<std::byte> head(100);
  std::vector<std::byte> tail(50);
  std::vector<cudf::host_span<std::byte>> spans;
  spans.emplace_back(head.data(), head.size());
  spans.emplace_back(tail.data(), tail.size());

  auto const [total, ep] =
    read_ranges_async(*ctx, *obj, ranges, std::span<cudf::host_span<std::byte>>{spans});

  REQUIRE(ep == nullptr);
  CHECK(total == 150);
  require_bytes_equal(head, local, 0);
  require_bytes_equal(tail, local, eof_crossing_offset);
}
