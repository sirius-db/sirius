/*
 * Copyright 2026, Sirius Contributors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * See the LICENSE file at the repo root for the full text.
 */

#include "catch.hpp"
#include "io/s3/sirius_httpfs.hpp"
#include "op/scan/iceberg_metadata_reader.hpp"
#include "op/scan/parquet_delete_reader.hpp"
#include "sirius_context.hpp"
#include "sirius_extension.hpp"
#include "transparent/parquet_footer_statistics.hpp"
#include "util/env_guard.hpp"
#include "utils/local_ioctx.hpp"
#include "utils/s3_backend.hpp"
#include "utils/s3_test_env.hpp"

#include <arpa/inet.h>  // IWYU pragma: keep
#include <core_functions_extension.hpp>
#include <cucascade/cudf/datasource.hpp>
#include <cucascade/io/io_errors.hpp>
#include <duckdb/common/file_system.hpp>
#include <duckdb/common/open_file_info.hpp>
#include <duckdb/common/virtual_file_system.hpp>
#include <duckdb/parser/parser.hpp>
#include <duckdb/planner/logical_operator.hpp>
#include <duckdb/planner/planner.hpp>
#include <duckdb/storage/external_file_cache/caching_file_system.hpp>
#include <netinet/in.h>
#include <parquet_extension.hpp>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

namespace {

using sirius::test::s3::env_or;
using sirius::test::s3::sql_quote;

namespace fs = std::filesystem;

struct s3_test_env {
  std::string endpoint;
  std::string region;
  std::string access_key;
  std::string secret_key;
  std::string bucket;
};

std::optional<s3_test_env> read_s3_test_env()
{
  if (!sirius::test::ensure_s3_test_env()) { return std::nullopt; }

  auto endpoint   = env_or("SIRIUS_TEST_S3_ENDPOINT");
  auto access_key = env_or("SIRIUS_TEST_S3_ACCESS_KEY");
  auto secret_key = env_or("SIRIUS_TEST_S3_SECRET_KEY");
  auto bucket     = env_or("SIRIUS_TEST_S3_BUCKET");

  if (endpoint.empty() || access_key.empty() || secret_key.empty() || bucket.empty()) {
    return std::nullopt;
  }

  return s3_test_env{std::move(endpoint),
                     env_or("SIRIUS_TEST_S3_REGION", "us-east-1"),
                     std::move(access_key),
                     std::move(secret_key),
                     std::move(bucket)};
}

bool skip_if_no_s3_env(std::optional<s3_test_env> const& env)
{
  return sirius::test::s3::skip_or_fail_unless(env.has_value(),
                                               "SIRIUS_TEST_S3_* environment is not configured");
}

std::string s3_uri(std::string_view bucket, std::string_view key)
{
  return "s3://" + std::string{bucket} + "/" + std::string{key};
}

std::string yaml_quote(std::string const& value) { return sql_quote(value); }

template <typename Fn>
std::string thrown_message(Fn&& fn)
{
  try {
    fn();
  } catch (std::exception const& e) {
    return e.what();
  } catch (...) {
    return "<non-std exception>";
  }
  return {};
}

class head_http_server {
 public:
  explicit head_http_server(std::size_t object_size,
                            std::string etag                 = {},
                            std::vector<std::uint8_t> object = {})
    : object_size_(object_size), etag_(std::move(etag)), object_(std::move(object))
  {
    if (!object_.empty() && object_.size() != object_size_) {
      throw std::invalid_argument("object size does not match the supplied bytes");
    }
    listen_fd_ = ::socket(AF_INET, SOCK_STREAM, 0);
    if (listen_fd_ < 0) { throw std::runtime_error("socket failed: " + errno_message()); }
    int one = 1;
    if (::setsockopt(listen_fd_, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one)) != 0) {
      throw std::runtime_error("setsockopt failed: " + errno_message());
    }

    sockaddr_in addr{};
    addr.sin_family      = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    addr.sin_port        = 0;
    if (::bind(listen_fd_, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0) {
      throw std::runtime_error("bind failed: " + errno_message());
    }
    if (::listen(listen_fd_, 4) != 0) {
      throw std::runtime_error("listen failed: " + errno_message());
    }
    socklen_t len = sizeof(addr);
    if (::getsockname(listen_fd_, reinterpret_cast<sockaddr*>(&addr), &len) != 0) {
      throw std::runtime_error("getsockname failed: " + errno_message());
    }
    port_   = ntohs(addr.sin_port);
    thread_ = std::thread([this] { accept_loop(); });
  }

  ~head_http_server()
  {
    stop_.store(true);
    if (listen_fd_ >= 0) { ::shutdown(listen_fd_, SHUT_RDWR); }
    if (thread_.joinable()) { thread_.join(); }
    if (listen_fd_ >= 0) { ::close(listen_fd_); }
  }

  head_http_server(head_http_server const&)            = delete;
  head_http_server& operator=(head_http_server const&) = delete;

  [[nodiscard]] std::string endpoint() const { return "http://127.0.0.1:" + std::to_string(port_); }
  [[nodiscard]] std::size_t head_count() const { return head_count_.load(); }
  [[nodiscard]] std::size_t list_count() const { return list_count_.load(); }
  void reject_suffix()
  {
    std::lock_guard guard(object_mutex_);
    reject_suffix_ = true;
  }
  [[nodiscard]] std::vector<std::string> get_ranges() const
  {
    std::lock_guard guard(object_mutex_);
    return get_ranges_;
  }
  void replace(std::vector<std::uint8_t> object, std::string etag)
  {
    std::lock_guard guard(object_mutex_);
    if (object.size() != object_size_) { throw std::invalid_argument("replacement size differs"); }
    object_ = std::move(object);
    etag_   = std::move(etag);
  }

 private:
  static std::string errno_message() { return std::strerror(errno); }

  static void send_all(int fd, std::string_view response)
  {
    std::size_t sent = 0;
    while (sent < response.size()) {
      auto const n = ::send(fd, response.data() + sent, response.size() - sent, MSG_NOSIGNAL);
      if (n <= 0) { return; }
      sent += static_cast<std::size_t>(n);
    }
  }

  void accept_loop()
  {
    while (!stop_.load()) {
      sockaddr_in client{};
      socklen_t len = sizeof(client);
      auto const fd = ::accept(listen_fd_, reinterpret_cast<sockaddr*>(&client), &len);
      if (fd < 0) {
        if (stop_.load()) { return; }
        continue;
      }
      timeval timeout{};
      timeout.tv_sec = 2;
      (void)::setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
      (void)::setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &timeout, sizeof(timeout));
      handle_client(fd);
      ::close(fd);
    }
  }

  void handle_client(int fd)
  {
    std::string request;
    std::array<char, 4096> buffer{};
    while (!stop_.load() && request.find("\r\n\r\n") == std::string::npos) {
      auto const n = ::recv(fd, buffer.data(), buffer.size(), 0);
      if (n <= 0 || request.size() + static_cast<std::size_t>(n) > 65536) { return; }
      request.append(buffer.data(), static_cast<std::size_t>(n));
    }
    std::lock_guard guard(object_mutex_);
    auto respond = [&](std::string const& status,
                       std::size_t size,
                       std::string const& extra,
                       std::string_view body) {
      auto response =
        "HTTP/1.1 " + status + "\r\nContent-Length: " + std::to_string(size) + "\r\n" + extra;
      if (!etag_.empty()) { response += "ETag: " + etag_ + "\r\n"; }
      response += "Connection: close\r\n\r\n";
      send_all(fd, response);
      send_all(fd, body);
    };
    if (request.starts_with("HEAD ")) {
      ++head_count_;
      respond("200 OK", object_size_, {}, {});
      return;
    }
    if (object_.empty() || !request.starts_with("GET ")) {
      respond("405 Method Not Allowed", 0, {}, {});
      return;
    }
    if (request.find("list-type=2") != std::string::npos) {
      ++list_count_;
      auto const xml =
        "<ListBucketResult><IsTruncated>false</IsTruncated><Contents>"
        "<Key>footer.parquet</Key><Size>" +
        std::to_string(object_size_) + "</Size><ETag>" + etag_ +
        "</ETag></Contents></ListBucketResult>";
      respond("200 OK", xml.size(), "Content-Type: application/xml\r\n", xml);
      return;
    }
    auto lower = request;
    std::transform(lower.begin(), lower.end(), lower.begin(), [](unsigned char c) {
      return static_cast<char>(std::tolower(c));
    });
    auto header = [&](std::string const& name) {
      auto pos = lower.find("\r\n" + name + ":");
      if (pos == std::string::npos) { return std::string{}; }
      pos += name.size() + 3;
      while (pos < request.size() && request[pos] == ' ') {
        ++pos;
      }
      return request.substr(pos, request.find("\r\n", pos) - pos);
    };
    auto const range = header("range");
    get_ranges_.push_back(range);
    if (reject_suffix_ && range.starts_with("bytes=-")) {
      respond("416 Range Not Satisfiable",
              0,
              "Content-Range: bytes */" + std::to_string(object_size_) + "\r\n",
              {});
      return;
    }
    auto const match = header("if-match");
    if (!match.empty() && match != etag_) {
      respond("412 Precondition Failed", 0, {}, {});
      return;
    }
    std::size_t begin = 0, end = object_size_ - 1;
    if (!range.empty()) {
      try {
        if (!range.starts_with("bytes=")) { throw std::invalid_argument("range"); }
        auto const spec = range.substr(6);
        auto const dash = spec.find('-');
        if (dash == std::string::npos) { throw std::invalid_argument("range"); }
        if (dash == 0) {
          auto const suffix = std::stoull(spec.substr(1));
          if (suffix == 0) { throw std::invalid_argument("range"); }
          begin = suffix >= object_size_ ? 0 : object_size_ - suffix;
        } else {
          begin = std::stoull(spec.substr(0, dash));
          if (dash + 1 < spec.size()) {
            end = std::min<std::size_t>(end, std::stoull(spec.substr(dash + 1)));
          }
        }
        if (begin > end) { throw std::invalid_argument("range"); }
      } catch (std::exception const&) {
        respond("416 Range Not Satisfiable",
                0,
                "Content-Range: bytes */" + std::to_string(object_size_) + "\r\n",
                {});
        return;
      }
    }
    auto const extra = range.empty()
                         ? std::string{}
                         : "Content-Range: bytes " + std::to_string(begin) + "-" +
                             std::to_string(end) + "/" + std::to_string(object_size_) + "\r\n";
    respond(
      range.empty() ? "200 OK" : "206 Partial Content",
      end - begin + 1,
      extra,
      std::string_view(reinterpret_cast<char const*>(object_.data() + begin), end - begin + 1));
  }

  std::size_t object_size_;
  std::string etag_;
  std::vector<std::uint8_t> object_;
  std::vector<std::string> get_ranges_;
  bool reject_suffix_{false};
  mutable std::mutex object_mutex_;
  int listen_fd_{-1};
  std::uint16_t port_{0};
  std::atomic<bool> stop_{false};
  std::atomic<std::size_t> head_count_{0};
  std::atomic<std::size_t> list_count_{0};
  std::thread thread_;
};

void load_sirius_extension(duckdb::DuckDB& db)
{
  try {
    db.LoadStaticExtension<duckdb::SiriusExtension>();
  } catch (std::exception const& e) {
    auto const msg = std::string{e.what()};
    if (msg.find("already exists") == std::string::npos &&
        msg.find("already loaded") == std::string::npos) {
      throw;
    }
  }
}

class sirius_httpfs_config_env_guard {
 public:
  explicit sirius_httpfs_config_env_guard(s3_test_env const& /*env*/,
                                          bool perf_instrumentation               = false,
                                          std::optional<bool> external_file_cache = {},
                                          std::size_t probe_bytes                 = 512 * 1024)
  {
    if (auto* current = std::getenv("SIRIUS_CONFIG_FILE"); current != nullptr) {
      had_original_config_env_ = true;
      original_config_env_     = current;
    }
    if (auto* current = std::getenv("SIRIUS_DISABLE"); current != nullptr) {
      had_original_disable_env_ = true;
      original_disable_env_     = current;
    }

    auto const unique = std::to_string(reinterpret_cast<std::uintptr_t>(this));
    dir_              = fs::temp_directory_path() / ("sirius_httpfs_" + unique);
    config_path_      = dir_ / "sirius.yaml";
    fs::create_directories(dir_);

    std::ofstream out(config_path_);
    out << "sirius:\n"
           "  space:\n"
           "    gpu:\n"
           "      - device_id: 0\n"
           "        per_stream_reservation: false\n"
           "        reservation_limit_fraction: 0.4\n"
           "        downgrade_trigger_fraction: 0.8\n"
           "        downgrade_stop_fraction: 0.6\n"
           "        memory_capacity: 256 MiB\n"
           "    host:\n"
           "      - numa_id: -1\n"
           "        reservation_limit_fraction: 0.9\n"
           "        downgrade_trigger_fraction: 0.8\n"
           "        downgrade_stop_fraction: 0.6\n"
           "        memory_capacity: 512 MiB\n"
           "        block_size: 1 MiB\n"
           "  executor:\n"
           "    scan_manager:\n";
    if (external_file_cache) {
      out << "      httpfs_external_file_cache: " << (*external_file_cache ? "true\n" : "false\n");
    }
    out << "      cache:\n"
           "        mode: none\n"
           "      rest:\n"
           "        request_timeout_s: 30\n";
    if (probe_bytes != 512 * 1024) { out << "        footer_probe_bytes: " << probe_bytes << '\n'; }
    out.close();
    REQUIRE(out);

    setenv("SIRIUS_CONFIG_FILE", config_path_.string().c_str(), 1);
    unsetenv("SIRIUS_DISABLE");
  }

  ~sirius_httpfs_config_env_guard()
  {
    if (had_original_config_env_) {
      setenv("SIRIUS_CONFIG_FILE", original_config_env_.c_str(), 1);
    } else {
      unsetenv("SIRIUS_CONFIG_FILE");
    }
    if (had_original_disable_env_) {
      setenv("SIRIUS_DISABLE", original_disable_env_.c_str(), 1);
    } else {
      unsetenv("SIRIUS_DISABLE");
    }
    std::error_code ec;
    fs::remove_all(dir_, ec);
  }

 private:
  fs::path dir_;
  fs::path config_path_;
  std::string original_config_env_;
  std::string original_disable_env_;
  bool had_original_config_env_{false};
  bool had_original_disable_env_{false};
};

class sirius_httpfs_fixture {
 public:
  explicit sirius_httpfs_fixture(s3_test_env const& env,
                                 std::optional<bool> external_file_cache = {},
                                 std::size_t probe_bytes                 = 512 * 1024)
    : config_env(env, false, external_file_cache, probe_bytes), db(nullptr), con(db)
  {
    load_sirius_extension(db);
    auto context = con.context->registered_state->Get<duckdb::SiriusContext>("sirius_state");
    REQUIRE(context);
    cucascade::io::object_store_config object_store;
    object_store.endpoint   = env.endpoint;
    object_store.region     = env.region;
    object_store.access_key = env.access_key;
    object_store.secret_key = env.secret_key;
    object_store.tls_verify = false;
    context->get_config().set_object_store_config(object_store);
    context->get_scan_manager().install_s3_config("s3://" + env.bucket, object_store);
    setenv("SIRIUS_DISABLE", "1", 1);
  }

  sirius_httpfs_config_env_guard config_env;
  duckdb::DuckDB db;
  duckdb::Connection con;
};

std::unique_ptr<duckdb::QueryResult> require_query_ok(duckdb::Connection& con,
                                                      std::string const& sql)
{
  auto result = con.Query(sql);
  REQUIRE(result);
  INFO((result->HasError() ? result->GetError() : ""));
  REQUIRE_FALSE(result->HasError());
  return std::unique_ptr<duckdb::QueryResult>(static_cast<duckdb::QueryResult*>(result.release()));
}

void set_gpu_execution(duckdb::Connection& con, bool enabled)
{
  auto result = con.Query(std::string{"SET gpu_execution = "} + (enabled ? "true" : "false"));
  REQUIRE(result);
  INFO((result->HasError() ? result->GetError() : ""));
  REQUIRE_FALSE(result->HasError());
}

std::shared_ptr<cucascade::io::datasource> require_rest_datasource(sirius_httpfs_fixture& fixture,
                                                                   std::string const& uri)
{
  auto sirius_ctx =
    fixture.con.context->registered_state->Get<duckdb::SiriusContext>("sirius_state");
  REQUIRE(sirius_ctx);
  auto datasource = sirius_ctx->get_scan_manager().create_datasource(uri);
  sirius::test::s3::require_rest_ioctx(datasource);
  return datasource;
}

duckdb::SiriusContext& require_sirius_context(sirius_httpfs_fixture& fixture)
{
  auto sirius_ctx =
    fixture.con.context->registered_state->Get<duckdb::SiriusContext>("sirius_state");
  REQUIRE(sirius_ctx);
  return *sirius_ctx;
}

class exposed_sirius_httpfs final : public sirius::io::s3::sirius_httpfs {
 public:
  using sirius::io::s3::sirius_httpfs::SupportsOpenFileExtended;
};

class deadline_probe_httpfs final : public sirius::io::s3::sirius_httpfs {
 public:
  explicit deadline_probe_httpfs(bool bypass) : bypass_(bypass) {}
  duckdb::optional<duckdb::timestamp_t> GetCacheValidUntil(duckdb::FileHandle&) override
  {
    ++deadline_calls;
    if (bypass_) { return duckdb::timestamp_t::ninfinity(); }
    return {};
  }
  std::atomic<std::size_t> deadline_calls{0};

 private:
  bool bypass_;
};

std::vector<std::uint8_t> footer_probe_parquet_bytes()
{
  sirius::util::env_guard disabled{"SIRIUS_DISABLE", "1"};
  char pattern[] = "/tmp/sirius-footer-cache-XXXXXX";
  auto* path     = ::mkdtemp(pattern);
  REQUIRE(path != nullptr);
  struct cleanup {
    fs::path path;
    ~cleanup()
    {
      std::error_code ec;
      fs::remove_all(path, ec);
    }
  } remove_directory{path};
  auto const file = (remove_directory.path / "footer.parquet").string();
  duckdb::DuckDB db(nullptr);
  db.LoadStaticExtension<duckdb::CoreFunctionsExtension>();
  db.LoadStaticExtension<duckdb::ParquetExtension>();
  duckdb::Connection con(db);
  require_query_ok(con,
                   "COPY (SELECT 'data.parquet' AS file_path, i::BIGINT AS pos, "
                   "repeat(md5(i::VARCHAR), 16) AS padding FROM range(1200) t(i)) TO " +
                     sql_quote(file) +
                     " (FORMAT PARQUET, COMPRESSION UNCOMPRESSED, ROW_GROUP_SIZE 2048)");
  std::ifstream input(file, std::ios::binary);
  REQUIRE(input.good());
  std::vector<std::uint8_t> bytes(std::istreambuf_iterator<char>{input}, {});
  REQUIRE(bytes.size() > 512 * 1024);
  REQUIRE(bytes.size() < 1024 * 1024);
  return bytes;
}

int64_t httpfs_cached_bytes(duckdb::Connection& con, std::string const& uri)
{
  auto rows = require_query_ok(
    con,
    "SELECT coalesce(sum(nr_bytes), 0)::BIGINT FROM duckdb_external_file_cache() WHERE path = " +
      sql_quote(uri));
  REQUIRE(rows->RowCount() == 1);
  return rows->Collection().GetValue(0, 0).GetValue<int64_t>();
}

}  // namespace

namespace {

enum class footer_cache_fixture_mode { probe, engine_default, configured };

void check_footer_cache_consumers(bool bypass,
                                  bool validate,
                                  bool statistics,
                                  footer_cache_fixture_mode mode)
{
  CAPTURE(bypass, validate, statistics);
  static std::atomic<unsigned> sequence{0};
  auto const id   = std::to_string(::getpid()) + "-" + std::to_string(sequence++);
  auto const etag = "\"footer-" + id + "\"";
  auto bytes      = footer_probe_parquet_bytes();
  head_http_server server(bytes.size(), etag, bytes);
  s3_test_env env{server.endpoint(), "us-east-1", "test-key", "test-secret", "footer-" + id};
  auto const policy =
    mode == footer_cache_fixture_mode::configured ? std::optional<bool>{!bypass} : std::nullopt;
  sirius_httpfs_fixture fixture(env, policy);
  set_gpu_execution(fixture.con, true);
  auto cache_enabled =
    require_query_ok(fixture.con, "SELECT current_setting('enable_external_file_cache')");
  REQUIRE(cache_enabled->Collection().GetValue(0, 0).GetValue<bool>());
  require_query_ok(fixture.con,
                   std::string{"SET validate_external_file_cache="} +
                     (validate ? "'VALIDATE_ALL'" : "'NO_VALIDATION'"));
  require_query_ok(fixture.con,
                   std::string{"SET parquet_footer_statistics="} + (statistics ? "true" : "false"));
  deadline_probe_httpfs* observed = nullptr;
  if (mode == footer_cache_fixture_mode::probe) {
    auto probe = duckdb::make_uniq<deadline_probe_httpfs>(bypass);
    observed   = probe.get();
    fixture.db.GetFileSystem().UnregisterSubSystem("SiriusHttpFS");
    fixture.db.GetFileSystem().RegisterSubSystem(std::move(probe));
  }
  auto const uri  = s3_uri(env.bucket, "footer.parquet");
  auto const glob = s3_uri(env.bucket, "*.parquet");
  // Observe bind before the optional pass opens another handle for its row total.
  fixture.con.BeginTransaction();
  duckdb::Parser parser(*fixture.con.context);
  parser.ParseQuery("SELECT pos FROM read_parquet(" + sql_quote(glob) + ")");
  duckdb::Planner planner(*fixture.con.context);
  planner.CreatePlan(std::move(parser.statements[0]));
  auto plan = std::move(planner.plan);
  fixture.con.Commit();
  REQUIRE(plan);
  if (observed) { REQUIRE(observed->deadline_calls.load() > 0); }
  auto const ranges = server.get_ranges();
  CAPTURE(ranges);
  std::vector<std::string> expected_ranges{"bytes=-524288"};
  if (!bypass) { expected_ranges.push_back("bytes=0-" + std::to_string(bytes.size() - 1)); }
  CHECK(ranges == expected_ranges);
  CHECK(httpfs_cached_bytes(fixture.con, uri) == (bypass ? 0 : int64_t(bytes.size())));
  if (statistics) {
    fixture.con.BeginTransaction();
    (void)sirius::transparent::add_parquet_footer_statistics(*fixture.con.context, *plan);
    fixture.con.Commit();
    CHECK(httpfs_cached_bytes(fixture.con, uri) == (bypass ? 0 : int64_t(bytes.size())));
  }

  auto& filesystem = duckdb::FileSystem::GetFileSystem(*fixture.con.context);
  auto const files = filesystem.Glob(glob);
  REQUIRE(files.size() == 1);
  REQUIRE(files[0].extended_info);
  auto const& fields = files[0].extended_info->options;
  REQUIRE(fields.contains("file_size"));
  REQUIRE(fields.contains("etag"));
  CHECK(fields.at("file_size").GetValue<int64_t>() == int64_t(bytes.size()));
  CHECK(fields.at("etag").GetValue<std::string>() == etag);
  for (bool extended : {false, true}) {
    CAPTURE(extended);
    auto handle = filesystem.OpenFile(extended ? files[0] : duckdb::OpenFileInfo(uri),
                                      duckdb::FileFlags::FILE_FLAGS_READ);
    REQUIRE(handle->file_system.GetName() == "SiriusHttpFS");
    auto const metadata = handle->Stats();
    if (bypass) {
      REQUIRE(metadata.cache_valid_until);
      CHECK(*metadata.cache_valid_until == duckdb::timestamp_t::ninfinity());
    } else {
      CHECK_FALSE(metadata.cache_valid_until);
    }
    CHECK(handle->file_system.GetVersionTag(*handle) == etag);
    CHECK(handle->file_system.GetFileSize(*handle) == int64_t(bytes.size()));
    CHECK(handle->file_system.GetLastModifiedTime(*handle) == duckdb::timestamp_t(0));
  }

  fixture.con.BeginTransaction();
  std::vector<int64_t> positions;
  auto read = [&](bool replace) {
    sirius::op::scan::read_parquet_delete_batches(
      *fixture.con.context,
      duckdb::OpenFileInfo{uri},
      {},
      [&](auto const&, uint64_t rows) {
        REQUIRE(rows == 1200);
        if (replace) {
          auto replacement = bytes;
          replacement[4] ^= 1;
          server.replace(std::move(replacement), "\"replacement-" + id + "\"");
        }
        return std::vector<uint64_t>{1};
      },
      [&] {
        return [&](duckdb::DataChunk& chunk) {
          for (duckdb::idx_t row = 0; row < chunk.size(); ++row) {
            positions.push_back(chunk.GetValue(0, row).GetValue<int64_t>());
          }
        };
      });
  };
  read(false);
  REQUIRE(positions.size() == 1200);
  for (int64_t i = 0; i < 1200; ++i) {
    REQUIRE(positions[i] == i);
  }
  CHECK(httpfs_cached_bytes(fixture.con, uri) == (bypass ? 0 : int64_t(bytes.size())));
  positions.clear();
  REQUIRE_THROWS_WITH(read(true), Catch::Matchers::ContainsSubstring("object changed"));
  CHECK(positions.empty());
  fixture.con.Rollback();
}

void check_footer_cache_fallback(std::optional<bool> policy, std::string const& shape)
{
  CAPTURE(shape);
  static std::atomic<unsigned> sequence{0};
  auto const id    = std::to_string(::getpid()) + "-" + std::to_string(sequence++);
  auto const bytes = footer_probe_parquet_bytes();
  head_http_server server(bytes.size(), "\"fallback-" + id + "\"", bytes);
  if (shape == "unusable_suffix") { server.reject_suffix(); }
  s3_test_env env{server.endpoint(), "us-east-1", "test-key", "test-secret", "fallback-" + id};
  sirius_httpfs_fixture fixture(env, policy, shape == "large_footer" ? 8 : 512 * 1024);
  set_gpu_execution(fixture.con, true);
  auto const uri  = s3_uri(env.bucket, "footer.parquet");
  auto const path = shape == "no_list_size" ? uri : s3_uri(env.bucket, "*.parquet");
  if (shape == "no_list_size") {
    // Exercise the no-hint open directly; SQL planning can issue unrelated LISTs later.
    auto caching_fs = duckdb::CachingFileSystem::Get(*fixture.con.context);
    auto handle =
      caching_fs.OpenFile(duckdb::OpenFileInfo(uri), duckdb::FileFlags::FILE_FLAGS_READ);
    REQUIRE(handle);
    auto trailer = handle->Read(8, bytes.size() - 8);
    std::array<std::uint8_t, 8> actual{};
    trailer.CopyTo(actual.data(), actual.size());
    CHECK(std::equal(actual.begin(), actual.end(), bytes.end() - 8));
    CHECK(server.list_count() == 0);
    CHECK(server.head_count() > 0);
  }
  auto result = require_query_ok(
    fixture.con,
    "SELECT count(*)::BIGINT, sum(pos)::BIGINT, min(pos), max(pos) FROM read_parquet(" +
      sql_quote(path) + ")");
  REQUIRE(result->RowCount() == 1);
  auto const expected = std::array<int64_t, 4>{1200, 719400, 0, 1199};
  for (duckdb::idx_t column = 0; column < expected.size(); ++column) {
    CHECK(result->Collection().GetValue(column, 0).GetValue<int64_t>() == expected[column]);
  }
  auto const ranges = server.get_ranges();
  REQUIRE_FALSE(ranges.empty());
  if (shape == "unusable_suffix") {
    CHECK(server.head_count() > 0);
    CHECK(std::find(ranges.begin(), ranges.end(), "bytes=-524288") != ranges.end());
  }
  if (shape == "large_footer") {
    CHECK(std::find(ranges.begin(), ranges.end(), "bytes=-8") != ranges.end());
  }
  if (!policy.value_or(false)) { CHECK(httpfs_cached_bytes(fixture.con, uri) == 0); }
  WARN("footer-cache shape=" << shape
                             << " policy=" << (policy ? (*policy ? "true" : "false") : "default")
                             << " GETs=" << ranges.size() << " HEADs=" << server.head_count()
                             << " LISTs=" << server.list_count());
}

}  // namespace

TEST_CASE("Sirius filesystem deadline probe reaches LIST binds and Parquet deletes",
          "[.][s3][integration][filesystem][httpfs_efc_probe]")
{
  auto const bypass     = GENERATE(false, true);
  auto const validate   = GENERATE(false, true);
  auto const statistics = GENERATE(false, true);
  check_footer_cache_consumers(bypass, validate, statistics, footer_cache_fixture_mode::probe);
}

TEST_CASE("Sirius HTTPFS bypasses cached bytes by default and preserves validators",
          "[.][s3][integration][filesystem][footer_defaults]")
{
  auto const validate   = GENERATE(false, true);
  auto const statistics = GENERATE(false, true);
  check_footer_cache_consumers(
    true, validate, statistics, footer_cache_fixture_mode::engine_default);
}

TEST_CASE("Sirius HTTPFS startup cache policy governs bind and delete reads",
          "[.][s3][integration][filesystem][httpfs_efc_feature][footer_defaults]")
{
  auto const bypass     = GENERATE(false, true);
  auto const validate   = GENERATE(false, true);
  auto const statistics = GENERATE(false, true);
  check_footer_cache_consumers(bypass, validate, statistics, footer_cache_fixture_mode::configured);
}

TEST_CASE("Sirius HTTPFS reads fallback footer shapes without retaining bytes by default",
          "[.][s3][integration][filesystem][footer_defaults]")
{
  auto const shape = std::string(GENERATE("large_footer", "no_list_size", "unusable_suffix"));
  check_footer_cache_fallback(std::nullopt, shape);
}

TEST_CASE("Sirius HTTPFS configured cache policies read fallback footer shapes",
          "[.][s3][integration][filesystem][httpfs_efc_feature]")
{
  auto const cache = GENERATE(false, true);
  auto const shape = std::string(GENERATE("large_footer", "no_list_size", "unusable_suffix"));
  CAPTURE(cache);
  check_footer_cache_fallback(cache, shape);
}

TEST_CASE("Sirius S3 filesystem keeps precedence over later S3 handlers", "[s3][filesystem]")
{
  class later_s3_fs final : public duckdb::FileSystem {
   public:
    bool CanHandleFile(const std::string& path) override { return path.rfind("s3://", 0) == 0; }
    std::string GetName() const override { return "LaterS3FS"; }
    duckdb::unique_ptr<duckdb::FileHandle> OpenFile(
      const std::string&, duckdb::FileOpenFlags, duckdb::optional_ptr<duckdb::FileOpener>) override
    {
      throw duckdb::IOException("later S3 handler was selected");
    }
  };

  duckdb::VirtualFileSystem fs;
  fs.RegisterSubSystem(duckdb::make_uniq<sirius::io::s3::sirius_httpfs>());
  fs.RegisterSubSystem(duckdb::make_uniq<later_s3_fs>());
  auto const message = thrown_message([&] {
    fs.OpenFile("s3://bucket/object.parquet", duckdb::FileFlags::FILE_FLAGS_READ, nullptr);
  });
  CHECK(message.find("[sirius_httpfs]") != std::string::npos);
}

TEST_CASE("sirius_httpfs claims only valid S3 object paths", "[s3][filesystem]")
{
  sirius::io::s3::sirius_httpfs fs;

  CHECK(fs.CanHandleFile("s3://bucket/key.parquet"));
  CHECK(fs.CanHandleFile("S3://bucket/key.parquet"));
  CHECK_FALSE(fs.CanHandleFile("s3://bucket"));
  CHECK_FALSE(fs.CanHandleFile("file:///tmp/key.parquet"));
  CHECK_FALSE(fs.CanHandleFile("/tmp/key.parquet"));
  CHECK(fs.CanSeek());
}

TEST_CASE("sirius_httpfs accepts exact keys and gates wildcard expansion on a Sirius opener",
          "[s3][filesystem]")
{
  exposed_sirius_httpfs fs;
  CHECK(fs.SupportsOpenFileExtended());

  auto wildcard_message = thrown_message([&] { (void)fs.Glob("s3://bucket/prefix/*.parquet"); });
  REQUIRE_FALSE(wildcard_message.empty());
  CHECK(wildcard_message.find("no ClientContext") != std::string::npos);

  auto exact = fs.Glob("s3://bucket/key.parquet");
  REQUIRE(exact.size() == 1);
  CHECK(exact[0].path == "s3://bucket/key.parquet");

  CHECK(fs.Glob("s3://bucket").empty());
}

TEST_CASE("sirius_httpfs glob helper throws instead of silently truncating matched files",
          "[.][s3][integration][filesystem][glob]")
{
  auto env = read_s3_test_env();
  if (skip_if_no_s3_env(env)) { return; }

  sirius_httpfs_fixture fixture(*env);
  auto& manager = require_sirius_context(fixture).get_scan_manager();

  CHECK_THROWS_WITH(sirius::io::s3::expand_glob(s3_uri(env->bucket, "glob/multi/nation_*.parquet"),
                                                manager,
                                                /*max_matches=*/1),
                    Catch::Matchers::ContainsSubstring("narrow the glob prefix"));
}

TEST_CASE("sirius_httpfs rejects write opens before opener resolution", "[s3][filesystem]")
{
  sirius::io::s3::sirius_httpfs fs;

  auto const write_message = thrown_message([&] {
    auto handle =
      fs.OpenFile("s3://bucket/key.parquet", duckdb::FileFlags::FILE_FLAGS_WRITE, nullptr);
    (void)handle;
  });
  REQUIRE_FALSE(write_message.empty());
  CHECK(write_message.find("read-only") != std::string::npos);
  CHECK(write_message.find("no ClientContext") == std::string::npos);

  auto const read_message = thrown_message([&] {
    auto handle =
      fs.OpenFile("s3://bucket/key.parquet", duckdb::FileFlags::FILE_FLAGS_READ, nullptr);
    (void)handle;
  });
  REQUIRE_FALSE(read_message.empty());
  CHECK(read_message.find("no ClientContext") != std::string::npos);
}

TEST_CASE("sirius_httpfs opens through FileOpener and reads positional ranges",
          "[.][s3][integration][filesystem]")
{
  auto env = read_s3_test_env();
  if (skip_if_no_s3_env(env)) { return; }

  sirius_httpfs_fixture fixture(*env);
  auto const uri = s3_uri(env->bucket, "parquet/nation.parquet");
  auto& fs       = duckdb::FileSystem::GetFileSystem(*fixture.con.context);
  auto handle    = fs.OpenFile(uri, duckdb::FileFlags::FILE_FLAGS_READ);
  REQUIRE(handle != nullptr);

  auto datasource = require_rest_datasource(fixture, uri);
  auto const size = handle->GetFileSize();
  REQUIRE(size > 16);
  CHECK(static_cast<std::size_t>(size) == datasource->get_io_object().size());
  CHECK_FALSE(handle->OnDiskFile());

  std::vector<std::uint8_t> fs_bytes(8);
  std::vector<std::uint8_t> direct_bytes(8);
  constexpr std::size_t offset = 4;

  handle->Read(fs_bytes.data(), fs_bytes.size(), offset);
  auto const direct_count = datasource->host_read(offset, direct_bytes.size(), direct_bytes.data());

  CHECK(direct_count == direct_bytes.size());
  CHECK(fs_bytes == direct_bytes);
}

TEST_CASE("sirius_httpfs positional reads fail on short reads and negative sizes",
          "[.][s3][integration][filesystem]")
{
  auto env = read_s3_test_env();
  if (skip_if_no_s3_env(env)) { return; }

  sirius_httpfs_fixture fixture(*env);
  auto const uri = s3_uri(env->bucket, "parquet/nation.parquet");
  auto& fs       = duckdb::FileSystem::GetFileSystem(*fixture.con.context);
  auto handle    = fs.OpenFile(uri, duckdb::FileFlags::FILE_FLAGS_READ);
  REQUIRE(handle != nullptr);

  auto const size = handle->GetFileSize();
  REQUIRE(size > 100);

  std::vector<std::uint8_t> in_range(16);
  REQUIRE_NOTHROW(fs.Read(*handle, in_range.data(), static_cast<int64_t>(in_range.size()), 0));

  std::vector<std::uint8_t> eof_crossing(100);
  CHECK_THROWS_AS(fs.Read(*handle,
                          eof_crossing.data(),
                          static_cast<int64_t>(eof_crossing.size()),
                          static_cast<duckdb::idx_t>(size - 10)),
                  duckdb::IOException);

  std::vector<std::uint8_t> whole_object(static_cast<std::size_t>(size));
  CHECK_THROWS_AS(fs.Read(*handle, whole_object.data(), -1, 0), duckdb::IOException);
}

TEST_CASE("sirius_httpfs exposes S3 ETags as DuckDB version tags",
          "[.][s3][integration][filesystem]")
{
  SECTION("plain and glob opens preserve the quoted SeaweedFS ETag")
  {
    auto env = read_s3_test_env();
    if (skip_if_no_s3_env(env)) { return; }

    sirius_httpfs_fixture fixture(*env);
    set_gpu_execution(fixture.con, true);
    auto& fs       = duckdb::FileSystem::GetFileSystem(*fixture.con.context);
    auto const uri = s3_uri(env->bucket, "parquet/nation.parquet");

    auto plain = fs.OpenFile(uri, duckdb::FileFlags::FILE_FLAGS_READ);
    REQUIRE(plain != nullptr);
    auto const plain_tag = plain->file_system.GetVersionTag(*plain);
    REQUIRE(plain_tag.size() >= 2);
    CHECK(plain_tag.front() == '"');
    CHECK(plain_tag.back() == '"');

    auto matches = fs.Glob(s3_uri(env->bucket, "parquet/nation*.parquet"));
    REQUIRE(matches.size() == 1);
    REQUIRE(matches[0].extended_info != nullptr);
    auto glob = fs.OpenFile(matches[0], duckdb::FileFlags::FILE_FLAGS_READ);
    REQUIRE(glob != nullptr);
    auto const glob_tag = glob->file_system.GetVersionTag(*glob);
    REQUIRE(glob_tag.size() >= 2);
    CHECK(glob_tag.front() == '"');
    CHECK(glob_tag.back() == '"');
    CHECK(glob_tag == plain_tag);
  }

  SECTION("an ETag-free backend keeps the version tag empty")
  {
    head_http_server server(4096);
    s3_test_env env{server.endpoint(), "us-east-1", "access", "secret", "bucket"};
    sirius_httpfs_fixture fixture(env);
    set_gpu_execution(fixture.con, true);
    auto& fs = duckdb::FileSystem::GetFileSystem(*fixture.con.context);

    auto handle = fs.OpenFile("s3://bucket/no-etag.bin", duckdb::FileFlags::FILE_FLAGS_READ);

    REQUIRE(handle != nullptr);
    CHECK(handle->GetFileSize() == 4096);
    CHECK(handle->file_system.GetVersionTag(*handle).empty());
    CHECK(server.head_count() == 1);
  }
}

TEST_CASE("DuckDB external file cache invalidates an overwritten S3 range by ETag",
          "[.][s3][integration][filesystem]")
{
  auto env = read_s3_test_env();
  if (skip_if_no_s3_env(env)) { return; }

  constexpr std::size_t object_size = 4096;
  constexpr std::size_t read_offset = 128;
  constexpr std::size_t read_size   = 512;
  std::vector<std::uint8_t> first(object_size);
  std::vector<std::uint8_t> second(object_size);
  for (std::size_t i = 0; i < object_size; ++i) {
    first[i]  = static_cast<std::uint8_t>((i * 17U + 3U) & 0xffU);
    second[i] = static_cast<std::uint8_t>((i * 29U + 11U) & 0xffU);
  }

  std::string const key = "efc/overwrite-invalidation.bin";
  if (!sirius::test::put_s3_test_object(key, first)) {
    SUCCEED("managed SeaweedFS is required for the overwrite invalidation test");
    return;
  }

  sirius_httpfs_fixture fixture(*env, true);
  set_gpu_execution(fixture.con, true);
  require_query_ok(fixture.con, "SET enable_external_file_cache = true");
  auto const uri = s3_uri(env->bucket, key);

  auto caching_fs        = duckdb::CachingFileSystem::Get(*fixture.con.context);
  auto read_cached_range = [&] {
    auto handle =
      caching_fs.OpenFile(duckdb::OpenFileInfo{uri}, duckdb::FileFlags::FILE_FLAGS_READ);
    auto pin = handle->Read(read_size, read_offset);
    std::vector<std::uint8_t> data(read_size);
    pin.CopyTo(data.data(), data.size());
    return data;
  };

  auto first_read = read_cached_range();
  CHECK(std::equal(first_read.begin(), first_read.end(), first.begin() + read_offset));

  auto cache_rows = require_query_ok(
    fixture.con,
    "SELECT count(*) FROM duckdb_external_file_cache() WHERE path = " + sql_quote(uri) +
      " AND loaded AND location <= " + std::to_string(read_offset) +
      " AND location + nr_bytes >= " + std::to_string(read_offset + read_size));
  REQUIRE(cache_rows->RowCount() == 1);
  CHECK(cache_rows->Collection().GetValue(0, 0).GetValue<std::int64_t>() >= 1);

  REQUIRE(sirius::test::put_s3_test_object(key, second));
  auto second_read = read_cached_range();
  CHECK(std::equal(second_read.begin(), second_read.end(), second.begin() + read_offset));
}

TEST_CASE("Sirius S3 Parquet position delete reader preserves results and admission",
          "[.][s3][integration][parquet_delete_sirius_s3]")
{
  auto env = read_s3_test_env();
  if (skip_if_no_s3_env(env)) return;
  auto directory = fs::temp_directory_path() / ("sirius-s3-delete-" + std::to_string(::getpid()));
  fs::create_directories(directory);
  struct cleanup {
    fs::path directory;
    ~cleanup()
    {
      std::error_code ec;
      fs::remove_all(directory, ec);
    }
  } remove_directory{directory};
  auto local_path = (directory / "deletes.parquet").string();
  {
    duckdb::DuckDB db(nullptr);
    duckdb::Connection con(db);
    auto result = con.Query(
      "COPY (SELECT 'data.parquet' AS file_path, i::BIGINT AS pos FROM range(10000) t(i)) TO " +
      sql_quote(local_path) + " (FORMAT PARQUET, ROW_GROUP_SIZE 2048)");
    REQUIRE_FALSE(result->HasError());
  }
  std::ifstream input(local_path, std::ios::binary);
  std::vector<uint8_t> bytes(std::istreambuf_iterator<char>{input}, {});
  auto const key = "parquet-delete-reader/sirius-positions.parquet";
  if (!sirius::test::put_s3_test_object(key, bytes))
    SKIP("Uploads only to the managed local S3 backend");

  sirius_httpfs_fixture fixture(*env);
  set_gpu_execution(fixture.con, true);
  require_query_ok(fixture.con, "SET enable_external_file_cache=false");
  auto uri = s3_uri(env->bucket, key);
  // Prove the filesystem actually selected is Sirius's production S3 adapter.
  auto& filesystem = duckdb::FileSystem::GetFileSystem(*fixture.con.context);
  auto handle      = filesystem.OpenFile(uri, duckdb::FileFlags::FILE_FLAGS_READ);
  REQUIRE(handle->file_system.GetName() == "SiriusHttpFS");
  handle.reset();

  sirius::op::scan::iceberg_delete_discovery discovery;
  discovery.positional_delete_files.push_back(uri);
  auto unused_equality_io = sirius::test::make_test_local_ioctx();
  auto read               = [&] {
    return sirius::op::scan::load_delete_payload(
      *fixture.con.context, "s3-table", unused_equality_io, discovery);
  };
  fixture.con.BeginTransaction();
  auto result           = read();
  auto const& positions = result->positional_deletes.at("data.parquet");
  REQUIRE(positions.size() == 10000);
  REQUIRE(positions.front() == 0);
  REQUIRE(positions.back() == 9999);
  fixture.con.Commit();

  set_gpu_execution(fixture.con, false);
  fixture.con.BeginTransaction();
  REQUIRE_THROWS_WITH(read(), Catch::Matchers::ContainsSubstring("requires GPU execution"));
  fixture.con.Rollback();
  set_gpu_execution(fixture.con, true);
  fixture.con.BeginTransaction();
  REQUIRE(read()->positional_deletes.at("data.parquet").size() == 10000);
  fixture.con.Commit();

  auto replacement_path = (directory / "replacement.parquet").string();
  {
    duckdb::DuckDB db(nullptr);
    duckdb::Connection con(db);
    require_query_ok(con,
                     "COPY (SELECT 'data.parquet' AS file_path, (i+10000)::BIGINT AS pos "
                     "FROM range(10000) t(i)) TO " +
                       sql_quote(replacement_path) + " (FORMAT PARQUET, ROW_GROUP_SIZE 2048)");
  }
  std::ifstream replacement_input(replacement_path, std::ios::binary);
  std::vector<uint8_t> replacement(std::istreambuf_iterator<char>{replacement_input}, {});
  bool const replace_during_batches = GENERATE(false, true);
  CAPTURE(replace_during_batches);
  bool replaced       = false;
  auto replace_object = [&] {
    if (!sirius::test::put_s3_test_object(key, replacement))
      throw std::runtime_error("S3 replacement requires managed test backend");
    replaced = true;
  };
  auto const threads = GENERATE(1, 2);
  CAPTURE(threads);
  require_query_ok(fixture.con, "SET threads=" + std::to_string(threads));
  // Leave later row groups unread until the first batch replaces the object.
  require_query_ok(fixture.con, "SET read_ahead_depth=1");
  std::mutex consume_mutex;
  uint64_t consumed  = 0;
  auto read_complete = [&] {
    std::vector<int64_t> pending;
    sirius::op::scan::read_parquet_delete_batches(
      *fixture.con.context,
      duckdb::OpenFileInfo(uri),
      {},
      [&](auto const&, uint64_t) {
        if (!replace_during_batches) replace_object();
        return std::vector<uint64_t>{1};
      },
      [&] {
        return [&](duckdb::DataChunk& chunk) {
          std::lock_guard lock(consume_mutex);
          for (duckdb::idx_t i = 0; i < chunk.size(); ++i)
            pending.push_back(chunk.GetValue(0, i).GetValue<int64_t>());
          consumed += chunk.size();
          if (replace_during_batches && !replaced) replace_object();
        };
      });
    return pending;
  };
  fixture.con.BeginTransaction();
  std::vector<int64_t> published;
  if (replace_during_batches) {
    REQUIRE_THROWS_AS(published = read_complete(), cucascade::io::object_changed_error);
    REQUIRE(consumed > 0);
  } else {
    REQUIRE_THROWS_WITH(published = read_complete(),
                        Catch::Matchers::ContainsSubstring("object changed"));
    REQUIRE(consumed == 0);
  }
  REQUIRE(replaced);
  REQUIRE(published.empty());
  fixture.con.Rollback();

  // A rejected read must leave the connection usable for the new object.
  fixture.con.BeginTransaction();
  auto recovered            = read();
  auto const& new_positions = recovered->positional_deletes.at("data.parquet");
  REQUIRE(new_positions.size() == 10000);
  for (int64_t i = 0; i < 10000; ++i)
    REQUIRE(new_positions[i] == i + 10000);
  fixture.con.Commit();
}
