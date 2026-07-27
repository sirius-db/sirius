/*
 * Copyright 2026, Sirius Contributors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * See the LICENSE file at the repo root for the full text.
 */

#include "catch.hpp"
#include "io/rest/rest_ioctx.hpp"
#include "io/s3/sirius_httpfs.hpp"
#include "io/sirius_datasource.hpp"
#include "io/uri_parser.hpp"
#include "memory/topology_index.hpp"
#include "scan/test_utils.hpp"
#include "scan_manager/sirius_scan_manager.hpp"
#include "sirius_context.hpp"
#include "sirius_extension.hpp"
#include "utils/s3_container.hpp"

#include <arpa/inet.h>
#include <duckdb.hpp>
#include <duckdb/common/file_system.hpp>
#include <duckdb/common/open_file_info.hpp>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

namespace {

namespace fs = std::filesystem;

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
};

std::optional<s3_test_env> read_s3_test_env()
{
  if (!sirius::test::ensure_s3_container_env()) { return std::nullopt; }

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
  if (env) { return false; }
  if (truthy_env("SIRIUS_TEST_S3_STRICT")) {
    FAIL("SIRIUS_TEST_S3_* environment is required in strict mode");
  }
  SUCCEED("SIRIUS_TEST_S3_* not set; skipping live SiriusHttpFS test");
  return true;
}

std::string s3_uri(std::string_view bucket, std::string_view key)
{
  return "s3://" + std::string{bucket} + "/" + std::string{key};
}

std::string sql_quote(std::string_view value)
{
  std::string out{"'"};
  for (char c : value) {
    if (c == '\'') { out.push_back('\''); }
    out.push_back(c);
  }
  out.push_back('\'');
  return out;
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

std::string read_text_file(fs::path const& path)
{
  std::ifstream in(path);
  if (!in) { throw std::runtime_error("cannot open source file: " + path.string()); }
  return {std::istreambuf_iterator<char>{in}, std::istreambuf_iterator<char>{}};
}

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
  explicit sirius_httpfs_config_env_guard(s3_test_env const& env)
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
           "    scan_manager:\n"
           "      object_store:\n"
           "        endpoint: "
        << yaml_quote(env.endpoint)
        << "\n"
           "        region: "
        << yaml_quote(env.region)
        << "\n"
           "        access_key: "
        << yaml_quote(env.access_key)
        << "\n"
           "        secret_key: "
        << yaml_quote(env.secret_key)
        << "\n"
           "        tls_verify: false\n"
           "      rest:\n"
           "        max_connections: 8\n"
           "        request_timeout_s: 30\n";
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
  explicit sirius_httpfs_fixture(s3_test_env const& env) : config_env(env), db(nullptr), con(db)
  {
    load_sirius_extension(db);
    REQUIRE(con.context->registered_state->Get<duckdb::SiriusContext>("sirius_state"));
    setenv("SIRIUS_DISABLE", "1", 1);
  }

  sirius_httpfs_config_env_guard config_env;
  duckdb::DuckDB db;
  duckdb::Connection con;
};

std::unique_ptr<duckdb::MaterializedQueryResult> require_query_ok(duckdb::Connection& con,
                                                                  std::string const& sql)
{
  auto result = con.Query(sql);
  REQUIRE(result);
  INFO((result->HasError() ? result->GetError() : ""));
  REQUIRE_FALSE(result->HasError());
  return std::unique_ptr<duckdb::MaterializedQueryResult>(
    static_cast<duckdb::MaterializedQueryResult*>(result.release()));
}

std::vector<std::vector<std::string>> collect_rows(duckdb::MaterializedQueryResult& result)
{
  std::vector<std::vector<std::string>> rows;
  for (duckdb::idx_t r = 0; r < result.RowCount(); ++r) {
    std::vector<std::string> row;
    row.reserve(result.ColumnCount());
    for (duckdb::idx_t c = 0; c < result.ColumnCount(); ++c) {
      row.push_back(result.GetValue(c, r).ToString());
    }
    rows.push_back(std::move(row));
  }
  return rows;
}

void set_gpu_execution(duckdb::Connection& con, bool enabled)
{
  auto result = con.Query(std::string{"SET gpu_execution = "} + (enabled ? "true" : "false"));
  REQUIRE(result);
  INFO((result->HasError() ? result->GetError() : ""));
  REQUIRE_FALSE(result->HasError());
}

std::shared_ptr<sirius::io::sirius_datasource> require_rest_datasource(
  sirius_httpfs_fixture& fixture, std::string const& uri)
{
  auto sirius_ctx =
    fixture.con.context->registered_state->Get<duckdb::SiriusContext>("sirius_state");
  REQUIRE(sirius_ctx);
  auto datasource = sirius_ctx->get_scan_manager().create_datasource(uri);
  REQUIRE(datasource != nullptr);
  REQUIRE(datasource->io_ctx() != nullptr);
  CHECK(datasource->io_ctx()->type() == sirius::io::io_context_type::restful);
  auto* rest_ctx = dynamic_cast<sirius::io::rest::rest_ioctx*>(datasource->io_ctx().get());
  REQUIRE(rest_ctx != nullptr);
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

struct glob_scan_manager_fixture {
  std::unique_ptr<sirius::memory::sirius_memory_reservation_manager> memory =
    initialize_memory_manager(1);
  std::shared_ptr<const sirius::memory::topology_index> topology = discover_topology_index(*memory);
};

sirius::scan_manager::scan_manager_config make_fake_list_config(std::string endpoint)
{
  sirius::scan_manager::scan_manager_config cfg{};
  cfg.use_sirius_datasource        = true;
  cfg.object_store.endpoint        = std::move(endpoint);
  cfg.object_store.region          = "us-east-1";
  cfg.object_store.access_key      = "glob-stress-access-key";
  cfg.object_store.secret_key      = "glob-stress-secret-key";
  cfg.object_store.tls_verify      = false;
  cfg.rest.request_timeout_s       = 2;
  cfg.rest.max_connections         = 1;
  cfg.rest.max_retry_attempts      = 1;
  cfg.rest.max_auth_retry_attempts = 1;
  cfg.rest.retry_backoff_base      = std::chrono::milliseconds{0};
  cfg.rest.retry_jitter            = std::chrono::milliseconds{0};
  cfg.rest.honor_retry_after       = false;
  cfg.rest_n_reactors              = 1;
  cfg.enable_prefetch_cache        = false;
  return cfg;
}

class generated_list_http_server {
 public:
  explicit generated_list_http_server(std::vector<std::string> keys) : _keys(std::move(keys))
  {
    _listen_fd = ::socket(AF_INET, SOCK_STREAM, 0);
    if (_listen_fd < 0) { throw std::runtime_error("socket failed: " + errno_message()); }

    int one = 1;
    if (::setsockopt(_listen_fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one)) != 0) {
      throw std::runtime_error("setsockopt failed: " + errno_message());
    }

    sockaddr_in addr{};
    addr.sin_family      = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    addr.sin_port        = 0;
    if (::bind(_listen_fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0) {
      throw std::runtime_error("bind failed: " + errno_message());
    }
    if (::listen(_listen_fd, 8) != 0) {
      throw std::runtime_error("listen failed: " + errno_message());
    }

    socklen_t len = sizeof(addr);
    if (::getsockname(_listen_fd, reinterpret_cast<sockaddr*>(&addr), &len) != 0) {
      throw std::runtime_error("getsockname failed: " + errno_message());
    }
    _port   = ntohs(addr.sin_port);
    _thread = std::thread([this] { accept_loop(); });
  }

  ~generated_list_http_server()
  {
    _stop.store(true);
    if (_listen_fd >= 0) {
      ::shutdown(_listen_fd, SHUT_RDWR);
      ::close(_listen_fd);
      _listen_fd = -1;
    }
    if (_thread.joinable()) { _thread.join(); }
    for (auto& worker : _workers) {
      if (worker.joinable()) { worker.join(); }
    }
  }

  generated_list_http_server(generated_list_http_server const&)            = delete;
  generated_list_http_server& operator=(generated_list_http_server const&) = delete;

  [[nodiscard]] std::string endpoint() const { return "http://127.0.0.1:" + std::to_string(_port); }

 private:
  static std::string errno_message() { return std::strerror(errno); }

  static std::string request_target(std::string const& request)
  {
    auto const first_space = request.find(' ');
    if (first_space == std::string::npos) { return {}; }
    auto const second_space = request.find(' ', first_space + 1);
    if (second_space == std::string::npos) { return {}; }
    return request.substr(first_space + 1, second_space - first_space - 1);
  }

  static int from_hex(char c)
  {
    if (c >= '0' && c <= '9') { return c - '0'; }
    if (c >= 'a' && c <= 'f') { return c - 'a' + 10; }
    if (c >= 'A' && c <= 'F') { return c - 'A' + 10; }
    return -1;
  }

  static std::string percent_decode(std::string_view value)
  {
    std::string out;
    out.reserve(value.size());
    for (std::size_t i = 0; i < value.size(); ++i) {
      if (value[i] == '%' && i + 2 < value.size()) {
        auto const hi = from_hex(value[i + 1]);
        auto const lo = from_hex(value[i + 2]);
        if (hi >= 0 && lo >= 0) {
          out.push_back(static_cast<char>((hi << 4) | lo));
          i += 2;
          continue;
        }
      }
      out.push_back(value[i]);
    }
    return out;
  }

  static std::string query_value(std::string const& target, std::string_view key)
  {
    auto const query_pos = target.find('?');
    if (query_pos == std::string::npos) { return {}; }
    auto const query  = std::string_view{target}.substr(query_pos + 1);
    auto const needle = std::string{key} + "=";
    for (std::size_t pos = 0; pos < query.size();) {
      auto amp = query.find('&', pos);
      if (amp == std::string_view::npos) { amp = query.size(); }
      auto const part        = query.substr(pos, amp - pos);
      auto const needle_view = std::string_view{needle};
      if (part.size() >= needle_view.size() && part.substr(0, needle_view.size()) == needle_view) {
        return percent_decode(part.substr(needle_view.size()));
      }
      pos = amp + 1;
    }
    return {};
  }

  static std::string xml_escape(std::string_view value)
  {
    std::string out;
    out.reserve(value.size());
    for (char c : value) {
      switch (c) {
        case '&': out += "&amp;"; break;
        case '<': out += "&lt;"; break;
        case '>': out += "&gt;"; break;
        case '"': out += "&quot;"; break;
        case '\'': out += "&apos;"; break;
        default: out.push_back(c); break;
      }
    }
    return out;
  }

  static void send_all(int fd, std::string_view bytes)
  {
    std::size_t sent = 0;
    while (sent < bytes.size()) {
      auto const written = ::send(fd, bytes.data() + sent, bytes.size() - sent, MSG_NOSIGNAL);
      if (written <= 0) { return; }
      sent += static_cast<std::size_t>(written);
    }
  }

  void accept_loop()
  {
    while (!_stop.load()) {
      sockaddr_in client{};
      socklen_t len = sizeof(client);
      auto const fd = ::accept(_listen_fd, reinterpret_cast<sockaddr*>(&client), &len);
      if (fd < 0) {
        if (_stop.load()) { return; }
        continue;
      }
      _workers.emplace_back([this, fd] {
        handle_client(fd);
        ::close(fd);
      });
    }
  }

  void handle_client(int fd)
  {
    std::string request(16 * 1024, '\0');
    auto const received = ::recv(fd, request.data(), request.size(), 0);
    if (received <= 0) { return; }
    request.resize(static_cast<std::size_t>(received));

    auto const target = request_target(request);
    if (request.rfind("GET ", 0) != 0 || target.find("list-type=2") == std::string::npos) {
      send_all(fd,
               "HTTP/1.1 405 Method Not Allowed\r\nContent-Length: 0\r\nConnection: close\r\n\r\n");
      return;
    }

    auto const prefix = query_value(target, "prefix");
    std::ostringstream xml;
    xml << "<?xml version=\"1.0\" encoding=\"UTF-8\"?>"
           "<ListBucketResult xmlns=\"http://s3.amazonaws.com/doc/2006-03-01/\">"
           "<IsTruncated>false</IsTruncated>";
    for (auto const& key : _keys) {
      if (key.rfind(prefix, 0) != 0) { continue; }
      xml << "<Contents><Key>" << xml_escape(key) << "</Key><Size>" << key.size()
          << "</Size></Contents>";
    }
    xml << "</ListBucketResult>";

    auto const body     = xml.str();
    auto const response = "HTTP/1.1 200 OK\r\nContent-Type: application/xml\r\nContent-Length: " +
                          std::to_string(body.size()) + "\r\nConnection: close\r\n\r\n" + body;
    send_all(fd, response);
  }

  int _listen_fd{-1};
  std::uint16_t _port{0};
  std::vector<std::string> _keys;
  std::atomic<bool> _stop{false};
  std::thread _thread;
  std::vector<std::thread> _workers;
};

std::string deep_key(std::string_view prefix, std::size_t depth, std::string_view tail)
{
  std::string key{prefix};
  for (std::size_t i = 0; i < depth; ++i) {
    key += "segment-" + std::to_string(i) + "/";
  }
  key += tail;
  return key;
}

std::vector<std::string> glob_paths(sirius::scan_manager::sirius_scan_manager& manager,
                                    std::string const& pattern)
{
  auto files = sirius::io::s3::expand_glob(pattern, manager);
  std::vector<std::string> paths;
  paths.reserve(files.size());
  for (auto const& file : files) {
    paths.push_back(file.path);
  }
  return paths;
}

}  // namespace

TEST_CASE("S3 LIST keys retain literal identity when embedded in object URIs",
          "[s3][filesystem][glob]")
{
  for (auto const key : {std::string_view{"a%2Fb.p"},
                         std::string_view{"a%20b.p"},
                         std::string_view{"a#b.p"},
                         std::string_view{"a?b.p"},
                         std::string_view{"100%x.p"}}) {
    DYNAMIC_SECTION("key=" << key)
    {
      auto const parsed = sirius::io::parse("s3://bkt/" + std::string{key});

      CHECK(parsed.host == "bkt");
      CHECK(parsed.path == key);
    }
  }
}

TEST_CASE("S3 object URI parsing preserves ordinary keys byte for byte", "[s3][filesystem][glob]")
{
  for (auto const key : {std::string_view{"plain.parquet"},
                         std::string_view{"year=2026/part 0.parquet"},
                         std::string_view{"nested/path/file.parquet"}}) {
    DYNAMIC_SECTION("key=" << key)
    {
      CHECK(sirius::io::parse("s3://bkt/" + std::string{key}).path == key);
    }
  }
}

TEST_CASE("S3 glob matcher handles deep crawl patterns without backtracking blowup",
          "[s3][filesystem][glob]")
{
  auto const stress_key = deep_key("stress/", 28, "y.parquet");
  std::vector<std::string> const keys{
    "prefix/x.parquet",
    "prefix/a/b/x.parquet",
    "prefix/a/b/y.parquet",
    "prefix/part-alpha/filea1.parquet",
    "prefix/deep/part-z/fileb2.parquet",
    "prefix/deep/notpart-z/filea3.parquet",
    "prefix/deep/part-z/filec4.parquet",
    "prefix/deep/part-z/filea12.parquet",
    "tail.parquet",
    "leading/a/tail.parquet",
    "leading/a/not-tail.csv",
    stress_key,
  };
  generated_list_http_server server(keys);
  glob_scan_manager_fixture fixture;
  sirius::scan_manager::sirius_scan_manager manager{
    make_fake_list_config(server.endpoint()), *fixture.memory, fixture.topology};

  CHECK(
    glob_paths(manager, "s3://bucket/prefix/**/x.parquet") ==
    std::vector<std::string>{"s3://bucket/prefix/a/b/x.parquet", "s3://bucket/prefix/x.parquet"});

  CHECK(glob_paths(manager, "s3://bucket/prefix/**/part-*/file[ab]?.parquet") ==
        std::vector<std::string>{"s3://bucket/prefix/deep/part-z/fileb2.parquet",
                                 "s3://bucket/prefix/part-alpha/filea1.parquet"});

  CHECK(glob_paths(manager, "s3://bucket/**/tail.parquet") ==
        std::vector<std::string>{"s3://bucket/leading/a/tail.parquet", "s3://bucket/tail.parquet"});

  CHECK(glob_paths(manager, "s3://bucket/prefix/**") ==
        std::vector<std::string>{"s3://bucket/prefix/a/b/x.parquet",
                                 "s3://bucket/prefix/a/b/y.parquet",
                                 "s3://bucket/prefix/deep/notpart-z/filea3.parquet",
                                 "s3://bucket/prefix/deep/part-z/filea12.parquet",
                                 "s3://bucket/prefix/deep/part-z/fileb2.parquet",
                                 "s3://bucket/prefix/deep/part-z/filec4.parquet",
                                 "s3://bucket/prefix/part-alpha/filea1.parquet",
                                 "s3://bucket/prefix/x.parquet"});

  auto const started = std::chrono::steady_clock::now();
  auto const stress_matches =
    glob_paths(manager, "s3://bucket/stress/**/**/**/**/**/**/**/**/**/**/x.parquet");
  auto const elapsed = std::chrono::steady_clock::now() - started;

  INFO("memoized deep-crawl mismatch completed in "
       << std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count() << " ms");
  CHECK(stress_matches.empty());
  if (elapsed > std::chrono::seconds{1}) {
    WARN("deep-crawl glob match exceeded the expected memoized runtime");
  }
}

TEST_CASE("S3 literal-key implementation contains no retired URI escape or percent guard",
          "[s3][filesystem][glob]")
{
  auto const root   = fs::path{SIRIUS_PROJECT_ROOT};
  auto const source = read_text_file(root / "src" / "io" / "s3" / "sirius_httpfs.cpp");
  auto const header = read_text_file(root / "src" / "include" / "io" / "s3" / "sirius_httpfs.hpp");
  auto const implementation = source + header;

  auto const retired_escape  = std::string{"escape_s3_key_"} + "for_uri";
  auto const retired_guard   = std::string{"key_has_percent_encoded_"} + "sequence";
  auto const retired_wording = std::string{"containing a percent-"} + "encoded sequence";

  CHECK(implementation.find(retired_escape) == std::string::npos);
  CHECK(implementation.find(retired_guard) == std::string::npos);
  CHECK(implementation.find(retired_wording) == std::string::npos);
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
  CHECK(wildcard_message.find("glob/wildcard patterns are not supported") == std::string::npos);

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
                    Catch::Contains("narrow the glob prefix"));
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
  CHECK(static_cast<std::size_t>(size) == datasource->io_object().size());
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

TEST_CASE("transparent read_parquet over S3 routes through Sirius without httpfs",
          "[.][s3][integration][filesystem][sql][gpu_execution]")
{
  auto env = read_s3_test_env();
  if (skip_if_no_s3_env(env)) { return; }

  sirius_httpfs_fixture fixture(*env);
  set_gpu_execution(fixture.con, true);

  auto const uri = s3_uri(env->bucket, "parquet/nation.parquet");
  auto s3_result = require_query_ok(fixture.con,
                                    "SELECT n_nationkey, n_name, n_regionkey "
                                    "FROM read_parquet(" +
                                      sql_quote(uri) + ") ORDER BY n_nationkey");

  duckdb::DuckDB local_db(nullptr);
  duckdb::Connection local_con(local_db);
  auto local_result =
    require_query_ok(local_con,
                     "SELECT n_nationkey, n_name, n_regionkey "
                     "FROM read_parquet(" +
                       sql_quote((fs::path(SIRIUS_PROJECT_ROOT) / "test" / "cpp" / "integration" /
                                  "data" / "parquet" / "nation.parquet")
                                   .string()) +
                       ") ORDER BY n_nationkey");

  REQUIRE(s3_result->RowCount() == 25);
  REQUIRE(s3_result->ColumnCount() == 3);
  CHECK(s3_result->GetValue(0, 0).GetValue<int32_t>() == 0);
  CHECK(s3_result->GetValue(1, 0).ToString() == "ALGERIA");
  CHECK(s3_result->GetValue(2, 0).GetValue<int32_t>() == 0);
  CHECK(collect_rows(*s3_result) == collect_rows(*local_result));
}
