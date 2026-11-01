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

#include <catch.hpp>
#include <core_functions_extension.hpp>
#include <duckdb.hpp>
#include <duckdb/common/file_opener.hpp>
#include <duckdb/common/file_system.hpp>
#include <duckdb/main/client_context.hpp>
#include <io/kvikio/kvikio_context.hpp>
#include <op/scan/iceberg_metadata_reader.hpp>
#include <op/scan/parquet_delete_reader.hpp>
#include <parquet_extension.hpp>
#include <unistd.h>
#include <utils/s3_backend.hpp>

#include <barrier>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <future>

namespace {
using namespace sirius::op::scan;
struct delete_fixture {
  duckdb::DuckDB db{nullptr};
  duckdb::Connection connection{db};
  std::filesystem::path directory;
  std::string path;
  delete_fixture()
  {
    db.LoadStaticExtension<duckdb::CoreFunctionsExtension>();
    db.LoadStaticExtension<duckdb::ParquetExtension>();
    char temp[] = "/tmp/sirius-delete-reader-XXXXXX";
    auto name   = mkdtemp(temp);
    REQUIRE(name != nullptr);
    directory = name;
    path      = (directory / "delete's.parquet").string();
  }
  ~delete_fixture()
  {
    std::error_code ec;
    std::filesystem::remove_all(directory, ec);
  }
  void write(uint64_t rows)
  {
    auto escaped = duckdb::StringUtil::Replace(path, "'", "''");
    auto result  = connection.Query(
      "COPY (SELECT i::BIGINT AS pos, 'data.parquet' AS file_path, "
       "(i % 3)::BIGINT AS _ducklake_internal_snapshot_id FROM range(" +
      std::to_string(rows) + ") t(i)) TO '" + escaped + "' (FORMAT PARQUET, ROW_GROUP_SIZE 2048)");
    REQUIRE_FALSE(result->HasError());
  }
  void read(parquet_delete_read_options const& options,
            parquet_delete_bind const& bind,
            std::function<void(duckdb::DataChunk&)> const& consume)
  {
    read_parquet_delete_batches(
      *connection.context, duckdb::OpenFileInfo(path), options, bind, consume);
  }
};
auto position_projection = [](auto const&, uint64_t) { return std::vector<uint64_t>{0}; };
}  // namespace

TEST_CASE("Parquet deletes project reordered columns across row groups", "[parquet_delete_reader]")
{
  delete_fixture f;
  f.write(10000);
  uint64_t count = 0, batches = 0;
  f.read(
    {},
    [](auto const& columns, uint64_t rows) {
      REQUIRE(rows == 10000);
      REQUIRE(columns[0].name == "pos");
      REQUIRE(columns[0].type == duckdb::LogicalType::BIGINT);
      return std::vector<uint64_t>{1, 0, 2};
    },
    [&](duckdb::DataChunk& chunk) {
      ++batches;
      REQUIRE(chunk.ColumnCount() == 3);
      for (duckdb::idx_t i = 0; i < chunk.size(); ++i) {
        REQUIRE(chunk.GetValue(0, i).GetValue<std::string>() == "data.parquet");
        REQUIRE(chunk.GetValue(1, i).GetValue<int64_t>() == int64_t(count));
        REQUIRE(chunk.GetValue(2, i).GetValue<int64_t>() == int64_t(count % 3));
        ++count;
      }
    });
  REQUIRE(count == 10000);
  REQUIRE(batches > 1);
}

TEST_CASE("Parquet deletes validate empty files and binding claims", "[parquet_delete_reader]")
{
  delete_fixture f;
  f.write(0);
  uint64_t batches = 0;
  f.read({}, position_projection, [&](auto&) { ++batches; });
  REQUIRE(batches == 0);
  parquet_delete_read_options options;
  options.expected_size = std::filesystem::file_size(f.path) + 1;
  REQUIRE_THROWS(f.read(options, position_projection, [](auto&) {}));
  options.expected_size.reset();
  options.expected_footer_size = 0;
  REQUIRE_THROWS(f.read(options, position_projection, [](auto&) {}));
  REQUIRE_THROWS(
    f.read({}, [](auto const&, auto) { return std::vector<uint64_t>{99}; }, [](auto&) {}));
  REQUIRE_THROWS(
    f.read({}, [](auto const&, auto) { return std::vector<uint64_t>{0, 0}; }, [](auto&) {}));
}

TEST_CASE("Parquet deletes stop before open and between batches", "[parquet_delete_reader]")
{
  delete_fixture f;
  bool cancelled = true;
  parquet_delete_read_options options;
  options.check_stop = [&] {
    if (cancelled) throw std::runtime_error("cancelled");
  };
  REQUIRE_THROWS_WITH(f.read(options, position_projection, [](auto&) {}), "cancelled");
  f.write(10000);
  cancelled        = false;
  uint64_t batches = 0;
  REQUIRE_THROWS_WITH(f.read(options,
                             position_projection,
                             [&](auto&) {
                               ++batches;
                               cancelled = true;
                             }),
                      "cancelled");
  REQUIRE(batches == 1);
  cancelled = false;
  REQUIRE_NOTHROW(f.read(options, position_projection, [](auto&) {}));
  f.connection.context->Interrupt();
  REQUIRE_THROWS(f.read({}, position_projection, [](auto&) {}));
  f.connection.context->ClearInterrupt();
}

TEST_CASE("Parquet deletes reject changes and release handles after consumer failure",
          "[parquet_delete_reader]")
{
  delete_fixture f;
  f.write(10000);
  REQUIRE_THROWS(f.read({}, position_projection, [&](auto&) {
    std::ofstream changed(f.path, std::ios::binary | std::ios::app);
    changed.put('x');
  }));
  std::filesystem::remove(f.path);
  f.write(10000);
  REQUIRE_THROWS_WITH(f.read({},
                             position_projection,
                             [](auto&) { throw std::runtime_error("consumer budget exhausted"); }),
                      "consumer budget exhausted");
  REQUIRE_NOTHROW(f.read({}, position_projection, [](auto&) {}));
}

TEST_CASE("Iceberg consumes shared Parquet batches and rejects malformed positions",
          "[parquet_delete_reader][iceberg]")
{
  delete_fixture f;
  f.write(10000);
  sirius::io::kvikio_context io;
  iceberg_delete_discovery discovery;
  discovery.positional_delete_files.push_back(f.path);
  auto result = load_delete_payload(*f.connection.context, "test-table", &io, {}, discovery);
  REQUIRE(result->positional_deletes.size() == 1);
  auto const& positions = result->positional_deletes.at("data.parquet");
  REQUIRE(positions.size() == 10000);
  REQUIRE(positions.front() == 0);
  REQUIRE(positions.back() == 9999);

  for (auto expression : {"NULL::BIGINT", "-1::BIGINT", "'bad'::VARCHAR"}) {
    std::filesystem::remove(f.path);
    auto escaped = duckdb::StringUtil::Replace(f.path, "'", "''");
    auto written =
      f.connection.Query("COPY (SELECT 'data.parquet' AS file_path, " + std::string(expression) +
                         " AS pos) TO '" + escaped + "' (FORMAT PARQUET)");
    REQUIRE_FALSE(written->HasError());
    REQUIRE_THROWS(load_delete_payload(*f.connection.context, "test-table", &io, {}, discovery));
  }
}

// A remote-shaped filesystem with per-open version tags and conditional reads.
// No device/inode metadata: this exercises the same branch as HTTP/S3.
namespace {
class delete_remote_fs : public duckdb::FileSystem {
 public:
  struct remote_handle : duckdb::FileHandle {
    remote_handle(delete_remote_fs& fs, std::string path, duckdb::FileOpenFlags flags)
      : FileHandle(fs, std::move(path), flags), owner(fs), generation(fs.generation)
    {
      ++owner.live_handles;
    }
    ~remote_handle() override { --owner.live_handles; }
    void Close() override {}
    delete_remote_fs& owner;
    uint64_t generation;
  };
  std::string bytes;
  uint64_t generation                     = 1;
  int live_handles                        = 0;
  bool versioned                          = true;
  duckdb::ClientContext* expected_context = nullptr;
  explicit delete_remote_fs(std::string const& path)
  {
    std::ifstream input(path, std::ios::binary);
    bytes.assign(std::istreambuf_iterator<char>(input), {});
  }
  duckdb::unique_ptr<duckdb::FileHandle> OpenFile(
    std::string const& path,
    duckdb::FileOpenFlags flags,
    duckdb::optional_ptr<duckdb::FileOpener> opener) override
  {
    if (duckdb::FileOpener::TryGetClientContext(opener).get() != expected_context)
      throw duckdb::IOException("Wrong caller context");
    return duckdb::make_uniq<remote_handle>(*this, path, flags);
  }
  void Read(duckdb::FileHandle& h, void* buffer, int64_t count, duckdb::idx_t offset) override
  {
    auto& handle = static_cast<remote_handle&>(h);
    if (handle.generation != generation) throw duckdb::IOException("Remote precondition failed");
    if (count < 0 || offset > bytes.size() || uint64_t(count) > bytes.size() - offset)
      throw duckdb::IOException("Remote short read");
    std::memcpy(buffer, bytes.data() + offset, count);
  }
  int64_t GetFileSize(duckdb::FileHandle&) override { return bytes.size(); }
  duckdb::timestamp_t GetLastModifiedTime(duckdb::FileHandle&) override
  {
    return duckdb::timestamp_t(0);
  }
  std::string GetVersionTag(duckdb::FileHandle& h) override
  {
    return versioned ? std::to_string(static_cast<remote_handle&>(h).generation) : "";
  }
  duckdb::FileType GetFileType(duckdb::FileHandle&) override
  {
    return duckdb::FileType::FILE_TYPE_REGULAR;
  }
  bool CanHandleFile(std::string const& path) override
  {
    return path.starts_with("delete-remote://");
  }
  bool CanSeek() override { return true; }
  bool OnDiskFile(duckdb::FileHandle&) override { return false; }
  std::string GetName() const override { return "delete-remote"; }
};
}  // namespace

TEST_CASE("Remote Parquet deletes preserve caller context and native filesystem support",
          "[parquet_delete_reader][remote]")
{
  delete_fixture f;
  f.write(10000);
  auto fs                 = duckdb::make_uniq<delete_remote_fs>(f.path);
  auto& remote            = *fs;
  remote.expected_context = f.connection.context.get();
  f.db.GetFileSystem().RegisterSubSystem(std::move(fs));
  f.path = "delete-remote://bucket/deletes.parquet";
  SECTION("versioned remote object") {}
  SECTION("unversioned remote object remains readable") { remote.versioned = false; }
  sirius::io::kvikio_context io;
  iceberg_delete_discovery discovery;
  discovery.positional_delete_files.push_back(f.path);
  auto result = load_delete_payload(*f.connection.context, "remote-table", &io, {}, discovery);
  REQUIRE(result->positional_deletes.at("data.parquet").size() == 10000);
  REQUIRE(remote.live_handles == 0);
}

TEST_CASE("Remote Parquet deletes reject replacement and release handles on failure",
          "[parquet_delete_reader][remote]")
{
  delete_fixture f;
  f.write(10000);
  auto fs                 = duckdb::make_uniq<delete_remote_fs>(f.path);
  auto& remote            = *fs;
  remote.expected_context = f.connection.context.get();
  f.db.GetFileSystem().RegisterSubSystem(std::move(fs));
  f.path = "delete-remote://bucket/deletes.parquet";
  SECTION("replacement between footer and scan")
  {
    REQUIRE_THROWS_WITH(f.read(
                          {},
                          [&](auto const&, uint64_t) {
                            ++remote.generation;
                            return std::vector<uint64_t>{0};
                          },
                          [](auto&) {}),
                        Catch::Matchers::ContainsSubstring("object changed"));
  }
  SECTION("replacement during batches")
  {
    REQUIRE_THROWS(f.read({}, position_projection, [&](auto&) { ++remote.generation; }));
  }
  SECTION("additional bound identity rejects before publishing")
  {
    parquet_delete_read_options options;
    options.verify_remote = [](auto&) { throw std::runtime_error("bound identity mismatch"); };
    REQUIRE_THROWS_WITH(
      f.read(options, position_projection, [](auto&) { FAIL("must not publish"); }),
      "bound identity mismatch");
  }
  SECTION("cancellation between remote batches")
  {
    bool stop = false;
    parquet_delete_read_options options;
    options.check_stop = [&] {
      if (stop) throw std::runtime_error("cancelled");
    };
    REQUIRE_THROWS_WITH(f.read(options, position_projection, [&](auto&) { stop = true; }),
                        "cancelled");
  }
  REQUIRE(remote.live_handles == 0);
}

TEST_CASE("HTTP Parquet position deletes match SQL read_parquet", "[.][parquet_delete_http]")
{
  auto root      = std::getenv("SIRIUS_DELETE_HTTP_ROOT");
  auto url       = std::getenv("SIRIUS_DELETE_HTTP_URL");
  auto extension = std::getenv("SIRIUS_DELETE_HTTPFS");
  REQUIRE(root);
  REQUIRE(url);
  REQUIRE(extension);
  delete_fixture f;
  f.write(10000);
  auto loaded =
    f.connection.Query("LOAD '" + duckdb::StringUtil::Replace(extension, "'", "''") + "'");
  INFO((loaded->HasError() ? loaded->GetError() : "loaded"));
  REQUIRE_FALSE(loaded->HasError());
  std::filesystem::copy_file(f.path,
                             std::filesystem::path(root) / "deletes.parquet",
                             std::filesystem::copy_options::overwrite_existing);
  REQUIRE_FALSE(f.connection.Query("SET enable_external_file_cache=false")->HasError());
  f.path = url;
  f.connection.BeginTransaction();
  auto baseline = f.connection.Query("SELECT count(*), sum(pos) FROM read_parquet('" +
                                     duckdb::StringUtil::Replace(f.path, "'", "''") + "')");
  INFO((baseline->HasError() ? baseline->GetError() : "queried"));
  REQUIRE_FALSE(baseline->HasError());
  auto expected = baseline->Fetch();
  REQUIRE(expected);
  REQUIRE(expected->GetValue(0, 0).GetValue<int64_t>() == 10000);
  sirius::io::kvikio_context io;
  iceberg_delete_discovery discovery;
  discovery.positional_delete_files.push_back(f.path);
  auto result = load_delete_payload(*f.connection.context, "remote-table", &io, {}, discovery);
  auto const& positions = result->positional_deletes.at("data.parquet");
  REQUIRE(positions.size() == 10000);
  int64_t sum = 0;
  for (auto position : positions)
    sum += position;
  REQUIRE(sum == expected->GetValue(1, 0).GetValue<int64_t>());
  f.connection.Commit();
}

TEST_CASE("S3 Parquet position deletes match SQL and enforce credentials", "[.][parquet_delete_s3]")
{
  if (!sirius::test::ensure_s3_test_env()) SKIP("S3 test backend is not enabled");
  auto env = [](char const* name) {
    auto value = std::getenv(name);
    REQUIRE(value);
    return std::string(value);
  };
  auto quote = [](std::string const& value) {
    return "'" + duckdb::StringUtil::Replace(value, "'", "''") + "'";
  };
  delete_fixture f;
  f.write(10000);
  std::ifstream input(f.path, std::ios::binary);
  std::vector<uint8_t> bytes(std::istreambuf_iterator<char>{input}, {});
  auto key = "parquet-delete-reader/positions.parquet";
  if (!sirius::test::put_s3_test_object(key, bytes))
    SKIP("This test uploads fixtures only to the managed local S3 backend");

  // This comparison targets DuckDB HTTPFS. The standard test host also
  // registers SiriusHttpFS, whose precedence and admission are tested separately.
  auto& files = f.db.GetFileSystem();
  for (auto const& name : files.ListSubSystems()) {
    if (name == "SiriusHttpFS") files.UnregisterSubSystem(name);
  }
  auto loaded = f.connection.Query("LOAD httpfs");
  INFO((loaded->HasError() ? loaded->GetError() : "httpfs loaded"));
  REQUIRE_FALSE(loaded->HasError());
  REQUIRE_FALSE(f.connection.Query("SET enable_external_file_cache=false")->HasError());
  auto endpoint = env("SIRIUS_TEST_S3_ENDPOINT");
  REQUIRE(endpoint.starts_with("http://"));
  endpoint.erase(0, 7);
  auto bucket    = env("SIRIUS_TEST_S3_BUCKET");
  auto uri       = "s3://" + bucket + "/" + key;
  auto configure = [&](std::string const& secret) {
    auto result = f.connection.Query(
      "CREATE OR REPLACE SECRET delete_reader_s3 (TYPE S3, KEY_ID " +
      quote(env("SIRIUS_TEST_S3_ACCESS_KEY")) + ", SECRET " + quote(secret) + ", REGION " +
      quote(env("SIRIUS_TEST_S3_REGION")) + ", ENDPOINT " + quote(endpoint) +
      ", URL_STYLE 'path', USE_SSL false, SCOPE " + quote("s3://" + bucket) + ")");
    REQUIRE_FALSE(result->HasError());
  };
  auto credential = env("SIRIUS_TEST_S3_SECRET_KEY");
  configure(credential);
  f.connection.BeginTransaction();
  auto baseline =
    f.connection.Query("SELECT count(*), sum(pos) FROM read_parquet(" + quote(uri) + ")");
  INFO((baseline->HasError() ? baseline->GetError() : "SQL baseline read"));
  REQUIRE_FALSE(baseline->HasError());
  auto expected = baseline->Fetch();
  REQUIRE(expected);
  REQUIRE(expected->GetValue(0, 0).GetValue<int64_t>() == 10000);
  auto expected_sum = expected->GetValue(1, 0).GetValue<int64_t>();

  sirius::io::kvikio_context io;
  iceberg_delete_discovery discovery;
  discovery.positional_delete_files.push_back(uri);
  auto read = [&] {
    return load_delete_payload(*f.connection.context, "s3-table", &io, {}, discovery);
  };
  auto result           = read();
  auto const& positions = result->positional_deletes.at("data.parquet");
  REQUIRE(positions.size() == 10000);
  int64_t sum = 0;
  for (auto position : positions)
    sum += position;
  REQUIRE(sum == expected_sum);
  f.connection.Commit();

  // Reusing the same URI must not bypass authentication via warmed caches.
  configure("intentionally-invalid-test-secret");
  f.connection.BeginTransaction();
  REQUIRE_THROWS(read());
  f.connection.Rollback();

  configure(credential);
  f.connection.BeginTransaction();
  REQUIRE(read()->positional_deletes.at("data.parquet").size() == 10000);
  f.connection.Commit();

  // Secrets are database-wide. Use session settings for two distinct identities
  // sharing one database, URL and external file cache.
  REQUIRE_FALSE(f.connection.Query("DROP SECRET delete_reader_s3")->HasError());
  REQUIRE_FALSE(f.connection.Query("SET enable_external_file_cache=true")->HasError());
  duckdb::Connection good(f.db), bad(f.db);
  auto configure_session = [&](duckdb::Connection& con, std::string const& secret) {
    for (auto const& [name, value] : std::vector<std::pair<std::string, std::string>>{
           {"s3_access_key_id", env("SIRIUS_TEST_S3_ACCESS_KEY")},
           {"s3_secret_access_key", secret},
           {"s3_region", env("SIRIUS_TEST_S3_REGION")},
           {"s3_endpoint", endpoint},
           {"s3_url_style", "path"}}) {
      REQUIRE_FALSE(con.Query("SET " + name + "=" + quote(value))->HasError());
    }
    REQUIRE_FALSE(con.Query("SET s3_use_ssl=false")->HasError());
    REQUIRE_FALSE(con.Query("SET http_retries=0")->HasError());
  };
  configure_session(good, credential);
  configure_session(bad, "intentionally-invalid-test-secret");
  auto read_connection = [&](duckdb::Connection& con) {
    return load_delete_payload(*con.context, "s3-table", &io, {}, discovery);
  };
  good.BeginTransaction();
  auto warmed = read_connection(good);
  good.Commit();
  auto cached = f.connection.Query(
    "SELECT count(*) FROM duckdb_external_file_cache() WHERE path=" + quote(uri) + " AND loaded");
  REQUIRE_FALSE(cached->HasError());
  REQUIRE(cached->Fetch()->GetValue(0, 0).GetValue<int64_t>() > 0);

  std::barrier start(2);
  using outcome = std::pair<std::shared_ptr<const IcebergDeleteData>, std::string>;
  auto exercise = [&](duckdb::Connection& con) {
    std::vector<outcome> outcomes;
    for (int round = 0; round < 4; ++round) {
      start.arrive_and_wait();
      try {
        con.BeginTransaction();
        auto data = read_connection(con);
        con.Commit();
        outcomes.emplace_back(std::move(data), "");
      } catch (std::exception const& error) {
        outcomes.emplace_back(nullptr, error.what());
        con.Query("ROLLBACK");
      }
    }
    return outcomes;
  };
  auto valid          = std::async(std::launch::async, exercise, std::ref(good));
  auto denied         = std::async(std::launch::async, exercise, std::ref(bad));
  auto valid_results  = valid.get();
  auto denied_results = denied.get();
  for (auto const& [data, error] : valid_results) {
    INFO(error);
    REQUIRE(data);
    REQUIRE(data->positional_deletes == warmed->positional_deletes);
  }
  for (auto const& [data, error] : denied_results) {
    INFO(error);
    REQUIRE_FALSE(data);
    REQUIRE(error.find("403") != std::string::npos);
  }
  configure_session(bad, credential);
  bad.BeginTransaction();
  REQUIRE(read_connection(bad)->positional_deletes == warmed->positional_deletes);
  bad.Commit();
}
