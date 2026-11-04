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
#include <duckdb/common/file_opener.hpp>
#include <duckdb/common/file_system.hpp>
#include <duckdb/main/client_context.hpp>
#include <duckdb/main/client_context_state.hpp>
#include <io/kvikio/kvikio_context.hpp>
#include <op/scan/iceberg_metadata_reader.hpp>
#include <op/scan/parquet_delete_reader.hpp>
#include <parquet_extension.hpp>
#include <parquet_reader.hpp>
#include <utils/s3_backend.hpp>

#include <algorithm>
#include <barrier>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <future>
#include <set>
#include <thread>

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
      *connection.context, duckdb::OpenFileInfo(path), options, bind, [&] { return consume; });
  }
};
struct worker_rendezvous {
  std::mutex mutex;
  std::condition_variable cv;
  std::set<std::thread::id> threads;

  bool wait_for_peer()
  {
    std::unique_lock lock(mutex);
    bool first = threads.empty();
    threads.insert(std::this_thread::get_id());
    cv.notify_all();
    if (!cv.wait_for(lock, std::chrono::seconds(10), [&] { return threads.size() == 2; }))
      throw std::runtime_error("second worker did not reach rendezvous");
    return first;
  }
};

auto position_projection = [](auto const&, uint64_t) { return std::vector<uint64_t>{0}; };
}  // namespace

TEST_CASE("Parquet deletes project reordered columns across row groups", "[parquet_delete_reader]")
{
  delete_fixture f;
  REQUIRE_FALSE(f.connection.Query("SET threads=1")->HasError());
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

TEST_CASE("Parquet delete workers claim each group once and propagate consumer failures",
          "[parquet_delete_reader]")
{
  delete_fixture f;
  f.write(10000);
  REQUIRE_FALSE(f.connection.Query("SET threads=2")->HasError());
  const bool fail = GENERATE(false, true);
  size_t workers  = 0;
  REQUIRE_FALSE(f.connection.Query("SET disable_parquet_prefetching=true")->HasError());
  struct task_exit : duckdb::ClientContextState {
    std::promise<void> first_exit;
    std::once_flag once;
    void OnTaskStop(duckdb::ClientContext&) override
    {
      std::call_once(once, [&] { first_exit.set_value(); });
    }
  };
  auto& states = *f.connection.context->registered_state;
  auto exit    = states.GetOrCreate<task_exit>("delete_task_exit");
  auto stopped = exit->first_exit.get_future();
  worker_rendezvous rendezvous;
  std::atomic<unsigned> consumed{0};
  parquet_delete_read_options options;
  options.after_decode_for_testing = [&] {
    // Hold one decoded batch until its peer has published the consumer error.
    if (!rendezvous.wait_for_peer() && fail &&
        stopped.wait_for(std::chrono::seconds(10)) != std::future_status::ready)
      throw std::runtime_error("failing decoder did not exit");
  };
  worker_rendezvous consumers;
  std::vector<int64_t> positions[2];
  auto scan = [&] {
    read_parquet_delete_batches(
      *f.connection.context, duckdb::OpenFileInfo(f.path), options, position_projection, [&] {
        auto worker = workers++;
        REQUIRE(worker < 2);
        return [&, worker](duckdb::DataChunk& chunk) {
          if (fail) {
            ++consumed;
            throw std::runtime_error("consumer failed");
          }
          if (positions[worker].empty()) consumers.wait_for_peer();
          for (duckdb::idx_t i = 0; i < chunk.size(); ++i)
            positions[worker].push_back(chunk.GetValue(0, i).GetValue<int64_t>());
        };
      });
  };
  if (fail) {
    REQUIRE_THROWS_WITH(scan(), Catch::Matchers::ContainsSubstring("consumer failed"));
    REQUIRE(consumed == 1);
  } else {
    scan();
    REQUIRE(consumers.threads.size() == 2);
    positions[0].insert(positions[0].end(), positions[1].begin(), positions[1].end());
    std::sort(positions[0].begin(), positions[0].end());
    REQUIRE(positions[0].size() == 10000);
    for (int64_t i = 0; i < 10000; ++i)
      REQUIRE(positions[0][i] == i);
  }
  REQUIRE(workers == 2);
  REQUIRE(rendezvous.threads.size() == 2);
  states.Remove("delete_task_exit");
}

TEST_CASE("Parquet deletes reserve footer rows across workers before consuming",
          "[parquet_delete_reader]")
{
  delete_fixture f;
  REQUIRE_FALSE(f.connection.Query("SET threads=2")->HasError());
  REQUIRE_FALSE(f.connection.Query("SET parquet_metadata_cache=true")->HasError());
  f.write(4096);
  duckdb::OpenFileInfo file(f.path);
  auto& context = *f.connection.context;
  duckdb::ParquetReader reader(context, file, duckdb::ParquetOptions(context));
  REQUIRE(reader.NumRowGroups() == 2);
  auto cached = duckdb::ParquetReader::GetMetadataCacheEntry(context, file);
  REQUIRE(cached);
  auto metadata      = duckdb::make_uniq<duckdb_parquet::FileMetaData>(*cached->metadata);
  metadata->num_rows = 2048;
  cached->metadata   = std::move(metadata);
  worker_rendezvous rendezvous;
  parquet_delete_read_options options;
  options.after_decode_for_testing = [&] { rendezvous.wait_for_peer(); };
  std::atomic<uint64_t> consumed{0};
  REQUIRE_THROWS_WITH(
    f.read(options, position_projection, [&](auto& chunk) { consumed += chunk.size(); }),
    Catch::Matchers::ContainsSubstring("exceed footer count"));
  REQUIRE(consumed <= 2048);
}

TEST_CASE("Parquet deletes drain workers before choosing the original error",
          "[parquet_delete_reader]")
{
  delete_fixture f;
  f.write(4096);
  REQUIRE_FALSE(f.connection.Query("SET threads=2")->HasError());
  REQUIRE_FALSE(f.connection.Query("SET disable_parquet_prefetching=true")->HasError());
  struct task_failure : duckdb::ClientContextState {
    bool also_fail_consumer = false;
    std::atomic<unsigned> starts{0};
    std::promise<void> consuming;
    void OnTaskStart(duckdb::ClientContext&) override
    {
      if (also_fail_consumer) {
        if (starts++ == 0) return;
        if (consuming.get_future().wait_for(std::chrono::seconds(10)) != std::future_status::ready)
          throw std::runtime_error("consumer did not start");
      }
      throw std::runtime_error("executor task failed");
    }
  };
  auto& states              = *f.connection.context->registered_state;
  auto state                = states.GetOrCreate<task_failure>("delete_test");
  state->also_fail_consumer = GENERATE(false, true);
  auto scan                 = [&] {
    f.read({}, position_projection, [&](auto&) {
      state->consuming.set_value();
      throw std::logic_error("original consumer error");
    });
  };
  if (state->also_fail_consumer) {
    REQUIRE_THROWS_AS(scan(), std::logic_error);
  } else {
    REQUIRE_THROWS_WITH(scan(), Catch::Matchers::ContainsSubstring("executor task failed"));
  }
  states.Remove("delete_test");
  REQUIRE_NOTHROW(f.read({}, position_projection, [](auto&) {}));
}

TEST_CASE("Parquet deletes validate empty files and binding claims", "[parquet_delete_reader]")
{
  delete_fixture f;
  REQUIRE_FALSE(f.connection.Query("SET threads=1")->HasError());
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
  REQUIRE_FALSE(f.connection.Query("SET threads=1")->HasError());
  std::atomic<bool> cancelled{true};
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
  REQUIRE_FALSE(f.connection.Query("SET threads=1")->HasError());
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

  for (auto expression : {"NULL::BIGINT", "'bad'::VARCHAR"}) {
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
  std::function<void()> before_read;
  std::string bytes;
  duckdb::timestamp_t last_modified{0};
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
    if (before_read) before_read();
    auto& handle = static_cast<remote_handle&>(h);
    if (handle.generation != generation) throw duckdb::IOException("Remote precondition failed");
    if (count < 0 || offset > bytes.size() || uint64_t(count) > bytes.size() - offset)
      throw duckdb::IOException("Remote short read");
    std::memcpy(buffer, bytes.data() + offset, count);
  }
  int64_t GetFileSize(duckdb::FileHandle&) override { return bytes.size(); }
  duckdb::timestamp_t GetLastModifiedTime(duckdb::FileHandle&) override { return last_modified; }
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

TEST_CASE("Remote Parquet deletes honor read-ahead settings with one decode worker",
          "[parquet_delete_reader][remote]")
{
  delete_fixture f;
  f.write(10000);
  REQUIRE_FALSE(f.connection.Query("SET threads=1")->HasError());
  auto mode =
    std::string(GENERATE("enabled", "no_async_threads", "zero_depth", "disable_prefetch"));
  CAPTURE(mode);
  REQUIRE_FALSE(
    f.connection.Query(mode == "no_async_threads" ? "SET async_threads=0" : "SET async_threads=2")
      ->HasError());
  REQUIRE_FALSE(
    f.connection.Query(mode == "zero_depth" ? "SET read_ahead_depth=0" : "SET read_ahead_depth=2")
      ->HasError());
  if (mode == "disable_prefetch")
    REQUIRE_FALSE(f.connection.Query("SET disable_parquet_prefetching=true")->HasError());
  REQUIRE_FALSE(f.connection.Query("SET enable_external_file_cache=false")->HasError());
  std::atomic<bool> scanning{false};
  std::atomic<unsigned> reads{0};
  std::promise<void> second_read;
  auto overlap            = second_read.get_future();
  auto fs                 = duckdb::make_uniq<delete_remote_fs>(f.path);
  auto& remote            = *fs;
  remote.expected_context = f.connection.context.get();
  const auto caller       = std::this_thread::get_id();
  remote.before_read      = [&] {
    if (!scanning) return;
    auto n = ++reads;
    if (mode != "enabled") {
      if (std::this_thread::get_id() != caller)
        throw std::runtime_error("disabled read-ahead issued a background read");
      return;
    }
    if (n == 1 && overlap.wait_for(std::chrono::seconds(10)) != std::future_status::ready)
      throw std::runtime_error("second range read did not start");
    if (n == 2) second_read.set_value();
  };
  f.db.GetFileSystem().RegisterSubSystem(std::move(fs));
  f.path         = "delete-remote://bucket/deletes.parquet";
  uint64_t count = 0;
  f.read(
    {},
    [&](auto const&, uint64_t) {
      scanning = true;
      return std::vector<uint64_t>{0};
    },
    [&](duckdb::DataChunk& chunk) {
      for (duckdb::idx_t i = 0; i < chunk.size(); ++i) {
        REQUIRE(chunk.GetValue(0, i).GetValue<int64_t>() == int64_t(count));
        ++count;
      }
    });
  REQUIRE(count == 10000);
  REQUIRE(reads >= 2);
  REQUIRE(remote.live_handles == 0);

  std::atomic<bool> cancelled{false};
  parquet_delete_read_options options;
  options.check_stop = [&] {
    if (cancelled) throw duckdb::InterruptException();
  };
  REQUIRE_THROWS_AS(f.read(options, position_projection, [&](auto&) { cancelled = true; }),
                    duckdb::InterruptException);
  REQUIRE(cancelled);
  REQUIRE(remote.live_handles == 0);
}

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
  REQUIRE_FALSE(f.connection.Query("SET threads=1")->HasError());
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
  SECTION("bound trailer is read from the current object even when the file cache is warm")
  {
    // The custom scheme needs local-file caching to exercise DuckDB's external cache.
    REQUIRE_FALSE(f.connection.Query("SET cache_local_files=true")->HasError());
    REQUIRE_FALSE(f.connection.Query("SET enable_external_file_cache=true")->HasError());
    remote.versioned     = false;
    remote.last_modified = duckdb::timestamp_t(1000000);
    auto offset          = remote.bytes.size() - 8;
    parquet_delete_read_options options;
    options.expected_footer_size =
      duckdb::Load<uint32_t>(duckdb::const_data_ptr_cast(remote.bytes.data() + offset));
    f.read(options, position_projection, [](auto&) {});

    // Replace the trailer while retaining the size and modification time.
    ++remote.generation;
    remote.bytes[offset] ^= 1;
    REQUIRE_THROWS_WITH(f.read(options, position_projection, [](auto&) {}),
                        Catch::Matchers::ContainsSubstring("footer differs from binding"));
  }
  SECTION("replacement during batches")
  {
    REQUIRE_FALSE(f.connection.Query("SET read_ahead_depth=1")->HasError());
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
    std::atomic<bool> stop{false};
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
