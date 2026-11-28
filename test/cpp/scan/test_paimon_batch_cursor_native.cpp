/*
 * Copyright 2026, Sirius Contributors.
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
#include "op/scan/paimon/paimon_batch_cursor.hpp"
#include "op/scan/paimon/paimon_native_read.hpp"
#include "scan/paimon_native_test_fixture.hpp"
#include "scan_manager/preparation_ledger.hpp"

#include <duckdb/common/arrow/arrow.hpp>
#include <paimon/fs/file_system_factory.h>
#include <paimon/memory/memory_pool.h>
#include <paimon/scan_context.h>
#include <paimon/table/source/data_split.h>
#include <paimon/table/source/plan.h>
#include <paimon/table/source/table_scan.h>

#include <algorithm>
#include <functional>
#include <mutex>
#include <stdexcept>
#include <stop_token>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

using namespace sirius::test::paimon;
namespace {
class fault_pool final : public paimon::MemoryPool {
 public:
  explicit fault_pool(bool reject) : reject(reject), upstream(paimon::GetMemoryPool()) {}
  bool reject;
  unsigned attempts = 0;
  void* Malloc(uint64_t n, uint64_t alignment) override
  {
    if (reject) {
      ++attempts;
      throw std::bad_alloc();
    }
    return upstream->Malloc(n, alignment);
  }
  void* Realloc(void* p, size_t old_size, size_t new_size, uint64_t alignment) override
  {
    if (reject) {
      ++attempts;
      throw std::bad_alloc();
    }
    return upstream->Realloc(p, old_size, new_size, alignment);
  }
  void Free(void* p, uint64_t n) override { upstream->Free(p, n); }
  void Free(void* p, uint64_t n, uint64_t alignment) override { upstream->Free(p, n, alignment); }
  uint64_t CurrentUsage() const override { return upstream->CurrentUsage(); }
  uint64_t MaxMemoryUsage() const override { return upstream->MaxMemoryUsage(); }

 private:
  std::unique_ptr<paimon::MemoryPool> upstream;
};

struct parquet_stream_counts {
  std::mutex mutex;
  size_t live = 0;
  size_t peak = 0;
};

// A raw Parquet file reader in the stock SDK retains this input stream until close.
class counted_parquet_stream final : public paimon::InputStream {
 public:
  counted_parquet_stream(std::unique_ptr<paimon::InputStream> input,
                         std::shared_ptr<parquet_stream_counts> counts)
    : _input(std::move(input)), _counts(std::move(counts))
  {
    std::lock_guard lock(_counts->mutex);
    _counts->peak = std::max(_counts->peak, ++_counts->live);
  }

  ~counted_parquet_stream() override
  {
    _input.reset();
    retire();
  }

  paimon::Status Close() override
  {
    auto status = _input->Close();
    if (status.ok()) retire();
    return status;
  }

  paimon::Status Seek(int64_t offset, paimon::SeekOrigin origin) override
  {
    return _input->Seek(offset, origin);
  }
  paimon::Result<int64_t> GetPos() const override { return _input->GetPos(); }
  paimon::Result<int64_t> Read(char* buffer, int64_t size) override
  {
    return _input->Read(buffer, size);
  }
  paimon::Result<int64_t> Read(char* buffer, int64_t size, int64_t offset) override
  {
    return _input->Read(buffer, size, offset);
  }
  void ReadAsync(char* buffer,
                 int64_t size,
                 int64_t offset,
                 std::function<void(paimon::Status)>&& callback) override
  {
    _input->ReadAsync(buffer, size, offset, std::move(callback));
  }
  paimon::Result<std::string> GetUri() const override { return _input->GetUri(); }
  paimon::Result<int64_t> Length() const override { return _input->Length(); }

 private:
  void retire()
  {
    if (_counts) {
      auto counts = std::move(_counts);
      std::lock_guard lock(counts->mutex);
      --counts->live;
    }
  }

  std::unique_ptr<paimon::InputStream> _input;
  std::shared_ptr<parquet_stream_counts> _counts;
};

class counted_parquet_filesystem final : public paimon::FileSystem {
 public:
  explicit counted_parquet_filesystem(std::shared_ptr<paimon::FileSystem> upstream)
    : _upstream(std::move(upstream))
  {
  }

  std::pair<size_t, size_t> live_and_peak() const
  {
    std::lock_guard lock(_counts->mutex);
    return {_counts->live, _counts->peak};
  }

  paimon::Result<std::unique_ptr<paimon::InputStream>> Open(std::string const& path) const override
  {
    return track(path, _upstream->Open(path));
  }
  paimon::Result<std::unique_ptr<paimon::InputStream>> Open(
    paimon::FileStatus const& status) const override
  {
    return track(status.GetPath(), _upstream->Open(status));
  }
  paimon::Result<std::unique_ptr<paimon::OutputStream>> Create(std::string const& path,
                                                               bool overwrite) const override
  {
    return _upstream->Create(path, overwrite);
  }
  paimon::Status Mkdirs(std::string const& path) const override { return _upstream->Mkdirs(path); }
  paimon::Status Rename(std::string const& from, std::string const& to) const override
  {
    return _upstream->Rename(from, to);
  }
  paimon::Status Delete(std::string const& path, bool recursive) const override
  {
    return _upstream->Delete(path, recursive);
  }
  paimon::Result<paimon::FileStatus> GetFileStatus(std::string const& path) const override
  {
    return _upstream->GetFileStatus(path);
  }
  paimon::Status ListDir(std::string const& path,
                         std::vector<paimon::BasicFileStatus>* entries) const override
  {
    return _upstream->ListDir(path, entries);
  }
  paimon::Status ListFileStatus(std::string const& path,
                                std::vector<paimon::FileStatus>* entries) const override
  {
    return _upstream->ListFileStatus(path, entries);
  }
  paimon::Result<bool> Exists(std::string const& path) const override
  {
    return _upstream->Exists(path);
  }

 private:
  paimon::Result<std::unique_ptr<paimon::InputStream>> track(
    std::string const& path, paimon::Result<std::unique_ptr<paimon::InputStream>> input) const
  {
    if (!input.ok() || !path.ends_with(".parquet")) return input;
    return std::unique_ptr<paimon::InputStream>(
      std::make_unique<counted_parquet_stream>(std::move(input).value(), _counts));
  }

  std::shared_ptr<paimon::FileSystem> _upstream;
  std::shared_ptr<parquet_stream_counts> _counts = std::make_shared<parquet_stream_counts>();
};

void append_ids(ArrowSchema const& schema, ArrowArray const& array, std::vector<int64_t>& actual)
{
  REQUIRE(std::string(schema.format) == "+s");
  REQUIRE(schema.n_children == array.n_children);
  REQUIRE(array.offset == 0);
  for (int64_t c = 0; c < schema.n_children; ++c) {
    auto const& field = *schema.children[c];
    if (!field.name || std::string(field.name) != "id") continue;
    auto const& column = *array.children[c];
    REQUIRE(std::string(field.format) == "l");
    REQUIRE(column.offset == 0);
    REQUIRE(column.length == array.length);
    REQUIRE(column.null_count == 0);
    REQUIRE(column.n_buffers == 2);
    auto data = static_cast<int64_t const*>(column.buffers[1]);
    actual.insert(actual.end(), data, data + column.length);
    return;
  }
  FAIL("Missing projected id field");
}
}  // namespace

TEST_CASE("Paimon cursor reads retained snapshot from real append files",
          "[paimon_cursor][paimon_native]")
{
  auto owners            = std::make_shared<native_fixture>();
  auto pool              = owners->pool;
  auto const snapshot_id = GENERATE(1, 2);
  {
    auto metadata        = fixture_metadata();
    metadata.snapshot_id = snapshot_id;
    std::shared_ptr<paimon::FileSystem> filesystem;
    if (snapshot_id == 2) {
      auto local = paimon::FileSystemFactory::Get("local", metadata.path, {});
      REQUIRE(local.ok());
      filesystem = std::move(local).value();
      // The explicitly supplied filesystem must serve both planning and data reads.
      metadata.options["file-system"]              = "unavailable-for-paimon-native-test";
      metadata.options["source.split.target-size"] = "1B";
      auto options                                 = metadata.options;
      options["scan.mode"]                         = "from-snapshot-full";
      options["scan.snapshot-id"]                  = "2";
      auto context                                 = paimon::ScanContextBuilder(metadata.path)
                       .SetOptions(options)
                       .SetTableSchema(metadata.schema_json)
                       .WithFileSystem(filesystem)
                       .WithMemoryPool(pool)
                       .Finish();
      REQUIRE(context.ok());
      auto scan = paimon::TableScan::Create(std::move(context).value());
      REQUIRE(scan.ok());
      auto plan = scan.value()->CreatePlan();
      REQUIRE(plan.ok());
      REQUIRE(plan.value()->SnapshotId() == 2);
      REQUIRE(plan.value()->Splits().size() == 2);
    }
    auto cursor =
      sirius::op::scan::open_paimon_read(metadata, {0}, pool, owners, 1, {}, filesystem);
    REQUIRE(cursor);
    std::vector<int64_t> actual;
    size_t batches = 0;
    auto accept    = [&](ArrowSchema const& schema, ArrowArray const& array) {
      REQUIRE(++batches <= 8);
      append_ids(schema, array, actual);
    };
    while (cursor->consume({}, 1, accept)) {}
    REQUIRE(batches > 1);
    // Independent corpus expectations: append_a at snapshot 1, append_b at snapshot 2.
    std::sort(actual.begin(), actual.end());
    auto const expected =
      snapshot_id == 1 ? std::vector<int64_t>{1, 1, 2} : std::vector<int64_t>{1, 1, 2, 3};
    REQUIRE(actual == expected);
    REQUIRE_FALSE(cursor->consume({}, 1, accept));
  }
  owners.reset();
  REQUIRE(pool->CurrentUsage() == 0);
}

TEST_CASE("Paimon native cursor prepares one file at a time within a packed split",
          "[paimon_cursor][paimon_native]")
{
  auto owners                                  = std::make_shared<native_fixture>();
  auto metadata                                = fixture_metadata();
  metadata.snapshot_id                         = 2;
  metadata.options["source.split.target-size"] = "128MB";
  metadata.options["read.preparation-window"]  = "4";
  auto local = paimon::FileSystemFactory::Get("local", metadata.path, {});
  REQUIRE(local.ok());
  auto filesystem      = std::make_shared<counted_parquet_filesystem>(std::move(local).value());
  auto options         = metadata.options;
  options["scan.mode"] = "from-snapshot-full";
  options["scan.snapshot-id"] = "2";
  auto context                = paimon::ScanContextBuilder(metadata.path)
                   .SetOptions(options)
                   .SetTableSchema(metadata.schema_json)
                   .WithFileSystem(filesystem)
                   .WithMemoryPool(owners->pool)
                   .Finish();
  REQUIRE(context.ok());
  auto scan = paimon::TableScan::Create(std::move(context).value());
  REQUIRE(scan.ok());
  auto plan = scan.value()->CreatePlan();
  REQUIRE(plan.ok());
  REQUIRE(plan.value()->SnapshotId() == 2);
  REQUIRE(plan.value()->Splits().size() == 1);
  auto split = std::dynamic_pointer_cast<paimon::DataSplit>(plan.value()->Splits().front());
  REQUIRE(split);
  auto const files = split->GetFileList();
  REQUIRE(files.size() == 2);
  REQUIRE(files[0].file_path.ends_with(".parquet"));
  REQUIRE(files[1].file_path.ends_with(".parquet"));
  REQUIRE(files[0].file_path != files[1].file_path);

  auto cursor =
    sirius::op::scan::open_paimon_read(metadata, {0}, owners->pool, owners, 1, {}, filesystem);
  REQUIRE(cursor);
  std::vector<int64_t> actual;
  size_t first_batch_live = 0;
  size_t batches          = 0;
  auto accept             = [&](ArrowSchema const& schema, ArrowArray const& array) {
    if (++batches == 1) first_batch_live = filesystem->live_and_peak().first;
    REQUIRE(batches <= 4);
    append_ids(schema, array, actual);
  };
  while (cursor->consume({}, 1, accept)) {}
  std::sort(actual.begin(), actual.end());
  REQUIRE(actual == std::vector<int64_t>{1, 1, 2, 3});
  auto const [live, peak] = filesystem->live_and_peak();
  REQUIRE(live == 0);
  CHECK(first_batch_live == 1);
  CHECK(peak == 1);
}

TEST_CASE("Paimon native binding preserves empty and refuses inconsistent selectors",
          "[paimon_cursor][paimon_native]")
{
  auto owners   = std::make_shared<native_fixture>();
  auto metadata = fixture_metadata();
  SECTION("empty binding never plans latest")
  {
    metadata.snapshot_id.reset();
    REQUIRE_FALSE(sirius::op::scan::open_paimon_read(metadata, {0}, owners->pool, owners, 1, {}));
  }
  SECTION("missing retained snapshot is an error")
  {
    metadata.snapshot_id = 9999;
    REQUIRE_THROWS(sirius::op::scan::open_paimon_read(metadata, {0}, owners->pool, owners, 1, {}));
  }
  SECTION("conflicting branch selector")
  {
    metadata.options["branch"] = "other";
    REQUIRE_THROWS_AS(
      sirius::op::scan::open_paimon_read(metadata, {0}, owners->pool, owners, 1, {}),
      std::invalid_argument);
  }
  SECTION("conflicting numeric selector")
  {
    metadata.options["scan.snapshot-id"] = "2";
    REQUIRE_THROWS_AS(
      sirius::op::scan::open_paimon_read(metadata, {0}, owners->pool, owners, 1, {}),
      std::invalid_argument);
  }
  SECTION("non-numeric selector cannot override retained binding")
  {
    metadata.options["scan.timestamp-millis"] = "1";
    REQUIRE_THROWS_AS(
      sirius::op::scan::open_paimon_read(metadata, {0}, owners->pool, owners, 1, {}),
      std::invalid_argument);
  }
  SECTION("empty binding still validates projection")
  {
    metadata.snapshot_id.reset();
    REQUIRE_THROWS(sirius::op::scan::open_paimon_read(metadata, {99}, owners->pool, owners, 1, {}));
  }
  SECTION("empty binding still validates schema")
  {
    metadata.snapshot_id.reset();
    metadata.schema_json = "{}";
    REQUIRE_THROWS(sirius::op::scan::open_paimon_read(metadata, {0}, owners->pool, owners, 1, {}));
  }
  SECTION("stop precedes native initialization")
  {
    std::stop_source source;
    source.request_stop();
    REQUIRE_THROWS_AS(sirius::op::scan::open_paimon_read(
                        metadata,
                        {0},
                        owners->pool,
                        owners,
                        1,
                        sirius::scan_manager::preparation_stop_token(source.get_token())),
                      std::system_error);
  }
  REQUIRE(owners->pool->CurrentUsage() == 0);
}

TEST_CASE("Paimon native planning allocation refusal cannot become EOF",
          "[paimon_cursor][paimon_native]")
{
  auto owners   = std::make_shared<native_fixture>();
  auto metadata = fixture_metadata();
  auto pool     = std::make_shared<fault_pool>(true);
  REQUIRE_THROWS(sirius::op::scan::open_paimon_read(metadata, {0}, pool, owners, 1, {}));
  REQUIRE(pool->attempts > 0);
  REQUIRE(pool->CurrentUsage() == 0);
  // The failed attempt must not poison a subsequent healthy reader.
  auto cursor = sirius::op::scan::open_paimon_read(metadata, {0}, owners->pool, owners, 1, {});
  REQUIRE(cursor);
  REQUIRE(cursor->consume({}, 1, [](ArrowSchema const&, ArrowArray const&) {}));
  cursor->close();
  cursor.reset();
  REQUIRE(owners->pool->CurrentUsage() == 0);
}

TEST_CASE("Paimon native read allocation refusal is terminal and drains",
          "[paimon_cursor][paimon_native]")
{
  auto owners   = std::make_shared<native_fixture>();
  auto metadata = fixture_metadata();
  auto pool     = std::make_shared<fault_pool>(false);
  auto cursor   = sirius::op::scan::open_paimon_read(metadata, {0}, pool, owners, 1, {});
  REQUIRE(cursor);
  pool->reject = true;
  REQUIRE_THROWS(cursor->consume({}, 1, [](ArrowSchema const&, ArrowArray const&) {}));
  REQUIRE(pool->attempts > 0);
  auto failures = pool->attempts;
  REQUIRE_THROWS(cursor->consume({}, 1, [](ArrowSchema const&, ArrowArray const&) {}));
  REQUIRE(pool->attempts == failures);
  cursor->close();
  cursor.reset();
  REQUIRE(pool->CurrentUsage() == 0);
}
