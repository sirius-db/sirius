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

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <stop_token>
#include <string>
#include <string_view>
#include <system_error>
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
}  // namespace

#if defined(__GLIBCXX__) && _GLIBCXX_USE_CXX11_ABI
TEST_CASE("Paimon admits retained snapshot input before allocating its buffer",
          "[paimon_cursor][paimon_native][paimon_snapshot_admission]")
{
  using sirius::op::scan::open_paimon_read;
  using sirius::op::scan::paimon_snapshot_input_budget;
  namespace fs = std::filesystem;

  auto owners   = std::make_shared<native_fixture>();
  auto metadata = fixture_metadata();
  auto pattern  = (fs::temp_directory_path() / "sirius-paimon-snapshot-XXXXXX").string();
  REQUIRE(::mkdtemp(pattern.data()));
  struct temp_directory {
    fs::path path;
    ~temp_directory()
    {
      std::error_code ignored;
      fs::remove_all(path, ignored);
    }
  } temp{pattern};
  auto root   = temp.path / "table";
  auto source = fs::path(metadata.path);
  fs::copy(source, root, fs::copy_options::recursive);
  metadata.path      = root.string();
  auto snapshot      = root / "snapshot/snapshot-1";
  auto original_size = fs::file_size(snapshot);
  auto filesystem    = paimon::FileSystemFactory::Get("local", metadata.path, {});
  REQUIRE(filesystem.ok());
  uint64_t remaining = 256 * 1024;
  uint64_t requested = 0;
  uint64_t reserved  = 0;
  unsigned requests  = 0;
  bool replace_path  = false;
  bool shorten_input = false;
  paimon_snapshot_input_budget budget{
    std::move(filesystem).value(), [&](uint64_t bytes) -> std::shared_ptr<void> {
      ++requests;
      requested = bytes;
      if (bytes > remaining) return {};
      REQUIRE(reserved == 0);
      if (replace_path) {
        fs::rename(snapshot, snapshot.string() + ".held");
        std::ofstream replacement(snapshot);
        replacement << "{}";
        REQUIRE(replacement.good());
      }
      if (shorten_input) fs::resize_file(snapshot, original_size - 1);
      auto lease = std::shared_ptr<void>(&reserved, [&](void*) { reserved = 0; });
      reserved   = bytes;
      return lease;
    }};
  auto open = [&] { return open_paimon_read(metadata, {0}, owners->pool, owners, 1, {}, &budget); };

  SECTION("oversized valid JSON refuses before read and a healthy retry succeeds")
  {
    constexpr uint64_t padded_size = 270337;
    {
      std::ofstream padded(snapshot, std::ios::app);
      padded << std::string(padded_size - original_size, ' ');
      REQUIRE(padded.good());
    }
    REQUIRE(fs::file_size(snapshot) == padded_size);
    try {
      (void)open();
      FAIL("oversized snapshot published a cursor");
    } catch (sirius::scan_manager::preparation_resource_error const& error) {
      REQUIRE(error.bound_exceeded);
    }
    REQUIRE(requests == 1);
    REQUIRE(requested == padded_size + 1);
    REQUIRE(reserved == 0);
    REQUIRE(owners->pool->CurrentUsage() == 0);
    fs::resize_file(snapshot, original_size);
  }
  SECTION("an admitted malformed snapshot releases its lease on failure")
  {
    {
      std::ofstream invalid(snapshot, std::ios::trunc);
      invalid << std::string(original_size, '?');
      REQUIRE(invalid.good());
    }
    try {
      (void)open();
      FAIL("malformed snapshot published a cursor");
    } catch (sirius::scan_manager::preparation_resource_error const&) {
      FAIL("malformed snapshot was classified as resource exhaustion");
    } catch (std::runtime_error const&) {
    }
    REQUIRE(requests == 1);
    REQUIRE(requested == original_size + 1);
    REQUIRE(reserved == 0);
    REQUIRE(owners->pool->CurrentUsage() == 0);
    fs::copy_file(source / "snapshot/snapshot-1", snapshot, fs::copy_options::overwrite_existing);
  }
  SECTION("an admitted short snapshot read is an I/O failure and retires its lease")
  {
    shorten_input = true;
    try {
      (void)open();
      FAIL("short snapshot read published a cursor");
    } catch (sirius::scan_manager::preparation_resource_error const&) {
      FAIL("short snapshot read was classified as resource exhaustion");
    } catch (std::runtime_error const& error) {
      REQUIRE(std::string_view(error.what()).find("actual read len") != std::string_view::npos);
    }
    REQUIRE(requests == 1);
    REQUIRE(requested == original_size + 1);
    REQUIRE(reserved == 0);
    REQUIRE(owners->pool->CurrentUsage() == 0);
    shorten_input = false;
    fs::copy_file(source / "snapshot/snapshot-1", snapshot, fs::copy_options::overwrite_existing);
  }
  SECTION("planning and data reads use the supplied filesystem")
  {
    metadata.options["file-system"] = "unavailable-for-snapshot-admission-test";
  }
  SECTION("size and read use the same open file") { replace_path = true; }
  SECTION("system-table aliases cannot change the retained data-table root")
  {
    metadata.path += "$ro";
    REQUIRE_THROWS_AS(open(), std::invalid_argument);
    REQUIRE(requests == 0);
    return;
  }

  remaining   = original_size + 1;
  auto cursor = open();
  REQUIRE(cursor);
  REQUIRE(requested == remaining);
  REQUIRE(reserved == 0);
  std::vector<int64_t> rows;
  while (cursor->consume({}, [&](ArrowSchema const& schema, ArrowArray const& batch) {
    REQUIRE(schema.n_children == 2);
    REQUIRE(std::string(schema.children[1]->name) == "id");
    auto const& column = *batch.children[1];
    REQUIRE(column.length == 1);
    rows.push_back(static_cast<int64_t const*>(column.buffers[1])[column.offset]);
  })) {}
  std::sort(rows.begin(), rows.end());
  REQUIRE(rows == std::vector<int64_t>{1, 1, 2});
  cursor.reset();
  REQUIRE(owners->pool->CurrentUsage() == 0);
}
#endif

TEST_CASE("Paimon cursor reads retained snapshot from real append files",
          "[paimon_cursor][paimon_native]")
{
  auto owners = std::make_shared<native_fixture>();
  auto pool   = owners->pool;
  {
    auto metadata = fixture_metadata();
    auto cursor   = sirius::op::scan::open_paimon_read(metadata, {0}, owners->pool, owners, 1, {});
    REQUIRE(cursor);
    std::vector<int64_t> actual;
    size_t batches = 0;
    auto accept    = [&](ArrowSchema const& schema, ArrowArray const& array) {
      REQUIRE(++batches <= 8);
      REQUIRE(std::string(schema.format) == "+s");
      REQUIRE(schema.n_children == array.n_children);
      REQUIRE(array.offset == 0);
      bool found = false;
      for (int64_t c = 0; c < schema.n_children; ++c) {
        auto const& field = *schema.children[c];
        if (!field.name || std::string(field.name) != "id") continue;
        found              = true;
        auto const& column = *array.children[c];
        REQUIRE(std::string(field.format) == "l");
        REQUIRE(column.offset == 0);
        REQUIRE(column.length == array.length);
        REQUIRE(column.null_count == 0);
        REQUIRE(column.n_buffers == 2);
        auto data = static_cast<int64_t const*>(column.buffers[1]);
        actual.insert(actual.end(), data, data + column.length);
      }
      REQUIRE(found);
    };
    while (cursor->consume({}, accept)) {}
    REQUIRE(batches > 1);
    // Independent checked-in corpus expectation: append_a at snapshot A (1).
    // LATEST is snapshot 2 and has an additional id=3; accidentally reopening latest fails.
    std::sort(actual.begin(), actual.end());
    REQUIRE(actual == std::vector<int64_t>{1, 1, 2});
    REQUIRE_FALSE(cursor->consume({}, accept));
  }
  owners.reset();
  REQUIRE(pool->CurrentUsage() == 0);
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
  REQUIRE(cursor->consume({}, [](ArrowSchema const&, ArrowArray const&) {}));
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
  REQUIRE_THROWS(cursor->consume({}, [](ArrowSchema const&, ArrowArray const&) {}));
  REQUIRE(pool->attempts > 0);
  auto failures = pool->attempts;
  REQUIRE_THROWS(cursor->consume({}, [](ArrowSchema const&, ArrowArray const&) {}));
  REQUIRE(pool->attempts == failures);
  cursor->close();
  cursor.reset();
  REQUIRE(pool->CurrentUsage() == 0);
}
