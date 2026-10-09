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
#include "io/path_utils.hpp"

#include <cucascade/io/kvikio/kvikio_context.hpp>
#include <unistd.h>

#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>

//===----------------------------------------------------------------------===//
// strip_file_scheme: Sirius's duckdb::Path-based variant. It strips every legal `file:` URI form
// and folds `.`, `..` and empty segments; src/io/path_utils.hpp has the rationale.
//===----------------------------------------------------------------------===//

using sirius::io::strip_file_scheme;

TEST_CASE("strip_file_scheme handles every legal file URI form", "[path_utils]")
{
  // https://iceberg.apache.org/spec/#paths-in-metadata points at the file URI scheme, which has
  // three spellings. An unstripped one fails during EXECUTION, taking the runtime fallback that
  // poisons the connection rather than declining at plan time.
  CHECK(strip_file_scheme("file:/abs/path.parquet") == "/abs/path.parquet");
  CHECK(strip_file_scheme("file:///abs/path.parquet") == "/abs/path.parquet");
  CHECK(strip_file_scheme("file:///var/tmp/t/data/00000-0-abc.parquet") ==
        "/var/tmp/t/data/00000-0-abc.parquet");
  CHECK(strip_file_scheme("file://localhost/abs/path.parquet") == "/abs/path.parquet");
}

TEST_CASE("strip_file_scheme folds dot, dot-dot and empty segments", "[path_utils]")
{
  // The duckdb::Path normalization cuCascade's scheme-only strip lacks: callers key caches and
  // match iceberg delete files on the result, so one file must have one spelling.
  CHECK(strip_file_scheme("file:///abs/./a//b/../c.parquet") == "/abs/a/c.parquet");
  // A leading `..` cannot climb above the root.
  CHECK(strip_file_scheme("file:///../x.parquet") == "/x.parquet");
}

TEST_CASE("strip_file_scheme percent-decodes a file URI", "[path_utils]")
{
  // Only what was stripped is a URI. A bare path or object-store key keeps a literal `%`, which
  // the case below asserts.
  CHECK(strip_file_scheme("file:///abs/a%20b/data.parquet") == "/abs/a b/data.parquet");
  CHECK(strip_file_scheme("file:///abs/100%25.parquet") == "/abs/100%.parquet");
  // A malformed escape is not a reason to fail an open: keep the stripped bytes as they are.
  CHECK(strip_file_scheme("file:///abs/a%2.parquet") == "/abs/a%2.parquet");
}

TEST_CASE("strip_file_scheme is case-insensitive", "[path_utils]")
{
  // Manifests spell the scheme FILE:// and File:// too, and iceberg delete files are matched on the
  // stripped path: a missed match silently returns deleted rows.
  CHECK(strip_file_scheme("FILE:///abs/path.parquet") == "/abs/path.parquet");
  CHECK(strip_file_scheme("File:///abs/path.parquet") == "/abs/path.parquet");
  CHECK(strip_file_scheme("fILe:///abs/path.parquet") == "/abs/path.parquet");
}

TEST_CASE("strip_file_scheme leaves everything else byte-identical", "[path_utils]")
{
  // Safe to apply unconditionally at an I/O boundary: object-store URIs must reach their
  // backend untouched, and s3 keys are taken literally (no percent-decoding), so this must
  // not normalize anything it does not strip.
  CHECK(strip_file_scheme("/abs/bare/path.parquet") == "/abs/bare/path.parquet");
  CHECK(strip_file_scheme("s3://bucket/key.parquet") == "s3://bucket/key.parquet");
  CHECK(strip_file_scheme("s3://bucket/a%20b") == "s3://bucket/a%20b");
  CHECK(strip_file_scheme("gs://bucket/key") == "gs://bucket/key");
  CHECK(strip_file_scheme("relative/path.parquet") == "relative/path.parquet");
  CHECK(strip_file_scheme("") == "");
}

TEST_CASE("strip_file_scheme does not throw on input a URI parser rejects", "[path_utils]")
{
  // Deliberately not implemented via a URI parser (cucascade::io::parse rejects these): it runs
  // on every open and must be total.
  CHECK_NOTHROW(strip_file_scheme(""));
  CHECK_NOTHROW(strip_file_scheme("file://"));
  CHECK_NOTHROW(strip_file_scheme("://"));
  // The non-standard "double-slash path" form: duckdb::Path rejects it, but this repo's fixtures
  // use it for repo-RELATIVE paths, so it keeps the plain strip.
  CHECK(strip_file_scheme("file://relative/path") == "relative/path");
  // `file:/` is the host-omitted spelling of the root path. `file://` alone has no path and no
  // localhost authority, so the original bytes come back.
  CHECK(strip_file_scheme("file:/") == "/");
  CHECK(strip_file_scheme("file://") == "file://");
}

//===----------------------------------------------------------------------===//
// open_datasource
//
// The io_object's raw_file_cache_id keys the prefetching cache and the metadata_store, so one
// file reached under two spellings must get one id. cuCascade's own open strips only the scheme
// (no `.`/`..` folding), which is why every Sirius open goes through sirius::io::open_datasource.
//===----------------------------------------------------------------------===//

namespace {

/// Removes a scratch directory tree when it goes out of scope, pass or fail.
class remove_all_guard {
 public:
  explicit remove_all_guard(std::filesystem::path root) : _root(std::move(root)) {}
  ~remove_all_guard()
  {
    std::error_code ignored;
    std::filesystem::remove_all(_root, ignored);
  }
  remove_all_guard(remove_all_guard const&)            = delete;
  remove_all_guard& operator=(remove_all_guard const&) = delete;

  [[nodiscard]] std::filesystem::path const& path() const noexcept { return _root; }

 private:
  std::filesystem::path _root;
};

/// <tmp>/sirius-path-utils-<pid>: the scratch root of the open_datasource tests.
std::filesystem::path open_datasource_root()
{
  return std::filesystem::temp_directory_path() /
         ("sirius-path-utils-" + std::to_string(::getpid()));
}

/// A regular file at <root>/dir/data.bin.
std::filesystem::path make_open_datasource_file(std::filesystem::path const& root)
{
  auto const dir = root / "dir";
  std::filesystem::create_directories(dir);
  auto const file = dir / "data.bin";
  std::ofstream{file, std::ios::binary} << "open_datasource normalization\n";
  REQUIRE(std::filesystem::is_regular_file(file));
  return file;
}

}  // namespace

TEST_CASE("open_datasource keys a dotted file URI like the bare path", "[path_utils]")
{
  remove_all_guard const scratch{open_datasource_root()};
  auto const file  = make_open_datasource_file(scratch.path());
  auto const ioctx = std::make_shared<cucascade::io::kvikio_context>();
  auto const dotted =
    "file://" + (file.parent_path() / ".." / "dir" / "." / file.filename()).string();

  auto const bare     = sirius::io::open_datasource(ioctx, file.string());
  auto const from_uri = sirius::io::open_datasource(ioctx, dotted);
  REQUIRE(bare);
  REQUIRE(from_uri);
  CHECK(bare->get_io_object().raw_file_cache_id() == file.string());
  CHECK(from_uri->get_io_object().raw_file_cache_id() == file.string());
}

TEST_CASE("open_datasource normalizes the path it forwards with an open hint", "[path_utils]")
{
  remove_all_guard const scratch{open_datasource_root()};
  auto const file  = make_open_datasource_file(scratch.path());
  auto const ioctx = std::make_shared<cucascade::io::kvikio_context>();
  // The host-omitted `file:/abs` spelling, with an empty segment.
  auto const uri = "file:" + file.parent_path().string() + "//" + file.filename().string();

  auto const datasource =
    sirius::io::open_datasource(ioctx, uri, cucascade::io::open_hint::parquet_footer_probe);
  REQUIRE(datasource);
  CHECK(datasource->get_io_object().raw_file_cache_id() == file.string());
  CHECK(datasource->size() == std::filesystem::file_size(file));
}
