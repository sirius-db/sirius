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

#include <string>
#include <string_view>

//===----------------------------------------------------------------------===//
// strip_file_scheme
//
// Sirius's own duckdb::Path-based variant (src/io/path_utils.hpp), applied where Sirius
// normalizes and keys paths before they reach the io layer: normalize_path in the scan
// manager, the iceberg/puffin readers and plan_get. Iceberg manifests written by the Apache
// implementations record fully-qualified URIs (file:///abs/path.parquet) — Java and Spark
// writers always do — while the local backends only open bare paths. An un-stripped URI reaches
// create_io_object and throws "unsupported path", which happens during execution and so takes the
// RUNTIME fallback rather than declining at plan time.
//
// This is the in-tree coverage for that rule. The corresponding end-to-end case needs a
// table whose manifests carry an absolute file:// URI, which cannot be expressed with the
// repo-relative paths a committed fixture requires; generate it with
// data/iceberg_conformance/gen_corpus.py using an absolute output path.
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
  // The ad-hoc stripper this replaced in the iceberg delete-key matcher was case-SENSITIVE.
  // A manifest written FILE:// would have failed to match its data file, and a failed match
  // finds no deletes for that file — returning deleted rows while looking healthy.
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
