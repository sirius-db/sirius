/*
 * Copyright 2026, Sirius Contributors.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "io/cache/metadata_store.hpp"
#include "op/scan/parquet_metadata.hpp"

#include <catch2/catch_test_macros.hpp>

TEST_CASE("A footer alias retains evidence after cache eviction", "[metadata_retention]")
{
  using sirius::op::scan::parquet_metadata;
  auto parsed = std::make_shared<cudf::io::parquet::FileMetaData>();
  parsed->row_groups.resize(2);
  parsed->row_groups[0].columns.resize(3);
  auto metadata                        = std::make_shared<parquet_metadata>(parsed, 64);
  std::weak_ptr<parquet_metadata> weak = metadata;
  auto footer                          = metadata->file_metadata();
  metadata.reset();
  parsed.reset();
  CHECK_FALSE(weak.expired());
  CHECK(footer->row_groups.size() == 2);
  footer.reset();
  CHECK(weak.expired());
}

TEST_CASE("Metadata store validates reopened objects and isolates access namespaces",
          "[metadata_retention][object_identity]")
{
  using namespace sirius::io;
  struct object : io_object {
    std::string path = "same-path", raw = "backend-locator", etag;
    const std::string& object_path() const noexcept override { return path; }
    const std::string& raw_file_cache_id() const noexcept override { return raw; }
    size_t size() const noexcept override { return 64; }
    std::string_view validation_tag() const noexcept override { return etag; }
  };
  object first, reopened;
  cache::metadata_store store, other_scope;
  auto parsed   = std::make_shared<cudf::io::parquet::FileMetaData>();
  auto metadata = std::make_shared<sirius::op::scan::parquet_metadata>(parsed, 64);
  store.register_metadata(first, metadata);
  CHECK(store.get_metadata(first) == metadata);
  CHECK_FALSE(store.get_metadata(reopened));
  CHECK(first.identity_cache_key() != reopened.identity_cache_key());
  first.etag = reopened.etag = "opaque-multipart-tag-42";
  store.register_metadata(first, metadata);
  CHECK(store.get_metadata(reopened) == metadata);
  CHECK_FALSE(other_scope.get_metadata(first));
  CHECK(first.identity_cache_key() == reopened.identity_cache_key());
  reopened.etag = "new-tag";
  CHECK_FALSE(store.get_metadata(reopened));
  CHECK(first.identity_cache_key() != reopened.identity_cache_key());
  reopened.etag = first.etag;
  reopened.raw  = "different-backend-locator";
  CHECK(first.identity_cache_key() != reopened.identity_cache_key());
}
