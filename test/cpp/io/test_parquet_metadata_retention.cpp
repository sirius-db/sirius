/*
 * Copyright 2026, Sirius Contributors.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "op/scan/parquet_metadata.hpp"

#include <catch2/catch_test_macros.hpp>

TEST_CASE("A footer alias retains evidence and charge after cache eviction", "[metadata_retention]")
{
  using sirius::io::io_object_metadata;
  using sirius::op::scan::parquet_metadata;
  auto baseline = io_object_metadata::live_retained_bytes();
  auto parsed   = std::make_shared<cudf::io::parquet::FileMetaData>();
  parsed->row_groups.resize(2);
  parsed->row_groups[0].columns.resize(3);
  auto metadata = std::make_shared<parquet_metadata>(parsed, 64);
  auto bytes    = metadata->account_retention();
  CHECK(bytes > sizeof(parquet_metadata) + sizeof(*parsed));
  std::weak_ptr<parquet_metadata> weak = metadata;
  auto footer                          = metadata->file_metadata();
  metadata.reset();
  parsed.reset();
  CHECK_FALSE(weak.expired());
  CHECK(footer->row_groups.size() == 2);
  CHECK(io_object_metadata::live_retained_bytes() == baseline + bytes);
  footer.reset();
  CHECK(weak.expired());
  CHECK(io_object_metadata::live_retained_bytes() == baseline);
}
