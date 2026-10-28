/*
 * Copyright 2026, Sirius Contributors.
 * SPDX-License-Identifier: Apache-2.0
 */
// Hidden integration microbenchmark: real local files and the production footer
// resolver. Copies share the same small schema; this is not a TPC benchmark.
#include "op/scan/parquet_metadata.hpp"

#include <catch2/catch_test_macros.hpp>
#include <cucascade/cudf/datasource.hpp>
#include <utils/local_ioctx.hpp>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <vector>

TEST_CASE("Ten thousand real Parquet files reuse verified footer records",
          "[.][parquet_footer_cache_bench]")
{
  namespace fs   = std::filesystem;
  char pattern[] = "/tmp/sirius-parquet-footers-XXXXXX";
  auto directory = ::mkdtemp(pattern);
  REQUIRE(directory);
  struct cleanup {
    fs::path path;
    ~cleanup()
    {
      std::error_code error;
      fs::remove_all(path, error);
    }
  } owner{directory};
  auto fixture = fs::path(SIRIUS_PROJECT_ROOT) / "test/cpp/integration/data/parquet/nation.parquet";
  std::vector<std::string> paths;
  for (int i = 0; i < 10000; ++i) {
    auto path = owner.path / (std::to_string(i) + ".parquet");
    fs::copy_file(fixture, path);
    paths.push_back(path.string());
  }
  // Files were just copied: OS page cache is warm in both passes. Only the
  // metadata cache changes from cold to warm; raw prefetch is not initialized.
  auto context = sirius::test::make_test_local_ioctx();
  auto options = cudf::io::parquet_reader_options::builder().build();
  for (int pass = 0; pass < 2; ++pass) {
    auto start  = std::chrono::steady_clock::now();
    size_t hits = 0;
    for (auto const& path : paths) {
      auto source   = cucascade::io::open_datasource(context, path);
      bool hit      = false;
      auto metadata = sirius::op::scan::resolve_parquet_metadata(*source, 0, path, options, &hit);
      REQUIRE(metadata->file_metadata()->num_rows == 25);
      hits += hit;
    }
    auto ms =
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    std::printf("real_footer pass=%d files=%zu hits=%zu ms=%.3f\n", pass, paths.size(), hits, ms);
    CHECK(hits == (pass == 0 ? 0 : paths.size()));
  }
  context.reset();
}
