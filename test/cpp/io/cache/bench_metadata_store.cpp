/*
 * Copyright 2026, Sirius Contributors.
 * SPDX-License-Identifier: Apache-2.0
 */
// Hidden host-only workload. This measures store overhead, not Parquet parsing
// or TPC-H/TPC-DS performance. Run '[metadata_store_bench]' explicitly.
#include "io/cache/metadata_store.hpp"

#include <catch2/catch_test_macros.hpp>

#include <barrier>
#include <cstdio>
#include <thread>
#include <vector>

namespace {
struct benchmark_metadata : sirius::io::io_object_metadata {
  std::vector<char> bytes = std::vector<char>(4096);
};

class benchmark_object final : public sirius::io::io_object {
 public:
  benchmark_object(std::string path, sirius::io::object_identity identity)
    : path_(std::move(path)), identity_(std::move(identity))
  {
  }
  [[nodiscard]] const std::string& raw_file_cache_id() const noexcept override { return path_; }
  [[nodiscard]] const std::string& object_path() const noexcept override { return path_; }
  [[nodiscard]] size_t size() const noexcept override { return identity_.size; }
  [[nodiscard]] sirius::io::object_identity identity() const override { return identity_; }

 private:
  std::string path_;
  sirius::io::object_identity identity_;
};
}  // namespace

TEST_CASE("Metadata store synthetic 10000-file workload", "[.][metadata_store_bench]")
{
  using namespace sirius::io;
  cache::metadata_store store;
  object_identity identity{identity_kind::object_tag, 1024 * 1024, "benchmark-version"};
  std::vector<std::unique_ptr<benchmark_object>> objects;
  objects.reserve(10000);
  for (int i = 0; i < 10000; ++i) {
    objects.push_back(std::make_unique<benchmark_object>(
      "s3://benchmark-bucket/dataset/part-" + std::to_string(i) + ".parquet", identity));
  }
  auto start = std::chrono::steady_clock::now();
  for (auto const& object : objects)
    store.register_metadata(*object, std::make_shared<benchmark_metadata>());
  auto elapsed = [&] {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start)
      .count();
  };
  std::printf(
    "cold_insert files=10000 ms=%.3f entries=%zu\n", elapsed(), store.statistics().entries);
  REQUIRE(store.statistics().entries == objects.size());
  for (int workers : {1, 4, 8}) {
    for (int sample = 0; sample < 3; ++sample) {
      constexpr int operations = 100000;
      std::barrier gate(workers + 1);
      std::atomic<int> hits{0};
      std::vector<std::thread> threads;
      for (int worker = 0; worker < workers; ++worker)
        threads.emplace_back([&, worker] {
          gate.arrive_and_wait();
          int local_hits = 0;
          for (int i = worker; i < operations; i += workers) {
            auto metadata = store.get_metadata(*objects[i % objects.size()]);
            local_hits += static_cast<bool>(metadata);
          }
          hits.fetch_add(local_hits, std::memory_order_relaxed);
        });
      start = std::chrono::steady_clock::now();
      gate.arrive_and_wait();
      for (auto& thread : threads)
        thread.join();
      std::printf(
        "warm_hit workers=%d sample=%d ops=%d ms=%.3f\n", workers, sample, operations, elapsed());
      CHECK(hits == operations);
    }
  }
  CHECK(store.statistics().entries == objects.size());
}
