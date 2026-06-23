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

// Unit tests for io::make_gpu_ingestible — the free-function factory
// that the scan_manager calls on every prepare_for_query. The factory's
// only job is to validate non-null table_info and dispatch through the
// virtual ingestible_table_info::make_ingestible. The concrete production
// implementations (parquet_ingestible_table_info, duckdb_native_ingestible_table_info)
// are covered by their own tests; this file uses a fake subclass to
// exercise the dispatch path in isolation.

#include "catch.hpp"

#include <cudf/table/table.hpp>

#include <op/scan/gpu_ingestible.hpp>
#include <scan_manager/sirius_scan_manager.hpp>

#include <atomic>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace {

// Minimal fake gpu_ingestible — every virtual is a no-op or trivial throw.
// Sufficient to confirm make_gpu_ingestible routes through the table_info's
// virtual factory and that the resulting shared_ptr is returned to the
// caller intact.
class fake_gpu_ingestible : public sirius::io::gpu_ingestible {
 public:
  using sirius::io::gpu_ingestible::gpu_ingestible;

  [[nodiscard]] bool has_more_splits() const override { return false; }
  std::function<std::vector<std::unique_ptr<sirius::op::operator_data>>()> next_split_provider()
    override
  {
    return nullptr;
  }
  sirius::io::filtered_table materialize_table(sirius::io::scan_info const&,
                                               cucascade::memory::memory_space const&,
                                               rmm::cuda_stream_view) override
  {
    throw std::runtime_error("not reachable in this test");
  }

  std::unique_ptr<cudf::table> post_filter_and_project(
    std::unique_ptr<cudf::table>,
    filter_state state,
    sirius::io::post_filter_and_projection_info const&,
    cucascade::memory::memory_space const&,
    rmm::cuda_stream_view) override
  {
    throw std::runtime_error("not reachable in this test");
  }
};

// Fake bind data. Counts how many times make_ingestible is called so the
// test can verify the factory's dispatch is exactly once-per-call (no
// retries, no extra dispatches).
class fake_table_info : public sirius::io::ingestible_table_info {
 public:
  fake_table_info() = default;

  std::shared_ptr<sirius::io::gpu_ingestible> make_ingestible(
    std::unique_ptr<sirius::io::ingestible_table_info> self,
    sirius::scan_manager::sirius_scan_manager const&) override
  {
    dispatch_count.fetch_add(1, std::memory_order_relaxed);
    return std::make_shared<fake_gpu_ingestible>(std::move(self));
  }

  [[nodiscard]] std::span<std::string const> file_paths() const override { return {}; }

  std::atomic<int> dispatch_count{0};
};

}  // namespace

TEST_CASE("make_gpu_ingestible - throws on null table_info", "[io][gpu_ingestible]")
{
  sirius::scan_manager::sirius_scan_manager mgr({});
  REQUIRE_THROWS_AS(sirius::io::make_gpu_ingestible(/*info=*/nullptr, mgr), std::runtime_error);
}

TEST_CASE("make_gpu_ingestible - dispatches through table_info::make_ingestible",
          "[io][gpu_ingestible]")
{
  // Hold a raw pointer to the fake so we can read dispatch_count after the
  // factory moves the unique_ptr into the constructed ingestible's base.
  auto info        = std::make_unique<fake_table_info>();
  auto* info_alias = info.get();

  sirius::scan_manager::sirius_scan_manager mgr({});

  auto ingestible = sirius::io::make_gpu_ingestible(std::move(info), mgr);
  REQUIRE(ingestible != nullptr);
  REQUIRE(info_alias->dispatch_count.load() == 1);

  // The base's table_info() accessor must still resolve to the same fake
  // (the unique_ptr was moved through the factory into gpu_ingestible's
  // base and never thrown away).
  REQUIRE(&ingestible->table_info() == info_alias);
}
