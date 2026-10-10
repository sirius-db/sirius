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
#include "io/io_request.hpp"

// One logical read may expand into many physical operations across reactors.
// grouped_coordinator exists to make that fan-out look like one future while
// stopping new dispatch on the first error and still draining published work.
// The tests below exercise those ordering and ownership requirements directly.

#include "utils/cold_file_cache.hpp"

#include <atomic>
#include <cstddef>
#include <exception>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <thread>
#include <vector>

TEST_CASE("physical reads retain their phase across workers and retries", "[physical_reads]")
{
  using namespace sirius::io;
  auto planning  = std::make_shared<physical_read_statistics>();
  auto execution = std::make_shared<physical_read_statistics>();
  std::shared_ptr<grouped_coordinator> first;
  {
    scoped_physical_reads scope(planning);
    first = std::make_shared<grouped_coordinator>(8, 1);
    {
      scoped_physical_reads nested(execution);
      CHECK(std::make_shared<grouped_coordinator>(8, 1)->physical_reads == execution);
    }
    CHECK(physical_reads_for_testing == planning);
  }
  CHECK_FALSE(physical_reads_for_testing);
  std::thread worker([&] {
    scoped_physical_reads current(execution);
    physical_read_attempt read;
    read.submitted(first->physical_reads, 8);
    read.completed(-EINTR);
    read.submitted(first->physical_reads, 8);
    read.completed(3);
    read.submitted(first->physical_reads, 5);
    read.completed(5);
    read.completed(5);  // Duplicate completion must not inflate bytes.
  });
  worker.join();
  CHECK(planning->requests == 3);
  CHECK(planning->completions == 3);
  CHECK(planning->bytes_requested == 21);
  CHECK(planning->bytes_returned == 8);
  CHECK(planning->failures == 1);
  CHECK(planning->short_reads == 1);
  CHECK(planning->retries == 2);
  CHECK(execution->requests == 0);
}

TEST_CASE("unreaped physical reads remain explicitly unobserved", "[physical_reads]")
{
  using namespace sirius::io;
  auto stats = std::make_shared<physical_read_statistics>();
  {
    physical_read_attempt pending;
    pending.submitted(stats, 4096);
    physical_read_attempt moved(std::move(pending));
  }
  CHECK(stats->requests == 1);
  CHECK(stats->completions == 0);
  CHECK(stats->bytes_returned == 0);
  CHECK(stats->unobserved_completions == 1);
}

TEST_CASE("cold cache verifies eviction and rejects invalid manifests", "[physical_reads]")
{
  // /tmp may be tmpfs, whose resident pages cannot be evicted with DONTNEED.
  char directory[] = "./sirius-cold-XXXXXX";
  REQUIRE(::mkdtemp(directory));
  auto absolute = std::filesystem::absolute(directory).string();
  auto data     = absolute + "/data";
  auto manifest = absolute + "/files";
  struct cleanup_files {
    std::string data, manifest, directory;
    ~cleanup_files()
    {
      ::unlink(data.c_str());
      ::unlink(manifest.c_str());
      ::rmdir(directory.c_str());
    }
  } cleanup{data, manifest, directory};
  {
    std::ofstream output(data);
    output << std::string(8193, 'x');
  }
  {
    std::ofstream output(manifest);
    output << data << '\n';
  }
  auto result = sirius::test::evict_file_cache(manifest);
  CHECK(result.files == 1);
  CHECK(result.bytes == 8193);
  CHECK(result.pages > 0);
  {
    auto fd = ::open(data.c_str(), O_RDONLY);
    REQUIRE(fd >= 0);
    auto* pinned = ::mmap(nullptr, 8193, PROT_READ, MAP_SHARED, fd, 0);
    ::close(fd);
    REQUIRE(pinned != MAP_FAILED);
    struct unmap {
      void* address;
      ~unmap()
      {
        ::munlock(address, 8193);
        ::munmap(address, 8193);
      }
    } cleanup_mapping{pinned};
    REQUIRE(::mlock(pinned, 8193) == 0);
    CHECK_THROWS_WITH(sirius::test::evict_file_cache(manifest),
                      "cold-cache eviction left resident pages");
  }
  {
    std::ofstream output(manifest);
  }
  CHECK_THROWS_WITH(sirius::test::evict_file_cache(manifest), "empty cold-cache file list");
  {
    std::ofstream output(manifest);
    output << "relative/path\n";
  }
  CHECK_THROWS_WITH(sirius::test::evict_file_cache(manifest),
                    "cold-cache file list requires absolute local paths");
  {
    std::ofstream output(manifest);
    output << data << "-missing\n";
  }
  CHECK_THROWS(sirius::test::evict_file_cache(manifest));
}

using sirius::io::grouped_coordinator;
using sirius::io::io_op_request;
using sirius::io::prepared_io_completion;

TEST_CASE("an empty grouped coordinator is ready with zero bytes", "[io][coordinator]")
{
  grouped_coordinator coordinator{0, 0};
  auto future = coordinator.get_future();

  CHECK(future.is_ready());
  CHECK(std::move(future).get() == 0);
}

TEST_CASE("slice expansion adds credits before children complete", "[io][coordinator]")
{
  grouped_coordinator coordinator{4096, 1};
  auto future = coordinator.get_future();

  coordinator.add_tasks(3);
  CHECK(coordinator.tasks_remaining() == 4);

  std::vector<std::thread> completions;
  completions.reserve(4);
  for (int i = 0; i < 4; ++i) {
    completions.emplace_back([&] { coordinator.on_complete(); });
  }
  for (auto& completion : completions) {
    completion.join();
  }

  CHECK(future.is_ready());
  CHECK(std::move(future).get() == 4096);
}

TEST_CASE("the first error stops dispatch but fulfillment waits for drain", "[io][coordinator]")
{
  grouped_coordinator coordinator{128, 2};
  auto future = coordinator.get_future();
  auto error  = std::make_exception_ptr(std::runtime_error("first physical read failed"));

  coordinator.report_error(error);

  CHECK_FALSE(coordinator.should_continue());
  CHECK(coordinator.has_error());
  CHECK(coordinator.tasks_remaining() == 1);
  CHECK_FALSE(future.is_ready());

  coordinator.on_complete();

  CHECK(future.is_ready());
  CHECK_THROWS_WITH(std::move(future).get(), "first physical read failed");
}

TEST_CASE("physical completion publishes cache state before the future", "[io][coordinator]")
{
  auto coordinator = std::make_shared<grouped_coordinator>(64, 1);
  auto future      = coordinator->get_future();
  std::atomic<bool> callback_ran{false};

  io_op_request operation;
  operation.coordinator = coordinator;
  operation.on_complete = std::make_shared<prepared_io_completion>(
    [&callback_ran](std::span<sirius::io::cache::cached_chunk* const>, bool success) noexcept {
      callback_ran.store(success, std::memory_order_release);
    });

  operation.finish_success();

  CHECK(callback_ran.load(std::memory_order_acquire));
  CHECK(future.is_ready());
  CHECK(std::move(future).get() == 64);
  CHECK(operation.terminal());
}

TEST_CASE("host-valid data is publishable even when its device copy fails", "[io][coordinator]")
{
  auto coordinator = std::make_shared<grouped_coordinator>(64, 1);
  auto future      = coordinator->get_future();
  std::atomic<bool> host_data_valid{false};

  io_op_request operation;
  operation.coordinator = coordinator;
  operation.on_complete = std::make_shared<prepared_io_completion>(
    [&host_data_valid](std::span<sirius::io::cache::cached_chunk* const>, bool success) noexcept {
      host_data_valid.store(success, std::memory_order_release);
    });

  operation.finish_error(std::make_exception_ptr(std::runtime_error("H2D event failed")), true);

  CHECK(host_data_valid.load(std::memory_order_acquire));
  CHECK(future.is_ready());
  CHECK_THROWS_WITH(std::move(future).get(), "H2D event failed");
}
