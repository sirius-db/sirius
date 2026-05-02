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
#include "io/sirius_datasource.hpp"
#include "io/types.hpp"
#include "io/uring/uring_ioctx.hpp"

#include <cudf/io/datasource.hpp>
#include <cudf/io/text/byte_range_info.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <cstddef>
#include <cstdint>
#include <exception>
#include <future>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

using sirius::io::io_datasource;
using sirius::io::sirius_datasource;
using sirius::io::sirius_io_object;
using sirius::io::sirius_ioctx;
using sirius::io::uring_ioctx;

namespace {

// Test-only io_object: these tests only exercise sirius_datasource surface
// flags, so the backing object can be inert.
class mock_io_object : public sirius_io_object {
 public:
  [[nodiscard]] std::string const& raw_file_cache_id() const noexcept override { return _id; }
  [[nodiscard]] size_t size() const noexcept override { return 0; }

 private:
  std::string _id{"mock"};
};

// Minimal test-only sirius_ioctx implementing the new IO-framework contract.
// All read entry points throw because these tests only care about the
// sirius_datasource capability surface, not the read path itself.
class probe_ioctx : public sirius_ioctx {
 public:
  void shutdown() override {}

  std::unique_ptr<cudf::io::datasource> make_datasource(std::shared_ptr<sirius_io_object>) override
  {
    throw std::logic_error("probe_ioctx::make_datasource: not exercised");
  }

  size_t host_read(sirius_io_object&, size_t, size_t, uint8_t*) override
  {
    throw std::logic_error("unused");
  }
  std::unique_ptr<cudf::io::datasource::buffer> host_read(sirius_io_object&,
                                                          size_t,
                                                          size_t) override
  {
    throw std::logic_error("unused");
  }
  void host_read_async(
    sirius_io_object&, size_t, size_t, uint8_t*, sirius::io::io_completion_handler) override
  {
    throw std::logic_error("unused");
  }
  std::unique_ptr<cudf::io::datasource::buffer> device_read_io(sirius_io_object&,
                                                               size_t,
                                                               size_t,
                                                               rmm::cuda_stream_view) override
  {
    throw std::logic_error("unused");
  }
  size_t device_read_io(sirius_io_object&, size_t, size_t, uint8_t*, rmm::cuda_stream_view) override
  {
    throw std::logic_error("unused");
  }
  void device_read_io_async(sirius_io_object&,
                            size_t,
                            size_t,
                            uint8_t*,
                            rmm::cuda_stream_view,
                            sirius::io::io_completion_handler) override
  {
    throw std::logic_error("unused");
  }
  void host_read_ranges_async(sirius_io_object&,
                              std::vector<cudf::io::text::byte_range_info> const&,
                              std::span<cudf::host_span<std::byte>>,
                              sirius::io::io_completion_handler) override
  {
    throw std::logic_error("unused");
  }
  size_t host_read_ranges(sirius_io_object&,
                          std::vector<cudf::io::text::byte_range_info> const&,
                          std::span<cudf::host_span<std::byte>>) override
  {
    throw std::logic_error("unused");
  }
  cudf::io::text::byte_range_info compute_physical_range(cudf::io::text::byte_range_info logical,
                                                         size_t) const override
  {
    return logical;
  }
};

std::shared_ptr<uring_ioctx> try_make_uring_ioctx()
{
  try {
    return std::make_shared<uring_ioctx>(/*host_ring_depth=*/2,
                                         /*ring_entries=*/8,
                                         /*n_reactors=*/1,
                                         /*bounce_slot_size=*/1UL << 20);
  } catch (std::exception const& e) {
    WARN("uring_ioctx construction failed: " << e.what());
    return nullptr;
  }
}

}  // namespace

TEST_CASE("uring-backed sirius_datasource advertises device reads", "[io_caps]")
{
  auto ctx = try_make_uring_ioctx();
  if (!ctx) {
    SUCCEED("Skipping: io_uring not supported on this runner");
    return;
  }

  sirius_datasource ds{ctx, std::make_shared<mock_io_object>()};

  // In the new IO framework, sirius_datasource always exposes a device-read
  // path. Backends may use direct device IO or a host-bounce implementation,
  // but callers no longer branch on per-ioctx capability flags.
  CHECK(ds.supports_device_read());
  CHECK(ds.is_device_read_preferred(0));
  CHECK(ds.is_device_read_preferred(1UL << 10));
  CHECK(ds.is_device_read_preferred(1UL << 30));
}

TEST_CASE("sirius_datasource device-read flags are backend-agnostic", "[io_caps]")
{
  auto ctx = std::make_shared<probe_ioctx>();
  sirius_datasource ds{ctx, std::make_shared<mock_io_object>()};

  CHECK(ds.io_ctx().get() == ctx.get());
  CHECK(ds.supports_device_read());
  CHECK(ds.is_device_read_preferred(0));
  CHECK(ds.is_device_read_preferred(1UL << 20));
}
