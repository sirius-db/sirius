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

// Test-only io_object: sirius_datasource stores a unique_ptr<sirius_io_object>
// but the capability-flag paths never touch it, so size()/cache_id() just
// return inert values.
class mock_io_object : public sirius_io_object {
 public:
  [[nodiscard]] std::string const& raw_file_cache_id() const noexcept override { return _id; }
  [[nodiscard]] size_t size() const noexcept override { return 0; }

 private:
  std::string _id{"mock"};
};

// Test-only sirius_ioctx whose device-read capability flags can be toggled
// independently. All read methods throw — the sirius_datasource forwarding
// tests only probe the two cap hooks, never the IO path.
class cap_probe_ioctx : public sirius_ioctx {
 public:
  cap_probe_ioctx(bool supports, bool preferred) : _supports(supports), _preferred(preferred) {}

  [[nodiscard]] bool supports_device_read() const override { return _supports; }
  [[nodiscard]] bool is_device_read_preferred(size_t) const override { return _preferred; }

  void shutdown() override {}

  std::unique_ptr<cudf::io::datasource> make_datasource(std::unique_ptr<sirius_io_object>) override
  {
    throw std::logic_error("cap_probe_ioctx::make_datasource: not exercised");
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
  std::future<size_t> host_read_async(sirius_io_object&, size_t, size_t, uint8_t*) override
  {
    throw std::logic_error("unused");
  }
  std::future<std::unique_ptr<cudf::io::datasource::buffer>> host_read_async(sirius_io_object&,
                                                                             size_t,
                                                                             size_t) override
  {
    throw std::logic_error("unused");
  }
  std::unique_ptr<cudf::io::datasource::buffer> device_read(sirius_io_object&,
                                                            size_t,
                                                            size_t,
                                                            rmm::cuda_stream_view) override
  {
    throw std::logic_error("unused");
  }
  size_t device_read(sirius_io_object&, size_t, size_t, uint8_t*, rmm::cuda_stream_view) override
  {
    throw std::logic_error("unused");
  }
  std::future<size_t> device_read_async(
    sirius_io_object&, size_t, size_t, uint8_t*, rmm::cuda_stream_view) override
  {
    throw std::logic_error("unused");
  }
  std::future<size_t> host_read_ranges_async(sirius_io_object&,
                                             std::vector<cudf::io::text::byte_range_info> const&,
                                             std::span<cudf::host_span<std::byte>>) override
  {
    throw std::logic_error("unused");
  }
  size_t host_read_ranges(sirius_io_object&,
                          std::vector<cudf::io::text::byte_range_info> const&,
                          std::span<cudf::host_span<std::byte>>) override
  {
    throw std::logic_error("unused");
  }

 private:
  bool _supports;
  bool _preferred;
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

TEST_CASE("uring_ioctx_does_not_prefer_device_read", "[io_caps]")
{
  auto ctx = try_make_uring_ioctx();
  if (!ctx) {
    SUCCEED("Skipping: io_uring not supported on this runner");
    return;
  }

  // supports_device_read() stays true — the bounce-buffer path can still
  // land bytes in device memory if a caller explicitly asks for it.
  CHECK(ctx->supports_device_read());

  // But is_device_read_preferred() must be false: the uring path is a pinned
  // host bounce + cudaMemcpyAsync, strictly slower than a plain host_read.
  // Flipping this to false is what PR4 is about — we do not want cuDF or any
  // other caller to silently route through the bounce path when host_read
  // would do.
  CHECK_FALSE(ctx->is_device_read_preferred(0));
  CHECK_FALSE(ctx->is_device_read_preferred(1UL << 10));
  CHECK_FALSE(ctx->is_device_read_preferred(1UL << 30));
}

TEST_CASE("sirius_datasource_forwards_caps_from_ioctx", "[io_caps]")
{
  // Each permutation of the two flags must round-trip through sirius_datasource
  // unchanged. This is the load-bearing contract for PR6/PR9/PR10: per-backend
  // ioctx decides its own caps, sirius_datasource is a pure forwarder.
  struct case_t {
    bool supports;
    bool preferred;
  };
  for (auto const c :
       {case_t{true, true}, case_t{true, false}, case_t{false, true}, case_t{false, false}}) {
    auto ctx = std::make_shared<cap_probe_ioctx>(c.supports, c.preferred);
    sirius_datasource ds{ctx, std::make_unique<mock_io_object>()};

    CHECK(ds.supports_device_read() == c.supports);
    CHECK(ds.is_device_read_preferred(0) == c.preferred);
    CHECK(ds.is_device_read_preferred(1UL << 20) == c.preferred);
  }
}
