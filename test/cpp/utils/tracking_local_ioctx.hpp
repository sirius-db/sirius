/* Copyright 2026, Sirius Contributors. SPDX-License-Identifier: Apache-2.0 */
#pragma once
#include <cucascade/io/byte_range.hpp>
#include <cucascade/io/io_context.hpp>
#include <utils/local_ioctx.hpp>

#include <atomic>
#include <memory>

// Real local I/O with observable reads; optionally exercises the production bulk route.
namespace sirius::test {
struct io_counts {
  std::atomic<size_t> calls{0}, bytes{0};
};
// Forwards to a started local ioctx (make_test_local_ioctx) while reporting its own capabilities:
// device reads only, no vectored or host-to-device reads, and bulk I/O when asked. Counts each
// forwarded host read and vectored slice, and the bytes they request.
class tracking_local_ioctx : public cucascade::io::ioctx {
  std::shared_ptr<cucascade::io::ioctx> inner = make_test_local_ioctx();
  bool bulk_;

 public:
  std::shared_ptr<io_counts> counts = std::make_shared<io_counts>();
  tracking_local_ioctx(bool bulk = false) : bulk_(bulk) {}
  ~tracking_local_ioctx() override { pre_destroy(); }
  cucascade::io::io_context_type type() const noexcept override { return inner->type(); }
  void shutdown() noexcept override {}
  bool prefers_bulk_io() const noexcept override { return bulk_; }
  bool supports(std::string_view p) const noexcept override { return inner->supports(p); }
  bool supports_device_read() const noexcept override { return true; }
  bool supports_host_to_device_read() const noexcept override { return false; }
  bool supports_vector_host_read() const noexcept override { return false; }
  bool supports_device_range_read() const noexcept override { return false; }
  std::vector<cucascade::io::byte_range> align_and_coalesce(
    std::span<cucascade::io::byte_range const> r, std::optional<size_t>) const noexcept override
  {
    return {r.begin(), r.end()};
  }
  size_t host_read_io(cucascade::io::io_object const& o,
                      size_t off,
                      size_t n,
                      uint8_t* dst) override
  {
    counts->calls.fetch_add(1);
    counts->bytes.fetch_add(n);
    return inner->host_read_io(o, off, n, dst);
  }
  cucascade::exec::semi_future<size_t> mixed_readv_async_io(
    cucascade::io::io_object const& o,
    std::vector<cucascade::io::prepared_io_slice>&& s) noexcept override
  {
    for (auto const& slice : s) {
      counts->calls.fetch_add(1);
      counts->bytes.fetch_add(slice.size());
    }
    return inner->mixed_readv_async_io(o, std::move(s));
  }

 protected:
  std::shared_ptr<cucascade::io::io_object> create_io_object(std::string p) override
  {
    return inner->open_io_object(std::move(p));
  }
};
}  // namespace sirius::test
