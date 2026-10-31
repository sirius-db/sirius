/* Copyright 2026, Sirius Contributors. SPDX-License-Identifier: Apache-2.0 */
#pragma once
#include "io/kvikio/kvikio_context.hpp"
#include "io/sirius_datasource.hpp"

#include <atomic>

// Real local I/O with observable reads; optionally exercises the production bulk route.
namespace sirius::test {
struct io_counts {
  std::atomic<size_t> calls{0}, bytes{0};
};
class counted_object : public sirius::io::kvikio_object {
  std::shared_ptr<sirius::io::kvikio_object const> inner;
  std::shared_ptr<io_counts> counts;

 public:
  counted_object(std::shared_ptr<sirius::io::kvikio_object const> in, std::shared_ptr<io_counts> c)
    : inner(std::move(in)), counts(std::move(c))
  {
  }
  std::string const& raw_file_cache_id() const noexcept override
  {
    return inner->raw_file_cache_id();
  }
  std::string const& object_path() const noexcept override { return inner->object_path(); }
  size_t size() const noexcept override { return inner->size(); }
  sirius::io::object_identity identity() const override { return inner->identity(); }
  size_t read_at(void* dst, size_t size, size_t offset) const override
  {
    auto got = inner->read_at(dst, size, offset);
    counts->calls.fetch_add(1);
    counts->bytes.fetch_add(got);
    return got;
  }
};
class tracking_kvikio_context : public sirius::io::ioctx {
  std::shared_ptr<sirius::io::kvikio_context> inner;
  bool bulk_;

 public:
  std::shared_ptr<io_counts> counts = std::make_shared<io_counts>();
  tracking_kvikio_context(bool bulk = false) : bulk_(bulk)
  {
    sirius::io::kvikio_config config;
    config.compat_mode = kvikio::CompatMode::ON;
    inner              = std::make_shared<sirius::io::kvikio_context>(config);
  }
  ~tracking_kvikio_context() override { pre_destroy(); }
  sirius::io::io_context_type type() const noexcept override { return inner->type(); }
  void shutdown() noexcept override {}
  bool prefers_bulk_io() const noexcept override { return bulk_; }
  bool supports(std::string_view p) const noexcept override { return inner->supports(p); }
  bool supports_device_read() const noexcept override { return true; }
  bool supports_host_to_device_read() const noexcept override { return false; }
  bool supports_vector_host_read() const noexcept override { return false; }
  bool supports_device_range_read() const noexcept override { return false; }
  std::vector<cudf::io::text::byte_range_info> align_and_coalesce(
    std::span<cudf::io::text::byte_range_info const> r,
    std::optional<size_t>) const noexcept override
  {
    return {r.begin(), r.end()};
  }
  size_t host_read_io(sirius::io::io_object const& o, size_t off, size_t n, uint8_t* dst) override
  {
    return inner->host_read_io(o, off, n, dst);
  }
  sirius::exec::semi_future<size_t> mixed_readv_async_io(
    sirius::io::io_object const& o,
    std::vector<sirius::io::prepared_io_slice>&& s) noexcept override
  {
    return inner->mixed_readv_async_io(o, std::move(s));
  }

 protected:
  std::shared_ptr<sirius::io::io_object> create_io_object(std::string p) override
  {
    auto ds = inner->open_datasource(std::move(p));
    return std::make_shared<counted_object>(
      std::dynamic_pointer_cast<sirius::io::kvikio_object const>(
        ds->get_io_object().shared_from_this()),
      counts);
  }
};
}  // namespace sirius::test
