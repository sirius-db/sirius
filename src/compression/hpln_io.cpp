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

#include "hpln_io.hpp"

#include "io/sirius_datasource.hpp"
#include "io/types.hpp"

#include <log/logging.hpp>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>

namespace sirius {

namespace {

/// True for a path that names a scheme rather than a filesystem location. Matches the test the
/// parquet source uses, so both sources route the same strings the same way.
bool has_uri_scheme(std::string const& p) { return p.find("://") != std::string::npos; }

std::string describe(std::uint64_t offset, std::uint64_t bytes)
{
  return "offset " + std::to_string(offset) + " (" + std::to_string(bytes) + " B)";
}

}  // namespace

//===----------------------------------------------------------------------===//
// planning
//===----------------------------------------------------------------------===//

std::uint64_t hpln_extent::bytes() const noexcept
{
  std::uint64_t n = 0;
  for (auto const& d : dst) {
    n += d.bytes;
  }
  return n;
}

hpln_read_plan plan_hpln_reads(std::vector<hpln_extent> extents, hpln_io_policy const& policy)
{
  hpln_read_plan plan;

  std::erase_if(extents, [](hpln_extent const& e) { return e.bytes() == 0; });
  if (extents.empty()) { return plan; }
  std::sort(extents.begin(), extents.end(), [](hpln_extent const& a, hpln_extent const& b) {
    return a.offset < b.offset;
  });

  for (std::size_t i = 0; i < extents.size(); ++i) {
    plan.wanted_bytes += extents[i].bytes();
    if (i > 0 && extents[i].offset < extents[i - 1].offset + extents[i - 1].bytes()) {
      throw std::runtime_error("[hpln io] overlapping read extents at " +
                               describe(extents[i].offset, extents[i].bytes()));
    }
  }

  // Pass 1 -- decide the runs. A run is a maximal set of consecutive extents whose gaps are worth
  // bridging and whose total stays within one request's target size. Recording the gaps first,
  // and sizing the scratch from their sum, is what lets pass 2 hand out stable pointers into it:
  // growing the scratch while filling it would dangle the destinations already issued.
  struct run {
    std::size_t first    = 0;  ///< index of the first extent
    std::size_t last     = 0;  ///< index of the last extent, inclusive
    std::uint64_t offset = 0;
    std::uint64_t bytes  = 0;
  };
  std::vector<run> runs;
  for (std::size_t i = 0; i < extents.size(); ++i) {
    auto const off = extents[i].offset;
    auto const len = extents[i].bytes();
    bool start_new = runs.empty();
    if (!start_new) {
      auto& back        = runs.back();
      auto const gap    = off - (back.offset + back.bytes);
      auto const merged = (off + len) - back.offset;
      // A run never grows past the target: a bigger read would only be split again below, and
      // splitting a bridged gap in the middle wastes the bridge.
      start_new = gap > policy.max_gap_bytes || merged > policy.target_request_bytes;
    }
    if (start_new) {
      runs.push_back({i, i, off, len});
      continue;
    }
    auto& back = runs.back();
    plan.bridged_bytes += off - (back.offset + back.bytes);
    back.last  = i;
    back.bytes = (off + len) - back.offset;
  }
  plan.scratch.assign(static_cast<std::size_t>(plan.bridged_bytes), std::uint8_t{0});

  // Pass 2 -- lay each run out as a destination list covering it contiguously, gaps included,
  // then cut the list into requests of at most target_request_bytes. Cutting is done on the
  // destination list rather than on the extents so a single huge extent is split too.
  std::size_t scratch_used = 0;
  std::vector<hpln_dst> run_dst;
  for (auto const& r : runs) {
    run_dst.clear();
    std::uint64_t cursor = r.offset;
    for (std::size_t i = r.first; i <= r.last; ++i) {
      if (extents[i].offset > cursor) {
        auto const gap = extents[i].offset - cursor;
        run_dst.push_back({plan.scratch.data() + scratch_used, gap});
        scratch_used += static_cast<std::size_t>(gap);
        cursor += gap;
      }
      for (auto const& d : extents[i].dst) {
        if (d.bytes == 0) { continue; }
        if (d.data == nullptr) {
          throw std::runtime_error("[hpln io] read extent at " + describe(extents[i].offset, 0) +
                                   " has a null destination");
        }
        run_dst.push_back(d);
        cursor += d.bytes;
      }
    }

    // Emit requests. `target_request_bytes == 0` means "do not split" (the verbatim control).
    auto const cap = policy.target_request_bytes == 0 ? std::numeric_limits<std::uint64_t>::max()
                                                      : policy.target_request_bytes;
    hpln_request req;
    req.offset = r.offset;
    for (auto piece : run_dst) {
      while (piece.bytes > 0) {
        auto const take = std::min(piece.bytes, cap - req.bytes);
        req.dst.push_back({piece.data, take});
        req.bytes += take;
        piece.data += take;
        piece.bytes -= take;
        if (req.bytes == cap) {
          auto const next_offset = req.offset + req.bytes;
          plan.requests.push_back(std::move(req));
          req        = {};
          req.offset = next_offset;
        }
      }
    }
    if (req.bytes > 0) { plan.requests.push_back(std::move(req)); }
  }
  return plan;
}

//===----------------------------------------------------------------------===//
// hpln_source
//===----------------------------------------------------------------------===//

hpln_source::~hpln_source() = default;

void hpln_source::read_extents(std::vector<hpln_extent> extents,
                               hpln_io_policy const& policy,
                               char const* what)
{
  _stats.extents += extents.size();
  auto const plan = plan_hpln_reads(std::move(extents), policy);
  if (plan.requests.empty()) { return; }
  _stats.requests += plan.requests.size();
  _stats.bytes_wanted += plan.wanted_bytes;
  for (auto const& r : plan.requests) {
    _stats.bytes_read += r.bytes;
  }
  submit(plan.requests, policy, what);
}

void hpln_source::read_at(std::uint64_t offset, std::uint64_t bytes, void* dst, char const* what)
{
  if (bytes == 0) { return; }
  hpln_extent e;
  e.offset = offset;
  e.dst.push_back({static_cast<std::uint8_t*>(dst), bytes});
  read_extents({std::move(e)}, hpln_io_policy{}, what);
}

std::vector<std::uint8_t> hpln_source::read_range(std::uint64_t offset,
                                                  std::uint64_t bytes,
                                                  char const* what)
{
  std::vector<std::uint8_t> buf(static_cast<std::size_t>(bytes));
  read_at(offset, bytes, buf.data(), what);
  return buf;
}

std::vector<std::uint8_t> hpln_source::read_tail(std::uint64_t want, char const* what)
{
  auto const n = std::min(want, size());
  return read_range(size() - n, n, what);
}

std::vector<std::uint8_t> hpln_source::read_prefix(std::uint64_t want, char const* what)
{
  return read_range(0, std::min(want, size()), what);
}

namespace {

//===----------------------------------------------------------------------===//
// filesystem transport
//===----------------------------------------------------------------------===//

/// Local reads with no io_context. Kept for the callers that have no engine around them (the pin
/// path, host unit tests): the coalescing above is pointless here, but the parsing above it is
/// the same code, which is the point.
class ifstream_hpln_source final : public hpln_source {
 public:
  ifstream_hpln_source(std::string path, char const* who) : _path(std::move(path))
  {
    namespace fs = std::filesystem;
    std::error_code ec;
    _size = static_cast<std::uint64_t>(fs::file_size(_path, ec));
    if (ec) {
      throw std::runtime_error(std::string("[") + who + "] cannot stat '" + _path +
                               "': " + ec.message());
    }
    _file.open(_path, std::ios::binary);
    if (!_file) {
      throw std::runtime_error(std::string("[") + who + "] cannot open '" + _path + "'");
    }
    _stats.transport = "ifstream";
  }

  [[nodiscard]] std::uint64_t size() const noexcept override { return _size; }
  [[nodiscard]] std::string_view transport() const noexcept override { return "ifstream"; }

  void submit(std::span<hpln_request const> requests,
              hpln_io_policy const& /*policy*/,
              char const* what) override
  {
    for (auto const& r : requests) {
      _file.seekg(static_cast<std::streamoff>(r.offset));
      for (auto const& d : r.dst) {
        _file.read(reinterpret_cast<char*>(d.data), static_cast<std::streamsize>(d.bytes));
        if (!_file) {
          throw std::runtime_error("[hpln io] '" + _path + "': short read of the " + what + " at " +
                                   describe(r.offset, r.bytes));
        }
      }
    }
  }

 private:
  std::string _path;
  std::uint64_t _size = 0;
  std::ifstream _file;
};

//===----------------------------------------------------------------------===//
// io_context transport
//===----------------------------------------------------------------------===//

char const* backend_name(io::io_context_type type)
{
  switch (type) {
    case io::io_context_type::uring: return "uring";
    case io::io_context_type::restful: return "rest";
    case io::io_context_type::kvikio: return "kvikio";
  }
  return "unknown";
}

/// Reads through a sirius io_context, which is what makes `s3://` readable and what puts the
/// ranged reads the zone maps decide on in front of an object store.
class ioctx_hpln_source final : public hpln_source {
 public:
  ioctx_hpln_source(std::string path, std::shared_ptr<io::sirius_ioctx> io_ctx, char const* who)
    : _path(std::move(path)), _io_ctx(std::move(io_ctx))
  {
    if (!_io_ctx) {
      throw std::runtime_error(std::string("[") + who + "] '" + _path + "': null io_context");
    }
    try {
      _ds = _io_ctx->open_datasource(_path);
    } catch (std::exception const& e) {
      throw std::runtime_error(std::string("[") + who + "] cannot open '" + _path +
                               "' through the " + backend_name(_io_ctx->type()) +
                               " io_context: " + e.what());
    }
    if (!_ds) {
      throw std::runtime_error(std::string("[") + who + "] cannot open '" + _path +
                               "' through the " + backend_name(_io_ctx->type()) + " io_context");
    }
    _size            = _ds->size();
    _stats.transport = std::string("io_context:") + backend_name(_io_ctx->type());
  }

  [[nodiscard]] std::uint64_t size() const noexcept override { return _size; }
  [[nodiscard]] std::string_view transport() const noexcept override { return _stats.transport; }

  /// Straight through to the datasource, which resolves the io_context's metadata store by the
  /// io_object's cache id -- the same path parquet's footer cache takes.
  [[nodiscard]] std::shared_ptr<io::sirius_io_object_metadata> metadata() const override
  {
    return _ds ? _ds->metadata() : nullptr;
  }

  bool store_metadata(std::shared_ptr<io::sirius_io_object_metadata> metadata) override
  {
    return _ds && _ds->store_metadata(std::move(metadata));
  }

  void submit(std::span<hpln_request const> requests,
              hpln_io_policy const& policy,
              char const* what) override
  {
    for (auto const& r : requests) {
      if (r.offset + r.bytes > _size) {
        // A range past the end is a metadata bug, not an I/O condition: the backends CLIP at EOF,
        // so letting it through would leave the tail of a buffer holding whatever was there.
        throw std::runtime_error("[hpln io] '" + _path + "': the " + what + " at " +
                                 describe(r.offset, r.bytes) + " runs past the end of a " +
                                 std::to_string(_size) + " B object");
      }
    }
    if (_ds->supports_vector_host_read()) {
      submit_vectored(requests, policy, what);
    } else {
      submit_one_by_one(requests, what);
    }
  }

 private:
  /// One dispatch per batch of requests, batched so no more than `max_bytes_in_flight` is
  /// outstanding. The segment list has to outlive the future -- the reactor references its iovecs
  /// until the reads are reaped -- which is why a batch is awaited before the next is built.
  void submit_vectored(std::span<hpln_request const> requests,
                       hpln_io_policy const& policy,
                       char const* what)
  {
    std::size_t i = 0;
    while (i < requests.size()) {
      std::vector<io::io_object_segment> segments;
      std::uint64_t batch_bytes = 0;
      while (i < requests.size() &&
             (segments.empty() || batch_bytes + requests[i].bytes <= policy.max_bytes_in_flight)) {
        auto const& r = requests[i];
        io::io_object_segment seg(static_cast<std::size_t>(r.offset),
                                  static_cast<std::size_t>(r.dst.front().bytes),
                                  r.dst.front().data);
        for (std::size_t d = 1; d < r.dst.size(); ++d) {
          seg.append(
            iovec{static_cast<void*>(r.dst[d].data), static_cast<std::size_t>(r.dst[d].bytes)});
        }
        batch_bytes += r.bytes;
        segments.push_back(std::move(seg));
        ++i;
      }
      auto fut              = _io_ctx->host_read_ranges_async_io(_ds->io_object(), segments);
      std::size_t const got = std::move(fut).get();
      if (got != batch_bytes) {
        throw std::runtime_error("[hpln io] '" + _path + "': short read of the " + what + ": got " +
                                 std::to_string(got) + " of " + std::to_string(batch_bytes) +
                                 " B over " + std::to_string(segments.size()) + " ranges");
      }
    }
  }

  /// Backends without a vector read (kvikio) still have to work; they simply pay per range.
  void submit_one_by_one(std::span<hpln_request const> requests, char const* what)
  {
    for (auto const& r : requests) {
      std::uint64_t at = r.offset;
      for (auto const& d : r.dst) {
        auto const got = _io_ctx->host_read_io(_ds->io_object(),
                                               static_cast<std::size_t>(at),
                                               static_cast<std::size_t>(d.bytes),
                                               d.data);
        if (got != d.bytes) {
          throw std::runtime_error("[hpln io] '" + _path + "': short read of the " + what + " at " +
                                   describe(at, d.bytes) + ": got " + std::to_string(got));
        }
        at += d.bytes;
      }
    }
  }

  std::string _path;
  std::shared_ptr<io::sirius_ioctx> _io_ctx;
  std::unique_ptr<io::sirius_datasource> _ds;
  std::uint64_t _size = 0;
};

}  // namespace

std::unique_ptr<hpln_source> open_hpln_source(std::string const& path,
                                              std::shared_ptr<io::sirius_ioctx> io_ctx,
                                              char const* who)
{
  if (io_ctx) { return std::make_unique<ioctx_hpln_source>(path, std::move(io_ctx), who); }
  if (has_uri_scheme(path)) {
    throw std::runtime_error(std::string("[") + who + "] '" + path +
                             "' names a remote object but no io_context was supplied; a scheme "
                             "path cannot be read from the local filesystem");
  }
  return std::make_unique<ifstream_hpln_source>(path, who);
}

}  // namespace sirius
