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

#include "simpatico_file_ingest.hpp"

#include <log/logging.hpp>

#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <vector>

namespace sirius {

namespace {

/// First guess at how many bytes of a .hpln are header. Wide enough that a normal table parses in
/// one read; the loop below handles the rest.
constexpr std::uint64_t kHeaderProbeBytes = 1u << 20;  // 1 MiB
/// Past this a "header" is not plausible and a malformed file is the likelier explanation than a
/// very wide schema, so stop rather than read the whole object looking for one.
constexpr std::uint64_t kHeaderProbeMax = 64u << 20;  // 64 MiB
/// Tail read that should contain the trailer and the whole postscript in one go.
constexpr std::uint64_t kTailProbeBytes = 64u << 10;  // 64 KiB

std::vector<std::uint8_t> read_prefix(std::ifstream& f, std::uint64_t want, std::uint64_t file_size)
{
  auto const n = static_cast<std::size_t>(std::min(want, file_size));
  std::vector<std::uint8_t> buf(n);
  f.seekg(0);
  f.read(reinterpret_cast<char*>(buf.data()), static_cast<std::streamsize>(n));
  if (!f) { throw std::runtime_error("[hpln ingest] short read while reading the header"); }
  return buf;
}

std::vector<std::uint8_t> read_tail(std::ifstream& f, std::uint64_t want, std::uint64_t file_size)
{
  auto const n = static_cast<std::size_t>(std::min(want, file_size));
  std::vector<std::uint8_t> buf(n);
  f.seekg(static_cast<std::streamoff>(file_size - n));
  f.read(reinterpret_cast<char*>(buf.data()), static_cast<std::streamsize>(n));
  if (!f) { throw std::runtime_error("[hpln ingest] short read while reading the trailer"); }
  return buf;
}

}  // namespace

ingested_hpln read_hpln_into_pinned(std::string const& path,
                                    cucascade::memory::memory_space& host_space)
{
  namespace fs = std::filesystem;
  std::error_code ec;
  auto const file_size = static_cast<std::uint64_t>(fs::file_size(path, ec));
  if (ec) { throw std::runtime_error("[hpln ingest] cannot stat '" + path + "': " + ec.message()); }

  std::ifstream f(path, std::ios::binary);
  if (!f) { throw std::runtime_error("[hpln ingest] cannot open '" + path + "'"); }

  // Preferred path: the trailer says exactly where the header is, so one tail read locates
  // everything. Falls back to growing a speculative prefix for pre-v7 files, which is what the
  // trailer exists to avoid -- over a network the guess costs a round trip every time it is short.
  ingested_hpln out;
  std::vector<std::uint8_t> prefix;
  std::uint64_t header_at = 0;
  bool located            = false;
  {
    std::vector<simpatico::hpln_segment_ref> segs;
    std::uint64_t need = 0;
    auto tail          = read_tail(f, kTailProbeBytes, file_size);
    auto err           = simpatico::read_hpln_postscript(tail, file_size, segs, &need);
    if (!err.empty() && need > tail.size() && need <= file_size) {
      tail = read_tail(f, need, file_size);  // exact re-read, never a second guess
      err  = simpatico::read_hpln_postscript(tail, file_size, segs, nullptr);
    }
    if (err.empty()) {
      for (auto const& sg : segs) {
        if (sg.kind != simpatico::hpln_segment::header) { continue; }
        prefix.resize(static_cast<std::size_t>(sg.bytes));
        f.seekg(static_cast<std::streamoff>(sg.offset));
        f.read(reinterpret_cast<char*>(prefix.data()), static_cast<std::streamsize>(sg.bytes));
        if (!f) { throw std::runtime_error("[hpln ingest] short read of the header segment"); }
        header_at = sg.offset;
        located   = true;
        break;
      }
    }
  }
  if (located) {
    auto const err = simpatico::describe_compressed_table_header(prefix, out.schema);
    if (!err.empty()) {
      throw std::runtime_error("[hpln ingest] '" + path +
                               "' header segment does not parse: " + err);
    }
  } else {
    std::string err;
    for (std::uint64_t want = kHeaderProbeBytes;; want *= 2) {
      prefix = read_prefix(f, want, file_size);
      err    = simpatico::describe_compressed_table_header(prefix, out.schema);
      if (err.empty()) { break; }
      if (prefix.size() >= file_size || want >= kHeaderProbeMax) {
        throw std::runtime_error("[hpln ingest] '" + path + "' is not a readable .hpln: " + err);
      }
    }
  }

  auto const header_bytes  = out.schema.header_bytes;
  auto const payload_bytes = out.schema.payload_bytes;
  if (header_at + header_bytes + payload_bytes > file_size) {
    throw std::runtime_error("[hpln ingest] '" + path + "' is truncated: header says " +
                             std::to_string(header_at + header_bytes + payload_bytes) +
                             " bytes, file has " + std::to_string(file_size));
  }

  auto* host_mr = host_space.get_memory_resource_of<cucascade::memory::Tier::HOST>();
  if (host_mr == nullptr) {
    throw std::runtime_error("[hpln ingest] target host space has no host memory resource");
  }

  out.blob = std::make_shared<pinned_compressed_blob>();
  out.blob->header.assign(prefix.begin(),
                          prefix.begin() + static_cast<std::ptrdiff_t>(header_bytes));
  auto payload_res        = host_space.make_reservation_or_null(payload_bytes);
  out.blob->payload       = host_mr->allocate_multiple_blocks(payload_bytes, payload_res.get());
  out.blob->payload_bytes = payload_bytes;

  // Straight into the pinned blocks, one block at a time -- the allocation is not contiguous, and
  // going through an intermediate host buffer would double both the copy and the footprint.
  f.seekg(static_cast<std::streamoff>(header_at + header_bytes));
  auto const block   = out.blob->payload->block_size();
  std::uint64_t done = 0;
  std::size_t idx    = 0;
  while (done < payload_bytes) {
    auto const n = static_cast<std::size_t>(std::min<std::uint64_t>(block, payload_bytes - done));
    f.read(reinterpret_cast<char*>(out.blob->payload->at(idx).data()),
           static_cast<std::streamsize>(n));
    if (!f) {
      throw std::runtime_error("[hpln ingest] short read at payload offset " +
                               std::to_string(done));
    }
    done += n;
    ++idx;
  }

  SIRIUS_LOG_DEBUG("[hpln ingest] '{}': {} columns, {} rows, header {} B, payload {} B",
                   path,
                   out.schema.columns.size(),
                   out.schema.columns.empty() ? 0 : out.schema.columns.front().num_rows,
                   header_bytes,
                   payload_bytes);
  return out;
}

}  // namespace sirius
