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

#include <cudf/io/text/byte_range_info.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <cuda_runtime_api.h>

#include <catch.hpp>
#include <io/rest/rest_reactor.hpp>
#include <io/rest/types.hpp>
#include <io/types.hpp>

#include <array>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <vector>

using cudf::io::text::byte_range_info;
using sirius::io::device_cpy_request;
using sirius::io::io_device_range;
using sirius::io::io_host_device_range;
using sirius::io::io_object_segment;
using sirius::io::rest::rest_chunked_rx_request;
using sirius::io::rest::rest_io_object;
using sirius::io::rest::rest_reactor;

namespace {

// Non-null buffer base for segments; the pure prep/coalesce logic never
// dereferences it.
uint8_t* fake_ptr(uintptr_t v) { return reinterpret_cast<uint8_t*>(v); }

// Drive a prep result to clean completion so the shared request_manager
// destructor's invariants (bytes_read >= bytes_requested, chunks_completed ==
// total_chunks) are satisfied — one chunk_complete per emitted chunk.
void complete(std::vector<std::unique_ptr<rest_chunked_rx_request>>& chunks)
{
  for (auto& c : chunks) {
    c->manager->chunk_complete(c->chunk.size);
  }
}

std::vector<byte_range_info> coalesce(std::vector<byte_range_info> ranges,
                                      std::optional<size_t> alignment = std::nullopt)
{
  return rest_reactor::align_and_coalesce(
    std::span<const byte_range_info>(ranges.data(), ranges.size()), alignment);
}

}  // namespace

TEST_CASE("rest_reactor::supports only accepts s3 URLs", "[rest]")
{
  CHECK(rest_reactor::supports("s3://bucket/key"));
  CHECK(rest_reactor::supports("s3://bucket/path/to/obj.parquet"));
  CHECK_FALSE(rest_reactor::supports("file:///tmp/x"));
  CHECK_FALSE(rest_reactor::supports("https://host/obj"));
  CHECK_FALSE(rest_reactor::supports("/local/abs/path"));
  CHECK_FALSE(rest_reactor::supports("not a uri"));
}

TEST_CASE("align_and_coalesce coalesces without alignment by default", "[rest]")
{
  SECTION("empty input") { CHECK(coalesce({}).empty()); }
  SECTION("zero-size ranges dropped")
  {
    auto out = coalesce({byte_range_info{100, 0}});
    CHECK(out.empty());
  }
  SECTION("disjoint ranges stay separate and sorted")
  {
    auto out = coalesce({byte_range_info{200, 50}, byte_range_info{0, 50}});
    REQUIRE(out.size() == 2);
    CHECK(out[0].offset() == 0);
    CHECK(out[0].size() == 50);
    CHECK(out[1].offset() == 200);
    CHECK(out[1].size() == 50);
  }
  SECTION("overlapping ranges merge")
  {
    auto out = coalesce({byte_range_info{0, 100}, byte_range_info{50, 100}});
    REQUIRE(out.size() == 1);
    CHECK(out[0].offset() == 0);
    CHECK(out[0].size() == 150);
  }
  SECTION("adjacent ranges merge")
  {
    auto out = coalesce({byte_range_info{0, 100}, byte_range_info{100, 100}});
    REQUIRE(out.size() == 1);
    CHECK(out[0].offset() == 0);
    CHECK(out[0].size() == 200);
  }
}

TEST_CASE("align_and_coalesce honors a caller alignment as a lower bound", "[rest]")
{
  // align=4096: [100,200) -> [0,4096); [9000,9100) -> [8192,12288).  The two
  // rounded ranges leave a gap (4096..8192), so they stay separate.
  auto out = coalesce({byte_range_info{100, 100}, byte_range_info{9000, 100}}, 4096);
  REQUIRE(out.size() == 2);
  CHECK(out[0].offset() == 0);
  CHECK(out[0].size() == 4096);
  CHECK(out[1].offset() == 8192);
  CHECK(out[1].size() == 4096);

  // After rounding, [100,200) and [3000,3100) both land in [0,4096) and merge.
  auto merged = coalesce({byte_range_info{100, 100}, byte_range_info{3000, 100}}, 4096);
  REQUIRE(merged.size() == 1);
  CHECK(merged[0].offset() == 0);
  CHECK(merged[0].size() == 4096);
}

TEST_CASE("prep_host_rx_request builds a single chunk for the segment", "[rest]")
{
  sirius::io::rest::config cfg;  // pure primitives; shared services live on the context
  rest_io_object const file("s3://bkt/key", "bkt", "key", /*size=*/1 << 20);

  SECTION("non-empty segment")
  {
    auto req = rest_reactor::prep_host_rx_request(
      cfg, file, io_object_segment{4096, 8192, fake_ptr(0x1000)});
    REQUIRE(req->size() == 1);
    auto chunks = req->get_all_chunks();
    REQUIRE(chunks.size() == 1);
    CHECK(chunks[0]->object.bucket == "bkt");
    CHECK(chunks[0]->object.key == "key");
    CHECK(chunks[0]->chunk.offset == 4096);
    CHECK(chunks[0]->chunk.size == 8192);
    CHECK_FALSE(chunks[0]->is_device());
  }
  SECTION("zero-size segment yields no chunks")
  {
    auto req =
      rest_reactor::prep_host_rx_request(cfg, file, io_object_segment{0, 0, fake_ptr(0x1)});
    CHECK(req->size() == 0);
  }
}

TEST_CASE("prep_host_rxv_request builds one chunk per non-empty segment", "[rest]")
{
  sirius::io::rest::config cfg;
  rest_io_object const file("s3://bkt/key", "bkt", "key", /*size=*/10000);

  SECTION("three in-range segments, all within merge_max_gap of each other")
  {
    std::vector<io_object_segment> segs{io_object_segment{0, 100, fake_ptr(0x1)},
                                        io_object_segment{500, 100, fake_ptr(0x2)},
                                        io_object_segment{9000, 100, fake_ptr(0x3)}};
    auto req = rest_reactor::prep_host_rxv_request(cfg, file, segs);
    REQUIRE(req->size() == 1);  // the 400B and 8400B gaps are both bridged
    auto chunks = req->get_all_chunks();
    REQUIRE(chunks.size() == 1);
    CHECK(chunks[0]->object.bucket == "bkt");
    CHECK_FALSE(chunks[0]->is_device());
    CHECK(chunks[0]->chunk.offset == 0);
    CHECK(chunks[0]->chunk.size == 9100);
    CHECK(chunks[0]->chunk.n_chunks() == 5);  // real, hole, real, hole, real
  }
  SECTION("three in-range segments with merging off")
  {
    cfg.merge_max_gap = 0;
    std::vector<io_object_segment> segs{io_object_segment{0, 100, fake_ptr(0x1)},
                                        io_object_segment{500, 100, fake_ptr(0x2)},
                                        io_object_segment{9000, 100, fake_ptr(0x3)}};
    auto req = rest_reactor::prep_host_rxv_request(cfg, file, segs);
    REQUIRE(req->size() == 3);
    auto chunks = req->get_all_chunks();
    REQUIRE(chunks.size() == 3);
    for (auto const& c : chunks) {
      CHECK(c->object.bucket == "bkt");
      CHECK_FALSE(c->is_device());
    }
  }
  SECTION("segment past EOF is clamped away")
  {
    std::vector<io_object_segment> segs{io_object_segment{0, 100, fake_ptr(0x1)},
                                        io_object_segment{20000, 100, fake_ptr(0x2)}};
    auto req = rest_reactor::prep_host_rxv_request(cfg, file, segs);
    REQUIRE(req->size() == 1);  // the past-EOF segment contributes nothing
  }
  SECTION("segment straddling EOF is clamped to the file end")
  {
    std::vector<io_object_segment> segs{io_object_segment{9900, 1000, fake_ptr(0x1)}};
    auto req = rest_reactor::prep_host_rxv_request(cfg, file, segs);
    REQUIRE(req->size() == 1);
    auto chunks = req->get_all_chunks();
    REQUIRE(chunks.size() == 1);
    CHECK(chunks[0]->chunk.offset == 9900);
    CHECK(chunks[0]->chunk.size == 100);  // clamped from 1000 to 10000-9900
  }
  SECTION("empty segment list yields no chunks")
  {
    std::vector<io_object_segment> segs;
    auto req = rest_reactor::prep_host_rxv_request(cfg, file, segs);
    CHECK(req->size() == 0);
  }
}

TEST_CASE("prep_host_rx_request splits at the max_chunk_size floor", "[rest]")
{
  constexpr size_t kMiB     = 1UL << 20;
  constexpr uintptr_t kBase = 0x10000;
  rest_io_object const file("s3://bkt/key", "bkt", "key", /*size=*/2UL << 30);

  SECTION("a read below the floor stays a single GET")
  {
    sirius::io::rest::config cfg;
    auto req = rest_reactor::prep_host_rx_request(
      cfg, file, io_object_segment{0, 15 * kMiB, fake_ptr(kBase)});
    CHECK(req->size() == 1);
  }

  SECTION("a read under twice the floor stays a single GET")
  {
    // The floor divides with truncation, so 31 MiB is one 31 MiB GET rather than
    // two 15.5 MiB ones — splitting here would buy nothing but a second round
    // trip.
    sirius::io::rest::config cfg;
    auto req = rest_reactor::prep_host_rx_request(
      cfg, file, io_object_segment{0, 31 * kMiB, fake_ptr(kBase)});
    REQUIRE(req->size() == 1);
    CHECK(req->get_all_chunks().front()->chunk.size == 31 * kMiB);
  }

  SECTION("a read past twice the floor splits in two, balanced")
  {
    sirius::io::rest::config cfg;
    auto req = rest_reactor::prep_host_rx_request(
      cfg, file, io_object_segment{0, 33 * kMiB, fake_ptr(kBase)});
    REQUIRE(req->size() == 2);
    auto chunks = req->get_all_chunks();
    // 33 MiB over two requests: both above the 16 MiB floor, and the bytes are
    // balanced rather than left as a full chunk plus a 1 MiB straggler.
    CHECK(chunks.front()->chunk.size == 16 * kMiB + kMiB / 2);
    CHECK(chunks.back()->chunk.size == 16 * kMiB + kMiB / 2);
  }

  SECTION("one request is created per floor-sized piece")
  {
    sirius::io::rest::config cfg;
    auto req = rest_reactor::prep_host_rx_request(
      cfg, file, io_object_segment{0, 64 * kMiB, fake_ptr(kBase)});
    REQUIRE(req->size() == 4);
    auto chunks  = req->get_all_chunks();
    size_t total = 0;
    size_t pos   = 0;
    for (auto const& c : chunks) {
      CHECK(c->chunk.offset == pos);
      CHECK(c->chunk.size == 16 * kMiB);
      CHECK(c->chunk.n_chunks() == 1);  // each piece is a plain single-buffer GET
      CHECK(reinterpret_cast<uintptr_t>(c->chunk.data()) == kBase + pos);
      pos += c->chunk.size;
      total += c->chunk.size;
    }
    CHECK(total == 64 * kMiB);  // pieces cover the whole range, contiguously
  }

  SECTION("the fan-out is not capped: a large read keeps floor-sized chunks")
  {
    sirius::io::rest::config cfg;
    auto req = rest_reactor::prep_host_rx_request(
      cfg, file, io_object_segment{0, 1024 * kMiB, fake_ptr(kBase)});
    REQUIRE(req->size() == 64);  // 1 GiB / 16 MiB — chunks stay at the floor
    auto chunks = req->get_all_chunks();
    for (auto const& c : chunks) {
      CHECK(c->chunk.size == 16 * kMiB);
    }
  }

  SECTION("an uneven read spreads its remainder one byte at a time")
  {
    sirius::io::rest::config cfg;
    size_t const size = 512 * kMiB + 3;
    auto req =
      rest_reactor::prep_host_rx_request(cfg, file, io_object_segment{1000, size, fake_ptr(kBase)});
    REQUIRE(req->size() == 32);
    auto chunks  = req->get_all_chunks();
    size_t total = 0;
    size_t pos   = 0;
    for (auto const& c : chunks) {
      CHECK(c->chunk.offset == 1000 + pos);
      CHECK(reinterpret_cast<uintptr_t>(c->chunk.data()) == kBase + pos);
      pos += c->chunk.size;
      total += c->chunk.size;
    }
    CHECK(total == size);                                // every byte covered exactly once
    CHECK(chunks.front()->chunk.size == 16 * kMiB + 1);  // leading remainder byte
    CHECK(chunks.back()->chunk.size == 16 * kMiB);       // chunks differ by at most one byte
  }

  SECTION("max_chunk_size sets the floor")
  {
    sirius::io::rest::config cfg;
    cfg.max_chunk_size = 4 * kMiB;
    auto req = rest_reactor::prep_host_rx_request(
      cfg, file, io_object_segment{0, 64 * kMiB, fake_ptr(kBase)});
    REQUIRE(req->size() == 16);
    CHECK(req->get_all_chunks().front()->chunk.size == 4 * kMiB);
  }
}

TEST_CASE("prep_host_rxv_request coalesces near segments into a scatter GET", "[rest]")
{
  sirius::io::rest::config cfg;
  rest_io_object const file("s3://bkt/key", "bkt", "key", /*size=*/1 << 20);

  SECTION("three contiguous segments, separate buffers -> one multi-buffer chunk")
  {
    std::vector<io_object_segment> segs{io_object_segment{0, 100, fake_ptr(0xA00)},
                                        io_object_segment{100, 100, fake_ptr(0xB00)},
                                        io_object_segment{200, 100, fake_ptr(0xC00)}};
    auto req = rest_reactor::prep_host_rxv_request(cfg, file, segs);
    REQUIRE(req->size() == 1);
    auto chunks = req->get_all_chunks();
    REQUIRE(chunks.size() == 1);
    CHECK(chunks[0]->chunk.offset == 0);
    CHECK(chunks[0]->chunk.size == 300);      // one contiguous range
    CHECK(chunks[0]->chunk.n_chunks() == 3);  // scattered across 3 buffers
    CHECK(chunks[0]->chunk.is_vectored());
  }

  SECTION("a gap under merge_max_gap is bridged by a hole buffer")
  {
    // [0,100) and [500,600) are 400 bytes apart: one GET covers [0,600) and the
    // bridged 400 bytes land in a null (hole) buffer that write_to_sink drops.
    std::vector<io_object_segment> segs{io_object_segment{0, 100, fake_ptr(0xA00)},
                                        io_object_segment{500, 100, fake_ptr(0xB00)}};
    auto req = rest_reactor::prep_host_rxv_request(cfg, file, segs);
    REQUIRE(req->size() == 1);
    auto chunks = req->get_all_chunks();
    auto const& c = chunks[0]->chunk;
    CHECK(c.offset == 0);
    CHECK(c.size == 600);        // the GET spans the gap
    CHECK(c.n_chunks() == 3);    // real, hole, real
    CHECK(c.buffers[0].iov_base == fake_ptr(0xA00));
    CHECK(c.buffers[0].iov_len == 100);
    CHECK(c.buffers[1].iov_base == nullptr);  // the hole
    CHECK(c.buffers[1].iov_len == 400);
    CHECK(c.buffers[2].iov_base == fake_ptr(0xB00));
    CHECK(c.buffers[2].iov_len == 100);
    // Only the caller's bytes are reported; the bridged ones are not delivered.
    CHECK(chunks[0]->manager->bytes_requested == 200);
  }

  SECTION("a gap over merge_max_gap stays two GETs")
  {
    cfg.merge_max_gap = 100;
    std::vector<io_object_segment> segs{io_object_segment{0, 100, fake_ptr(0xA00)},
                                        io_object_segment{500, 100, fake_ptr(0xB00)}};
    auto req = rest_reactor::prep_host_rxv_request(cfg, file, segs);
    REQUIRE(req->size() == 2);
    auto chunks = req->get_all_chunks();
    CHECK(chunks[0]->chunk.size == 100);
    CHECK(chunks[1]->chunk.offset == 500);
    CHECK(chunks[1]->chunk.size == 100);
  }

  SECTION("merge_max_gap of zero fuses only genuinely adjacent segments")
  {
    cfg.merge_max_gap = 0;
    std::vector<io_object_segment> segs{io_object_segment{0, 100, fake_ptr(0xA00)},
                                        io_object_segment{100, 100, fake_ptr(0xB00)},
                                        io_object_segment{201, 100, fake_ptr(0xC00)}};
    auto req = rest_reactor::prep_host_rxv_request(cfg, file, segs);
    REQUIRE(req->size() == 2);
    auto chunks = req->get_all_chunks();
    CHECK(chunks[0]->chunk.size == 200);
    CHECK(chunks[0]->chunk.n_chunks() == 2);
    CHECK(chunks[1]->chunk.offset == 201);
  }

  SECTION("a coalesced span past twice the floor is cut on a buffer boundary")
  {
    cfg.max_chunk_size = 100;
    std::vector<io_object_segment> segs{io_object_segment{0, 100, fake_ptr(0xA00)},
                                        io_object_segment{100, 100, fake_ptr(0xB00)},
                                        io_object_segment{200, 100, fake_ptr(0xC00)}};
    auto req = rest_reactor::prep_host_rxv_request(cfg, file, segs);
    REQUIRE(req->size() == 3);  // 300 / 100 -> three chunks, one buffer each
    auto chunks = req->get_all_chunks();
    std::array<uintptr_t, 3> const bases{0xA00, 0xB00, 0xC00};
    for (size_t i = 0; i < 3; ++i) {
      CHECK(chunks[i]->chunk.offset == i * 100);
      CHECK(chunks[i]->chunk.size == 100);
      CHECK(chunks[i]->chunk.n_chunks() == 1);
      CHECK(reinterpret_cast<uintptr_t>(chunks[i]->chunk.data()) == bases[i]);
    }
  }

  SECTION("a cut falling inside a buffer divides it between two chunks")
  {
    cfg.max_chunk_size = 150;
    std::vector<io_object_segment> segs{io_object_segment{0, 100, fake_ptr(0xA00)},
                                        io_object_segment{100, 100, fake_ptr(0xB00)},
                                        io_object_segment{200, 100, fake_ptr(0xC00)}};
    auto req = rest_reactor::prep_host_rxv_request(cfg, file, segs);
    REQUIRE(req->size() == 2);  // 300 / 150 -> two 150-byte chunks
    auto chunks = req->get_all_chunks();

    // [0,150): all of buffer A, then the first half of buffer B.
    CHECK(chunks[0]->chunk.offset == 0);
    CHECK(chunks[0]->chunk.size == 150);
    REQUIRE(chunks[0]->chunk.n_chunks() == 2);
    CHECK(chunks[0]->chunk.buffers[0].iov_base == fake_ptr(0xA00));
    CHECK(chunks[0]->chunk.buffers[0].iov_len == 100);
    CHECK(chunks[0]->chunk.buffers[1].iov_base == fake_ptr(0xB00));
    CHECK(chunks[0]->chunk.buffers[1].iov_len == 50);

    // [150,300): the second half of buffer B, then all of buffer C.
    CHECK(chunks[1]->chunk.offset == 150);
    CHECK(chunks[1]->chunk.size == 150);
    REQUIRE(chunks[1]->chunk.n_chunks() == 2);
    CHECK(chunks[1]->chunk.buffers[0].iov_base == fake_ptr(0xB00 + 50));
    CHECK(chunks[1]->chunk.buffers[0].iov_len == 50);
    CHECK(chunks[1]->chunk.buffers[1].iov_base == fake_ptr(0xC00));
    CHECK(chunks[1]->chunk.buffers[1].iov_len == 100);
  }

  SECTION("a single oversized segment is cut at the floor")
  {
    cfg.max_chunk_size = 4096;
    std::vector<io_object_segment> segs{io_object_segment{0, 3 * 4096, fake_ptr(0xA00)}};
    auto req = rest_reactor::prep_host_rxv_request(cfg, file, segs);
    REQUIRE(req->size() == 3);  // 3 x 4096 / 4096 -> three floor-sized chunks
    auto chunks = req->get_all_chunks();
    for (size_t i = 0; i < 3; ++i) {
      CHECK(chunks[i]->chunk.offset == i * 4096);
      CHECK(chunks[i]->chunk.size == 4096);
      CHECK(reinterpret_cast<uintptr_t>(chunks[i]->chunk.data()) == 0xA00 + i * 4096);
    }
  }
}

TEST_CASE("rx_request splits batches by bytes rather than chunk count", "[rest]")
{
  sirius::io::rest::config cfg;
  cfg.merge_max_gap = 0;  // keep one chunk per segment so the split is what is under test
  rest_io_object const file("s3://bkt/key", "bkt", "key", /*size=*/1 << 20);

  // Four small reads and one large one, 800 bytes in total.  A chunk-count
  // split would put three chunks (300 bytes) against two (500); the byte split
  // puts four chunks (400) against one (400) — deliberately lopsided in count.
  std::vector<io_object_segment> segs{io_object_segment{0, 100, fake_ptr(0xA00)},
                                      io_object_segment{200, 100, fake_ptr(0xB00)},
                                      io_object_segment{400, 100, fake_ptr(0xC00)},
                                      io_object_segment{600, 100, fake_ptr(0xD00)},
                                      io_object_segment{800, 400, fake_ptr(0xE00)}};
  auto req = rest_reactor::prep_host_rxv_request(cfg, file, segs);
  REQUIRE(req->size() == 5);

  auto parts = sirius::io::rest::rest_rx_request::splits(std::move(req), 2);
  REQUIRE(parts.size() == 2);

  std::array<size_t, 2> bytes{0, 0};
  std::array<size_t, 2> counts{0, 0};
  for (size_t p = 0; p < 2; ++p) {
    auto chunks = parts[p]->get_all_chunks();
    counts[p]   = chunks.size();
    for (auto const& c : chunks) {
      bytes[p] += c->chunk.size;
    }
    complete(chunks);
  }

  CHECK(bytes[0] == 400);
  CHECK(bytes[1] == 400);
  CHECK(counts[0] == 4);
  CHECK(counts[1] == 1);

}

TEST_CASE("rx_request splits never produces an empty batch", "[rest]")
{
  sirius::io::rest::config cfg;
  cfg.merge_max_gap = 0;
  rest_io_object const file("s3://bkt/key", "bkt", "key", /*size=*/1 << 20);

  // Fewer chunks than splits: the result is short, not padded with empties (an
  // empty batch would enqueue a request that can never complete its manager).
  std::vector<io_object_segment> segs{io_object_segment{0, 100, fake_ptr(0xA00)},
                                      io_object_segment{200, 100, fake_ptr(0xB00)}};
  auto req   = rest_reactor::prep_host_rxv_request(cfg, file, segs);
  auto parts = sirius::io::rest::rest_rx_request::splits(std::move(req), 8);
  REQUIRE(parts.size() == 2);
  for (auto& p : parts) {
    auto chunks = p->get_all_chunks();
    CHECK(chunks.size() == 1);
    complete(chunks);
  }
}

TEST_CASE("prep_host_to_device fuses contiguous segments into a multi-copy chunk", "[rest]")
{
  sirius::io::rest::config cfg;  // default max_chunk_size (16 MiB) / merge_max_gap (0.5 MiB)
  rest_io_object const file("s3://bkt/key", "bkt", "key", /*size=*/1 << 20);
  constexpr uintptr_t kDst = 0x100000;
  constexpr uintptr_t kB0 = 0xA000, kB1 = 0xB000, kB2 = 0xC000;

  SECTION("three contiguous buffers, full overlap -> one chunk, three copies")
  {
    std::vector<io_object_segment> segs{io_object_segment{0, 100, fake_ptr(kB0)},
                                        io_object_segment{100, 100, fake_ptr(kB1)},
                                        io_object_segment{200, 100, fake_ptr(kB2)}};
    auto req = rest_reactor::prep_host_to_device_rx_request(
      cfg, file, segs, fake_ptr(kDst), /*offset=*/0, /*size=*/300, rmm::cuda_stream_view{}, 0);
    REQUIRE(req->size() == 1);
    auto chunks = req->get_all_chunks();
    REQUIRE(chunks.size() == 1);
    auto const& c = *chunks[0];
    CHECK(c.is_device());
    CHECK(c.chunk.offset == 0);
    CHECK(c.chunk.size == 300);      // one contiguous scatter GET
    CHECK(c.chunk.n_chunks() == 3);  // landing across three buffers
    REQUIRE(c.cpy_req != nullptr);
    REQUIRE(c.cpy_req->copies.size() == 3);  // one H2D copy per buffer
    std::array<uintptr_t, 3> const bufs{kB0, kB1, kB2};
    for (size_t i = 0; i < 3; ++i) {
      auto const& cp = c.cpy_req->copies[i];
      CHECK(reinterpret_cast<uintptr_t>(cp.dst) == kDst + i * 100);
      CHECK(reinterpret_cast<uintptr_t>(cp.src) == bufs[i]);  // absolute per-buffer src
      CHECK(cp.src_off == 0);
      CHECK(cp.size == 100);
    }
  }

  SECTION("partial device window clips each buffer's copy")
  {
    std::vector<io_object_segment> segs{io_object_segment{0, 100, fake_ptr(kB0)},
                                        io_object_segment{100, 100, fake_ptr(kB1)},
                                        io_object_segment{200, 100, fake_ptr(kB2)}};
    // Device window [50, 250): clips the first and last buffers.
    auto req = rest_reactor::prep_host_to_device_rx_request(
      cfg, file, segs, fake_ptr(kDst), /*offset=*/50, /*size=*/200, rmm::cuda_stream_view{}, 0);
    REQUIRE(req->size() == 1);
    auto chunks   = req->get_all_chunks();
    auto const& c = *chunks[0];
    REQUIRE(c.cpy_req->copies.size() == 3);
    // buffer0 file [0,100) intersects [50,250) as [50,100)
    CHECK(reinterpret_cast<uintptr_t>(c.cpy_req->copies[0].src) == kB0 + 50);
    CHECK(c.cpy_req->copies[0].src_off == 0);
    CHECK(reinterpret_cast<uintptr_t>(c.cpy_req->copies[0].dst) == kDst + 0);
    CHECK(c.cpy_req->copies[0].size == 50);
    // buffer1 file [100,200) fully inside -> [100,200)
    CHECK(reinterpret_cast<uintptr_t>(c.cpy_req->copies[1].src) == kB1);
    CHECK(c.cpy_req->copies[1].src_off == 0);
    CHECK(reinterpret_cast<uintptr_t>(c.cpy_req->copies[1].dst) == kDst + 50);
    CHECK(c.cpy_req->copies[1].size == 100);
    // buffer2 file [200,300) intersects [50,250) as [200,250)
    CHECK(reinterpret_cast<uintptr_t>(c.cpy_req->copies[2].src) == kB2);
    CHECK(c.cpy_req->copies[2].src_off == 0);
    CHECK(reinterpret_cast<uintptr_t>(c.cpy_req->copies[2].dst) == kDst + 150);
    CHECK(c.cpy_req->copies[2].size == 50);
  }

  SECTION("a cut inside a buffer splits its copy across the two chunks")
  {
    cfg.max_chunk_size = 150;
    std::vector<io_object_segment> segs{io_object_segment{0, 100, fake_ptr(kB0)},
                                        io_object_segment{100, 100, fake_ptr(kB1)},
                                        io_object_segment{200, 100, fake_ptr(kB2)}};
    auto req = rest_reactor::prep_host_to_device_rx_request(
      cfg, file, segs, fake_ptr(kDst), 0, 300, rmm::cuda_stream_view{}, 0);
    REQUIRE(req->size() == 2);  // 300 / 150
    auto chunks = req->get_all_chunks();

    // Buffer B straddles the cut, so each half copies from its own chunk.
    REQUIRE(chunks[0]->cpy_req->copies.size() == 2);
    CHECK(reinterpret_cast<uintptr_t>(chunks[0]->cpy_req->copies[0].src) == kB0);
    CHECK(chunks[0]->cpy_req->copies[0].size == 100);
    CHECK(reinterpret_cast<uintptr_t>(chunks[0]->cpy_req->copies[1].src) == kB1);
    CHECK(reinterpret_cast<uintptr_t>(chunks[0]->cpy_req->copies[1].dst) == kDst + 100);
    CHECK(chunks[0]->cpy_req->copies[1].size == 50);

    REQUIRE(chunks[1]->cpy_req->copies.size() == 2);
    CHECK(reinterpret_cast<uintptr_t>(chunks[1]->cpy_req->copies[0].src) == kB1 + 50);
    CHECK(reinterpret_cast<uintptr_t>(chunks[1]->cpy_req->copies[0].dst) == kDst + 150);
    CHECK(chunks[1]->cpy_req->copies[0].size == 50);
    CHECK(reinterpret_cast<uintptr_t>(chunks[1]->cpy_req->copies[1].src) == kB2);
    CHECK(chunks[1]->cpy_req->copies[1].size == 100);
  }

  SECTION("a near-neighbor gap is bridged and copies nothing for the hole")
  {
    std::vector<io_object_segment> segs{io_object_segment{0, 100, fake_ptr(kB0)},
                                        io_object_segment{500, 100, fake_ptr(kB1)}};
    auto req = rest_reactor::prep_host_to_device_rx_request(
      cfg, file, segs, fake_ptr(kDst), 0, 600, rmm::cuda_stream_view{}, 0);
    REQUIRE(req->size() == 1);
    auto chunks   = req->get_all_chunks();
    auto const& c = *chunks[0];
    CHECK(c.chunk.size == 600);
    CHECK(c.chunk.n_chunks() == 3);  // real, hole, real
    REQUIRE(c.cpy_req->copies.size() == 2);
    CHECK(reinterpret_cast<uintptr_t>(c.cpy_req->copies[0].src) == kB0);
    CHECK(reinterpret_cast<uintptr_t>(c.cpy_req->copies[0].dst) == kDst);
    CHECK(c.cpy_req->copies[0].size == 100);
    CHECK(reinterpret_cast<uintptr_t>(c.cpy_req->copies[1].src) == kB1);
    CHECK(reinterpret_cast<uintptr_t>(c.cpy_req->copies[1].dst) == kDst + 500);
    CHECK(c.cpy_req->copies[1].size == 100);
    // The bridged bytes are fetched but never delivered.
    CHECK(chunks[0]->manager->bytes_requested == 200);
  }

  SECTION("buffers further apart than merge_max_gap stay separate chunks")
  {
    cfg.merge_max_gap = 100;
    std::vector<io_object_segment> segs{io_object_segment{0, 100, fake_ptr(kB0)},
                                        io_object_segment{500, 100, fake_ptr(kB1)}};
    auto req = rest_reactor::prep_host_to_device_rx_request(
      cfg, file, segs, fake_ptr(kDst), 0, 600, rmm::cuda_stream_view{}, 0);
    REQUIRE(req->size() == 2);
    auto chunks = req->get_all_chunks();
    for (auto const& cp : chunks) {
      CHECK(cp->chunk.n_chunks() == 1);
      REQUIRE(cp->cpy_req->copies.size() == 1);
    }
  }
}

TEST_CASE("prep_host_to_device keeps null-buffer segments as standalone bounce-staged chunks",
          "[rest]")
{
  sirius::io::rest::config cfg;  // default max_chunk_size (16 MiB) / merge_max_gap (0.5 MiB)
  rest_io_object const file("s3://bkt/key", "bkt", "key", /*size=*/1 << 20);
  constexpr uintptr_t kDst = 0x100000;
  constexpr uintptr_t kB0 = 0xA000, kB1 = 0xB000;

  SECTION("real-null-real neighbors are not fused across the null segment")
  {
    std::vector<io_object_segment> segs{io_object_segment{100, 100, fake_ptr(kB0)},
                                        io_object_segment{200, 100, nullptr},
                                        io_object_segment{300, 100, fake_ptr(kB1)}};
    // Device window [150, 380): clips the first and last real buffers while the
    // null-buffer gap is staged later through a reactor-owned bounce slot.
    auto req = rest_reactor::prep_host_to_device_rx_request(
      cfg, file, segs, fake_ptr(kDst), /*offset=*/150, /*size=*/230, rmm::cuda_stream_view{}, 0);

    auto chunks = req->get_all_chunks();
    REQUIRE(chunks.size() == 3);
    for (auto const& chunk : chunks) {
      CHECK(chunk->is_device());
      CHECK(chunk->chunk.n_chunks() == 1);
      REQUIRE(chunk->cpy_req != nullptr);
      REQUIRE(chunk->cpy_req->copies.size() == 1);
    }

    auto const& first = chunks[0]->cpy_req->copies[0];
    CHECK(chunks[0]->chunk.offset == 100);
    CHECK(chunks[0]->chunk.size == 100);
    CHECK(reinterpret_cast<uintptr_t>(first.dst) == kDst);
    CHECK(reinterpret_cast<uintptr_t>(first.src) == kB0 + 50);
    CHECK(first.src_off == 0);
    CHECK(first.size == 50);

    auto const& gap = chunks[1]->cpy_req->copies[0];
    CHECK(chunks[1]->chunk.offset == 200);
    CHECK(chunks[1]->chunk.size == 100);
    CHECK(chunks[1]->chunk.data() == nullptr);
    CHECK(gap.dst == fake_ptr(kDst + 50));
    CHECK(gap.src == nullptr);
    CHECK(gap.src_off == 0);
    CHECK(gap.size == 100);

    auto const& last = chunks[2]->cpy_req->copies[0];
    CHECK(chunks[2]->chunk.offset == 300);
    CHECK(chunks[2]->chunk.size == 100);
    CHECK(reinterpret_cast<uintptr_t>(last.dst) == kDst + 150);
    CHECK(reinterpret_cast<uintptr_t>(last.src) == kB1);
    CHECK(last.src_off == 0);
    CHECK(last.size == 80);
  }

  SECTION("adjacent null-buffer segments are not fused")
  {
    std::vector<io_object_segment> segs{io_object_segment{100, 100, nullptr},
                                        io_object_segment{200, 100, nullptr}};
    // Device window [125, 275): the first null-buffer chunk starts 25 bytes into
    // its future bounce slot, proving the copy carries a src_off instead of a
    // near-null absolute pointer.
    auto req = rest_reactor::prep_host_to_device_rx_request(
      cfg, file, segs, fake_ptr(kDst), /*offset=*/125, /*size=*/150, rmm::cuda_stream_view{}, 0);

    auto chunks = req->get_all_chunks();
    REQUIRE(chunks.size() == 2);
    for (auto const& chunk : chunks) {
      CHECK(chunk->is_device());
      CHECK(chunk->chunk.n_chunks() == 1);
      CHECK(chunk->chunk.data() == nullptr);
      REQUIRE(chunk->cpy_req != nullptr);
      REQUIRE(chunk->cpy_req->copies.size() == 1);
      CHECK(chunk->cpy_req->copies[0].src == nullptr);
    }

    CHECK(chunks[0]->chunk.offset == 100);
    CHECK(chunks[0]->chunk.size == 100);
    CHECK(chunks[0]->cpy_req->copies[0].dst == fake_ptr(kDst));
    CHECK(chunks[0]->cpy_req->copies[0].src_off == 25);
    CHECK(chunks[0]->cpy_req->copies[0].size == 75);

    CHECK(chunks[1]->chunk.offset == 200);
    CHECK(chunks[1]->chunk.size == 100);
    CHECK(chunks[1]->cpy_req->copies[0].dst == fake_ptr(kDst + 75));
    CHECK(chunks[1]->cpy_req->copies[0].src_off == 0);
    CHECK(chunks[1]->cpy_req->copies[0].size == 75);
  }
}

TEST_CASE("prep_device_ranges builds one staged GET per small range", "[rest]")
{
  sirius::io::rest::config cfg;
  cfg.bounce_block_size = 4096;  // reactor-owned pinned staging slots
  rest_io_object const file("s3://bkt/key", "bkt", "key", /*size=*/1 << 20);
  constexpr uintptr_t kD0 = 0x100000, kD1 = 0x200000, kD2 = 0x300000;

  // Ranges below the bounce block: one GET and one H2D copy each, and — unlike
  // the single-range path — every copy targets its own device buffer.
  std::vector<io_device_range> ranges{
    {0, 100, fake_ptr(kD0)}, {500, 100, fake_ptr(kD1)}, {9000, 300, fake_ptr(kD2)}};
  auto req = rest_reactor::prep_device_rxv_request(cfg, file, ranges, rmm::cuda_stream_view{}, 0);
  REQUIRE(req->size() == 3);
  auto fut    = req->get_future();
  auto chunks = req->get_all_chunks();

  REQUIRE(chunks.size() == 3);
  CHECK(chunks[0]->manager->total_chunks == 3);
  CHECK(chunks[0]->manager->bytes_requested == 100 + 100 + 300);

  std::array<uintptr_t, 3> const dsts{kD0, kD1, kD2};
  std::array<size_t, 3> const offsets{0, 500, 9000};
  std::array<size_t, 3> const sizes{100, 100, 300};
  for (size_t i = 0; i < 3; ++i) {
    CHECK(chunks[i]->object.bucket == "bkt");
    CHECK(chunks[i]->object.key == "key");
    CHECK(chunks[i]->is_device());
    CHECK(chunks[i]->chunk.offset == offsets[i]);
    CHECK(chunks[i]->chunk.size == sizes[i]);   // no alignment: REST reads exactly the range
    CHECK(chunks[i]->chunk.data() == nullptr);  // staged through a bounce slot later
    REQUIRE(chunks[i]->cpy_req != nullptr);
    REQUIRE(chunks[i]->cpy_req->copies.size() == 1);
    auto const& cp = chunks[i]->cpy_req->copies[0];
    CHECK(reinterpret_cast<uintptr_t>(cp.dst) == dsts[i]);  // the range's own destination
    CHECK(cp.src == nullptr);                               // resolved late against the bounce slot
    CHECK(cp.src_off == 0);
    CHECK(cp.size == sizes[i]);
  }

  complete(chunks);
}

TEST_CASE("prep_device_ranges splits a range across bounce blocks", "[rest]")
{
  sirius::io::rest::config cfg;
  cfg.bounce_block_size = 100;  // force the first range over several blocks
  rest_io_object const file("s3://bkt/key", "bkt", "key", /*size=*/1 << 20);
  constexpr uintptr_t kBig = 0x100000, kSmall = 0x200000;

  std::vector<io_device_range> ranges{{0, 250, fake_ptr(kBig)}, {1000, 50, fake_ptr(kSmall)}};
  auto req = rest_reactor::prep_device_rxv_request(cfg, file, ranges, rmm::cuda_stream_view{}, 0);
  auto fut = req->get_future();
  auto chunks = req->get_all_chunks();

  REQUIRE(chunks.size() == 4);  // ceil(250/100) = 3 blocks, plus 1 for the small range
  CHECK(chunks[0]->manager->total_chunks == 4);
  CHECK(chunks[0]->manager->bytes_requested == 250 + 50);

  // The blocks tile the big range: each copy lands at its own offset in its
  // destination and together they cover the range exactly once.
  std::array<size_t, 3> const block_sizes{100, 100, 50};  // the tail block is short
  size_t covered = 0;
  for (size_t i = 0; i < 3; ++i) {
    CHECK(chunks[i]->chunk.offset == i * 100);
    CHECK(chunks[i]->chunk.size == block_sizes[i]);
    REQUIRE(chunks[i]->cpy_req->copies.size() == 1);
    auto const& cp = chunks[i]->cpy_req->copies[0];
    CHECK(reinterpret_cast<uintptr_t>(cp.dst) == kBig + i * 100);  // block_offset - range.offset
    CHECK(cp.src_off == 0);
    CHECK(cp.size == block_sizes[i]);
    covered += cp.size;
  }
  CHECK(covered == 250);

  // A sub-block range is unaffected by the split and keeps its own destination.
  CHECK(chunks[3]->chunk.offset == 1000);
  REQUIRE(chunks[3]->cpy_req->copies.size() == 1);
  CHECK(reinterpret_cast<uintptr_t>(chunks[3]->cpy_req->copies[0].dst) == kSmall);
  CHECK(chunks[3]->cpy_req->copies[0].size == 50);

  complete(chunks);
}

TEST_CASE("prep_device_ranges clamps a range at the object end and drops ranges past it", "[rest]")
{
  sirius::io::rest::config cfg;
  cfg.bounce_block_size = 4096;
  rest_io_object const file("s3://bkt/key", "bkt", "key", /*size=*/10000);
  constexpr uintptr_t kTail = 0x100000;

  std::vector<io_device_range> ranges{
    {9900, 1000, fake_ptr(kTail)},      // starts in the object, overhangs the end
    {10000, 100, fake_ptr(0x200000)},   // starts exactly at the end
    {20000, 100, fake_ptr(0x300000)}};  // starts past the end
  auto req = rest_reactor::prep_device_rxv_request(cfg, file, ranges, rmm::cuda_stream_view{}, 0);
  auto fut = req->get_future();
  auto chunks = req->get_all_chunks();

  // Only the in-object range survives; the at/past-end ranges emit no chunk and
  // do not inflate the manager's chunk count (which would strand the future).
  REQUIRE(chunks.size() == 1);
  CHECK(chunks[0]->manager->total_chunks == 1);
  CHECK(chunks[0]->manager->bytes_requested == 100);  // 10000 - 9900, not the requested 1000
  CHECK(chunks[0]->chunk.offset == 9900);
  CHECK(chunks[0]->chunk.size == 100);
  REQUIRE(chunks[0]->cpy_req->copies.size() == 1);
  CHECK(reinterpret_cast<uintptr_t>(chunks[0]->cpy_req->copies[0].dst) == kTail);
  CHECK(chunks[0]->cpy_req->copies[0].size == 100);

  complete(chunks);
}

TEST_CASE("prep_device_ranges drops zero-size ranges and null destinations", "[rest]")
{
  sirius::io::rest::config cfg;
  cfg.bounce_block_size = 4096;
  rest_io_object const file("s3://bkt/key", "bkt", "key", /*size=*/1 << 20);
  constexpr uintptr_t kDst = 0x100000;

  SECTION("skipped ranges neither emit chunks nor inflate the counts")
  {
    std::vector<io_device_range> ranges{{0, 0, fake_ptr(0x200000)},  // no bytes wanted
                                        {100, 100, nullptr},         // nowhere to put them
                                        {200, 100, fake_ptr(kDst)}};
    auto req = rest_reactor::prep_device_rxv_request(cfg, file, ranges, rmm::cuda_stream_view{}, 0);
    REQUIRE(req->size() == 1);
    auto fut    = req->get_future();
    auto chunks = req->get_all_chunks();

    REQUIRE(chunks.size() == 1);
    CHECK(chunks[0]->manager->total_chunks == 1);
    CHECK(chunks[0]->manager->bytes_requested == 100);
    CHECK(chunks[0]->chunk.offset == 200);
    REQUIRE(chunks[0]->cpy_req->copies.size() == 1);
    CHECK(reinterpret_cast<uintptr_t>(chunks[0]->cpy_req->copies[0].dst) == kDst);

    complete(chunks);
  }

  SECTION("an all-skipped vector degrades to a ready zero-byte request")
  {
    std::vector<io_device_range> ranges{{0, 0, fake_ptr(0x200000)}, {100, 100, nullptr}};
    auto req = rest_reactor::prep_device_rxv_request(cfg, file, ranges, rmm::cuda_stream_view{}, 0);
    REQUIRE(req != nullptr);
    CHECK(req->size() == 0);
    auto fut = req->get_future();
    CHECK(fut.is_ready());
    CHECK(std::move(fut).get() == 0);
  }
}

TEST_CASE("prep_device_ranges on an empty range list yields a ready zero-byte request", "[rest]")
{
  sirius::io::rest::config cfg;
  cfg.bounce_block_size = 4096;
  rest_io_object const file("s3://bkt/key", "bkt", "key", /*size=*/1 << 20);

  std::vector<io_device_range> ranges;
  auto req = rest_reactor::prep_device_rxv_request(cfg, file, ranges, rmm::cuda_stream_view{}, 0);
  REQUIRE(req != nullptr);
  CHECK(req->size() == 0);
  auto fut = req->get_future();
  CHECK(fut.is_ready());
  CHECK(std::move(fut).get() == 0);
}

TEST_CASE("prep_device_ranges reports the clamped byte total through the request future", "[rest]")
{
  // 10 KB object.  The caller asks for 1400 bytes across three ranges, but only
  // 400 of them exist: the future must report what was actually delivered, not
  // the sum of the input sizes (bytes_requested alone is just a ctor argument —
  // this drives the manager to completion and reads the value back out).
  sirius::io::rest::config cfg;
  cfg.bounce_block_size = 4096;
  rest_io_object const file("s3://bkt/key", "bkt", "key", /*size=*/10000);

  std::vector<io_device_range> ranges{
    {0, 300, fake_ptr(0x100000)},       // fully inside         -> 300
    {9900, 1000, fake_ptr(0x200000)},   // straddles the end    -> 100
    {20000, 100, fake_ptr(0x300000)}};  // past the end         -> dropped

  auto req = rest_reactor::prep_device_rxv_request(cfg, file, ranges, rmm::cuda_stream_view{}, 0);
  auto fut = req->get_future();
  auto chunks = req->get_all_chunks();

  REQUIRE(chunks.size() == 2);
  CHECK(chunks[0]->manager->bytes_requested == 300 + 100);

  complete(chunks);
  chunks.clear();  // drop the last manager reference: its dtor fulfills the future
  REQUIRE(fut.is_ready());
  CHECK(std::move(fut).get() == 300 + 100);  // not the 1400 bytes asked for
}

TEST_CASE("prep_device_ranges throws without a bounce block size", "[rest]")
{
  sirius::io::rest::config cfg;  // bounce_block_size defaults to 0
  rest_io_object const file("s3://bkt/key", "bkt", "key", /*size=*/1 << 20);

  SECTION("a device range needs the bounce resource to stage through")
  {
    // REST has no GPU-direct path, so without a staging block size there is no
    // way to land the bytes — same contract as the single-range prep.
    std::vector<io_device_range> ranges{{0, 100, fake_ptr(0x100000)}};
    CHECK_THROWS_AS(
      rest_reactor::prep_device_rxv_request(cfg, file, ranges, rmm::cuda_stream_view{}, 0),
      std::runtime_error);
  }

  SECTION("an empty range list short-circuits before the bounce check")
  {
    std::vector<io_device_range> ranges;
    auto req = rest_reactor::prep_device_rxv_request(cfg, file, ranges, rmm::cuda_stream_view{}, 0);
    CHECK(req->size() == 0);
  }
}

// --- vectored host-to-device ranges (read span vs. copy window per range) ---

TEST_CASE("prep_host_to_device_ranges clips the copy window inside an over-read segment", "[rest]")
{
  sirius::io::rest::config cfg;
  cfg.merge_max_gap = 0;  // isolate copy-window clipping from gap bridging
  rest_io_object const file("s3://bkt/key", "bkt", "key", /*size=*/1 << 20);
  constexpr uintptr_t kB0 = 0xA000, kB1 = 0xB000;
  constexpr uintptr_t kD0 = 0x100000, kD1 = 0x200000;

  // Each range GETs a whole cache-sized segment but wants only an interior slice
  // of it on the device: the read span is what lands in the caller's buffer, the
  // copy window is what reaches device_dst.
  std::vector<io_host_device_range> ranges{{/*offset=*/0,
                                            /*size=*/1000,
                                            /*copy_offset=*/200,
                                            /*copy_size=*/300,
                                            fake_ptr(kB0),
                                            fake_ptr(kD0)},
                                           {/*offset=*/5000,
                                            /*size=*/1000,
                                            /*copy_offset=*/5000,
                                            /*copy_size=*/100,
                                            fake_ptr(kB1),
                                            fake_ptr(kD1)}};

  auto req =
    rest_reactor::prep_host_to_device_rxv_request(cfg, file, ranges, rmm::cuda_stream_view{}, 0);
  auto fut    = req->get_future();
  auto chunks = req->get_all_chunks();

  REQUIRE(chunks.size() == 2);  // not file-adjacent and no bridging, so no fusion
  CHECK(chunks[0]->manager->total_chunks == 2);
  // The future reports the copy windows (400), never the 2000 bytes fetched.
  CHECK(chunks[0]->manager->bytes_requested == 300 + 100);

  // The GET still covers the whole segment...
  CHECK(chunks[0]->chunk.offset == 0);
  CHECK(chunks[0]->chunk.size == 1000);
  CHECK(chunks[0]->is_device());
  // ... while the copy takes only [200, 500) out of it.  device_dst addresses
  // copy_offset, so the copy lands at its head, not at a read-span offset.
  REQUIRE(chunks[0]->cpy_req->copies.size() == 1);
  auto const& c0 = chunks[0]->cpy_req->copies[0];
  CHECK(reinterpret_cast<uintptr_t>(c0.src) == kB0 + 200);  // absolute src into the caller buffer
  CHECK(c0.src_off == 0);
  CHECK(reinterpret_cast<uintptr_t>(c0.dst) == kD0);
  CHECK(c0.size == 300);  // the copy window, not the 1000-byte read

  CHECK(chunks[1]->chunk.size == 1000);
  REQUIRE(chunks[1]->cpy_req->copies.size() == 1);
  auto const& c1 = chunks[1]->cpy_req->copies[0];
  CHECK(reinterpret_cast<uintptr_t>(c1.src) == kB1);
  CHECK(reinterpret_cast<uintptr_t>(c1.dst) == kD1);
  CHECK(c1.size == 100);

  complete(chunks);
}

TEST_CASE("prep_host_to_device_ranges fuses contiguous caller buffers into one scatter GET",
          "[rest]")
{
  sirius::io::rest::config cfg;
  rest_io_object const file("s3://bkt/key", "bkt", "key", /*size=*/1 << 20);
  constexpr uintptr_t kB0 = 0xA000, kB1 = 0xB000, kB2 = 0xC000;
  constexpr uintptr_t kD0 = 0x100000, kD1 = 0x200000, kD2 = 0x300000;

  // Three file-adjacent ranges (copy window == read span) with unrelated device
  // destinations: they share one scatter GET, but each buffer keeps its own copy.
  std::vector<io_host_device_range> ranges{{0, 100, fake_ptr(kB0), fake_ptr(kD0)},
                                           {100, 100, fake_ptr(kB1), fake_ptr(kD1)},
                                           {200, 100, fake_ptr(kB2), fake_ptr(kD2)}};

  auto req =
    rest_reactor::prep_host_to_device_rxv_request(cfg, file, ranges, rmm::cuda_stream_view{}, 0);
  auto fut    = req->get_future();
  auto chunks = req->get_all_chunks();

  REQUIRE(chunks.size() == 1);
  CHECK(chunks[0]->chunk.offset == 0);
  CHECK(chunks[0]->chunk.size == 300);      // one contiguous range
  CHECK(chunks[0]->chunk.n_chunks() == 3);  // scattered across 3 buffers
  CHECK(chunks[0]->chunk.is_vectored());
  CHECK(chunks[0]->manager->total_chunks == 1);
  CHECK(chunks[0]->manager->bytes_requested == 300);

  std::array<uintptr_t, 3> const hosts{kB0, kB1, kB2};
  std::array<uintptr_t, 3> const devs{kD0, kD1, kD2};
  auto const& copies = chunks[0]->cpy_req->copies;
  REQUIRE(copies.size() == 3);
  for (size_t i = 0; i < 3; ++i) {
    CHECK(reinterpret_cast<uintptr_t>(copies[i].src) == hosts[i]);
    CHECK(copies[i].src_off == 0);
    CHECK(reinterpret_cast<uintptr_t>(copies[i].dst) == devs[i]);  // its own destination
    CHECK(copies[i].size == 100);
  }

  complete(chunks);
}

TEST_CASE("prep_host_to_device_ranges keeps a null-buffer range standalone between fused groups",
          "[rest]")
{
  sirius::io::rest::config cfg;
  cfg.bounce_block_size = 4096;  // the null range still fits one bounce block
  rest_io_object const file("s3://bkt/key", "bkt", "key", /*size=*/1 << 20);
  constexpr uintptr_t kB0 = 0xA000, kB1 = 0xB000, kB2 = 0xC000, kB3 = 0xD000;
  constexpr uintptr_t kD0 = 0x100000, kD1 = 0x200000, kDg = 0x300000, kD2 = 0x400000,
                      kD3 = 0x500000;

  // Five contiguous ranges; the middle one has no caller buffer (a prefetch-cache
  // gap), so it stages through a pinned bounce slot and can never be fused.  It
  // also wants only [325, 375) of its segment, to prove the bounce copy carries
  // the within-segment offset rather than an absolute src.
  std::vector<io_host_device_range> ranges{
    {100, 100, fake_ptr(kB0), fake_ptr(kD0)},
    {200, 100, fake_ptr(kB1), fake_ptr(kD1)},
    {300, 100, /*copy_offset=*/325, /*copy_size=*/50, /*host_buffer=*/nullptr, fake_ptr(kDg)},
    {400, 100, fake_ptr(kB2), fake_ptr(kD2)},
    {500, 100, fake_ptr(kB3), fake_ptr(kD3)}};

  auto req =
    rest_reactor::prep_host_to_device_rxv_request(cfg, file, ranges, rmm::cuda_stream_view{}, 0);
  auto fut    = req->get_future();
  auto chunks = req->get_all_chunks();

  // => scatter head [100,300), the null range alone, scatter tail [400,600).
  REQUIRE(chunks.size() == 3);
  CHECK(chunks[0]->manager->total_chunks == 3);
  CHECK(chunks[0]->manager->bytes_requested == 4 * 100 + 50);  // copy windows, not read spans
  CHECK(chunks[0]->chunk.offset == 100);
  CHECK(chunks[0]->chunk.n_chunks() == 2);
  CHECK(chunks[2]->chunk.offset == 400);
  CHECK(chunks[2]->chunk.n_chunks() == 2);

  // The bounce-staged range: null src, the within-segment offset in src_off, and
  // dst at the head of its own device destination (which addresses copy_offset).
  CHECK(chunks[1]->chunk.offset == 300);
  CHECK(chunks[1]->chunk.size == 100);
  CHECK(chunks[1]->chunk.n_chunks() == 1);
  CHECK(chunks[1]->chunk.data() == nullptr);
  REQUIRE(chunks[1]->cpy_req->copies.size() == 1);
  auto const& gap = chunks[1]->cpy_req->copies[0];
  CHECK(gap.src == nullptr);
  CHECK(gap.src_off == 325 - 300);  // data_lo - segment offset
  CHECK(reinterpret_cast<uintptr_t>(gap.dst) == kDg);
  CHECK(gap.size == 50);

  complete(chunks);
}

TEST_CASE("prep_host_to_device_ranges splits an oversized null-buffer range across bounce blocks",
          "[rest]")
{
  sirius::io::rest::config cfg;
  cfg.bounce_block_size = 100;  // force the 500-byte read span over several slots
  rest_io_object const file("s3://bkt/key", "bkt", "key", /*size=*/1 << 20);
  constexpr uintptr_t kDst = 0x100000;

  // Read [0, 500) through bounce slots but copy only [150, 430): the first
  // slot-sized piece holds none of the copy window and must not be fetched.
  std::vector<io_host_device_range> ranges{
    {0, 500, /*copy_offset=*/150, /*copy_size=*/280, /*host_buffer=*/nullptr, fake_ptr(kDst)}};

  auto req =
    rest_reactor::prep_host_to_device_rxv_request(cfg, file, ranges, rmm::cuda_stream_view{}, 0);
  auto fut    = req->get_future();
  auto chunks = req->get_all_chunks();

  REQUIRE(chunks.size() == 4);  // the [0,100) piece is dropped, not emitted
  CHECK(chunks[0]->manager->total_chunks == 4);
  CHECK(chunks[0]->manager->bytes_requested == 280);  // exactly the copy window

  // Every piece is standalone (a bounce slot each) and carries a clipped copy
  // whose dst is biased by how far into the copy window that piece starts.
  size_t copied = 0;
  for (auto const& c : chunks) {
    CHECK(c->chunk.n_chunks() == 1);
    CHECK(c->chunk.data() == nullptr);
    REQUIRE(c->cpy_req->copies.size() == 1);
    CHECK(c->cpy_req->copies[0].src == nullptr);
    copied += c->cpy_req->copies[0].size;
  }
  CHECK(copied == 280);

  CHECK(chunks[0]->chunk.offset == 100);
  CHECK(chunks[0]->cpy_req->copies[0].src_off == 150 - 100);  // head of the copy window
  CHECK(reinterpret_cast<uintptr_t>(chunks[0]->cpy_req->copies[0].dst) == kDst);
  CHECK(chunks[0]->cpy_req->copies[0].size == 50);
  CHECK(chunks[1]->chunk.offset == 200);
  CHECK(chunks[1]->cpy_req->copies[0].src_off == 0);
  CHECK(reinterpret_cast<uintptr_t>(chunks[1]->cpy_req->copies[0].dst) == kDst + 50);
  CHECK(chunks[1]->cpy_req->copies[0].size == 100);
  CHECK(chunks[3]->chunk.offset == 400);
  CHECK(reinterpret_cast<uintptr_t>(chunks[3]->cpy_req->copies[0].dst) == kDst + 250);
  CHECK(chunks[3]->cpy_req->copies[0].size == 30);  // clipped by the copy window end

  complete(chunks);
}

TEST_CASE("prep_host_to_device_ranges rejects an invalid or out-of-file copy window", "[rest]")
{
  sirius::io::rest::config cfg;
  rest_io_object const file("s3://bkt/key", "bkt", "key", /*size=*/10000);
  constexpr uintptr_t kB = 0xA000, kD = 0x100000;

  // A copy window outside its read span would copy bytes the GET never fetched;
  // one entirely past the object end would copy bytes that do not exist.
  SECTION("copy window starting before the read span")
  {
    std::vector<io_host_device_range> ranges{
      {100, 100, /*copy_offset=*/50, 20, fake_ptr(kB), fake_ptr(kD)}};
    CHECK_THROWS_AS(
      rest_reactor::prep_host_to_device_rxv_request(cfg, file, ranges, rmm::cuda_stream_view{}, 0),
      std::runtime_error);
  }

  SECTION("copy window running past the read span")
  {
    std::vector<io_host_device_range> ranges{
      {100, 100, /*copy_offset=*/150, 100, fake_ptr(kB), fake_ptr(kD)}};
    CHECK_THROWS_AS(
      rest_reactor::prep_host_to_device_rxv_request(cfg, file, ranges, rmm::cuda_stream_view{}, 0),
      std::runtime_error);
  }

  SECTION("copy window entirely past the end of the object")
  {
    std::vector<io_host_device_range> ranges{
      {10000, 100, /*copy_offset=*/10000, 100, fake_ptr(kB), fake_ptr(kD)}};
    CHECK_THROWS_AS(
      rest_reactor::prep_host_to_device_rxv_request(cfg, file, ranges, rmm::cuda_stream_view{}, 0),
      std::runtime_error);
  }

  SECTION("a null-buffer range without a bounce block size")
  {
    // REST has no GPU-direct path: with no staging block size there is nowhere
    // for a caller-bufferless range to land.
    std::vector<io_host_device_range> ranges{
      {100, 100, /*host_buffer=*/nullptr, fake_ptr(kD)}};  // cfg.bounce_block_size == 0
    CHECK_THROWS_AS(
      rest_reactor::prep_host_to_device_rxv_request(cfg, file, ranges, rmm::cuda_stream_view{}, 0),
      std::runtime_error);
  }
}

TEST_CASE("prep_host_to_device_ranges on an empty range list yields a ready zero-byte request",
          "[rest]")
{
  sirius::io::rest::config cfg;
  cfg.bounce_block_size = 4096;
  rest_io_object const file("s3://bkt/key", "bkt", "key", /*size=*/1 << 20);

  SECTION("no ranges at all")
  {
    std::vector<io_host_device_range> ranges;
    auto req =
      rest_reactor::prep_host_to_device_rxv_request(cfg, file, ranges, rmm::cuda_stream_view{}, 0);
    REQUIRE(req != nullptr);
    CHECK(req->size() == 0);
    auto fut = req->get_future();
    CHECK(fut.is_ready());
    CHECK(std::move(fut).get() == 0);
  }

  SECTION("only zero-length read spans")
  {
    std::vector<io_host_device_range> ranges{{0, 0, fake_ptr(0xA000), fake_ptr(0x100000)}};
    auto req =
      rest_reactor::prep_host_to_device_rxv_request(cfg, file, ranges, rmm::cuda_stream_view{}, 0);
    REQUIRE(req != nullptr);
    CHECK(req->size() == 0);
    auto fut = req->get_future();
    CHECK(fut.is_ready());
    CHECK(std::move(fut).get() == 0);
  }
}

TEST_CASE("device_cpy_request rejects null-derived host sources before cuda memcpy", "[rest][gpu]")
{
  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
    WARN("Skipping device_cpy_request CUDA guard: no CUDA device");
    return;
  }

  device_cpy_request invalid;
  invalid.stream    = rmm::cuda_stream_view{};
  invalid.device_id = 0;
  invalid.copies.push_back(device_cpy_request::copy{
    /*dst=*/fake_ptr(0x100000), /*src=*/nullptr, /*src_off=*/4, /*size=*/4});
  CHECK(invalid.copy_async(nullptr, 0) == cudaErrorInvalidValue);

  std::array<uint8_t, 8> host{0, 1, 2, 3, 4, 5, 6, 7};
  uint8_t* device_dst = nullptr;
  REQUIRE(cudaMalloc(reinterpret_cast<void**>(&device_dst), host.size()) == cudaSuccess);

  device_cpy_request valid;
  valid.stream    = rmm::cuda_stream_view{};
  valid.device_id = 0;
  valid.copies.push_back(device_cpy_request::copy{
    /*dst=*/device_dst, /*src=*/nullptr, /*src_off=*/0, /*size=*/host.size()});
  CHECK(valid.copy_async(host.data(), host.size()) == cudaSuccess);
  CHECK(cudaDeviceSynchronize() == cudaSuccess);
  CHECK(cudaFree(device_dst) == cudaSuccess);
}
