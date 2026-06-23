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

#include <catch.hpp>
#include <io/rest/rest_reactor.hpp>
#include <io/rest/types.hpp>
#include <io/types.hpp>

#include <array>
#include <cstdint>
#include <optional>
#include <span>
#include <vector>

using cudf::io::text::byte_range_info;
using sirius::io::io_object_segment;
using sirius::io::rest::rest_chunked_rx_request;
using sirius::io::rest::rest_io_object;
using sirius::io::rest::rest_reactor;

namespace {

// Non-null buffer base for segments; the pure prep/coalesce logic never
// dereferences it.
uint8_t* fake_ptr(uintptr_t v) { return reinterpret_cast<uint8_t*>(v); }

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
  rest_reactor::config cfg;  // pure primitives; shared services live on the context
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
  rest_reactor::config cfg;
  rest_io_object const file("s3://bkt/key", "bkt", "key", /*size=*/10000);

  SECTION("three in-range segments")
  {
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

TEST_CASE("prep_host_rx_request splits a contiguous read by max_read_split", "[rest]")
{
  constexpr size_t kMiB     = 1UL << 20;
  constexpr uintptr_t kBase = 0x10000;
  rest_io_object const file("s3://bkt/key", "bkt", "key", /*size=*/256 * kMiB);

  SECTION("a read below 2 MiB stays a single GET")
  {
    rest_reactor::config cfg;
    cfg.max_read_split = 16;
    auto req           = rest_reactor::prep_host_rx_request(
      cfg, file, io_object_segment{0, kMiB + kMiB / 2, fake_ptr(kBase)});  // 1.5 MiB
    CHECK(req->size() == 1);
  }

  SECTION("split count is capped by max_read_split")
  {
    rest_reactor::config cfg;
    cfg.max_read_split = 4;
    // 8 MiB / 1 MiB = 8 candidate pieces, but max_read_split caps it at 4.
    auto req = rest_reactor::prep_host_rx_request(
      cfg, file, io_object_segment{0, 8 * kMiB, fake_ptr(kBase)});
    REQUIRE(req->size() == 4);
    auto chunks  = req->get_all_chunks();
    size_t total = 0;
    size_t pos   = 0;
    for (auto const& c : chunks) {
      CHECK(c->chunk.offset == pos);
      CHECK(c->chunk.size == 2 * kMiB);
      CHECK(c->chunk.n_chunks() == 1);  // each piece is a plain single-buffer GET
      CHECK(reinterpret_cast<uintptr_t>(c->chunk.data()) == kBase + pos);
      pos += c->chunk.size;
      total += c->chunk.size;
    }
    CHECK(total == 8 * kMiB);  // pieces cover the whole range, contiguously
  }

  SECTION("pieces stay at least 1 MiB when max_read_split exceeds size / 1 MiB")
  {
    rest_reactor::config cfg;
    cfg.max_read_split = 16;
    // 5 MiB / 1 MiB = 5 pieces, fewer than the cap, so each piece is exactly 1 MiB.
    auto req = rest_reactor::prep_host_rx_request(
      cfg, file, io_object_segment{0, 5 * kMiB, fake_ptr(kBase)});
    REQUIRE(req->size() == 5);
    auto chunks = req->get_all_chunks();
    for (auto const& c : chunks) {
      CHECK(c->chunk.size == kMiB);
    }
  }

  SECTION("an uneven split spreads the remainder over the leading pieces")
  {
    rest_reactor::config cfg;
    cfg.max_read_split = 4;
    size_t const size  = 8 * kMiB + 3;  // 3 leading pieces get one extra byte
    auto req =
      rest_reactor::prep_host_rx_request(cfg, file, io_object_segment{1000, size, fake_ptr(kBase)});
    REQUIRE(req->size() == 4);
    auto chunks  = req->get_all_chunks();
    size_t total = 0;
    size_t pos   = 0;
    for (auto const& c : chunks) {
      CHECK(c->chunk.offset == 1000 + pos);
      CHECK(reinterpret_cast<uintptr_t>(c->chunk.data()) == kBase + pos);
      pos += c->chunk.size;
      total += c->chunk.size;
    }
    CHECK(total == size);                               // every byte covered exactly once
    CHECK(chunks.front()->chunk.size == 2 * kMiB + 1);  // leading piece took a remainder byte
    CHECK(chunks.back()->chunk.size == 2 * kMiB);       // trailing piece did not
  }
}

TEST_CASE("prep_host_rxv_request fuses file-adjacent segments into a scatter GET", "[rest]")
{
  rest_reactor::config cfg;
  rest_io_object const file("s3://bkt/key", "bkt", "key", /*size=*/1 << 20);

  SECTION("three contiguous segments, separate buffers -> one multi-buffer chunk")
  {
    cfg.chunk_size   = 1 << 20;  // big enough to fit all three
    cfg.max_n_chunks = 16;
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
  SECTION("max_n_chunks caps the fusion")
  {
    cfg.chunk_size   = 1 << 20;
    cfg.max_n_chunks = 2;  // at most 2 buffers per request
    std::vector<io_object_segment> segs{io_object_segment{0, 100, fake_ptr(0xA00)},
                                        io_object_segment{100, 100, fake_ptr(0xB00)},
                                        io_object_segment{200, 100, fake_ptr(0xC00)}};
    auto req = rest_reactor::prep_host_rxv_request(cfg, file, segs);
    REQUIRE(req->size() == 2);  // [0,200) over 2 buffers, then [200,300)
  }
  SECTION("chunk_size caps the fused span")
  {
    cfg.chunk_size   = 250;  // two 100B segments fit, the third spills over
    cfg.max_n_chunks = 16;
    std::vector<io_object_segment> segs{io_object_segment{0, 100, fake_ptr(0xA00)},
                                        io_object_segment{100, 100, fake_ptr(0xB00)},
                                        io_object_segment{200, 100, fake_ptr(0xC00)}};
    auto req = rest_reactor::prep_host_rxv_request(cfg, file, segs);
    REQUIRE(req->size() == 2);  // [0,200) then [200,300)
  }
  SECTION("non-contiguous segments stay separate")
  {
    cfg.chunk_size   = 1 << 20;
    cfg.max_n_chunks = 16;
    std::vector<io_object_segment> segs{io_object_segment{0, 100, fake_ptr(0xA00)},
                                        io_object_segment{500, 100, fake_ptr(0xB00)}};
    auto req = rest_reactor::prep_host_rxv_request(cfg, file, segs);
    REQUIRE(req->size() == 2);
  }
  SECTION("a large segment in the vector is split")
  {
    cfg.chunk_size   = 4096;
    cfg.max_n_chunks = 16;
    std::vector<io_object_segment> segs{io_object_segment{0, 3 * 4096, fake_ptr(0xA00)}};
    auto req = rest_reactor::prep_host_rxv_request(cfg, file, segs);
    REQUIRE(req->size() == 3);
  }
}

TEST_CASE("prep_host_to_device fuses contiguous segments into a multi-copy chunk", "[rest]")
{
  rest_reactor::config cfg;  // default chunk_size (8 MiB) / max_n_chunks (16)
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
    // buffer0 file [0,100) ∩ [50,250) = [50,100)
    CHECK(reinterpret_cast<uintptr_t>(c.cpy_req->copies[0].src) == kB0 + 50);
    CHECK(reinterpret_cast<uintptr_t>(c.cpy_req->copies[0].dst) == kDst + 0);
    CHECK(c.cpy_req->copies[0].size == 50);
    // buffer1 file [100,200) fully inside -> [100,200)
    CHECK(reinterpret_cast<uintptr_t>(c.cpy_req->copies[1].src) == kB1);
    CHECK(reinterpret_cast<uintptr_t>(c.cpy_req->copies[1].dst) == kDst + 50);
    CHECK(c.cpy_req->copies[1].size == 100);
    // buffer2 file [200,300) ∩ [50,250) = [200,250)
    CHECK(reinterpret_cast<uintptr_t>(c.cpy_req->copies[2].src) == kB2);
    CHECK(reinterpret_cast<uintptr_t>(c.cpy_req->copies[2].dst) == kDst + 150);
    CHECK(c.cpy_req->copies[2].size == 50);
  }
  SECTION("max_n_chunks caps the fused buffers per chunk")
  {
    cfg.max_n_chunks = 2;
    std::vector<io_object_segment> segs{io_object_segment{0, 100, fake_ptr(kB0)},
                                        io_object_segment{100, 100, fake_ptr(kB1)},
                                        io_object_segment{200, 100, fake_ptr(kB2)}};
    auto req = rest_reactor::prep_host_to_device_rx_request(
      cfg, file, segs, fake_ptr(kDst), 0, 300, rmm::cuda_stream_view{}, 0);
    REQUIRE(req->size() == 2);  // [0,200) over 2 buffers, then [200,300)
  }
  SECTION("non-contiguous buffers stay separate single-copy chunks")
  {
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
