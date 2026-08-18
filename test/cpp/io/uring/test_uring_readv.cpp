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

// test
#include <catch.hpp>

// sirius
#include <rmm/cuda_stream_view.hpp>

#include <exec/semi_future.hpp>
#include <io/types.hpp>
#include <io/uring/types.hpp>
#include <io/uring/uring_reactor.hpp>

// standard library
#include <fcntl.h>
#include <unistd.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <stdexcept>
#include <vector>

using sirius::io::contiguous;
using sirius::io::io_device_range;
using sirius::io::io_host_device_range;
using sirius::io::io_object_segment;
using sirius::io::uring::chunked_rx_request;
using sirius::io::uring::local_io_object;
using sirius::io::uring::request_manager;
using sirius::io::uring::uring_reactor;

namespace {

// A fake buffer base used only so io_object_segment::data() is non-null; the
// pure grouping/iovec logic never dereferences it.
uint8_t* fake_ptr(uintptr_t v) { return reinterpret_cast<uint8_t*>(v); }

// Build a local_io_object over a real temp file (two distinct O_RDONLY fds so
// odirect_handle() != buffered_handle()).  O_DIRECT is intentionally avoided so
// the test does not depend on the temp filesystem supporting it.
struct temp_file {
  std::filesystem::path path;
  std::unique_ptr<local_io_object> obj;

  explicit temp_file(size_t bytes)
  {
    path = std::filesystem::temp_directory_path() /
           ("sirius_readv_test_" + std::to_string(::getpid()) + "_" +
            std::to_string(reinterpret_cast<uintptr_t>(this)));
    {
      std::FILE* f = std::fopen(path.c_str(), "wb");
      REQUIRE(f != nullptr);
      std::vector<char> data(bytes, 'x');
      if (bytes > 0) { REQUIRE(std::fwrite(data.data(), 1, bytes, f) == bytes); }
      std::fclose(f);
    }
    sirius::io::file_descriptor fd{::open(path.c_str(), O_RDONLY)};
    sirius::io::file_descriptor fd_direct{::open(path.c_str(), O_RDONLY)};
    REQUIRE(static_cast<bool>(fd));
    REQUIRE(static_cast<bool>(fd_direct));
    obj =
      std::make_unique<local_io_object>(path.string(), std::move(fd), std::move(fd_direct), bytes);
  }

  ~temp_file()
  {
    obj.reset();
    std::error_code ec;
    std::filesystem::remove(path, ec);
  }
};

// Drive a prep result to clean completion so the shared request_manager
// destructor's invariants (bytes_read >= bytes_requested, chunks_completed ==
// total_chunks) are satisfied — one chunk_complete per emitted group.
void complete(std::vector<std::unique_ptr<chunked_rx_request>>& chunks)
{
  for (auto& c : chunks) {
    c->manager->chunk_complete(c->chunk.size);
  }
}

}  // namespace

TEST_CASE("io_object_segment single-buffer ctor seeds one buffer", "[uring_readv]")
{
  io_object_segment s{4096, 1024, fake_ptr(0x1000)};
  REQUIRE(s.n_chunks() == 1);
  CHECK_FALSE(s.is_vectored());
  CHECK(s.size == 1024);
  CHECK(s.data() == fake_ptr(0x1000));
  CHECK(s.buffers.front().iov_base == reinterpret_cast<void*>(0x1000));
  CHECK(s.buffers.front().iov_len == 1024);
}

TEST_CASE("io_object_segment::append fuses a buffer and grows size", "[uring_readv]")
{
  io_object_segment s{1024, 512, fake_ptr(0xA000)};
  s.append(iovec{fake_ptr(0xB000), 512});
  s.append(iovec{fake_ptr(0xC000), 256});

  CHECK(s.is_vectored());
  CHECK(s.offset == 1024);              // base offset unchanged
  CHECK(s.size == 512 + 512 + 256);     // total bytes across buffers
  CHECK(s.data() == fake_ptr(0xA000));  // first buffer base
  REQUIRE(s.n_chunks() == 3);
  CHECK(s.buffers[2].iov_len == 256);
}

TEST_CASE("contiguous detects adjacency", "[uring_readv]")
{
  io_object_segment a{0, 4096, fake_ptr(0x1000)};
  io_object_segment b{4096, 4096, fake_ptr(0x2000)};
  io_object_segment c{8193, 4096, fake_ptr(0x3000)};  // gap after b
  CHECK(contiguous(a, b));
  CHECK_FALSE(contiguous(b, c));
  CHECK_FALSE(contiguous(b, a));  // overlap / wrong order
}

TEST_CASE("fill_remaining_buffers resumes after a short read", "[uring_readv]")
{
  io_object_segment s{0, 100, fake_ptr(0x1000)};
  s.append(iovec{fake_ptr(0x2000), 200});
  s.append(iovec{fake_ptr(0x3000), 300});
  std::vector<iovec> r;

  SECTION("skip == 0 returns the whole list")
  {
    s.fill_remaining_buffers(0, r);
    REQUIRE(r.size() == 3);
    CHECK(r[0].iov_base == fake_ptr(0x1000));
    CHECK(r[0].iov_len == 100);
  }

  SECTION("skip lands exactly on an iovec boundary drops consumed entries")
  {
    s.fill_remaining_buffers(100, r);
    REQUIRE(r.size() == 2);
    CHECK(r[0].iov_base == fake_ptr(0x2000));
    CHECK(r[0].iov_len == 200);
  }

  SECTION("skip straddling an iovec advances into it")
  {
    s.fill_remaining_buffers(150, r);  // 100 + 50 into second
    REQUIRE(r.size() == 2);
    CHECK(r[0].iov_base == fake_ptr(0x2000 + 50));
    CHECK(r[0].iov_len == 150);
    CHECK(r[1].iov_base == fake_ptr(0x3000));
    CHECK(r[1].iov_len == 300);
  }

  SECTION("skip into the last iovec")
  {
    s.fill_remaining_buffers(350, r);  // 100 + 200 + 50
    REQUIRE(r.size() == 1);
    CHECK(r[0].iov_base == fake_ptr(0x3000 + 50));
    CHECK(r[0].iov_len == 250);
  }
}

TEST_CASE("prep_host_rxv_request fuses contiguous same-fd segments into one readv", "[uring_readv]")
{
  temp_file tf(1 << 20);
  sirius::io::uring::config cfg;
  cfg.use_odirect  = false;  // all segments share the buffered fd
  cfg.max_n_chunks = 16;

  std::vector<io_object_segment> segs{
    {0, 4096, fake_ptr(0x1000)}, {4096, 4096, fake_ptr(0x2000)}, {8192, 4096, fake_ptr(0x3000)}};
  auto req    = uring_reactor::prep_host_rxv_request(cfg, *tf.obj, segs);
  auto fut    = req->get_future();
  auto chunks = req->get_all_chunks();

  REQUIRE(chunks.size() == 1);
  CHECK(chunks[0]->is_vectored());
  CHECK(chunks[0]->chunk.n_chunks() == 3);
  CHECK(chunks[0]->chunk.offset == 0);
  CHECK(chunks[0]->chunk.size == 3 * 4096);
  CHECK(chunks[0]->manager->total_chunks == 1);
  CHECK(chunks[0]->manager->bytes_requested == 3 * 4096);
  CHECK(chunks[0]->fd == tf.obj->buffered_handle());

  complete(chunks);
}

TEST_CASE("prep_host_rxv_request starts a new group at a non-contiguous boundary", "[uring_readv]")
{
  temp_file tf(1 << 20);
  sirius::io::uring::config cfg;
  cfg.use_odirect  = false;
  cfg.max_n_chunks = 16;

  // [0,4k)+[4k,8k) contiguous; gap; [16k,20k) standalone.
  std::vector<io_object_segment> segs{
    {0, 4096, fake_ptr(0x1000)}, {4096, 4096, fake_ptr(0x2000)}, {16384, 4096, fake_ptr(0x3000)}};
  auto req    = uring_reactor::prep_host_rxv_request(cfg, *tf.obj, segs);
  auto fut    = req->get_future();
  auto chunks = req->get_all_chunks();

  REQUIRE(chunks.size() == 2);
  // First group: a fused readv of two segments.
  CHECK(chunks[0]->is_vectored());
  CHECK(chunks[0]->chunk.n_chunks() == 2);
  // Second group: a single segment degrades to a plain read.
  CHECK_FALSE(chunks[1]->is_vectored());
  CHECK(chunks[1]->chunk.offset == 16384);
  CHECK(chunks[0]->manager->total_chunks == 2);

  complete(chunks);
}

TEST_CASE("prep_host_rxv_request caps each group at max_n_chunks", "[uring_readv]")
{
  temp_file tf(1 << 20);
  sirius::io::uring::config cfg;
  cfg.use_odirect  = false;
  cfg.max_n_chunks = 2;  // force splitting a contiguous run

  std::vector<io_object_segment> segs{{0, 4096, fake_ptr(0x1000)},
                                      {4096, 4096, fake_ptr(0x2000)},
                                      {8192, 4096, fake_ptr(0x3000)},
                                      {12288, 4096, fake_ptr(0x4000)},
                                      {16384, 4096, fake_ptr(0x5000)}};
  auto req    = uring_reactor::prep_host_rxv_request(cfg, *tf.obj, segs);
  auto fut    = req->get_future();
  auto chunks = req->get_all_chunks();

  // 5 contiguous segments, cap 2 => groups of 2, 2, 1.
  REQUIRE(chunks.size() == 3);
  CHECK(chunks[0]->chunk.n_chunks() == 2);
  CHECK(chunks[1]->chunk.n_chunks() == 2);
  CHECK_FALSE(chunks[2]->is_vectored());  // trailing group of one
  CHECK(chunks[0]->manager->total_chunks == 3);

  complete(chunks);
}

TEST_CASE("prep_host_rxv_request does not fuse segments with different fds", "[uring_readv]")
{
  temp_file tf(1 << 20);
  sirius::io::uring::config cfg;
  cfg.use_odirect  = true;  // fd chosen per-segment by is_odirect_compatible
  cfg.max_n_chunks = 16;

  // Aligned buffer => odirect-compatible; misaligned buffer => buffered.  Both
  // segments are contiguous, so only the differing fd prevents fusion.
  void* aligned = std::aligned_alloc(sirius::io::IO_BLOCK_SIZE, sirius::io::IO_BLOCK_SIZE);
  REQUIRE(aligned != nullptr);
  auto* misaligned = reinterpret_cast<uint8_t*>(aligned) + 1;

  std::vector<io_object_segment> segs{
    {0, sirius::io::IO_BLOCK_SIZE, reinterpret_cast<uint8_t*>(aligned)},
    {sirius::io::IO_BLOCK_SIZE, sirius::io::IO_BLOCK_SIZE, misaligned}};
  REQUIRE(segs[0].is_odirect_compatible());
  REQUIRE_FALSE(segs[1].is_odirect_compatible());

  auto req    = uring_reactor::prep_host_rxv_request(cfg, *tf.obj, segs);
  auto fut    = req->get_future();
  auto chunks = req->get_all_chunks();

  REQUIRE(chunks.size() == 2);
  CHECK(chunks[0]->fd == tf.obj->odirect_handle());
  CHECK(chunks[1]->fd == tf.obj->buffered_handle());
  CHECK(chunks[0]->manager->total_chunks == 2);

  complete(chunks);
  std::free(aligned);
}

TEST_CASE("prep_host_rxv_request clamps bytes_requested at EOF", "[uring_readv]")
{
  // File is 6 KiB; a single 8 KiB segment over-hangs the end by 2 KiB.
  temp_file tf(6 * 1024);
  sirius::io::uring::config cfg;
  cfg.use_odirect  = false;
  cfg.max_n_chunks = 16;

  std::vector<io_object_segment> segs{{0, 8192, fake_ptr(0x1000)}};
  auto req    = uring_reactor::prep_host_rxv_request(cfg, *tf.obj, segs);
  auto fut    = req->get_future();
  auto chunks = req->get_all_chunks();

  REQUIRE(chunks.size() == 1);
  CHECK_FALSE(chunks[0]->is_vectored());                   // single segment => plain read
  CHECK(chunks[0]->manager->bytes_requested == 6 * 1024);  // clamped to file size

  complete(chunks);
}

TEST_CASE("prep_host_rxv_request on empty input yields a ready zero-byte request", "[uring_readv]")
{
  temp_file tf(4096);
  sirius::io::uring::config cfg;
  std::vector<io_object_segment> segs;
  auto req = uring_reactor::prep_host_rxv_request(cfg, *tf.obj, segs);
  REQUIRE(req != nullptr);
  CHECK(req->size() == 0);
}

// --- align_and_coalesce -----------------------------------------------------

using cudf::io::text::byte_range_info;

TEST_CASE("align_and_coalesce rounds ends out to the default O_DIRECT alignment", "[uring_readv]")
{
  std::vector<byte_range_info> in{{100, 200}};  // [100, 300)
  auto out = uring_reactor::align_and_coalesce(in);
  REQUIRE(out.size() == 1);
  CHECK(out[0].offset() == 0);   // 100 rounded down to 4 KiB
  CHECK(out[0].size() == 4096);  // 300 rounded up to 4 KiB
}

TEST_CASE("align_and_coalesce fuses ranges that overlap after alignment", "[uring_readv]")
{
  // [0,100) and [4000,4100): both touch the first 4 KiB block; the second also
  // spills into the second block, so they coalesce into [0, 8192).
  std::vector<byte_range_info> in{{0, 100}, {4000, 100}};
  auto out = uring_reactor::align_and_coalesce(in);
  REQUIRE(out.size() == 1);
  CHECK(out[0].offset() == 0);
  CHECK(out[0].size() == 8192);
}

TEST_CASE("align_and_coalesce keeps disjoint aligned ranges separate and sorted", "[uring_readv]")
{
  // Provided out of order; [0,4k) and [16k,20k) do not touch after alignment.
  std::vector<byte_range_info> in{{16384, 100}, {0, 100}};
  auto out = uring_reactor::align_and_coalesce(in);
  REQUIRE(out.size() == 2);
  CHECK(out[0].offset() == 0);
  CHECK(out[0].size() == 4096);
  CHECK(out[1].offset() == 16384);
  CHECK(out[1].size() == 4096);
}

TEST_CASE("align_and_coalesce honors a larger caller alignment", "[uring_readv]")
{
  std::vector<byte_range_info> in{{0, 100}};
  auto out = uring_reactor::align_and_coalesce(in, /*alignment=*/1 << 16);  // 64 KiB
  REQUIRE(out.size() == 1);
  CHECK(out[0].offset() == 0);
  CHECK(out[0].size() == (1 << 16));
}

TEST_CASE("align_and_coalesce ignores an alignment smaller than the reactor's", "[uring_readv]")
{
  std::vector<byte_range_info> in{{100, 50}};
  auto out = uring_reactor::align_and_coalesce(in, /*alignment=*/512);  // < IO_BLOCK_SIZE
  REQUIRE(out.size() == 1);
  CHECK(out[0].offset() == 0);
  CHECK(out[0].size() == 4096);  // still the 4 KiB default, not 512
}

TEST_CASE("align_and_coalesce drops empty ranges and handles empty input", "[uring_readv]")
{
  CHECK(uring_reactor::align_and_coalesce(std::vector<byte_range_info>{}).empty());
  std::vector<byte_range_info> in{{0, 0}, {4096, 0}};
  CHECK(uring_reactor::align_and_coalesce(in).empty());
}

// --- host-to-device (vectored bounce reads + batched H2D copy) --------------

TEST_CASE("prep_host_to_device fuses contiguous bounce buffers into one readv with a batched copy",
          "[uring_readv]")
{
  temp_file tf(1 << 20);
  sirius::io::uring::config cfg;
  cfg.use_odirect  = false;
  cfg.max_n_chunks = 16;

  // Three contiguous 4 KiB cached chunks covering the whole [0, 12 KiB) request.
  std::vector<io_object_segment> segs{
    {0, 4096, fake_ptr(0x10000)}, {4096, 4096, fake_ptr(0x20000)}, {8192, 4096, fake_ptr(0x30000)}};
  auto* dst = fake_ptr(0x40000000);
  auto req  = uring_reactor::prep_host_to_device_rx_request(
    cfg, *tf.obj, segs, dst, /*offset=*/0, /*size=*/3 * 4096, rmm::cuda_stream_view{}, 0);
  auto fut    = req->get_future();
  auto chunks = req->get_all_chunks();

  REQUIRE(chunks.size() == 1);
  CHECK(chunks[0]->is_vectored());
  CHECK(chunks[0]->chunk.n_chunks() == 3);
  CHECK(chunks[0]->chunk.offset == 0);
  CHECK(chunks[0]->chunk.size == 3 * 4096);
  CHECK(chunks[0]->manager->total_chunks == 1);
  CHECK(chunks[0]->manager->bytes_requested == 3 * 4096);  // bytes_covered

  // One copy per buffer: dst advances by 4 KiB, src is each buffer base, full size.
  REQUIRE(chunks[0]->cpy_req != nullptr);
  auto const& copies = chunks[0]->cpy_req->copies;
  REQUIRE(copies.size() == 3);
  CHECK(copies[0].dst == dst);
  CHECK(copies[0].src == fake_ptr(0x10000));
  CHECK(copies[0].size == 4096);
  CHECK(copies[1].dst == dst + 4096);
  CHECK(copies[1].src == fake_ptr(0x20000));
  CHECK(copies[2].dst == dst + 8192);
  CHECK(copies[2].src == fake_ptr(0x30000));

  complete(chunks);
}

TEST_CASE("prep_host_to_device keeps a null-buffer segment standalone between vectored runs",
          "[uring_readv]")
{
  temp_file tf(1 << 20);
  sirius::io::uring::config cfg;
  cfg.use_odirect  = false;
  cfg.max_n_chunks = 16;

  // Five contiguous segments; the middle one has no buffer, so it must read
  // through an internal bounce slot and cannot join a readv.
  std::vector<io_object_segment> segs{{0, 4096, fake_ptr(0x1000)},
                                      {4096, 4096, fake_ptr(0x2000)},
                                      {8192, 4096},  // null buffer
                                      {12288, 4096, fake_ptr(0x4000)},
                                      {16384, 4096, fake_ptr(0x5000)}};
  auto* dst = fake_ptr(0x40000000);
  auto req  = uring_reactor::prep_host_to_device_rx_request(
    cfg, *tf.obj, segs, dst, /*offset=*/0, /*size=*/5 * 4096, rmm::cuda_stream_view{}, 0);
  auto fut    = req->get_future();
  auto chunks = req->get_all_chunks();

  // => vectored head [0,8k), the null segment alone, vectored tail [12k,20k).
  REQUIRE(chunks.size() == 3);
  CHECK(chunks[0]->is_vectored());
  CHECK(chunks[0]->chunk.n_chunks() == 2);
  CHECK(chunks[0]->chunk.offset == 0);
  CHECK_FALSE(chunks[1]->is_vectored());
  CHECK(chunks[1]->chunk.offset == 8192);
  CHECK_FALSE(chunks[1]->chunk.is_buffer_allocated());
  CHECK(chunks[2]->is_vectored());
  CHECK(chunks[2]->chunk.n_chunks() == 2);
  CHECK(chunks[2]->chunk.offset == 12288);
  CHECK(chunks[0]->manager->total_chunks == 3);

  // The bounce-slot chunk defers its source: null src, offset carried in
  // src_off so copy_async can resolve it against the late-assigned slot.
  REQUIRE(chunks[1]->cpy_req != nullptr);
  auto const& bounce_copies = chunks[1]->cpy_req->copies;
  REQUIRE(bounce_copies.size() == 1);
  CHECK(bounce_copies[0].src == nullptr);
  CHECK(bounce_copies[0].src_off == 0);
  CHECK(bounce_copies[0].dst == dst + 8192);
  CHECK(chunks[1]->needs_event_for_synchronization());

  complete(chunks);
}

TEST_CASE("prep_host_to_device clips the batched copy to the request window", "[uring_readv]")
{
  temp_file tf(1 << 20);
  sirius::io::uring::config cfg;
  cfg.use_odirect  = false;
  cfg.max_n_chunks = 16;

  // Request [1 KiB, 11 KiB) overlaps the tail of chunk 0, all of chunk 1, and the
  // head of chunk 2 — the copies clip to the request, not the chunk bounds.
  std::vector<io_object_segment> segs{
    {0, 4096, fake_ptr(0x10000)}, {4096, 4096, fake_ptr(0x20000)}, {8192, 4096, fake_ptr(0x30000)}};
  auto* dst = fake_ptr(0x40000000);
  auto req  = uring_reactor::prep_host_to_device_rx_request(
    cfg, *tf.obj, segs, dst, /*offset=*/1024, /*size=*/10240, rmm::cuda_stream_view{}, 0);
  auto chunks = req->get_all_chunks();

  REQUIRE(chunks.size() == 1);
  CHECK(chunks[0]->manager->bytes_requested == 10240);  // bytes_covered == request size
  auto const& copies = chunks[0]->cpy_req->copies;
  REQUIRE(copies.size() == 3);
  // chunk 0: file [1024, 4096) -> dst[0..3072), src offset 1024 into buffer 0.
  CHECK(copies[0].dst == dst);
  CHECK(copies[0].src == fake_ptr(0x10000 + 1024));
  CHECK(copies[0].size == 3072);
  // chunk 1: file [4096, 8192) -> dst[3072..7168), all of buffer 1.
  CHECK(copies[1].dst == dst + 3072);
  CHECK(copies[1].src == fake_ptr(0x20000));
  CHECK(copies[1].size == 4096);
  // chunk 2: file [8192, 11264) -> dst[7168..10240), head of buffer 2.
  CHECK(copies[2].dst == dst + 7168);
  CHECK(copies[2].src == fake_ptr(0x30000));
  CHECK(copies[2].size == 3072);

  complete(chunks);
}

TEST_CASE("prep_host_to_device caps each readv group at max_n_chunks", "[uring_readv]")
{
  temp_file tf(1 << 20);
  sirius::io::uring::config cfg;
  cfg.use_odirect  = false;
  cfg.max_n_chunks = 2;  // force splitting a contiguous run

  std::vector<io_object_segment> segs{{0, 4096, fake_ptr(0x10000)},
                                      {4096, 4096, fake_ptr(0x20000)},
                                      {8192, 4096, fake_ptr(0x30000)},
                                      {12288, 4096, fake_ptr(0x40000)},
                                      {16384, 4096, fake_ptr(0x50000)}};
  auto* dst = fake_ptr(0x40000000);
  auto req  = uring_reactor::prep_host_to_device_rx_request(
    cfg, *tf.obj, segs, dst, /*offset=*/0, /*size=*/5 * 4096, rmm::cuda_stream_view{}, 0);
  auto chunks = req->get_all_chunks();

  // 5 contiguous segments, cap 2 => groups of 2, 2, 1.
  REQUIRE(chunks.size() == 3);
  CHECK(chunks[0]->chunk.n_chunks() == 2);
  CHECK(chunks[1]->chunk.n_chunks() == 2);
  CHECK_FALSE(chunks[2]->is_vectored());  // trailing group of one
  CHECK(chunks[0]->manager->total_chunks == 3);

  complete(chunks);
}

// --- vectored device ranges (one device destination per range) --------------

TEST_CASE("prep_device_ranges builds one bounce chunk per small range", "[uring_readv]")
{
  temp_file tf(1 << 20);
  sirius::io::uring::config cfg;
  cfg.use_odirect = false;
  cfg.bounce_size = 1 << 20;  // every range fits in a single window

  // Block-aligned, well-separated ranges: one read window and one H2D copy each,
  // and — unlike the single-range path — each copy targets its own device buffer.
  auto* dst0 = fake_ptr(0x40000000);
  auto* dst1 = fake_ptr(0x50000000);
  auto* dst2 = fake_ptr(0x60000000);
  std::vector<io_device_range> ranges{{0, 4096, dst0}, {8192, 4096, dst1}, {16384, 8192, dst2}};

  auto req =
    uring_reactor::prep_device_rxv_request(cfg, *tf.obj, ranges, rmm::cuda_stream_view{}, 0);
  auto fut    = req->get_future();
  auto chunks = req->get_all_chunks();

  REQUIRE(chunks.size() == 3);
  CHECK(chunks[0]->manager->total_chunks == 3);
  CHECK(chunks[0]->manager->bytes_requested == 4096 + 4096 + 8192);

  std::array<uint8_t*, 3> const dsts{dst0, dst1, dst2};
  std::array<size_t, 3> const offsets{0, 8192, 16384};
  std::array<size_t, 3> const sizes{4096, 4096, 8192};
  for (size_t i = 0; i < 3; ++i) {
    CHECK(chunks[i]->fd == tf.obj->buffered_handle());
    CHECK(chunks[i]->chunk.offset == offsets[i]);
    CHECK(chunks[i]->chunk.size == sizes[i]);
    CHECK(chunks[i]->chunk.data() == nullptr);  // staged through an internal bounce slot
    CHECK(chunks[i]->needs_event_for_synchronization());
    REQUIRE(chunks[i]->cpy_req != nullptr);
    REQUIRE(chunks[i]->cpy_req->copies.size() == 1);
    auto const& cp = chunks[i]->cpy_req->copies[0];
    CHECK(cp.dst == dsts[i]);  // the range's own destination, at its head
    CHECK(cp.src == nullptr);  // resolved late against the bounce slot
    CHECK(cp.src_off == 0);    // aligned range => no alignment overhang
    CHECK(cp.size == sizes[i]);
  }

  complete(chunks);
}

TEST_CASE("prep_device_ranges splits a range across bounce windows", "[uring_readv]")
{
  temp_file tf(1 << 20);
  sirius::io::uring::config cfg;
  cfg.use_odirect = false;
  cfg.bounce_size = 4096;  // force the first range over several windows

  auto* dst_big   = fake_ptr(0x40000000);
  auto* dst_small = fake_ptr(0x50000000);
  std::vector<io_device_range> ranges{{0, 3 * 4096, dst_big}, {32768, 1024, dst_small}};

  auto req =
    uring_reactor::prep_device_rxv_request(cfg, *tf.obj, ranges, rmm::cuda_stream_view{}, 0);
  auto fut    = req->get_future();
  auto chunks = req->get_all_chunks();

  REQUIRE(chunks.size() == 4);  // 3 windows for the big range, 1 for the small one
  CHECK(chunks[0]->manager->total_chunks == 4);
  CHECK(chunks[0]->manager->bytes_requested == 3 * 4096 + 1024);

  // The windows tile the big range: each copy lands at its own offset in dst_big
  // and together they cover it exactly once.
  size_t covered = 0;
  for (size_t i = 0; i < 3; ++i) {
    CHECK(chunks[i]->chunk.offset == i * 4096);
    CHECK(chunks[i]->chunk.size == 4096);
    REQUIRE(chunks[i]->cpy_req->copies.size() == 1);
    auto const& cp = chunks[i]->cpy_req->copies[0];
    CHECK(cp.dst == dst_big + chunks[i]->chunk.offset);  // window_offset - range.offset
    CHECK(cp.src_off == 0);
    covered += cp.size;
  }
  CHECK(covered == 3 * 4096);

  // A sub-window range is unaffected by the split and keeps its own destination.
  CHECK(chunks[3]->chunk.offset == 32768);
  REQUIRE(chunks[3]->cpy_req->copies.size() == 1);
  CHECK(chunks[3]->cpy_req->copies[0].dst == dst_small);
  CHECK(chunks[3]->cpy_req->copies[0].size == 1024);

  complete(chunks);
}

TEST_CASE("prep_device_ranges clamps a range at EOF and drops ranges past it", "[uring_readv]")
{
  // 6 KiB file: EOF falls inside the second 4 KiB block.
  temp_file tf(6 * 1024);
  sirius::io::uring::config cfg;
  cfg.use_odirect = false;
  cfg.bounce_size = 1 << 20;

  auto* dst_tail = fake_ptr(0x40000000);
  std::vector<io_device_range> ranges{
    {4096, 8192, dst_tail},                 // starts in the file, overhangs the end
    {6 * 1024, 512, fake_ptr(0x50000000)},  // starts exactly at EOF
    {8192, 512, fake_ptr(0x60000000)}};     // starts past EOF

  auto req =
    uring_reactor::prep_device_rxv_request(cfg, *tf.obj, ranges, rmm::cuda_stream_view{}, 0);
  auto fut    = req->get_future();
  auto chunks = req->get_all_chunks();

  // Only the in-file range survives; the at/past-EOF ranges emit no chunk and do
  // not inflate the manager's chunk count (which would strand the future).
  REQUIRE(chunks.size() == 1);
  CHECK(chunks[0]->manager->total_chunks == 1);
  CHECK(chunks[0]->manager->bytes_requested == 2048);  // 6 KiB - 4 KiB, not the requested 8 KiB
  CHECK(chunks[0]->chunk.offset == 4096);
  CHECK(chunks[0]->chunk.size == 4096);  // whole trailing block is read; the copy is clipped
  REQUIRE(chunks[0]->cpy_req->copies.size() == 1);
  auto const& cp = chunks[0]->cpy_req->copies[0];
  CHECK(cp.dst == dst_tail);
  CHECK(cp.src_off == 0);
  CHECK(cp.size == 2048);

  complete(chunks);
}

TEST_CASE("prep_device_ranges skips zero-size ranges and null device destinations", "[uring_readv]")
{
  temp_file tf(1 << 20);
  sirius::io::uring::config cfg;
  cfg.use_odirect = false;
  cfg.bounce_size = 1 << 20;

  SECTION("skipped ranges neither emit chunks nor inflate the counts")
  {
    auto* dst = fake_ptr(0x40000000);
    std::vector<io_device_range> ranges{{0, 0, fake_ptr(0x50000000)},  // no bytes wanted
                                        {4096, 4096, nullptr},         // nowhere to put them
                                        {8192, 4096, dst}};
    auto req =
      uring_reactor::prep_device_rxv_request(cfg, *tf.obj, ranges, rmm::cuda_stream_view{}, 0);
    auto fut    = req->get_future();
    auto chunks = req->get_all_chunks();

    REQUIRE(chunks.size() == 1);
    CHECK(chunks[0]->manager->total_chunks == 1);
    CHECK(chunks[0]->manager->bytes_requested == 4096);
    CHECK(chunks[0]->chunk.offset == 8192);
    REQUIRE(chunks[0]->cpy_req->copies.size() == 1);
    CHECK(chunks[0]->cpy_req->copies[0].dst == dst);

    complete(chunks);
  }

  SECTION("an all-skipped vector degrades to a ready zero-byte request")
  {
    std::vector<io_device_range> ranges{{0, 0, fake_ptr(0x50000000)}, {4096, 4096, nullptr}};
    auto req =
      uring_reactor::prep_device_rxv_request(cfg, *tf.obj, ranges, rmm::cuda_stream_view{}, 0);
    REQUIRE(req != nullptr);
    CHECK(req->size() == 0);
    auto fut = req->get_future();
    CHECK(fut.is_ready());
    CHECK(std::move(fut).get() == 0);
  }
}

TEST_CASE("prep_device_ranges on empty input yields a ready zero-byte request", "[uring_readv]")
{
  temp_file tf(4096);
  sirius::io::uring::config cfg;
  std::vector<io_device_range> ranges;
  auto req =
    uring_reactor::prep_device_rxv_request(cfg, *tf.obj, ranges, rmm::cuda_stream_view{}, 0);
  REQUIRE(req != nullptr);
  CHECK(req->size() == 0);
  auto fut = req->get_future();
  CHECK(fut.is_ready());
  CHECK(std::move(fut).get() == 0);
}

TEST_CASE("prep_device_ranges block-aligns every read window for an unaligned range",
          "[uring_readv]")
{
  temp_file tf(1 << 20);
  sirius::io::uring::config cfg;
  cfg.use_odirect = false;
  cfg.bounce_size = 4096;

  // [5000, 14000) is unaligned at both ends: O_DIRECT forces the physical read
  // out to [4096, 16384) — three 4 KiB windows — while the copies must still
  // land exactly the 9000 requested bytes at dst.
  auto* dst = fake_ptr(0x40000000);
  std::vector<io_device_range> ranges{{5000, 9000, dst}};

  auto req =
    uring_reactor::prep_device_rxv_request(cfg, *tf.obj, ranges, rmm::cuda_stream_view{}, 0);
  auto fut    = req->get_future();
  auto chunks = req->get_all_chunks();

  REQUIRE(chunks.size() == 3);
  CHECK(chunks[0]->manager->bytes_requested == 9000);  // the logical size, not the aligned 12288

  size_t copied = 0;
  for (auto const& c : chunks) {
    CHECK(c->chunk.offset % sirius::io::IO_BLOCK_SIZE == 0);  // O_DIRECT offset alignment
    CHECK(c->chunk.size % sirius::io::IO_BLOCK_SIZE == 0);    // ... and length (no EOF here)
    REQUIRE(c->cpy_req->copies.size() == 1);
    copied += c->cpy_req->copies[0].size;
  }
  CHECK(copied == 9000);  // the over-read blocks are never copied to the device

  // The first window starts a block early: the copy skips that overhang instead
  // of shifting dst.
  CHECK(chunks[0]->chunk.offset == 4096);
  CHECK(chunks[0]->cpy_req->copies[0].src_off == 5000 - 4096);
  CHECK(chunks[0]->cpy_req->copies[0].dst == dst);
  CHECK(chunks[0]->cpy_req->copies[0].size == 8192 - 5000);
  // Interior windows start inside the range: no overhang, dst advances instead.
  CHECK(chunks[1]->cpy_req->copies[0].src_off == 0);
  CHECK(chunks[1]->cpy_req->copies[0].dst == dst + (8192 - 5000));
  CHECK(chunks[1]->cpy_req->copies[0].size == 4096);
  // The last window is truncated by the range end, not by the block boundary.
  CHECK(chunks[2]->cpy_req->copies[0].size == 14000 - 12288);

  complete(chunks);
}

TEST_CASE("prep_device_ranges reports the clamped byte total through its future", "[uring_readv]")
{
  // 6 KiB file.  The caller asks for 9728 bytes across three ranges, but only
  // 3072 of them exist: the future must report what was actually delivered, not
  // the sum of the input sizes (bytes_requested alone is just a ctor argument —
  // this drives the manager to completion and reads the value back out).
  temp_file tf(6 * 1024);
  sirius::io::uring::config cfg;
  cfg.use_odirect = false;
  cfg.bounce_size = 1 << 20;

  std::vector<io_device_range> ranges{
    {0, 1024, fake_ptr(0x40000000)},     // fully inside  -> 1024
    {4096, 8192, fake_ptr(0x50000000)},  // straddles EOF -> 2048
    {8192, 512, fake_ptr(0x60000000)}};  // past EOF      -> dropped

  auto req =
    uring_reactor::prep_device_rxv_request(cfg, *tf.obj, ranges, rmm::cuda_stream_view{}, 0);
  auto fut    = req->get_future();
  auto chunks = req->get_all_chunks();

  REQUIRE(chunks.size() == 2);
  CHECK(chunks[0]->manager->bytes_requested == 1024 + 2048);

  complete(chunks);
  chunks.clear();  // drop the last manager reference: its dtor fulfills the future
  REQUIRE(fut.is_ready());
  CHECK(std::move(fut).get() == 1024 + 2048);  // not the 9728 bytes asked for
}

TEST_CASE("prep_device_ranges submits on the fd selected by use_odirect", "[uring_readv]")
{
  temp_file tf(1 << 20);
  sirius::io::uring::config cfg;
  cfg.bounce_size = 1 << 20;
  REQUIRE(tf.obj->odirect_handle() != tf.obj->buffered_handle());

  std::vector<io_device_range> ranges{{0, 4096, fake_ptr(0x40000000)}};

  SECTION("use_odirect reads through the O_DIRECT handle")
  {
    cfg.use_odirect = true;
    auto req =
      uring_reactor::prep_device_rxv_request(cfg, *tf.obj, ranges, rmm::cuda_stream_view{}, 0);
    auto fut    = req->get_future();
    auto chunks = req->get_all_chunks();
    REQUIRE(chunks.size() == 1);
    CHECK(chunks[0]->fd == tf.obj->odirect_handle());
    complete(chunks);
  }

  SECTION("without use_odirect it reads through the buffered handle")
  {
    cfg.use_odirect = false;
    auto req =
      uring_reactor::prep_device_rxv_request(cfg, *tf.obj, ranges, rmm::cuda_stream_view{}, 0);
    auto fut    = req->get_future();
    auto chunks = req->get_all_chunks();
    REQUIRE(chunks.size() == 1);
    CHECK(chunks[0]->fd == tf.obj->buffered_handle());
    complete(chunks);
  }
}

TEST_CASE("prep_device_ranges never coalesces two ranges inside one block", "[uring_readv]")
{
  temp_file tf(1 << 20);
  sirius::io::uring::config cfg;
  cfg.use_odirect = false;
  cfg.bounce_size = 1 << 20;

  // Both ranges align out to the very same [0, 4096) block, but each wants its
  // bytes in a different device buffer: the ranges are taken as given, so the
  // block is read once per range rather than shared between them.
  auto* dst_a = fake_ptr(0x40000000);
  auto* dst_b = fake_ptr(0x50000000);
  std::vector<io_device_range> ranges{{100, 200, dst_a}, {3000, 500, dst_b}};

  auto req =
    uring_reactor::prep_device_rxv_request(cfg, *tf.obj, ranges, rmm::cuda_stream_view{}, 0);
  auto fut    = req->get_future();
  auto chunks = req->get_all_chunks();

  REQUIRE(chunks.size() == 2);  // one read per range, not one shared read
  CHECK(chunks[0]->manager->total_chunks == 2);
  CHECK(chunks[0]->manager->bytes_requested == 200 + 500);
  CHECK(chunks[0]->chunk.offset == 0);
  CHECK(chunks[0]->chunk.size == 4096);
  CHECK(chunks[1]->chunk.offset == 0);  // the same block, read a second time
  CHECK(chunks[1]->chunk.size == 4096);

  // Each copy still carves its own range out of that block into its own buffer.
  REQUIRE(chunks[0]->cpy_req->copies.size() == 1);
  CHECK(chunks[0]->cpy_req->copies[0].dst == dst_a);
  CHECK(chunks[0]->cpy_req->copies[0].src_off == 100);
  CHECK(chunks[0]->cpy_req->copies[0].size == 200);
  REQUIRE(chunks[1]->cpy_req->copies.size() == 1);
  CHECK(chunks[1]->cpy_req->copies[0].dst == dst_b);
  CHECK(chunks[1]->cpy_req->copies[0].src_off == 3000);
  CHECK(chunks[1]->cpy_req->copies[0].size == 500);

  complete(chunks);
}

// --- vectored host-to-device ranges (read span vs. copy window per range) ---

TEST_CASE("prep_host_to_device_ranges clips the copy window inside an over-read span",
          "[uring_readv]")
{
  temp_file tf(1 << 20);
  sirius::io::uring::config cfg;
  cfg.use_odirect  = false;
  cfg.max_n_chunks = 16;
  cfg.bounce_size  = 1 << 20;

  // Each range reads a whole O_DIRECT block but wants only an interior slice of
  // it on the device: the read span is the caller's alignment over-read, the
  // copy window is what it actually asked for.
  auto* hb0 = fake_ptr(0x10000);
  auto* hb1 = fake_ptr(0x20000);
  auto* dd0 = fake_ptr(0x40000000);
  auto* dd1 = fake_ptr(0x50000000);
  std::vector<io_host_device_range> ranges{
    {/*offset=*/4096, /*size=*/4096, /*copy_offset=*/5000, /*copy_size=*/1000, hb0, dd0},
    {/*offset=*/16384, /*size=*/4096, /*copy_offset=*/16484, /*copy_size=*/116, hb1, dd1}};

  auto req = uring_reactor::prep_host_to_device_rxv_request(
    cfg, *tf.obj, ranges, rmm::cuda_stream_view{}, 0);
  auto fut    = req->get_future();
  auto chunks = req->get_all_chunks();

  REQUIRE(chunks.size() == 2);  // not file-adjacent, so no fusion
  CHECK(chunks[0]->manager->total_chunks == 2);
  // The future reports the copy windows (1116), never the 8192 bytes read.
  CHECK(chunks[0]->manager->bytes_requested == 1000 + 116);

  // The read still covers the whole block...
  CHECK(chunks[0]->chunk.offset == 4096);
  CHECK(chunks[0]->chunk.size == 4096);
  // ... while the copy takes only [5000, 6000) out of it.  device_dst addresses
  // copy_offset, so the copy lands at its head, not at a read-span offset.
  REQUIRE(chunks[0]->cpy_req->copies.size() == 1);
  auto const& c0 = chunks[0]->cpy_req->copies[0];
  CHECK(c0.src == hb0 + (5000 - 4096));  // absolute src into the caller's buffer
  CHECK(c0.src_off == 0);
  CHECK(c0.dst == dd0);
  CHECK(c0.size == 1000);  // the copy window, not the 4096-byte read

  CHECK(chunks[1]->chunk.size == 4096);
  REQUIRE(chunks[1]->cpy_req->copies.size() == 1);
  auto const& c1 = chunks[1]->cpy_req->copies[0];
  CHECK(c1.src == hb1 + (16484 - 16384));
  CHECK(c1.dst == dd1);
  CHECK(c1.size == 116);

  complete(chunks);
}

TEST_CASE("prep_host_to_device_ranges fuses contiguous caller buffers into one readv",
          "[uring_readv]")
{
  temp_file tf(1 << 20);
  sirius::io::uring::config cfg;
  cfg.use_odirect  = false;
  cfg.max_n_chunks = 16;
  cfg.bounce_size  = 1 << 20;

  // Three file-adjacent ranges (copy window == read span) with unrelated device
  // destinations: they share one readv, but each buffer keeps its own copy.
  auto* hb0 = fake_ptr(0x10000);
  auto* hb1 = fake_ptr(0x20000);
  auto* hb2 = fake_ptr(0x30000);
  auto* dd0 = fake_ptr(0x40000000);
  auto* dd1 = fake_ptr(0x50000000);
  auto* dd2 = fake_ptr(0x60000000);
  std::vector<io_host_device_range> ranges{
    {0, 4096, hb0, dd0}, {4096, 4096, hb1, dd1}, {8192, 4096, hb2, dd2}};

  auto req = uring_reactor::prep_host_to_device_rxv_request(
    cfg, *tf.obj, ranges, rmm::cuda_stream_view{}, 0);
  auto fut    = req->get_future();
  auto chunks = req->get_all_chunks();

  REQUIRE(chunks.size() == 1);
  CHECK(chunks[0]->is_vectored());
  CHECK(chunks[0]->chunk.n_chunks() == 3);
  CHECK(chunks[0]->chunk.offset == 0);
  CHECK(chunks[0]->chunk.size == 3 * 4096);
  CHECK(chunks[0]->manager->total_chunks == 1);
  CHECK(chunks[0]->manager->bytes_requested == 3 * 4096);
  CHECK(chunks[0]->fd == tf.obj->buffered_handle());

  // One copy per buffer, each to its OWN device destination — the copies are
  // batched, not contiguous in device memory.
  std::array<uint8_t*, 3> const hosts{hb0, hb1, hb2};
  std::array<uint8_t*, 3> const devs{dd0, dd1, dd2};
  auto const& copies = chunks[0]->cpy_req->copies;
  REQUIRE(copies.size() == 3);
  for (size_t i = 0; i < 3; ++i) {
    CHECK(copies[i].src == hosts[i]);
    CHECK(copies[i].src_off == 0);
    CHECK(copies[i].dst == devs[i]);
    CHECK(copies[i].size == 4096);
  }

  complete(chunks);
}

TEST_CASE("prep_host_to_device_ranges keeps a null-buffer range standalone between fused runs",
          "[uring_readv]")
{
  temp_file tf(1 << 20);
  sirius::io::uring::config cfg;
  cfg.use_odirect  = false;
  cfg.max_n_chunks = 16;
  cfg.bounce_size  = 1 << 20;  // the null range still fits one bounce window

  // Five contiguous ranges; the middle one has no caller buffer, so it must be
  // staged through an internal bounce slot and cannot join a readv.  It also
  // wants only [8692, 9692) of its block, to prove the bounce copy carries the
  // within-window offset rather than an absolute src.
  auto* hb0 = fake_ptr(0x10000);
  auto* hb1 = fake_ptr(0x20000);
  auto* hb3 = fake_ptr(0x30000);
  auto* hb4 = fake_ptr(0x40000);
  auto* dd0 = fake_ptr(0x40000000);
  auto* dd1 = fake_ptr(0x50000000);
  auto* ddg = fake_ptr(0x60000000);
  auto* dd3 = fake_ptr(0x70000000);
  auto* dd4 = fake_ptr(0x80000000);
  std::vector<io_host_device_range> ranges{
    {0, 4096, hb0, dd0},
    {4096, 4096, hb1, dd1},
    {8192, 4096, /*copy_offset=*/8692, /*copy_size=*/1000, /*host_buffer=*/nullptr, ddg},
    {12288, 4096, hb3, dd3},
    {16384, 4096, hb4, dd4}};

  auto req = uring_reactor::prep_host_to_device_rxv_request(
    cfg, *tf.obj, ranges, rmm::cuda_stream_view{}, 0);
  auto fut    = req->get_future();
  auto chunks = req->get_all_chunks();

  // => vectored head [0,8k), the null range alone, vectored tail [12k,20k).
  REQUIRE(chunks.size() == 3);
  CHECK(chunks[0]->manager->total_chunks == 3);
  CHECK(chunks[0]->manager->bytes_requested == 4 * 4096 + 1000);  // copy windows, not read spans
  CHECK(chunks[0]->is_vectored());
  CHECK(chunks[0]->chunk.n_chunks() == 2);
  CHECK(chunks[0]->chunk.offset == 0);
  CHECK(chunks[2]->is_vectored());
  CHECK(chunks[2]->chunk.n_chunks() == 2);
  CHECK(chunks[2]->chunk.offset == 12288);

  // The bounce-staged range: null src, the within-window offset in src_off, and
  // dst at the head of its own device destination (which addresses copy_offset).
  CHECK_FALSE(chunks[1]->is_vectored());
  CHECK(chunks[1]->chunk.offset == 8192);
  CHECK(chunks[1]->chunk.size == 4096);
  CHECK_FALSE(chunks[1]->chunk.is_buffer_allocated());
  CHECK(chunks[1]->needs_event_for_synchronization());
  REQUIRE(chunks[1]->cpy_req->copies.size() == 1);
  auto const& gap = chunks[1]->cpy_req->copies[0];
  CHECK(gap.src == nullptr);
  CHECK(gap.src_off == 8692 - 8192);  // data_lo - segment offset
  CHECK(gap.dst == ddg);
  CHECK(gap.size == 1000);

  complete(chunks);
}

TEST_CASE("prep_host_to_device_ranges splits an oversized null-buffer range across bounce windows",
          "[uring_readv]")
{
  temp_file tf(1 << 20);
  sirius::io::uring::config cfg;
  cfg.use_odirect  = false;
  cfg.max_n_chunks = 16;
  cfg.bounce_size  = 4096;  // force the 16 KiB read span over several slots

  // Read [0, 16384) through bounce slots but copy only [5000, 13000): the first
  // slot-sized piece holds none of the copy window and must not be read at all.
  auto* dd = fake_ptr(0x40000000);
  std::vector<io_host_device_range> ranges{
    {0, 4 * 4096, /*copy_offset=*/5000, /*copy_size=*/8000, /*host_buffer=*/nullptr, dd}};

  auto req = uring_reactor::prep_host_to_device_rxv_request(
    cfg, *tf.obj, ranges, rmm::cuda_stream_view{}, 0);
  auto fut    = req->get_future();
  auto chunks = req->get_all_chunks();

  REQUIRE(chunks.size() == 3);  // the [0,4096) piece is dropped, not emitted
  CHECK(chunks[0]->manager->total_chunks == 3);
  CHECK(chunks[0]->manager->bytes_requested == 8000);  // exactly the copy window

  // Every piece is standalone (a bounce slot each) and carries a clipped copy
  // whose dst is biased by how far into the copy window that piece starts.
  size_t copied = 0;
  for (auto const& c : chunks) {
    CHECK_FALSE(c->is_vectored());
    CHECK_FALSE(c->chunk.is_buffer_allocated());
    REQUIRE(c->cpy_req->copies.size() == 1);
    CHECK(c->cpy_req->copies[0].src == nullptr);
    copied += c->cpy_req->copies[0].size;
  }
  CHECK(copied == 8000);

  CHECK(chunks[0]->chunk.offset == 4096);
  CHECK(chunks[0]->cpy_req->copies[0].src_off == 5000 - 4096);  // head of the copy window
  CHECK(chunks[0]->cpy_req->copies[0].dst == dd);
  CHECK(chunks[0]->cpy_req->copies[0].size == 8192 - 5000);
  CHECK(chunks[1]->chunk.offset == 8192);
  CHECK(chunks[1]->cpy_req->copies[0].src_off == 0);
  CHECK(chunks[1]->cpy_req->copies[0].dst == dd + (8192 - 5000));
  CHECK(chunks[1]->cpy_req->copies[0].size == 4096);
  CHECK(chunks[2]->chunk.offset == 12288);
  CHECK(chunks[2]->cpy_req->copies[0].dst == dd + (12288 - 5000));
  CHECK(chunks[2]->cpy_req->copies[0].size == 13000 - 12288);

  complete(chunks);
}

TEST_CASE("prep_host_to_device_ranges rejects an invalid copy window", "[uring_readv]")
{
  temp_file tf(8192);
  sirius::io::uring::config cfg;
  cfg.use_odirect = false;
  cfg.bounce_size = 1 << 20;
  auto* hb        = fake_ptr(0x10000);
  auto* dd        = fake_ptr(0x40000000);

  // A copy window outside its read span would copy bytes the read never landed;
  // one entirely past EOF would copy bytes that do not exist.  Both are caller
  // errors, and both would otherwise desynchronize the plan from the buffers.
  SECTION("copy window starting before the read span")
  {
    std::vector<io_host_device_range> ranges{{4096, 4096, /*copy_offset=*/4000, 100, hb, dd}};
    CHECK_THROWS_AS(uring_reactor::prep_host_to_device_rxv_request(
                      cfg, *tf.obj, ranges, rmm::cuda_stream_view{}, 0),
                    std::runtime_error);
  }

  SECTION("copy window running past the read span")
  {
    std::vector<io_host_device_range> ranges{{4096, 4096, /*copy_offset=*/6000, 4096, hb, dd}};
    CHECK_THROWS_AS(uring_reactor::prep_host_to_device_rxv_request(
                      cfg, *tf.obj, ranges, rmm::cuda_stream_view{}, 0),
                    std::runtime_error);
  }

  SECTION("copy window entirely past the end of the file")
  {
    std::vector<io_host_device_range> ranges{{8192, 4096, /*copy_offset=*/8192, 100, hb, dd}};
    CHECK_THROWS_AS(uring_reactor::prep_host_to_device_rxv_request(
                      cfg, *tf.obj, ranges, rmm::cuda_stream_view{}, 0),
                    std::runtime_error);
  }
}

TEST_CASE("prep_host_to_device_ranges on empty input yields a ready zero-byte request",
          "[uring_readv]")
{
  temp_file tf(4096);
  sirius::io::uring::config cfg;

  SECTION("no ranges at all")
  {
    std::vector<io_host_device_range> ranges;
    auto req = uring_reactor::prep_host_to_device_rxv_request(
      cfg, *tf.obj, ranges, rmm::cuda_stream_view{}, 0);
    REQUIRE(req != nullptr);
    CHECK(req->size() == 0);
    auto fut = req->get_future();
    CHECK(fut.is_ready());
    CHECK(std::move(fut).get() == 0);
  }

  SECTION("only zero-length read spans")
  {
    std::vector<io_host_device_range> ranges{{0, 0, fake_ptr(0x10000), fake_ptr(0x40000000)}};
    auto req = uring_reactor::prep_host_to_device_rxv_request(
      cfg, *tf.obj, ranges, rmm::cuda_stream_view{}, 0);
    REQUIRE(req != nullptr);
    CHECK(req->size() == 0);
    auto fut = req->get_future();
    CHECK(fut.is_ready());
    CHECK(std::move(fut).get() == 0);
  }
}
