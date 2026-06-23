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

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <vector>

using sirius::io::contiguous;
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
  uring_reactor::config cfg;
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
  uring_reactor::config cfg;
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
  uring_reactor::config cfg;
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

TEST_CASE("prep_host_rxv_request keeps a null-buffer segment standalone between vectored runs",
          "[uring_readv]")
{
  temp_file tf(1 << 20);
  uring_reactor::config cfg;
  cfg.use_odirect  = false;
  cfg.max_n_chunks = 16;

  // Five contiguous segments; the middle one has no buffer (it must read through
  // an internal bounce slot, so it cannot join a readv).
  std::vector<io_object_segment> segs{{0, 4096, fake_ptr(0x1000)},
                                      {4096, 4096, fake_ptr(0x2000)},
                                      {8192, 4096},  // null buffer
                                      {12288, 4096, fake_ptr(0x4000)},
                                      {16384, 4096, fake_ptr(0x5000)}};
  auto req    = uring_reactor::prep_host_rxv_request(cfg, *tf.obj, segs);
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

  complete(chunks);
}

TEST_CASE("prep_host_rxv_request does not fuse segments with different fds", "[uring_readv]")
{
  temp_file tf(1 << 20);
  uring_reactor::config cfg;
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
  uring_reactor::config cfg;
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
  uring_reactor::config cfg;
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
  uring_reactor::config cfg;
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

TEST_CASE("prep_host_to_device clips the batched copy to the request window", "[uring_readv]")
{
  temp_file tf(1 << 20);
  uring_reactor::config cfg;
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
  uring_reactor::config cfg;
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
