// SPDX-License-Identifier: Apache-2.0
//
// decompress_column_rows: decoding only the rows a join left behind.
//
// The reference is the decode this one is supposed to replace — a full
// decompress, gathered on the host by the same rows. Differential rather than
// golden on purpose: the sparse walk launches a different grid, visits a
// different set of chunks and computes its output positions differently, so the
// one thing worth proving is that none of that changes the values. A golden
// vector would only prove the fixture matches itself.
//
// The geometries are chosen to be the ones the sparse grid exists for, and the
// ones most likely to break it: a selection touching a handful of chunks out of
// many, one survivor per touched chunk, the last row of a chunk and the first
// row of the next, and a partial tail chunk. All of them are shapes a mask
// walk would handle by visiting every chunk anyway, which is exactly why they
// are the interesting cases here.

#include "api/simpatico_codegen.hpp"
#include "codegen/selection/chunk_row_set.hpp"
#include "test_utils.hpp"

#include <cudf/table/table.hpp>
#include <cudf/types.hpp>

#include <rmm/device_buffer.hpp>
#include <rmm/mr/per_device_resource.hpp>

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace {

constexpr std::int64_t kChunk = ::codegen::kChunkSize;

/// Upload a chunk-CSR for `rows`, which must be ascending and within num_rows.
struct device_row_set {
  rmm::device_buffer chunk_ids, block_offsets, in_chunk_rows;
  sirius::codegen::chunk_row_set view{};

  device_row_set(std::vector<std::int64_t> const& rows,
                 std::int64_t num_rows,
                 rmm::cuda_stream_view stream)
  {
    std::vector<std::uint32_t> chunks, offsets{0};
    std::vector<std::uint16_t> positions;
    for (auto const r : rows) {
      auto const c = static_cast<std::uint32_t>(r / kChunk);
      if (chunks.empty() || chunks.back() != c) {
        if (!chunks.empty()) { offsets.push_back(static_cast<std::uint32_t>(positions.size())); }
        chunks.push_back(c);
      }
      positions.push_back(static_cast<std::uint16_t>(r % kChunk));
    }
    offsets.push_back(static_cast<std::uint32_t>(positions.size()));

    auto const mr = rmm::mr::get_current_device_resource_ref();
    auto upload   = [&](void const* src, std::size_t bytes) {
      rmm::device_buffer buf(bytes, stream, mr);
      if (bytes != 0) {
        cudaMemcpyAsync(buf.data(), src, bytes, cudaMemcpyHostToDevice, stream.value());
      }
      return buf;
    };
    chunk_ids     = upload(chunks.data(), chunks.size() * sizeof(std::uint32_t));
    block_offsets = upload(offsets.data(), offsets.size() * sizeof(std::uint32_t));
    in_chunk_rows = upload(positions.data(), positions.size() * sizeof(std::uint16_t));
    cudaStreamSynchronize(stream.value());

    view.chunk_ids     = static_cast<std::uint32_t const*>(chunk_ids.data());
    view.block_offsets = static_cast<std::uint32_t const*>(block_offsets.data());
    view.in_chunk_rows = static_cast<std::uint16_t const*>(in_chunk_rows.data());
    view.num_touched   = static_cast<std::int64_t>(chunks.size());
    view.num_survivors = static_cast<std::int64_t>(positions.size());
    view.num_rows      = num_rows;
  }
};

std::vector<std::int32_t> read_int32(cudf::column_view const& col, rmm::cuda_stream_view stream)
{
  std::vector<std::int32_t> host(static_cast<std::size_t>(col.size()));
  if (!host.empty()) {
    cudaMemcpyAsync(host.data(),
                    col.data<std::int32_t>(),
                    host.size() * sizeof(std::int32_t),
                    cudaMemcpyDeviceToHost,
                    stream.value());
    cudaStreamSynchronize(stream.value());
  }
  return host;
}

/// Decode `selected` sparsely and require it to equal the full decode's own
/// rows at those positions.
void check_against_full_decode(char const* label,
                               std::int64_t num_rows,
                               std::vector<std::int64_t> const& selected)
{
  auto const stream = cudf::get_default_stream();
  auto const mr     = rmm::mr::get_current_device_resource_ref();

  auto input = make_int32_table(1, static_cast<int>(num_rows), 7);
  auto ct =
    simpatico::compress_with_plan(input->view(), "input -> bitpack -> packed\n", stream, mr);

  auto const full     = simpatico::decompress(ct, stream, mr);
  auto const all_rows = read_int32(full->view().column(0), stream);

  device_row_set rs(selected, num_rows, stream);
  expect(rs.view.valid(), (std::string(label) + ": built an invalid row set").c_str());

  std::string err;
  auto got = simpatico::decompress_column_rows(ct, 0, rs.view, stream, mr, &err);
  expect(got != nullptr, (std::string(label) + ": sparse decode returned null: " + err).c_str());
  expect(got->size() == static_cast<cudf::size_type>(selected.size()),
         (std::string(label) + ": sparse decode row count").c_str());

  auto const sparse = read_int32(got->view(), stream);
  for (std::size_t i = 0; i < selected.size(); ++i) {
    auto const want = all_rows[static_cast<std::size_t>(selected[i])];
    if (sparse[i] != want) {
      std::fprintf(stderr,
                   "FAIL: %s: row %lld (position %zu) decoded %d, full decode has %d\n",
                   label,
                   static_cast<long long>(selected[i]),
                   i,
                   sparse[i],
                   want);
      throw std::runtime_error("sparse decode does not match the full decode");
    }
  }
}

// A few rows spread over a handful of chunks out of many — the shape a
// post-join selection actually has, and the one the sparse grid exists for.
void test_sparse()
{
  std::int64_t const num_rows = 64 * kChunk;
  std::vector<std::int64_t> selected;
  for (std::int64_t c : {std::int64_t{0}, std::int64_t{9}, std::int64_t{40}, std::int64_t{63}}) {
    for (int k = 0; k < 7; ++k) {
      selected.push_back(c * kChunk + k * 131);
    }
  }
  check_against_full_decode("sparse", num_rows, selected);
}

// One survivor per touched chunk: every block decodes a single row, which is
// what sf1000 q17 and q19 look like.
void test_one_per_chunk()
{
  std::int64_t const num_rows = 32 * kChunk;
  std::vector<std::int64_t> selected;
  for (std::int64_t c = 0; c < 32; c += 3) {
    selected.push_back(c * kChunk + 500);
  }
  check_against_full_decode("one per chunk", num_rows, selected);
}

// Chunk edges in both directions, where an off-by-one in the position or the
// output base lands in a neighbour's row.
void test_chunk_boundaries()
{
  std::int64_t const num_rows = 8 * kChunk;
  check_against_full_decode(
    "chunk boundaries", num_rows, {0, 1, 1023, 1024, 1025, 2047, 2048, 8 * kChunk - 1});
}

// A partial tail chunk: the last chunk is short, and its rows must still be
// addressed from its own row 0.
void test_partial_tail()
{
  std::int64_t const num_rows = 3 * kChunk + 261;
  check_against_full_decode(
    "partial tail", num_rows, {5, 3 * kChunk, 3 * kChunk + 1, 3 * kChunk + 260});
}

// Every row of every chunk: the sparse grid degenerates to the dense one and
// must still be right.
void test_all_rows()
{
  std::int64_t const num_rows = 2 * kChunk;
  std::vector<std::int64_t> selected;
  for (std::int64_t r = 0; r < num_rows; ++r) {
    selected.push_back(r);
  }
  check_against_full_decode("all rows", num_rows, selected);
}

}  // namespace

int main()
{
  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
    std::printf("SKIP: no CUDA device\n");
    return 0;
  }

  try {
    test_sparse();
    test_one_per_chunk();
    test_chunk_boundaries();
    test_partial_tail();
    test_all_rows();
  } catch (std::exception const& e) {
    std::fprintf(stderr, "FAIL: %s\n", e.what());
    return 1;
  }

  std::printf("ALL PASS\n");
  return 0;
}
