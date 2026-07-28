// SPDX-License-Identifier: Apache-2.0
//
// Production-path contract for Bitpack's layout boundary:
//   encode kernel: strided OverAllocate scratch
//   persisted rep: dense Compact words only
//   decode kernel: Compact words + synthesized bp_offsets

#include "api/compressed_table_io.hpp"
#include "api/simpatico_codegen.hpp"
#include "codegen/jit/fused_tree.hpp"
#include "codegen/plan/leaf_desc.hpp"
#include "test_utils.hpp"

#include <cudf/column/column_factories.hpp>
#include <cudf/table/table.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <rmm/mr/per_device_resource.hpp>

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr std::int32_t kPartialRows = 17;
constexpr std::int32_t kNumRows = 3 * static_cast<std::int32_t>(codegen::kChunkSize) + kPartialRows;

std::unique_ptr<cudf::table> make_layout_fixture()
{
  std::vector<std::int32_t> host(static_cast<std::size_t>(kNumRows));
  for (std::int32_t i = 0; i < kNumRows; ++i) {
    auto const chunk = i / codegen::kChunkSize;
    auto const pos   = i % codegen::kChunkSize;
    switch (chunk) {
      case 0:
        // Zero-bit chunk: all residuals from chunk_min are zero.
        host[static_cast<std::size_t>(i)] = 7;
        break;
      case 1:
        // One-bit chunk.
        host[static_cast<std::size_t>(i)] = 100 + (pos & 1);
        break;
      case 2:
        // Ten-bit chunk.
        host[static_cast<std::size_t>(i)] = pos;
        break;
      default:
        // Partial final chunk with a 16-bit range.
        host[static_cast<std::size_t>(i)] = pos == kPartialRows - 1 ? 65535 : 0;
        break;
    }
  }

  auto col = cudf::make_numeric_column(
    cudf::data_type{cudf::type_id::INT32}, kNumRows, cudf::mask_state::UNALLOCATED);
  auto const rc = cudaMemcpy(col->mutable_view().head<std::int32_t>(),
                             host.data(),
                             host.size() * sizeof(std::int32_t),
                             cudaMemcpyHostToDevice);
  if (rc != cudaSuccess) throw std::runtime_error("fixture HtoD copy failed");

  std::vector<std::unique_ptr<cudf::column>> cols;
  cols.push_back(std::move(col));
  return std::make_unique<cudf::table>(std::move(cols));
}

simpatico::leaf_desc const& find_bitpack_leaf(
  std::vector<std::vector<simpatico::leaf_desc>> const& columns)
{
  expect(columns.size() == 1, "describe: expected one column");
  auto const it = std::find_if(columns[0].begin(), columns[0].end(), [](auto const& leaf) {
    return leaf.kind == simpatico::OpId::Bitpack;
  });
  expect(it != columns[0].end(), "describe: missing bitpack leaf");
  return *it;
}

simpatico::leaf_buffer_desc const& find_buffer(simpatico::leaf_desc const& leaf, char const* name)
{
  auto const it = std::find_if(leaf.buffers.begin(), leaf.buffers.end(), [&](auto const& buffer) {
    return buffer.name == name;
  });
  expect(it != leaf.buffers.end(), (std::string("missing bitpack buffer: ") + name).c_str());
  return *it;
}

template <typename T>
std::vector<T> copy_buffer(simpatico::leaf_buffer_desc const& buffer)
{
  expect(buffer.size_bytes == buffer.num_rows * sizeof(T), "buffer byte/row size mismatch");
  std::vector<T> host(static_cast<std::size_t>(buffer.num_rows));
  if (!host.empty()) {
    auto const rc =
      cudaMemcpy(host.data(), buffer.device_ptr, buffer.size_bytes, cudaMemcpyDeviceToHost);
    if (rc != cudaSuccess) throw std::runtime_error("buffer DtoH copy failed");
  }
  return host;
}

void test_compact_persistence_and_decode()
{
  auto input        = make_layout_fixture();
  auto const stream = cudf::get_default_stream();
  auto const mr     = rmm::mr::get_current_device_resource_ref();

  auto compressed = simpatico::compress_with_plan(
    input->view(), "input -> bitpack -> chunk_min, chunk_count, chunk_bits, packed\n", stream, mr);

  auto const description  = compressed.describe(stream);
  auto const& leaf        = find_bitpack_leaf(description);
  auto const& counts_desc = find_buffer(leaf, "chunk_count");
  auto const& bits_desc   = find_buffer(leaf, "chunk_bits");
  auto const& packed_desc = find_buffer(leaf, "packed");

  auto const counts = copy_buffer<std::int32_t>(counts_desc);
  auto const bits   = copy_buffer<std::uint8_t>(bits_desc);
  expect(counts == std::vector<std::int32_t>({1024, 1024, 1024, kPartialRows}),
         "unexpected per-chunk counts");
  expect(bits == std::vector<std::uint8_t>({0, 1, 10, 16}), "unexpected per-chunk bit widths");

  std::uint64_t expected_words = 0;
  for (std::size_t i = 0; i < counts.size(); ++i) {
    if (counts[i] == 0) {
      // Match NwordsFromChunk: empty structural chunks retain one readable
      // word even though normal top-level input cannot produce such a chunk.
      ++expected_words;
      continue;
    }
    auto const live_bits =
      static_cast<std::uint64_t>(counts[i]) * static_cast<std::uint64_t>(bits[i]);
    expected_words += (live_bits + 31) / 32;
  }
  expect(expected_words == 361, "fixture no longer exercises the intended compact sizes");
  expect(packed_desc.num_rows == expected_words, "persisted packed column is not compact");
  expect(packed_desc.size_bytes == expected_words * sizeof(std::uint32_t),
         "persisted packed byte size includes OverAllocate stride or decode slack");

  // Header serialization must expose only the logical Compact bytes. The
  // private two-word decode over-read slack stays allocation-only.
  std::vector<std::uint8_t> header;
  std::vector<simpatico::payload_buffer_ref> payload_refs;
  std::uint64_t payload_bytes = 0;
  auto err                    = simpatico::build_compressed_table_header(
    compressed, header, payload_refs, payload_bytes, stream);
  expect(err.empty(), err.empty() ? "header build failed" : err.c_str());
  auto const packed_ref =
    std::find_if(payload_refs.begin(), payload_refs.end(), [&](auto const& ref) {
      return ref.device_ptr == packed_desc.device_ptr;
    });
  expect(packed_ref != payload_refs.end(), "serialized payload omitted packed buffer");
  expect(packed_ref->size_bytes == packed_desc.size_bytes,
         "serialized packed payload is not logically compact");
  expect(packed_ref->alloc_bytes >= packed_ref->size_bytes + 2 * sizeof(std::uint32_t),
         "reconstructed packed allocation lacks decode gather slack");
  expect(payload_bytes == std::accumulate(payload_refs.begin(),
                                          payload_refs.end(),
                                          std::uint64_t{0},
                                          [](std::uint64_t total, auto const& ref) {
                                            return total + ref.size_bytes;
                                          }),
         "payload byte count is not the sum of logical buffer sizes");

  std::vector<std::uint8_t> payload(static_cast<std::size_t>(payload_bytes));
  for (auto const& ref : payload_refs) {
    if (ref.size_bytes == 0) continue;
    auto const rc = cudaMemcpy(
      payload.data() + ref.offset, ref.device_ptr, ref.size_bytes, cudaMemcpyDeviceToHost);
    if (rc != cudaSuccess) throw std::runtime_error("payload DtoH copy failed");
  }

  simpatico::payload_fetch_fn fetch =
    [&](std::uint64_t offset, std::size_t size, void* dst, rmm::cuda_stream_view fetch_stream) {
      auto const rc = cudaMemcpyAsync(
        dst, payload.data() + offset, size, cudaMemcpyHostToDevice, fetch_stream.value());
      if (rc != cudaSuccess) throw std::runtime_error("payload HtoD fetch failed");
    };
  std::string read_error;
  auto restored =
    simpatico::read_compressed_table_from_memory(header, fetch, stream, mr, &read_error);
  expect(read_error.empty(), read_error.empty() ? "memory read failed" : read_error.c_str());
  expect(restored.num_columns() == 1, "memory read returned wrong column count");

  auto output = simpatico::decompress(restored, stream, mr);
  expect(output != nullptr, "compact decode returned null");
  expect(columns_equal(input->view().column(0), output->view().column(0)),
         "compact decode with synthesized bp_offsets changed data");
}

}  // namespace

int main()
{
  if (cudaSetDevice(0) != cudaSuccess) {
    std::fprintf(stderr, "test_bitpack_layout_contract: cudaSetDevice failed\n");
    return 1;
  }
  try {
    test_compact_persistence_and_decode();
    std::printf("test_bitpack_layout_contract: PASS\n");
    return 0;
  } catch (std::exception const& e) {
    std::fprintf(stderr, "test_bitpack_layout_contract: FAIL: %s\n", e.what());
    return 1;
  }
}
