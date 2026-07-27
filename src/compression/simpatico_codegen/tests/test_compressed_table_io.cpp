// SPDX-License-Identifier: Apache-2.0
//
// Tests for write_compressed_table / read_compressed_table (.hpln v6).
//
// Each test_* function throws std::runtime_error on failure; main() catches and
// reports.  Tests are intentionally independent so a failure in one does not
// mask others.

#include "api/compressed_table_io.hpp"
#include "api/simpatico_codegen.hpp"
#include "codegen/jit/nvrtc_compiler.hpp"
#include "codegen/plan/leaf_desc.hpp"
#include "test_utils.hpp"

#include <cudf/table/table.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <unistd.h>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

// RAII temp file: created on construction, deleted on destruction.
struct TmpFile {
  std::string path;
  TmpFile()
  {
    char buf[] = "/tmp/simpatico_io_test_XXXXXX";
    int fd     = mkstemp(buf);
    if (fd != -1) ::close(fd);
    path = std::string(buf) + ".hpln";
  }
  ~TmpFile() { std::remove(path.c_str()); }
};

// Flatten leaf kinds from describe() into a simple vector for comparison.
std::vector<simpatico::OpId> leaf_kinds(simpatico::compressed_table const& ct)
{
  std::vector<simpatico::OpId> out;
  for (auto const& descs : ct.describe())
    for (auto const& ld : descs)
      out.push_back(ld.kind);
  return out;
}

// Flatten buffer names from describe() into a simple vector for comparison.
std::vector<std::string> buf_names(simpatico::compressed_table const& ct)
{
  std::vector<std::string> out;
  for (auto const& descs : ct.describe())
    for (auto const& ld : descs)
      for (auto const& bd : ld.buffers)
        out.push_back(bd.name);
  return out;
}

// ---------------------------------------------------------------------------
// Core roundtrip helper
// ---------------------------------------------------------------------------

void io_roundtrip(char const* label,
                  cudf::table_view input,
                  std::string const& dsl,
                  std::vector<std::string> column_names = {})
{
  auto stream = cudf::get_default_stream();

  // Compress.
  simpatico::compressed_table ct1 = simpatico::compress_with_plan(
    input, dsl, stream, rmm::mr::get_current_device_resource_ref(), column_names);
  expect(ct1.columns.size() == static_cast<std::size_t>(input.num_columns()),
         (std::string(label) + ": compress column count").c_str());

  // Write.
  TmpFile tmp;
  std::string werr = simpatico::write_compressed_table(ct1, tmp.path);
  expect(werr.empty(), (std::string(label) + ": write error: " + werr).c_str());

  // Read.
  std::string rerr;
  simpatico::compressed_table ct2 = simpatico::read_compressed_table(
    tmp.path, stream, rmm::mr::get_current_device_resource_ref(), &rerr);
  expect(rerr.empty(), (std::string(label) + ": read error: " + rerr).c_str());
  expect(ct2.columns.size() == ct1.columns.size(),
         (std::string(label) + ": read column count").c_str());

  // Metadata survives.
  for (std::size_t i = 0; i < ct1.columns.size(); ++i) {
    auto const& c1 = ct1.columns[i];
    auto const& c2 = ct2.columns[i];
    expect(c1.dtype == c2.dtype, (std::string(label) + ": dtype col " + std::to_string(i)).c_str());
    expect(c1.num_rows == c2.num_rows,
           (std::string(label) + ": num_rows col " + std::to_string(i)).c_str());
    expect(c1.name == c2.name, (std::string(label) + ": name col " + std::to_string(i)).c_str());
    // The plan is persisted structurally (node array), not as DSL text, so the
    // compound's presence round-trips even though plan_dsl is not restored.
    expect((c1.compound != nullptr) == (c2.compound != nullptr),
           (std::string(label) + ": compound presence col " + std::to_string(i)).c_str());
  }

  // Leaf structure survives (same kinds and buffer names).
  expect(leaf_kinds(ct1) == leaf_kinds(ct2),
         (std::string(label) + ": leaf kinds mismatch").c_str());
  expect(buf_names(ct1) == buf_names(ct2),
         (std::string(label) + ": buffer names mismatch").c_str());

  // Decompress and compare pixel-exact to original.
  auto out = simpatico::decompress(ct2, stream, rmm::mr::get_current_device_resource_ref());
  expect(out != nullptr, (std::string(label) + ": decompress returned null").c_str());
  expect(out->num_columns() == input.num_columns(),
         (std::string(label) + ": decompressed column count").c_str());
  for (int i = 0; i < input.num_columns(); ++i)
    expect(columns_equal_any(input.column(i), out->view().column(i), stream),
           (std::string(label) + ": data mismatch col " + std::to_string(i)).c_str());
}

// In-memory (pinned-blob) roundtrip via the production pin-path entry points:
// build_compressed_table_header enumerates payload buffers, we assemble the
// payload host-side, then read_compressed_table_from_memory reconstructs
// through the same fetch seam pin_table uses.
void memory_roundtrip(char const* label, cudf::table_view input, std::string const& dsl)
{
  auto stream = cudf::get_default_stream();
  auto mr     = rmm::mr::get_current_device_resource_ref();

  simpatico::compressed_table ct = simpatico::compress_with_plan(input, dsl, stream, mr);

  std::vector<std::uint8_t> header;
  std::vector<simpatico::payload_buffer_ref> buffers;
  std::uint64_t payload_bytes = 0;
  auto const herr =
    simpatico::build_compressed_table_header(ct, header, buffers, payload_bytes, stream);
  expect(herr.empty(), (std::string(label) + ": header error: " + herr).c_str());

  std::vector<std::uint8_t> payload(payload_bytes);
  stream.synchronize();
  for (auto const& b : buffers) {
    if (b.size_bytes == 0 || b.device_ptr == nullptr) continue;
    expect(cudaMemcpy(payload.data() + b.offset,
                      b.device_ptr,
                      static_cast<std::size_t>(b.size_bytes),
                      cudaMemcpyDeviceToHost) == cudaSuccess,
           (std::string(label) + ": payload staging copy failed").c_str());
  }

  simpatico::payload_fetch_fn fetch =
    [&payload](std::uint64_t off, std::size_t sz, void* dst, rmm::cuda_stream_view s) {
      if (cudaMemcpyAsync(dst, payload.data() + off, sz, cudaMemcpyHostToDevice, s.value()) !=
          cudaSuccess)
        throw std::runtime_error("memory_roundtrip: fetch copy failed");
    };

  std::string rerr;
  simpatico::compressed_table ct2 =
    simpatico::read_compressed_table_from_memory(header, fetch, stream, mr, &rerr);
  expect(rerr.empty(), (std::string(label) + ": read error: " + rerr).c_str());

  auto out = simpatico::decompress(ct2, stream, mr);
  expect(out != nullptr, (std::string(label) + ": decompress returned null").c_str());
  expect(out->num_columns() == input.num_columns(),
         (std::string(label) + ": decompressed column count").c_str());
  for (int i = 0; i < input.num_columns(); ++i)
    expect(columns_equal_any(input.column(i), out->view().column(i), stream),
           (std::string(label) + ": data mismatch col " + std::to_string(i)).c_str());
}

void expect_selected_columns(char const* label,
                             cudf::table const& output,
                             cudf::table_view const& input,
                             std::span<const std::size_t> selected,
                             rmm::cuda_stream_view stream)
{
  expect(output.num_columns() == static_cast<cudf::size_type>(selected.size()),
         (std::string(label) + ": output column count").c_str());
  for (std::size_t i = 0; i < selected.size(); ++i) {
    expect(columns_equal_any(input.column(static_cast<cudf::size_type>(selected[i])),
                             output.view().column(static_cast<cudf::size_type>(i)),
                             stream),
           (std::string(label) + ": data mismatch at output column " + std::to_string(i)).c_str());
  }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

// 1. Fused delta→rle→bitpack plan: exercises codegen_fused_representation
//    serialisation and the multi-buffer fused detect path in rep_from_leaf_desc.
void test_fused_delta_rle_bitpack()
{
  auto t = make_int32_table(1, 4096, 7);
  io_roundtrip("fused_delta_rle_bitpack",
               t->view(),
               "input -> delta -> differences\n"
               "delta.differences -> rle -> values, runs\n"
               "delta.differences.values -> bitpack\n"
               "delta.differences.runs -> bitpack\n");
}

// 1b. Regression: tail-routed nested RLE on a bitpack metadata channel, at a
//     scale where the metadata channel is far shorter than the column.
//
//     `bitpack` is a fused-tree LEAF (build_fused_tree), so `bitpack.chunk_count
//     -> rle` is decoded as a SEPARATE codegen subtree whose true length is the
//     parent's per-chunk count (~ceil(col_rows/kChunkSize)), i.e. ~1000x below
//     the column row count. The decode grid is ceil(rep->num_rows / kChunkSize),
//     so the subtree's rep MUST carry its own length. Before the fix,
//     reconstruction gave every fused rep the *column* row count, so this
//     subtree launched ceil(col_rows/1024) blocks against per-chunk metadata
//     built for only ceil(chunk_count_len/1024) chunks — a device out-of-bounds
//     read (context-fatal on real data / hardware). ~300k rows yields ~293
//     parent chunks, so the buggy grid overran a 2-entry rle_runs_offsets by
//     ~290 blocks. The tiny hand-written plans elsewhere never dispatch a nested
//     metadata subtree at this scale, which is why this went uncaught.
void test_nested_metadata_rle_scale()
{
  auto t = make_int32_table(1, 300000, 5);
  io_roundtrip("nested_metadata_rle_scale",
               t->view(),
               "input -> bitpack -> chunk_min, chunk_count, chunk_bits, packed\n"
               "bitpack.chunk_min -> identity\n"
               "bitpack.chunk_bits -> identity\n"
               "bitpack.packed -> identity\n"
               "bitpack.chunk_count -> rle -> runs, values\n"
               "bitpack.chunk_count.runs -> rle -> runs, values\n"
               "bitpack.chunk_count.values -> rle -> runs, values\n");
}

// 2. FOR + fused bitpack on deltas: JIT fused region covers both ops.
void test_for_bitpack()
{
  auto t = make_int32_table(1, 4096, 11);
  io_roundtrip("for_bitpack",
               t->view(),
               "input -> for -> deltas, references\n"
               "for.deltas -> bitpack\n");
}

// 3. FOR only: deltas stored as Raw fixed-stride leaf (no downstream bitpack).
void test_for_only()
{
  auto t = make_int32_table(1, 4096, 13);
  io_roundtrip("for_only", t->view(), "input -> for -> deltas, references\n");
}

// 3b. ZigZag leaf: exercises the codegen_fused_representation("zigzag") channel
//     write/read (OpId::Zigzag -> make_fused_rep), terminal and with an
//     entropy tail on the stored channel.
void test_zigzag()
{
  auto t = make_int32_table(1, 4096, 113);
  io_roundtrip("zigzag_terminal",
               t->view(),
               "input -> delta -> differences\n"
               "delta.differences -> zigzag -> zigzag\n");
  io_roundtrip("zigzag_ans",
               t->view(),
               "input -> delta -> differences\n"
               "delta.differences -> zigzag -> zigzag\n"
               "delta.differences.zigzag -> ans\n");
}

// 4. ALP (floating-point): exercises alp_compressed_representation with its
//    four output channels and alp_rd leaf_meta on a FLOAT32 column.
void test_alp_f32()
{
  auto t = make_f32_table(1, 4096, 17);
  io_roundtrip("alp_f32", t->view(), "input -> alp\n");
}

// Bitextract: multi-output op whose planes are terminal channel leaves,
// exercising per-node output_names + terminal-slot attachment on read.
void test_bitextract_f32()
{
  auto t = make_f32_table(1, 4096, 29);
  io_roundtrip(
    "bitextract_f32", t->view(), "input -> bitextract_f32 -> sign, exponent, mantissa\n");
}

// Bitjoin DAG: the reconvergent node carries structured attrs (input node refs
// + channels) that must survive the structural node serialization.
void test_bitjoin_f32()
{
  auto t = make_f32_table(1, 4096, 31);
  io_roundtrip("bitjoin_f32",
               t->view(),
               "input -> bitextract_f32 -> sign, exponent, mantissa\n"
               "bitextract_f32.sign, bitextract_f32.exponent, bitextract_f32.mantissa "
               "-> bitjoin_f32 -> rejoined\n");
}

// ALP-RD (f64): six output channels + right_bw carried in a channel.
void test_alp_rd_f64()
{
  auto t = make_f64_table(1, 4096, 37);
  io_roundtrip("alp_rd_f64", t->view(), "input -> alp_rd\n");
}

// Dictionary (STRING): variable channel set on a STRING column, exercising the
// STRING column dtype tag and keys_offsets/keys_chars/indices channels.
void test_dictionary()
{
  auto t = make_string_table(4096, cudf::get_default_stream());
  io_roundtrip("dictionary", t->view(), "input -> dictionary\n");
}

// 5. Multi-column: three columns with three different plans in one file.
//    Verifies the column loop in write/read and that per-column plans don't bleed.
void test_multi_column()
{
  auto t = make_int32_table(3, 2048, 21);
  io_roundtrip("multi_column",
               t->view(),
               "input -> delta -> differences\n"
               "delta.differences -> rle -> values, runs\n"
               "delta.differences.values -> bitpack\n"
               "delta.differences.runs -> bitpack\n"
               "---\n"
               "input -> for -> deltas, references\n"
               "for.deltas -> bitpack\n"
               "---\n"
               "input -> for -> deltas, references\n");
}

// Selective decompression preserves requested order and duplicates, accepts an
// empty projection, rejects invalid indices, and never touches an unselected
// column. Exercise all three overload families.
void test_selective_decompression()
{
  auto stream = cudf::get_default_stream();
  auto mr     = rmm::mr::get_current_device_resource_ref();
  auto t      = make_int32_table(3, 2048, 29);
  auto ct     = simpatico::compress_with_plan(t->view(),
                                          "input -> for -> deltas, references\n"
                                              "---\n"
                                              "input -> delta -> differences\n"
                                              "delta.differences -> bitpack\n"
                                              "---\n"
                                              "input -> bitpack\n",
                                          stream,
                                          mr);

  // If an implementation accidentally decompresses every column before
  // projecting, this deliberately invalid unselected column makes that visible.
  ct.columns[1].compound.reset();

  std::vector<std::size_t> reordered_with_duplicate{2, 0, 2};
  auto sequential = simpatico::decompress(ct, reordered_with_duplicate, stream, mr);
  expect(sequential != nullptr, "selective sequential: decompress returned null");
  expect_selected_columns(
    "selective sequential", *sequential, t->view(), reordered_with_duplicate, stream);

  std::vector<std::size_t> reordered{2, 0};
  auto threaded = simpatico::decompress(ct, reordered, 2, mr);
  expect(threaded != nullptr, "selective threaded: decompress returned null");
  expect_selected_columns("selective threaded", *threaded, t->view(), reordered, stream);

  simpatico::stream_pool pool;
  expect(pool.init(2), "selective stream_pool: init failed");
  auto pooled = simpatico::decompress(ct, reordered, pool, mr);
  pool.sync_all();
  expect(pooled != nullptr, "selective stream_pool: decompress returned null");
  expect_selected_columns("selective stream_pool", *pooled, t->view(), reordered, stream);

  std::vector<std::size_t> empty_selection;
  auto empty = simpatico::decompress(ct, empty_selection, stream, mr);
  expect(empty != nullptr && empty->num_columns() == 0, "selective empty: expected an empty table");

  std::vector<std::size_t> invalid{0, 3};
  bool threw = false;
  try {
    (void)simpatico::decompress(ct, invalid, stream, mr);
  } catch (std::runtime_error const&) {
    threw = true;
  }
  expect(threw, "selective invalid: expected out-of-range exception");
}

// The subset in-memory reader must reconstruct only requested columns. Its
// payload callback is instrumented so this test checks I/O exclusion directly,
// in addition to metadata, order, duplicate, empty, and error behavior.
void test_memory_subset_read()
{
  auto stream = cudf::get_default_stream();
  auto mr     = rmm::mr::get_current_device_resource_ref();
  auto t      = make_int32_table(3, 2048, 37);
  auto ct     = simpatico::compress_with_plan(t->view(),
                                          "input -> for -> deltas, references\n"
                                              "---\n"
                                              "input -> delta -> differences\n"
                                              "delta.differences -> bitpack\n"
                                              "---\n"
                                              "input -> bitpack\n",
                                          stream,
                                          mr,
                                              {"first", "second", "third"});

  std::vector<std::uint8_t> header;
  std::vector<simpatico::payload_buffer_ref> buffers;
  std::uint64_t payload_bytes = 0;
  auto const herr =
    simpatico::build_compressed_table_header(ct, header, buffers, payload_bytes, stream);
  expect(herr.empty(), ("memory subset: header error: " + herr).c_str());

  std::vector<std::uint8_t> payload(payload_bytes);
  stream.synchronize();
  for (auto const& b : buffers) {
    if (b.size_bytes == 0 || b.device_ptr == nullptr) continue;
    expect(cudaMemcpy(payload.data() + b.offset,
                      b.device_ptr,
                      static_cast<std::size_t>(b.size_bytes),
                      cudaMemcpyDeviceToHost) == cudaSuccess,
           "memory subset: payload staging copy failed");
  }

  auto const descriptions = ct.describe(stream);
  std::vector<std::size_t> buffer_owner;
  for (std::size_t ci = 0; ci < ct.num_columns(); ++ci) {
    for (auto const& leaf : descriptions[ci]) {
      for (std::size_t bi = 0; bi < leaf.buffers.size(); ++bi)
        buffer_owner.push_back(ci);
    }
  }
  expect(buffer_owner.size() == buffers.size(), "memory subset: buffer owner map mismatch");

  bool fetched_first      = false;
  bool fetched_third      = false;
  std::size_t fetch_count = 0;
  simpatico::payload_fetch_fn fetch =
    [&](std::uint64_t off, std::size_t sz, void* dst, rmm::cuda_stream_view s) {
      ++fetch_count;
      auto const it = std::find_if(buffers.begin(), buffers.end(), [&](auto const& b) {
        return b.offset == off && b.size_bytes == sz;
      });
      expect(it != buffers.end(), "memory subset: fetched unknown payload range");
      auto const bi    = static_cast<std::size_t>(std::distance(buffers.begin(), it));
      auto const owner = buffer_owner[bi];
      expect(owner != 1, "memory subset: fetched unselected second column");
      fetched_first = fetched_first || owner == 0;
      fetched_third = fetched_third || owner == 2;
      if (cudaMemcpyAsync(dst, payload.data() + off, sz, cudaMemcpyHostToDevice, s.value()) !=
          cudaSuccess)
        throw std::runtime_error("memory subset: fetch copy failed");
    };

  std::vector<std::size_t> selected{2, 0, 2};
  std::string rerr;
  auto subset =
    simpatico::read_compressed_table_subset_from_memory(header, fetch, selected, stream, mr, &rerr);
  expect(rerr.empty(), ("memory subset: read error: " + rerr).c_str());
  expect(subset.num_columns() == selected.size(), "memory subset: compressed column count");
  expect(subset.columns[0].name == std::optional<std::string>{"third"} &&
           subset.columns[1].name == std::optional<std::string>{"first"} &&
           subset.columns[2].name == std::optional<std::string>{"third"},
         "memory subset: names do not preserve selection order and duplicates");
  expect(fetch_count > 0 && fetched_first && fetched_third,
         "memory subset: selected payloads were not fetched");

  auto out = simpatico::decompress(subset, stream, mr);
  expect(out != nullptr, "memory subset: decompress returned null");
  expect_selected_columns("memory subset", *out, t->view(), selected, stream);

  std::size_t empty_fetch_count = 0;
  simpatico::payload_fetch_fn empty_fetch =
    [&](std::uint64_t, std::size_t, void*, rmm::cuda_stream_view) { ++empty_fetch_count; };
  std::vector<std::size_t> empty_selection;
  rerr.clear();
  auto empty = simpatico::read_compressed_table_subset_from_memory(
    header, empty_fetch, empty_selection, stream, mr, &rerr);
  expect(rerr.empty(), "memory subset empty: unexpected read error");
  expect(empty.num_columns() == 0, "memory subset empty: expected zero columns");
  expect(empty_fetch_count == 0, "memory subset empty: payload callback was invoked");

  std::vector<std::size_t> invalid{0, 3};
  rerr.clear();
  auto invalid_result = simpatico::read_compressed_table_subset_from_memory(
    header, empty_fetch, invalid, stream, mr, &rerr);
  expect(!rerr.empty(), "memory subset invalid: expected an error");
  expect(invalid_result.num_columns() == 0, "memory subset invalid: expected empty result");
  expect(empty_fetch_count == 0, "memory subset invalid: payload callback was invoked");
}

// 6. Column names survive write→read.
void test_column_names_survive()
{
  auto t = make_int32_table(2, 1024, 3);
  io_roundtrip("column_names_survive",
               t->view(),
               "input -> for -> deltas, references\n"
               "---\n"
               "input -> for -> deltas, references\n",
               {"price", "volume"});
}

// 7. Zero-row table: edge case that should produce an empty but valid file
//    that round-trips without error.
void test_zero_rows()
{
  auto t = make_int32_table(1, 0, 0);
  io_roundtrip("zero_rows",
               t->view(),
               "input -> delta -> differences\n"
               "delta.differences -> rle -> values, runs\n"
               "delta.differences.values -> bitpack\n"
               "delta.differences.runs -> bitpack\n");
}

// 8. Error: file does not exist.
void test_error_not_found()
{
  std::string err;
  auto ct = simpatico::read_compressed_table("/nonexistent/path/that/does/not/exist.hpln",
                                             cudf::get_default_stream(),
                                             rmm::mr::get_current_device_resource_ref(),
                                             &err);
  expect(!err.empty(), "error_not_found: expected non-empty error");
  expect(ct.columns.empty(), "error_not_found: expected empty result");
}

// Error: a non-HPLN file (garbage) is rejected rather than crashing.
void test_error_garbage()
{
  TmpFile tmp;
  {
    std::ofstream f(tmp.path, std::ios::binary);
    f.write("garbage data", 12);
  }
  std::string err;
  auto ct = simpatico::read_compressed_table(
    tmp.path, cudf::get_default_stream(), rmm::mr::get_current_device_resource_ref(), &err);
  expect(!err.empty(), "error_garbage: expected non-empty error");
  expect(ct.columns.empty(), "error_garbage: expected empty result");
}

// Error: wrong magic bytes.
void test_error_bad_magic()
{
  TmpFile tmp;
  {
    std::ofstream f(tmp.path, std::ios::binary);
    char bad[] = {'X', 'X', 'X', 'X', 8, 0, 0};
    f.write(bad, sizeof(bad));
  }
  std::string err;
  auto ct = simpatico::read_compressed_table(
    tmp.path, cudf::get_default_stream(), rmm::mr::get_current_device_resource_ref(), &err);
  expect(!err.empty(), "error_bad_magic: expected non-empty error");
}

// Error: correct magic but unsupported version number.
void test_error_bad_version()
{
  TmpFile tmp;
  {
    std::ofstream f(tmp.path, std::ios::binary);
    char data[] = {'H', 'P', 'L', 'N', 99, 0, 0};  // version 99
    f.write(data, sizeof(data));
  }
  std::string err;
  auto ct = simpatico::read_compressed_table(
    tmp.path, cudf::get_default_stream(), rmm::mr::get_current_device_resource_ref(), &err);
  expect(!err.empty(), "error_bad_version: expected non-empty error");
}

// Error: identity on a STRING column has no single contiguous payload buffer;
// build_compressed_table_header must reject it loudly and clear its outputs.
// identity on a STRING column decomposes via str_split, so it round-trips to
// file (both the file path and the in-memory pin path).
void test_identity_string_roundtrip()
{
  auto stream = cudf::get_default_stream();
  auto input  = make_string_table(128, stream);
  io_roundtrip("identity_string", input->view(), "input -> identity\n");
  memory_roundtrip("identity_string_mem", input->view(), "input -> identity\n");
}

// The two production STRING plan shapes from plans/tpch_sf1000 (customer/
// supplier): variable-length "address" (offsets delta -> ans) and constant-
// length "phone" (offsets delta -> rle with terminal runs/values). Exercised
// through BOTH the file path and the in-memory pin path — the shapes the
// rerouted sf1000 plans serialize in production.
void test_str_split_plan_shapes_roundtrip()
{
  auto stream = cudf::get_default_stream();

  std::vector<std::string> addresses;
  std::vector<std::string> phones;
  addresses.reserve(512);
  phones.reserve(512);
  for (int i = 0; i < 512; ++i) {
    addresses.push_back("No. " + std::to_string((i * 37) % 990) + " Elm Street, Apt " +
                        std::to_string(i % 97));
    char buf[16];
    std::snprintf(buf,
                  sizeof(buf),
                  "%02d-%03d-%03d-%03d",
                  i % 100,
                  (i * 7) % 1000,
                  (i * 13) % 1000,
                  (i * 31) % 1000);
    phones.emplace_back(buf);  // constant length 14 — one RLE run of offset deltas
  }
  auto addr_tbl  = make_strings_table(addresses, {}, stream);
  auto phone_tbl = make_strings_table(phones, {}, stream);

  std::string const address_dsl =
    "input -> str_split -> offsets, chars\n"
    "str_split.offsets -> delta -> differences\n"
    "str_split.chars -> deflate\n"
    "str_split.offsets.differences -> ans\n";
  std::string const phone_dsl =
    "input -> str_split -> offsets, chars\n"
    "str_split.offsets -> delta -> differences\n"
    "str_split.offsets.differences -> rle -> runs, values\n"
    "str_split.chars -> bitpack -> chunk_min, chunk_count, chunk_bits, packed\n"
    "str_split.chars.packed -> deflate\n"
    "str_split.chars.chunk_count -> rle -> runs, values\n"
    "str_split.chars.chunk_bits -> bitcomp\n"
    "str_split.chars.chunk_min -> rle -> runs, values\n";

  io_roundtrip("str_split_address_shape", addr_tbl->view(), address_dsl);
  io_roundtrip("str_split_phone_shape", phone_tbl->view(), phone_dsl);
  memory_roundtrip("str_split_address_shape_mem", addr_tbl->view(), address_dsl);
  memory_roundtrip("str_split_phone_shape_mem", phone_tbl->view(), phone_dsl);

  // Bare terminal: the str_split rep itself is the node's stored leaf.
  std::string const bare_dsl = "input -> str_split\n";
  io_roundtrip("str_split_bare_terminal", addr_tbl->view(), bare_dsl);
  memory_roundtrip("str_split_bare_terminal_mem", addr_tbl->view(), bare_dsl);
}

}  // namespace

int main()
{
  if (cudaSetDevice(0) != cudaSuccess) {
    std::fprintf(stderr, "test_compressed_table_io: cudaSetDevice failed\n");
    return 1;
  }

  struct Case {
    char const* name;
    void (*fn)();
  };
  Case cases[] = {
    {"fused_delta_rle_bitpack", test_fused_delta_rle_bitpack},
    {"nested_metadata_rle_scale", test_nested_metadata_rle_scale},
    {"for_bitpack", test_for_bitpack},
    {"for_only", test_for_only},
    {"zigzag", test_zigzag},
    {"alp_f32", test_alp_f32},
    {"bitextract_f32", test_bitextract_f32},
    {"bitjoin_f32", test_bitjoin_f32},
    {"alp_rd_f64", test_alp_rd_f64},
    {"dictionary", test_dictionary},
    {"multi_column", test_multi_column},
    {"selective_decompression", test_selective_decompression},
    {"memory_subset_read", test_memory_subset_read},
    {"column_names_survive", test_column_names_survive},
    {"zero_rows", test_zero_rows},
    {"error_not_found", test_error_not_found},
    {"error_garbage", test_error_garbage},
    {"error_bad_magic", test_error_bad_magic},
    {"error_bad_version", test_error_bad_version},
    {"identity_string_roundtrip", test_identity_string_roundtrip},
    {"str_split_plan_shapes", test_str_split_plan_shapes_roundtrip},
  };

  int failures = 0;
  for (auto const& c : cases) {
    try {
      c.fn();
      std::printf("  PASS  %s\n", c.name);
    } catch (std::exception const& e) {
      std::fprintf(stderr, "  FAIL  %s: %s\n", c.name, e.what());
      ++failures;
    }
  }

  if (failures == 0) {
    std::printf("test_compressed_table_io: PASS (%zu cases)\n", sizeof(cases) / sizeof(cases[0]));
    return 0;
  }
  std::fprintf(stderr,
               "test_compressed_table_io: FAIL (%d/%zu cases failed)\n",
               failures,
               sizeof(cases) / sizeof(cases[0]));
  return 1;
}
