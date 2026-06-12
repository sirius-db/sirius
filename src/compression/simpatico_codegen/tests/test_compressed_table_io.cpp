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
std::vector<simpatico::PlanLeafKind> leaf_kinds(simpatico::compressed_table const& ct)
{
  std::vector<simpatico::PlanLeafKind> out;
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
    bool dsl1_empty = !c1.compound || c1.compound->plan_dsl.empty();
    bool dsl2_empty = !c2.compound || c2.compound->plan_dsl.empty();
    expect(dsl1_empty == dsl2_empty,
           (std::string(label) + ": dsl emptiness col " + std::to_string(i)).c_str());
    if (!dsl1_empty)
      expect(c1.compound->plan_dsl == c2.compound->plan_dsl,
             (std::string(label) + ": plan_dsl col " + std::to_string(i)).c_str());
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
    expect(columns_equal(input.column(i), out->view().column(i)),
           (std::string(label) + ": data mismatch col " + std::to_string(i)).c_str());
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

// 2. Hybrid FOR head + fused bitpack subtree: exercises for_compressed_representation
//    (legacy C++) alongside a fused JIT leaf in the same column.
void test_hybrid_for_bitpack()
{
  auto t = make_int32_table(1, 4096, 11);
  io_roundtrip("hybrid_for_bitpack",
               t->view(),
               "input -> for -> deltas, references, reference_offsets\n"
               "for.deltas -> bitpack\n");
}

// 3. FOR only: exercises for_compressed_representation without a fused tail.
void test_for_only()
{
  auto t = make_int32_table(1, 4096, 13);
  io_roundtrip("for_only", t->view(), "input -> for\n");
}

// 4. ALP (floating-point): exercises alp_compressed_representation with its
//    four output channels and alp_rd leaf_meta on a FLOAT32 column.
void test_alp_f32()
{
  auto t = make_f32_table(1, 4096, 17);
  io_roundtrip("alp_f32", t->view(), "input -> alp\n");
}

// 5. Multi-column: three columns with three different plans in one file.
//    Verifies the column loop in write/read and that per-column DSLs don't bleed.
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
               "input -> for -> deltas, references, reference_offsets\n"
               "for.deltas -> bitpack\n"
               "---\n"
               "input -> for\n");
}

// 6. Column names survive write→read.
void test_column_names_survive()
{
  auto t = make_int32_table(2, 1024, 3);
  io_roundtrip("column_names_survive",
               t->view(),
               "input -> for\n"
               "---\n"
               "input -> for\n",
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

// 9. Error: file exists but contains no end-marker (totally corrupt data).
void test_error_no_marker()
{
  TmpFile tmp;
  {
    std::ofstream f(tmp.path, std::ios::binary);
    f.write("garbage data", 12);
  }
  std::string err;
  auto ct = simpatico::read_compressed_table(
    tmp.path, cudf::get_default_stream(), rmm::mr::get_current_device_resource_ref(), &err);
  expect(!err.empty(), "error_no_marker: expected non-empty error");
  expect(ct.columns.empty(), "error_no_marker: expected empty result");
}

// 10. Error: end-marker present but binary header has wrong magic bytes.
void test_error_bad_magic()
{
  TmpFile tmp;
  {
    std::ofstream f(tmp.path, std::ios::binary);
    // DSL section + end marker
    std::string header = "\n---END-HEADERS-V6---\n";
    f.write(header.data(), static_cast<std::streamsize>(header.size()));
    // Wrong magic
    char bad[] = {'X', 'X', 'X', 'X', 6, 0, 0};
    f.write(bad, sizeof(bad));
  }
  std::string err;
  auto ct = simpatico::read_compressed_table(
    tmp.path, cudf::get_default_stream(), rmm::mr::get_current_device_resource_ref(), &err);
  expect(!err.empty(), "error_bad_magic: expected non-empty error");
}

// 11. Error: correct magic but unsupported version number.
void test_error_bad_version()
{
  TmpFile tmp;
  {
    std::ofstream f(tmp.path, std::ios::binary);
    std::string header = "\n---END-HEADERS-V6---\n";
    f.write(header.data(), static_cast<std::streamsize>(header.size()));
    char data[] = {'H', 'P', 'L', 'N', 99, 0, 0};  // version 99
    f.write(data, sizeof(data));
  }
  std::string err;
  auto ct = simpatico::read_compressed_table(
    tmp.path, cudf::get_default_stream(), rmm::mr::get_current_device_resource_ref(), &err);
  expect(!err.empty(), "error_bad_version: expected non-empty error");
}

}  // namespace

int main()
{
  if (cudaSetDevice(0) != cudaSuccess) {
    std::fprintf(stderr, "test_compressed_table_io: cudaSetDevice failed\n");
    return 1;
  }
  codegen::jit::ensure_cuda_context();

  struct Case {
    char const* name;
    void (*fn)();
  };
  Case cases[] = {
    {"fused_delta_rle_bitpack", test_fused_delta_rle_bitpack},
    {"hybrid_for_bitpack", test_hybrid_for_bitpack},
    {"for_only", test_for_only},
    {"alp_f32", test_alp_f32},
    {"multi_column", test_multi_column},
    {"column_names_survive", test_column_names_survive},
    {"zero_rows", test_zero_rows},
    {"error_not_found", test_error_not_found},
    {"error_no_marker", test_error_no_marker},
    {"error_bad_magic", test_error_bad_magic},
    {"error_bad_version", test_error_bad_version},
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
