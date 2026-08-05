#include "api/simpatico_codegen.hpp"
#include "codegen/jit/nvrtc_compiler.hpp"
#include "codegen/plan/plan_interpreter.hpp"
#include "codegen/plan/plan_tree.hpp"
#include "codegen/util/stream_pool.hpp"
#include "test_utils.hpp"

#include <cudf/copying.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/table/table.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/traits.hpp>

#include <rmm/mr/cuda_async_memory_resource.hpp>
#include <rmm/mr/per_device_resource.hpp>
#include <rmm/resource_ref.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <set>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

namespace {

using simpatico::compress_with_plan;
using simpatico::compressed_table;
using simpatico::decompress;
using simpatico::split_plan_dsl;

// Check: every compressed column must carry a natively-built PlanTree that
// OWNS every compressed representation (node.rep + node.channels). Every owned
// rep must also appear in the path map of the tree re-parsed from its own
// rendered DSL, confirming rep placement, structural wiring, and the renderer's
// paths all agree.
void verify_plan_tree(compressed_table const& ct, char const* label)
{
  for (auto const& col : ct.columns) {
    expect(col.plan_tree != nullptr, (std::string(label) + ": null plan_tree").c_str());
    auto const& tree = *col.plan_tree;
    expect(tree.nodes.size() >= 2,
           (std::string(label) + ": PlanTree not built (need >= 2 nodes)").c_str());
    expect(tree.nodes[0].op == "input",
           (std::string(label) + ": PlanTree root is not 'input'").c_str());
    auto rebuilt = simpatico::plan_tree_from_dsl(simpatico::render_plan_tree(tree), nullptr);
    expect(rebuilt.has_value(),
           (std::string(label) + ": rendered DSL does not parse to a PlanTree").c_str());

    // Every path the rebuilt tree produces (each node's output_paths, plus the
    // root "input"). Each owned rep's path must be one of these.
    std::unordered_set<std::string> known_paths{"input"};
    for (auto const& rn : rebuilt->nodes)
      for (auto const& op : rn.output_paths)
        known_paths.insert(op);

    // Collect every rep pointer the tree owns (node.rep + node.channels) and
    // verify each corresponds to a path the rebuilt tree produces.
    std::size_t owned_count = 0;
    for (auto const& node : tree.nodes) {
      if (node.rep) ++owned_count;
      for (auto const& [path, rep] : node.channels) {
        if (rep) {
          ++owned_count;
          expect(
            known_paths.count(path) > 0,
            (std::string(label) + ": channel rep has no path entry for '" + path + "'").c_str());
        }
      }
    }
    expect(owned_count > 0,
           (std::string(label) + ": tree owns no reps (re-home must have failed)").c_str());
  }
}

void test_split_mismatch()
{
  auto plans = split_plan_dsl("input -> delta -> differences\n");
  expect(plans.size() == 1, "single plan split");
  bool threw = false;
  try {
    auto t = make_int32_table(2, 1024, 1);
    compress_with_plan(t->view(), "input -> delta -> differences\n", 1);
  } catch (std::runtime_error const&) {
    threw = true;
  }
  expect(threw, "plan/table count mismatch should throw");
}

compressed_table roundtrip_once(cudf::table_view input,
                                std::string const& dsl,
                                int threads,
                                char const* label)
{
  std::fprintf(stderr, "[roundtrip_once] label='%s'\n", label);
  compressed_table ct =
    (threads <= 1)
      ? compress_with_plan(
          input, dsl, cudf::get_default_stream(), rmm::mr::get_current_device_resource_ref())
      : compress_with_plan(input, dsl, threads, rmm::mr::get_current_device_resource_ref());
  auto out =
    (threads <= 1)
      ? decompress(ct, cudf::get_default_stream(), rmm::mr::get_current_device_resource_ref())
      : decompress(ct, threads, rmm::mr::get_current_device_resource_ref());
  expect(out != nullptr, "decompress returned null");
  expect(out->num_columns() == input.num_columns(), "column count");
  for (int i = 0; i < input.num_columns(); ++i) {
    expect(columns_equal_any(input.column(i), out->view().column(i), cudf::get_default_stream()),
           (std::string(label) + ": column data mismatch at index " + std::to_string(i)).c_str());
  }
  verify_plan_tree(ct, label);
  return ct;
}

// Regression for the internal `int column_threads` overloads: their result
// buffers are built on leased pool streams and record those streams for the
// eventual async free. With the old per-call pool (destroyed on return) the
// returned buffers referenced dangling stream handles, so freeing them was a
// use-after-free. The process-lifetime stream cache keeps every handle valid for
// the life of the process.
//
// Runs under an async (stream-ordered) memory resource so the deallocs are really
// enqueued on each buffer's recorded stream, then frees the parallel compress and
// decompress results after their internal pools are gone. This exercises the exact
// UAF scenario end to end.
//
// NOTE on teeth: a destroyed-stream free is not deterministically observable in a
// Release build — RMM ignores the cudaFreeAsync return under NDEBUG, and CUDA does
// not reliably invalidate a destroyed stream handle (cudaStreamQuery on it may
// still return success), so neither a crash nor a query check bites here. The guard
// is real under compute-sanitizer or a Debug RMM build, where the invalid-handle
// free is reported; this case gives those tools the scenario to catch.
void test_async_mr_free_safety()
{
  rmm::mr::cuda_async_memory_resource async_mr{};
  rmm::device_async_resource_ref prev{rmm::mr::get_current_device_resource_ref()};
  rmm::mr::set_current_device_resource_ref(async_mr);
  try {
    auto t = make_int32_table(3, 8192, 21);
    std::string dsl =
      "input -> delta -> differences\n"
      "delta.differences -> rle -> values, runs\n"
      "delta.differences.values -> bitpack\n"
      "delta.differences.runs -> bitpack\n"
      "---\n"
      "input -> for -> deltas, references\n"
      "for.deltas -> bitpack\n"
      "---\n"
      "input -> delta -> differences\n"
      "delta.differences -> bitpack\n";
    {
      // Parallel compress (3 threads) → compressed_table; parallel decompress →
      // cudf::table. Both allocate on leased streams that are returned to the cache
      // (not destroyed) when these calls return, so both results are safe to free
      // at the end of this scope.
      auto ct  = compress_with_plan(t->view(), dsl, 3, rmm::mr::get_current_device_resource_ref());
      auto out = decompress(ct, 3, rmm::mr::get_current_device_resource_ref());
      expect(out != nullptr && out->num_columns() == 3, "async_mr: decompress shape");
      for (int i = 0; i < t->num_columns(); ++i)
        expect(
          columns_equal_any(t->view().column(i), out->view().column(i), cudf::get_default_stream()),
          "async_mr: column data mismatch");
    }
    expect(cudaDeviceSynchronize() == cudaSuccess, "async_mr: device sync after async free");
  } catch (...) {
    rmm::mr::set_current_device_resource_ref(prev);
    throw;
  }
  rmm::mr::set_current_device_resource_ref(prev);
}

}  // namespace

int main()
{
  if (cudaSetDevice(0) != cudaSuccess) {
    std::fprintf(stderr, "test_compress_with_plan_roundtrip: cudaSetDevice failed\n");
    return 1;
  }
  try {
    test_split_mismatch();

    {
      auto t = make_int32_table(2, 2048, 13);
      std::string dsl =
        "input -> delta -> differences\n"
        "delta.differences -> rle -> values, runs\n"
        "delta.differences.values -> bitpack\n"
        "delta.differences.runs -> bitpack\n"
        "---\n"
        "input -> for -> deltas, references\n"
        "for.deltas -> bitpack\n";
      auto ct = compress_with_plan(t->view(),
                                   dsl,
                                   cudf::get_default_stream(),
                                   rmm::mr::get_current_device_resource_ref(),
                                   std::vector<std::string>{"col_a", "col_b"});
      expect(ct.columns.size() == 2, "column_names: column count");
      expect(ct.columns[0].name && *ct.columns[0].name == "col_a", "column_names: col_a");
      expect(ct.columns[1].name && *ct.columns[1].name == "col_b", "column_names: col_b");
      bool threw = false;
      try {
        compress_with_plan(
          t->view(), dsl, 1, rmm::mr::get_current_device_resource_ref(), {"only_one"});
      } catch (std::runtime_error const&) {
        threw = true;
      }
      expect(threw, "column_names count mismatch should throw");
    }

    {
      auto t = make_int32_table(2, 4096, 7);
      std::string dsl =
        "input -> delta -> differences\n"
        "delta.differences -> rle -> values, runs\n"
        "delta.differences.values -> bitpack\n"
        "delta.differences.runs -> bitpack\n"
        "---\n"
        "input -> delta -> differences\n"
        "delta.differences -> rle -> values, runs\n"
        "delta.differences.values -> bitpack\n"
        "delta.differences.runs -> bitpack\n";
      roundtrip_once(t->view(), dsl, 1, "fused_head");
      roundtrip_once(t->view(), dsl, 2, "fused_head_mt");
    }

    {
      auto t = make_int32_table(1, 4096, 3);
      std::string dsl =
        "input -> delta -> differences\n"
        "delta.differences -> rle -> values, runs\n"
        "delta.differences.values -> bitpack -> chunk_min, chunk_count, chunk_bits, packed\n"
        "delta.differences.runs -> bitpack -> chunk_min, chunk_count, chunk_bits, packed\n"
        "delta.differences.values.packed -> ans\n"
        "delta.differences.values.chunk_min -> identity\n"
        "delta.differences.values.chunk_count -> identity\n"
        "delta.differences.values.chunk_bits -> identity\n"
        "delta.differences.runs.chunk_min -> identity\n"
        "delta.differences.runs.chunk_count -> identity\n"
        "delta.differences.runs.chunk_bits -> identity\n";
      roundtrip_once(t->view(), dsl, 1, "fused_ans_tail");
      roundtrip_once(t->view(), dsl, 2, "fused_ans_tail_mt");
    }

    {
      // FOR without a downstream bitpack — deltas stored as raw fixed-stride.
      auto t = make_int32_table(2, 2048, 11);
      std::string dsl =
        "input -> for -> deltas, references\n"
        "---\n"
        "input -> for -> deltas, references\n";
      roundtrip_once(t->view(), dsl, 1, "for_only");
      roundtrip_once(t->view(), dsl, 2, "for_only_mt");
    }

    {
      // FOR + fused bitpack on deltas: JIT fused region covers both ops.
      auto t = make_int32_table(2, 4096, 5);
      std::string dsl =
        "input -> for -> deltas, references\n"
        "for.deltas -> bitpack\n"
        "---\n"
        "input -> for -> deltas, references\n"
        "for.deltas -> bitpack\n";
      auto ct = roundtrip_once(t->view(), dsl, 1, "for_bitpack");
      expect(ct.columns.size() == 2, "for_bitpack: column count");
      for (auto const& col : ct.columns) {
        expect(col.plan_tree != nullptr, "for_bitpack: plan_tree");
      }
      roundtrip_once(t->view(), dsl, 2, "for_bitpack_mt");
    }

    // NOTE: `for.deltas.packed -> ans` IS supported: the JIT region root is FOR,
    // bitpack is interior, and its boundary channels (packed, chunk_min, etc.)
    // are exposed and can be further entropy-coded like any other bitpack output.

    {
      // bitextract: split a FLOAT32 column into sign/exponent/mantissa planes;
      // decode rebuilds the column from the planes (transform-op inverse).
      // Bit-exact roundtrip (columns_equal compares raw 4-byte payloads).
      auto t = make_f32_table(2, 4096, 9);
      std::string dsl =
        "input -> bitextract_f32 -> sign, exponent, mantissa\n"
        "---\n"
        "input -> bitextract_f32 -> sign, exponent, mantissa\n";
      roundtrip_once(t->view(), dsl, 1, "bitextract_f32");
      roundtrip_once(t->view(), dsl, 2, "bitextract_f32_mt");
    }

    {
      // bitjoin (DAG reconvergence): extract the f32 planes then immediately
      // rejoin them in the same plan. The bitjoin node has multiple in-edges
      // (one per plane); decode splits the packed leaf back into the planes,
      // exercising the multi-input decode branch + memoized reconvergence.
      auto t = make_f32_table(2, 4096, 4);
      std::string dsl =
        "input -> bitextract_f32 -> sign, exponent, mantissa\n"
        "bitextract_f32.sign, bitextract_f32.exponent, bitextract_f32.mantissa "
        "-> bitjoin_f32 -> rejoined\n"
        "---\n"
        "input -> bitextract_f32 -> sign, exponent, mantissa\n"
        "bitextract_f32.sign, bitextract_f32.exponent, bitextract_f32.mantissa "
        "-> bitjoin_f32 -> rejoined\n";
      roundtrip_once(t->view(), dsl, 1, "bitextract_bitjoin_f32");
      roundtrip_once(t->view(), dsl, 2, "bitextract_bitjoin_f32_mt");
    }

    {
      // The sign plane is consumed twice by one ranged bitjoin. Decode must
      // retain/copy this shared memo value rather than moving it prematurely.
      auto t = make_f32_table(1, 4096, 6);
      std::string dsl =
        "input -> bitextract_f32 -> sign, exponent, mantissa\n"
        "bitextract_f32.sign_0:0, bitextract_f32.sign_0:0 "
        "-> bitjoin_u8 -> duplicated_sign\n";
      roundtrip_once(t->view(), dsl, 1, "bitextract_shared_sign");
    }

    {
      // Corrupt the structural metadata (drop two of the bitjoin's three inputs)
      // so decode is asked to reconstruct from an inconsistent tree. This must
      // be a loud error, never an implicit/partial decode of a wrong column.
      auto t = make_f32_table(1, 4096, 8);
      std::string dsl =
        "input -> bitextract_f32 -> sign, exponent, mantissa\n"
        "bitextract_f32.sign, bitextract_f32.exponent, bitextract_f32.mantissa "
        "-> bitjoin_f32 -> rejoined\n";
      auto ct = compress_with_plan(
        t->view(), dsl, cudf::get_default_stream(), rmm::mr::get_current_device_resource_ref());
      auto& tree   = *ct.columns[0].plan_tree;
      auto bitjoin = std::find_if(tree.nodes.begin(), tree.nodes.end(), [](auto const& node) {
        return node.op == "bitjoin_f32";
      });
      expect(bitjoin != tree.nodes.end(), "consumed memo: bitjoin node missing");
      bitjoin->input_sources.resize(1);

      bool loud_error = false;
      try {
        (void)decompress(
          ct, cudf::get_default_stream(), rmm::mr::get_current_device_resource_ref());
      } catch (std::runtime_error const& e) {
        loud_error = std::string(e.what()).find("input count") != std::string::npos;
      }
      expect(loud_error, "corrupt bitjoin metadata must fail loudly");
    }

    {
      // ALP (Adaptive Lossless Floating-Point), FLOAT32. Terminal multi-output
      // operator: integers/exceptions/exception_positions/metadata. Lossless,
      // so the roundtrip must be bit-exact.
      auto t = make_f32_table(2, 4096, 21);
      std::string dsl =
        "input -> alp\n"
        "---\n"
        "input -> alp\n";
      roundtrip_once(t->view(), dsl, 1, "alp_f32");
      roundtrip_once(t->view(), dsl, 2, "alp_f32_mt");
    }

    {
      // ALP, FLOAT64 — exercises the 8-byte (INT64 integers) path and the
      // public compress path on a wide non-integer dtype.
      auto t = make_f64_table(2, 4096, 27);
      std::string dsl =
        "input -> alp\n"
        "---\n"
        "input -> alp\n";
      roundtrip_once(t->view(), dsl, 1, "alp_f64");
      roundtrip_once(t->view(), dsl, 2, "alp_f64_mt");
    }

    {
      // ALP-RD (Right-Dictionary), FLOAT32 — the non-decimal float path.
      auto t = make_f32_table(2, 4096, 33);
      std::string dsl =
        "input -> alp_rd\n"
        "---\n"
        "input -> alp_rd\n";
      roundtrip_once(t->view(), dsl, 1, "alp_rd_f32");
      roundtrip_once(t->view(), dsl, 2, "alp_rd_f32_mt");
    }

    {
      // bitcomp (nvcomp), INT32 — bare alias (algorithm 0), the explicit
      // `bitcomp_default` alias, and the `bitcomp_sparse` (algorithm 1) variant.
      // All three must decode losslessly via the rep's decompress() path.
      auto t                 = make_int32_table(2, 4096, 39);
      char const* variants[] = {"bitcomp", "bitcomp_default", "bitcomp_sparse"};
      for (char const* op : variants) {
        std::string dsl = std::string("input -> ") + op + "\n---\ninput -> " + op + "\n";
        roundtrip_once(t->view(), dsl, 1, op);
        roundtrip_once(t->view(), dsl, 2, (std::string(op) + "_mt").c_str());
      }
    }

    {
      // Caller-owned stream_pool overload: compress + decompress through a
      // pool the caller supplies (vs. the int column_threads overload that
      // builds one internally). Uses a 4-column table with a 2-stream pool so
      // the work-queue distributes more columns than streams, and reuses the
      // SAME pool across two roundtrips to prove it is safe to keep.
      simpatico::stream_pool pool;
      expect(pool.init(2), "stream_pool init");
      std::string dsl =
        "input -> delta -> differences\n"
        "delta.differences -> rle -> values, runs\n"
        "delta.differences.values -> bitpack\n"
        "delta.differences.runs -> bitpack\n"
        "---\n"
        "input -> for -> deltas, references\n"
        "for.deltas -> bitpack\n"
        "---\n"
        "input -> bitcomp\n"
        "---\n"
        "input -> delta -> differences\n"
        "delta.differences -> bitpack\n";
      for (int round = 0; round < 2; ++round) {
        auto t = make_int32_table(4, 4096, 41 + round);
        compressed_table ct =
          compress_with_plan(t->view(), dsl, pool, rmm::mr::get_current_device_resource_ref());
        auto out = decompress(ct, pool, rmm::mr::get_current_device_resource_ref());
        expect(out != nullptr, "stream_pool: decompress null");
        expect(out->num_columns() == t->num_columns(), "stream_pool: column count");
        for (int i = 0; i < t->num_columns(); ++i) {
          expect(columns_equal(t->view().column(i), out->view().column(i)),
                 "stream_pool: column mismatch");
        }
        verify_plan_tree(ct, "stream_pool");
      }
    }

    // --- Additional FOR roundtrip tests (int32/int64/references entropy tail) ---

    {
      // FOR + bitpack on deltas, int32, larger data set.
      auto t = make_int32_table(2, 4096, 53);
      std::string dsl =
        "input -> for -> deltas, references\n"
        "for.deltas -> bitpack\n"
        "---\n"
        "input -> for -> deltas, references\n"
        "for.deltas -> bitpack\n";
      roundtrip_once(t->view(), dsl, 1, "jit_for_bp_int32");
      roundtrip_once(t->view(), dsl, 2, "jit_for_bp_int32_mt");
    }

    {
      // FOR + bitpack on deltas, int64.
      auto t = make_int64_table(2, 4096, 59);
      std::string dsl =
        "input -> for -> deltas, references\n"
        "for.deltas -> bitpack\n"
        "---\n"
        "input -> for -> deltas, references\n"
        "for.deltas -> bitpack\n";
      roundtrip_once(t->view(), dsl, 1, "jit_for_bp_int64");
      roundtrip_once(t->view(), dsl, 2, "jit_for_bp_int64_mt");
    }

    {
      // FOR without a downstream bitpack — deltas stored as Raw fixed-stride leaf.
      auto t = make_int32_table(2, 2048, 61);
      std::string dsl =
        "input -> for -> deltas, references\n"
        "---\n"
        "input -> for -> deltas, references\n";
      roundtrip_once(t->view(), dsl, 1, "jit_for_terminal_int32");
      roundtrip_once(t->view(), dsl, 2, "jit_for_terminal_int32_mt");
    }

    {
      // FOR + bitpack on deltas + entropy tail on references channel.
      auto t = make_int32_table(2, 4096, 67);
      std::string dsl =
        "input -> for -> deltas, references\n"
        "for.deltas -> bitpack\n"
        "for.references -> identity\n"
        "---\n"
        "input -> for -> deltas, references\n"
        "for.deltas -> bitpack\n"
        "for.references -> identity\n";
      roundtrip_once(t->view(), dsl, 1, "jit_for_ref_identity_int32");
      roundtrip_once(t->view(), dsl, 2, "jit_for_ref_identity_int32_mt");
    }

    {
      // ZigZag leaf at the region root — stored as its own fixed-stride
      // `zigzag` channel.  int32 and int64.
      auto t32 = make_int32_table(2, 4096, 71);
      std::string dsl32 =
        "input -> zigzag -> zigzag\n"
        "---\n"
        "input -> zigzag -> zigzag\n";
      roundtrip_once(t32->view(), dsl32, 1, "jit_zigzag_root_int32");
      roundtrip_once(t32->view(), dsl32, 2, "jit_zigzag_root_int32_mt");

      auto t64 = make_int64_table(2, 4096, 73);
      std::string dsl64 =
        "input -> zigzag -> zigzag\n"
        "---\n"
        "input -> zigzag -> zigzag\n";
      roundtrip_once(t64->view(), dsl64, 1, "jit_zigzag_root_int64");
      roundtrip_once(t64->view(), dsl64, 2, "jit_zigzag_root_int64_mt");
    }

    {
      // ZigZag -> Bitpack (FUSED): ZigZag is the region root acting as an
      // inline transformer (stores nothing), bitpack is the covered leaf that
      // packs the ZigZag-mapped codes.  Exercises the transformer encode path
      // + the closed-form (Bitpack leaf) inverse decode path.  int32 and int64.
      auto t32 = make_int32_table(2, 4096, 171);
      std::string dsl32 =
        "input -> zigzag -> zigzag\n"
        "zigzag.zigzag -> bitpack\n"
        "---\n"
        "input -> zigzag -> zigzag\n"
        "zigzag.zigzag -> bitpack\n";
      roundtrip_once(t32->view(), dsl32, 1, "jit_zigzag_bitpack_int32");
      roundtrip_once(t32->view(), dsl32, 2, "jit_zigzag_bitpack_int32_mt");

      auto t64 = make_int64_table(2, 4096, 173);
      std::string dsl64 =
        "input -> zigzag -> zigzag\n"
        "zigzag.zigzag -> bitpack\n"
        "---\n"
        "input -> zigzag -> zigzag\n"
        "zigzag.zigzag -> bitpack\n";
      roundtrip_once(t64->view(), dsl64, 1, "jit_zigzag_bitpack_int64");
      roundtrip_once(t64->view(), dsl64, 2, "jit_zigzag_bitpack_int64_mt");
    }

    {
      // Delta -> ZigZag -> Bitpack (FUSED): delta is the region root (inline
      // scan), ZigZag is an interior transformer, bitpack is the covered leaf.
      // Exercises ZigZag fused BELOW another transformer (decode materialises
      // ZigZag via the generic value_source slab path).
      auto t = make_int32_table(2, 4096, 177);
      std::string dsl =
        "input -> delta -> differences\n"
        "delta.differences -> zigzag -> zigzag\n"
        "delta.differences.zigzag -> bitpack\n"
        "---\n"
        "input -> delta -> differences\n"
        "delta.differences -> zigzag -> zigzag\n"
        "delta.differences.zigzag -> bitpack\n";
      roundtrip_once(t->view(), dsl, 1, "jit_delta_zigzag_bitpack_int32");
      roundtrip_once(t->view(), dsl, 2, "jit_delta_zigzag_bitpack_int32_mt");

      // Regression: delta over raw float32 bit patterns can produce a diff
      // that overflows int32 (adjacent floats need not be numerically close
      // as bit patterns, e.g. a sign flip). Delta/FOR's diff-then-reconstruct
      // relies on wraparound, which must be done in unsigned arithmetic
      // (signed overflow is UB) -- this shape is what surfaced it.
      auto tf = make_f32_table(2, 4096, 3);
      roundtrip_once(tf->view(), dsl, 1, "jit_delta_zigzag_bitpack_f32");
    }

    {
      // Delta -> ZigZag: delta is the region root (inline scan), ZigZag is the
      // covered leaf that stores the ZigZagged differences.  This is the
      // canonical signed-residual pipeline.
      auto t = make_int32_table(2, 4096, 77);
      std::string dsl =
        "input -> delta -> differences\n"
        "delta.differences -> zigzag -> zigzag\n"
        "---\n"
        "input -> delta -> differences\n"
        "delta.differences -> zigzag -> zigzag\n";
      roundtrip_once(t->view(), dsl, 1, "jit_delta_zigzag_int32");
      roundtrip_once(t->view(), dsl, 2, "jit_delta_zigzag_int32_mt");
    }

    {
      // Delta -> ZigZag -> ANS: the headline use case.  ZigZag's stored
      // channel is entropy-tail-routed to ANS (small signed deltas become
      // small unsigned codes that the entropy coder compresses well).
      auto t = make_int32_table(2, 4096, 83);
      std::string dsl =
        "input -> delta -> differences\n"
        "delta.differences -> zigzag -> zigzag\n"
        "delta.differences.zigzag -> ans\n"
        "---\n"
        "input -> delta -> differences\n"
        "delta.differences -> zigzag -> zigzag\n"
        "delta.differences.zigzag -> ans\n";
      roundtrip_once(t->view(), dsl, 1, "jit_delta_zigzag_ans_int32");
      roundtrip_once(t->view(), dsl, 2, "jit_delta_zigzag_ans_int32_mt");
    }

    {
      // FOR + bitpack on deltas + ans entropy tail on packed: mirrors the
      // delta->rle->bitpack->ans pattern in fused_ans_tail.
      auto t = make_int32_table(2, 4096, 71);
      std::string dsl =
        "input -> for -> deltas, references\n"
        "for.deltas -> bitpack -> chunk_min, chunk_count, chunk_bits, packed\n"
        "for.deltas.packed -> ans\n"
        "for.deltas.chunk_min -> identity\n"
        "for.deltas.chunk_count -> identity\n"
        "for.deltas.chunk_bits -> identity\n"
        "---\n"
        "input -> for -> deltas, references\n"
        "for.deltas -> bitpack -> chunk_min, chunk_count, chunk_bits, packed\n"
        "for.deltas.packed -> ans\n"
        "for.deltas.chunk_min -> identity\n"
        "for.deltas.chunk_count -> identity\n"
        "for.deltas.chunk_bits -> identity\n";
      roundtrip_once(t->view(), dsl, 1, "jit_for_bp_ans_int32");
      roundtrip_once(t->view(), dsl, 2, "jit_for_bp_ans_int32_mt");
    }

    // -----------------------------------------------------------------
    // Dual-mode entropy-tail tests: delta, for, rle channels -> ans
    // -----------------------------------------------------------------

    {
      // Delta.differences -> ANS (no zigzag): the differences stream is
      // materialized via a Raw passthrough leaf and entropy-coded directly.
      // int32 and int64.
      auto t32 = make_int32_table(2, 4096, 191);
      std::string dsl32 =
        "input -> delta -> differences\n"
        "delta.differences -> ans\n"
        "---\n"
        "input -> delta -> differences\n"
        "delta.differences -> ans\n";
      roundtrip_once(t32->view(), dsl32, 1, "jit_delta_differences_ans_int32");
      roundtrip_once(t32->view(), dsl32, 2, "jit_delta_differences_ans_int32_mt");

      auto t64 = make_int64_table(2, 4096, 193);
      std::string dsl64 =
        "input -> delta -> differences\n"
        "delta.differences -> ans\n"
        "---\n"
        "input -> delta -> differences\n"
        "delta.differences -> ans\n";
      roundtrip_once(t64->view(), dsl64, 1, "jit_delta_differences_ans_int64");
      roundtrip_once(t64->view(), dsl64, 2, "jit_delta_differences_ans_int64_mt");
    }

    {
      // FOR.deltas -> ANS: residuals (delta channel) entropy-tail-routed,
      // references stored as identity. int32 and int64.
      auto t32 = make_int32_table(2, 4096, 197);
      std::string dsl32 =
        "input -> for -> deltas, references\n"
        "for.deltas -> ans\n"
        "---\n"
        "input -> for -> deltas, references\n"
        "for.deltas -> ans\n";
      roundtrip_once(t32->view(), dsl32, 1, "jit_for_deltas_ans_int32");
      roundtrip_once(t32->view(), dsl32, 2, "jit_for_deltas_ans_int32_mt");

      auto t64 = make_int64_table(2, 4096, 199);
      std::string dsl64 =
        "input -> for -> deltas, references\n"
        "for.deltas -> ans\n"
        "---\n"
        "input -> for -> deltas, references\n"
        "for.deltas -> ans\n";
      roundtrip_once(t64->view(), dsl64, 1, "jit_for_deltas_ans_int64");
      roundtrip_once(t64->view(), dsl64, 2, "jit_for_deltas_ans_int64_mt");
    }

    {
      // RLE.values -> ANS: unique run values entropy-tail-routed to ANS.
      // The runs are bitpacked (fused). int32.
      auto t32 = make_int32_table(2, 1024, 203);
      std::string dsl32 =
        "input -> rle -> values, runs\n"
        "rle.runs -> bitpack\n"
        "rle.values -> ans\n"
        "---\n"
        "input -> rle -> values, runs\n"
        "rle.runs -> bitpack\n"
        "rle.values -> ans\n";
      roundtrip_once(t32->view(), dsl32, 1, "jit_rle_values_ans_int32");
      roundtrip_once(t32->view(), dsl32, 2, "jit_rle_values_ans_int32_mt");
    }

    {
      // RLE.runs -> ANS: run counts entropy-tail-routed to ANS.
      // The values are bitpacked (fused). int32.
      auto t32 = make_int32_table(2, 1024, 211);
      std::string dsl32 =
        "input -> rle -> values, runs\n"
        "rle.values -> bitpack\n"
        "rle.runs -> ans\n"
        "---\n"
        "input -> rle -> values, runs\n"
        "rle.values -> bitpack\n"
        "rle.runs -> ans\n";
      roundtrip_once(t32->view(), dsl32, 1, "jit_rle_runs_ans_int32");
      roundtrip_once(t32->view(), dsl32, 2, "jit_rle_runs_ans_int32_mt");
    }

    {
      // RLE with BOTH channels entropy-tail-routed: values -> ans,
      // runs -> snappy. All channels use Raw passthrough. int32.
      auto t32 = make_int32_table(2, 1024, 223);
      std::string dsl32 =
        "input -> rle -> values, runs\n"
        "rle.values -> ans\n"
        "rle.runs -> snappy\n"
        "---\n"
        "input -> rle -> values, runs\n"
        "rle.values -> ans\n"
        "rle.runs -> snappy\n";
      roundtrip_once(t32->view(), dsl32, 1, "jit_rle_both_tails_int32");
      roundtrip_once(t32->view(), dsl32, 2, "jit_rle_both_tails_int32_mt");
    }

    {
      // Delta -> RLE -> values (ans) + runs (snappy): a two-stage fused
      // region (delta inside rle) where both rle outputs are entropy tails.
      // DSL path convention: nested output paths are <parent_path>.<output_name>,
      // NOT <parent_path>.<op_name>.<output_name>.
      auto t32 = make_int32_table(2, 1024, 227);
      std::string dsl32 =
        "input -> delta -> differences\n"
        "delta.differences -> rle -> values, runs\n"
        "delta.differences.values -> ans\n"
        "delta.differences.runs -> snappy\n"
        "---\n"
        "input -> delta -> differences\n"
        "delta.differences -> rle -> values, runs\n"
        "delta.differences.values -> ans\n"
        "delta.differences.runs -> snappy\n";
      roundtrip_once(t32->view(), dsl32, 1, "jit_delta_rle_both_tails_int32");
      roundtrip_once(t32->view(), dsl32, 2, "jit_delta_rle_both_tails_int32_mt");
    }

    {
      // Fused->fused: for.references -> bitpack.
      // FOR fuses deltas with bitpack inline (one kernel); the per-chunk
      // reference minimums are bitpacked separately in a second kernel.
      // int32, int64.
      auto t32 = make_int32_table(2, 2048, 241);
      std::string dsl32 =
        "input -> for -> deltas, references\n"
        "for.deltas -> bitpack\n"
        "for.references -> bitpack\n"
        "---\n"
        "input -> for -> deltas, references\n"
        "for.deltas -> bitpack\n"
        "for.references -> bitpack\n";
      roundtrip_once(t32->view(), dsl32, 1, "jit_for_refs_bitpack_int32");
      roundtrip_once(t32->view(), dsl32, 2, "jit_for_refs_bitpack_int32_mt");

      auto t64 = make_int64_table(2, 2048, 251);
      std::string dsl64 =
        "input -> for -> deltas, references\n"
        "for.deltas -> bitpack\n"
        "for.references -> bitpack\n"
        "---\n"
        "input -> for -> deltas, references\n"
        "for.deltas -> bitpack\n"
        "for.references -> bitpack\n";
      roundtrip_once(t64->view(), dsl64, 1, "jit_for_refs_bitpack_int64");
    }

    {
      // Fused->fused: delta -> for -> references -> bitpack.
      // Delta fuses with FOR fuses with bitpack (deltas) inline; FOR's
      // reference channel is bitpacked by a second independent kernel.
      auto t32 = make_int32_table(2, 2048, 257);
      std::string dsl32 =
        "input -> delta -> differences\n"
        "delta.differences -> for -> deltas, references\n"
        "delta.differences.deltas -> bitpack\n"
        "delta.differences.references -> bitpack\n"
        "---\n"
        "input -> delta -> differences\n"
        "delta.differences -> for -> deltas, references\n"
        "delta.differences.deltas -> bitpack\n"
        "delta.differences.references -> bitpack\n";
      roundtrip_once(t32->view(), dsl32, 1, "jit_delta_for_refs_bitpack_int32");
    }

    {
      // Regression (operator-sweep depth 4): a fused-region channel
      // (rle.values) is entropy-tail-routed to a NON-codegen op (alp) whose own
      // output feeds a codegen op that has a further codegen tail
      // (bitpack -> chunk_min -> zigzag). Reconstructing rle.values resolves the
      // alp subtree, which must decode that nested bitpack->zigzag tail with an
      // empty decompressed map. f32 so alp applies.
      auto t = make_f32_table(1, 4096, 91);
      std::string dsl =
        "input -> rle -> runs, values\n"
        "rle.values -> alp -> integers, exceptions, exception_positions, metadata\n"
        "rle.values.integers -> bitpack -> chunk_min, chunk_count, chunk_bits, packed\n"
        "rle.values.integers.chunk_min -> zigzag -> zigzag\n";
      roundtrip_once(t->view(), dsl, 1, "jit_rle_alp_bitpack_zigzag_tail_f32");
    }

    {
      // str_split: decompose STRING into offsets/chars[/null_mask], compress each
      // channel independently (offsets -> delta -> bitpack fused; chars -> lz4),
      // reassemble via make_strings_column. Conditional arity: a non-null column
      // uses a 2-channel plan; a nullable column MUST route null_mask (guarded).
      // NOTE the nested path str_split.offsets.differences (offsets is non-root).
      auto stream                   = cudf::get_default_stream();
      std::vector<std::string> vals = {
        "apple", "banana", "apple", "cherry", "elderberry", "fig", "grape", "honeydew"};

      std::string const dsl2 =
        "input -> str_split -> offsets, chars\n"
        "str_split.offsets -> delta -> differences\n"
        "str_split.offsets.differences -> bitpack\n"
        "str_split.chars -> lz4\n";
      auto t = make_strings_table(vals, {}, stream);
      roundtrip_once(t->view(), dsl2, 1, "str_split");
      roundtrip_once(t->view(), dsl2, 2, "str_split_mt");

      // Nullable: 3-channel plan carries the mask (null_mask declared on line 1).
      std::string const dsl3 =
        "input -> str_split -> offsets, chars, null_mask\n"
        "str_split.offsets -> delta -> differences\n"
        "str_split.offsets.differences -> bitpack\n"
        "str_split.chars -> lz4\n"
        "str_split.null_mask -> identity\n";
      std::vector<bool> valid = {true, false, true, true, false, true, true, false};  // 3 nulls
      auto tn                 = make_strings_table(vals, valid, stream);
      auto ct                 = roundtrip_once(tn->view(), dsl3, 1, "str_split_nulls");
      // strings_equal ignores validity, so assert null preservation explicitly.
      auto decoded = decompress(ct, stream, rmm::mr::get_current_device_resource_ref());
      expect(decoded->view().column(0).null_count() == 3, "str_split_nulls: null_count preserved");

      // Guard: nullable column + 2-channel plan must ERROR, not drop nulls silently.
      bool threw = false;
      try {
        compress_with_plan(tn->view(), dsl2, stream, rmm::mr::get_current_device_resource_ref());
      } catch (std::runtime_error const&) {
        threw = true;
      }
      expect(threw, "str_split: nullable input without null_mask must throw");
    }

    {
      // Degenerate STRING shapes: zero-row / childless-empty / all-null /
      // all-empty-string inputs through dictionary and str_split. describe()
      // runs on every rep too — the zero-row full dictionary used to segfault
      // there (childless DICTIONARY32).
      auto stream = cudf::get_default_stream();
      auto mr     = rmm::mr::get_current_device_resource_ref();

      for (char const* plan : {"input -> dictionary\n", "input -> str_split\n"}) {
        // describe() runs on a FRESH compress (independent of the roundtrip below).
        auto t0 = make_strings_table({}, {}, stream);
        compress_with_plan(t0->view(), plan, stream, mr).describe(stream);
        roundtrip_once(t0->view(), plan, 1, "string_zero_row");

        auto tn = make_strings_table({"x", "y", "z", "w"}, {false, false, false, false}, stream);
        compress_with_plan(tn->view(), plan, stream, mr).describe(stream);
        auto ctn = compress_with_plan(tn->view(), plan, stream, mr);
        auto dn  = decompress(ctn, stream, mr);
        expect(dn->view().column(0).null_count() == 4, "string_all_null: null_count");
        expect(strings_equal(tn->view().column(0), dn->view().column(0), stream),
               "string_all_null: data");

        auto t_empty = make_strings_table({"", "", ""}, {}, stream);
        roundtrip_once(t_empty->view(), plan, 1, "string_all_empty");
      }

      // The canonical childless empty column (cudf::make_empty_column).
      {
        std::vector<std::unique_ptr<cudf::column>> cols;
        cols.push_back(cudf::make_empty_column(cudf::data_type{cudf::type_id::STRING}));
        cudf::table tbl(std::move(cols));
        roundtrip_once(tbl.view(), "input -> str_split\n", 1, "str_split_childless_empty");
      }
    }

    {
      // Sliced STRING views. Both head (offset=0) and tail (offset>0) slices must
      // roundtrip: leaf operators now read through data() which accounts for offset.
      auto stream = cudf::get_default_stream();
      auto full   = make_strings_column({"aa", "bb", "cc", "dd", "ee", "ff", "gg", "hh"},
                                        {true, false, true, true, true, true, true, false},
                                      stream);
      auto head   = cudf::slice(full->view(), {0, 5});
      roundtrip_once(
        cudf::table_view({head[0]}), "input -> str_split\n", 1, "str_split_head_slice");
      auto tail = cudf::slice(full->view(), {3, 8});
      roundtrip_once(
        cudf::table_view({tail[0]}), "input -> str_split\n", 1, "str_split_tail_slice");
    }

    {
      // Nullable STRING through the byte codecs must be REJECTED (the payload
      // carries no mask), and a decomposed dictionary plan that fails to
      // route null_mask must error rather than silently drop validity.
      auto stream = cudf::get_default_stream();
      auto mr     = rmm::mr::get_current_device_resource_ref();
      auto tn     = make_strings_table({"a", "b"}, {true, false}, stream);
      for (char const* plan : {"input -> ans\n", "input -> bitcomp\n"}) {
        bool threw = false;
        try {
          compress_with_plan(tn->view(), plan, stream, mr);
        } catch (std::exception const&) {
          threw = true;
        }
        expect(threw, "nullable STRING through ans/bitcomp must throw");
      }
      bool threw = false;
      try {
        compress_with_plan(
          tn->view(), "input -> dictionary -> keys_offsets, keys_chars, indices\n", stream, mr);
      } catch (std::exception const&) {
        threw = true;
      }
      expect(threw, "nullable decomposed dictionary without null_mask must throw");
    }

    {
      // Nullable NUMERIC input through a non-null-aware op must be REJECTED,
      // not silently stripped of validity. No numeric op carries a null_mask
      // channel, so both the fused codegen path (delta -> bitpack) and the
      // generic byte codecs (ans) must fail closed rather than decode garbage
      // under the null slots.
      auto stream = cudf::get_default_stream();
      auto mr     = rmm::mr::get_current_device_resource_ref();
      // Build a nullable INT32 column (offset 0, one null at row 5) directly so
      // the input carries a real validity bitmask.
      auto col = cudf::make_numeric_column(
        cudf::data_type{cudf::type_id::INT32}, 64, cudf::mask_state::ALL_VALID, stream, mr);
      std::vector<std::int32_t> host(64);
      for (int r = 0; r < 64; ++r)
        host[static_cast<std::size_t>(r)] = static_cast<std::int32_t>((r * 17 + 11) % 1000);
      if (cudaMemcpy(col->mutable_view().head<std::int32_t>(),
                     host.data(),
                     host.size() * sizeof(std::int32_t),
                     cudaMemcpyHostToDevice) != cudaSuccess)
        throw std::runtime_error("nullable numeric guard: cudaMemcpy failed");
      cudf::set_null_mask(col->mutable_view().null_mask(), 5, 6, /*valid=*/false, stream);
      col->set_null_count(1);
      cudf::table_view nullable_tbl({col->view()});
      expect(col->null_count() == 1, "nullable numeric guard: input has one null");
      for (char const* plan :
           {"input -> delta -> differences\n", "input -> ans\n", "input -> bitpack\n"}) {
        bool threw = false;
        try {
          compress_with_plan(nullable_tbl, plan, stream, mr);
        } catch (std::exception const&) {
          threw = true;
        }
        expect(threw, "nullable numeric input through a non-null-aware op must throw");
      }
    }

    {
      // Nullable decomposed dictionary WITH the null_mask channel routed:
      // validity must survive the from_outputs rebuild.
      auto stream             = cudf::get_default_stream();
      std::vector<bool> valid = {true, false, true, true};
      auto tn                 = make_strings_table({"apple", "", "cherry", "apple"}, valid, stream);
      auto ct                 = roundtrip_once(tn->view(),
                               "input -> dictionary -> keys_offsets, keys_chars, indices, "
                                               "null_mask\n"
                                               "dictionary.indices -> bitpack\n"
                                               "dictionary.null_mask -> identity\n",
                               1,
                               "dictionary_decomposed_nulls");
      auto decoded            = decompress(ct, stream, rmm::mr::get_current_device_resource_ref());
      expect(decoded->view().column(0).null_count() == 1, "dictionary_decomposed_nulls: count");
    }

    {
      // u8 codegen on string sub-channels: RLE on the (mostly 0xFF) null_mask
      // bytes and a fused delta->bitpack cascade on the chars bytes.
      auto stream = cudf::get_default_stream();
      std::vector<std::string> v64;
      std::vector<bool> b64;
      for (int i = 0; i < 64; ++i) {
        v64.push_back("val" + std::to_string(i % 7));
        b64.push_back(i != 13 && i != 47);
      }
      auto tn      = make_strings_table(v64, b64, stream);
      auto ct      = roundtrip_once(tn->view(),
                               "input -> str_split -> offsets, chars, null_mask\n"
                                    "str_split.offsets -> delta -> differences\n"
                                    "str_split.offsets.differences -> bitpack\n"
                                    "str_split.chars -> lz4\n"
                                    "str_split.null_mask -> rle -> runs, values\n",
                               1,
                               "str_split_null_mask_rle_u8");
      auto decoded = decompress(ct, stream, rmm::mr::get_current_device_resource_ref());
      expect(decoded->view().column(0).null_count() == 2, "str_split_null_mask_rle_u8: count");

      std::vector<std::string> vals8 = {"aa", "bb", "cc", "aa", "dd", "bb", "ee", "aa"};
      auto t8                        = make_strings_table(vals8, {}, stream);
      roundtrip_once(t8->view(),
                     "input -> str_split -> offsets, chars\n"
                     "str_split.offsets -> delta -> differences\n"
                     "str_split.offsets.differences -> bitpack\n"
                     "str_split.chars -> delta -> differences\n"
                     "str_split.chars.differences -> bitpack\n",
                     1,
                     "str_split_chars_fused_u8");
    }

    {
      // Chrono columns compress as their integer storage and restore the
      // logical type at the end of decompress (apply_stored_dtype) — the
      // comparators check dtype, so a pass proves the restoration. DATE32
      // through the classic date cascade; an int64 timestamp through a fused
      // cascade and codec leaves.
      auto td = make_chrono_table(cudf::type_id::TIMESTAMP_DAYS, 4096, 5);
      roundtrip_once(td->view(),
                     "input -> delta -> differences\n"
                     "delta.differences -> bitpack\n",
                     1,
                     "date32_delta_bitpack");
      roundtrip_once(td->view(), "input -> ans\n", 1, "date32_ans");

      auto tu = make_chrono_table(cudf::type_id::TIMESTAMP_MICROSECONDS, 4096, 9);
      roundtrip_once(tu->view(),
                     "input -> delta -> differences\n"
                     "delta.differences -> bitpack\n",
                     1,
                     "timestamp_us_delta_bitpack");
      roundtrip_once(tu->view(), "input -> nvcomp_cascaded\n", 1, "timestamp_us_cascaded");
    }

    test_async_mr_free_safety();

    std::printf("test_compress_with_plan_roundtrip: PASS\n");
    return 0;
  } catch (std::exception const& e) {
    std::fprintf(stderr, "test_compress_with_plan_roundtrip: FAIL: %s\n", e.what());
    return 1;
  }
}
