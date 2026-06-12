#include "api/simpatico_codegen.hpp"
#include "codegen/jit/nvrtc_compiler.hpp"
#include "codegen/plan/plan_interpreter.hpp"
#include "codegen/plan/plan_tree.hpp"
#include "codegen/util/stream_pool.hpp"
#include "test_utils.hpp"

#include <cudf/table/table.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/traits.hpp>

#include <rmm/mr/per_device_resource.hpp>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using simpatico::compress_with_plan;
using simpatico::compressed_table;
using simpatico::decompress;
using simpatico::split_plan_dsl;

// Check: every compressed column must carry a natively-built PlanTree that
// OWNS every compressed representation (node.rep + node.channels). Every
// owned rep must also appear in the DSL-rebuilt tree's path map, confirming
// that rep placement and structural wiring agree.
void verify_plan_tree(compressed_table const& ct, char const* label)
{
  for (auto const& col : ct.columns) {
    expect(col.compound != nullptr, (std::string(label) + ": null compound").c_str());
    auto const& tree = col.compound->tree;
    expect(tree.nodes.size() >= 2,
           (std::string(label) + ": PlanTree not built (need >= 2 nodes)").c_str());
    expect(tree.nodes[0].op == "input",
           (std::string(label) + ": PlanTree root is not 'input'").c_str());
    simpatico::PlanPathMap map;
    auto rebuilt = simpatico::plan_tree_from_dsl(col.compound->plan_dsl, nullptr, &map);
    expect(rebuilt.has_value(),
           (std::string(label) + ": stored DSL does not parse to a PlanTree").c_str());

    // Collect every rep pointer the tree owns (node.rep + node.channels) and
    // verify each has a corresponding path in the DSL-rebuilt map.
    std::size_t owned_count = 0;
    for (auto const& node : tree.nodes) {
      if (node.rep) {
        ++owned_count;
        expect(!node.rep_path.empty() && map.node.find(node.rep_path) != map.node.end(),
               (std::string(label) + ": node.rep has no map entry for path '" + node.rep_path + "'")
                 .c_str());
      }
      for (auto const& [path, rep] : node.channels) {
        if (rep) {
          ++owned_count;
          expect(map.node.find(path) != map.node.end(),
                 (std::string(label) + ": channel rep has no map entry for path '" + path + "'")
                   .c_str());
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
    expect(columns_equal(input.column(i), out->view().column(i)),
           (std::string(label) + ": column data mismatch at index " + std::to_string(i)).c_str());
  }
  verify_plan_tree(ct, label);
  return ct;
}

}  // namespace

int main()
{
  if (cudaSetDevice(0) != cudaSuccess) {
    std::fprintf(stderr, "test_compress_with_plan_roundtrip: cudaSetDevice failed\n");
    return 1;
  }
  codegen::jit::ensure_cuda_context();
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
        "input -> for -> deltas, references, reference_offsets\n"
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
      auto t = make_int32_table(2, 2048, 11);
      std::string dsl =
        "input -> for\n"
        "---\n"
        "input -> for\n";
      roundtrip_once(t->view(), dsl, 1, "for_only");
      roundtrip_once(t->view(), dsl, 2, "for_only_mt");
    }

    {
      // Hybrid FOR head + fused bitpack subtree: legacy FOR head produces the
      // `deltas` channel, which a fused bitpack subtree compresses; decode
      // dispatches the fused subtree, then inverts FOR. Full roundtrip.
      auto t = make_int32_table(2, 4096, 5);
      std::string dsl =
        "input -> for -> deltas, references, reference_offsets\n"
        "for.deltas -> bitpack\n"
        "---\n"
        "input -> for -> deltas, references, reference_offsets\n"
        "for.deltas -> bitpack\n";
      auto ct = roundtrip_once(t->view(), dsl, 1, "for_bitpack_subtree");
      expect(ct.columns.size() == 2, "for_bitpack_subtree: column count");
      for (auto const& col : ct.columns) {
        expect(col.compound != nullptr, "for_bitpack_subtree: compound");
      }
      roundtrip_once(t->view(), dsl, 2, "for_bitpack_subtree_mt");
    }

    // NOTE: a FOR head + fused bitpack subtree + nvcomp tail on the fused
    // bitpack's `packed` channel (e.g. `for.deltas.packed -> ans`) is NOT
    // supported: the fused subtree leaf does not expose chunk_min/chunk_count/
    // chunk_bits/packed as resolvable downstream paths the way a legacy bitpack
    // head does, so the plan fails to resolve. See decode-roundtrip-matrix in
    // docs/compress_with_plan_plan.md.

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
        "input -> for -> deltas, references, reference_offsets\n"
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

    std::printf("test_compress_with_plan_roundtrip: PASS\n");
    return 0;
  } catch (std::exception const& e) {
    std::fprintf(stderr, "test_compress_with_plan_roundtrip: FAIL: %s\n", e.what());
    return 1;
  }
}
