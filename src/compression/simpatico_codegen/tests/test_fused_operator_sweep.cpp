// Fused-operator sweep: encode -> decode -> equality across a systematic,
// depth-bounded FusedTree shape family spanning every fused op. (Companion to
// test_operator_sweep, which sweeps the full operator catalog x dtypes via the
// plan DSL; this one stresses composition in the fused IR directly.)
//
// Why this exists
// ---------------
// The encode and decode sides support {Bitpack, Delta, Rle, For, Zigzag,
// Raw passthrough} composed through the FusedTree IR.  The GPU encode kernel
// (`gpu_encode_tree`) and JIT decode kernel walk the same tree.  Per-shape
// ctests cover hand-picked compositions but only those — adding a new op to
// one side without the other (or silently breaking composability for a
// particular nesting depth) goes unnoticed until a downstream user trips it.
//
// This test enumerates the recursive grammar below up to a given depth, adds
// the non-recursive boundary forms separately, and runs the full pipeline on
// each:
//
//     FusedTree -> gpu_encode_tree -> jit_decode_tree -> equality
//
// Any divergence between encode and decode coverage manifests as
// either a renderer `RenderError`, a JIT compile error, a bind-time
// dtype/size mismatch, or a decode-equality miss — all surfaced as
// per-shape failures with the offending tag.
//
// Enumeration
// -----------
// Each shape has depth d in [1, max_depth] where depth = longest
// root-to-leaf path.  Build by induction on d:
//
//   * d = 1: {Bitpack}
//   * d > 1: for every shape c of depth d-1, add Delta(differences=c),
//            For(deltas=c), Zigzag(zigzag=c),
//            Rle(runs=Bitpack, values=c), Rle(runs=Raw, values=c).
//
// Counts up to d = 4: 1 + 5 + 25 + 125 = 156 recursively generated
// shapes, plus boundary cases for leaf/passthrough forms and RLE nesting in
// the runs branch. Set SIMPATICO_FUSED_SWEEP_DEPTH to override (default 4).
//
// Fixture
// -------
// int32_t throughout (matches the existing JIT-roundtrip tests).
// Two synthetic columns: `synth_data` (generic, no-RLE shapes), and
// `synth_rle_data` (chunk-varied RLE patterns).  Shapes containing
// any Rle node use the RLE-friendly fixture.

#include "codegen/jit/fused_tree.hpp"
#include "codegen/jit/kernel_cache.hpp"
#include "jit_decode.hpp"
#include "test_utils.hpp"

#include <cuda.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

namespace cc  = codegen;
namespace jit = codegen::jit;
using cc::OpKind;

namespace {

// ---------------------------------------------------------------------
// Small reporting + CUDA-error helpers (shared by every shape's run).
// ---------------------------------------------------------------------
std::string cu_err_str(CUresult r)
{
  const char* s = nullptr;
  cuGetErrorString(r, &s);
  return s ? std::string(s) : ("CUresult=" + std::to_string((int)r));
}

#define CU_RETURN_ERR(call, tag, what)                                                     \
  do {                                                                                     \
    CUresult _r = (call);                                                                  \
    if (_r != CUDA_SUCCESS) {                                                              \
      std::fprintf(stderr, "FAIL [%s]: %s (%s)\n", (tag), (what), cu_err_str(_r).c_str()); \
      return 1;                                                                            \
    }                                                                                      \
  } while (0)

// ---------------------------------------------------------------------
// Synthetic data — identical to test_jit_roundtrip / test_encode_*
// fixtures so a mismatch here is directly comparable.
// ---------------------------------------------------------------------
std::vector<int32_t> synth_data(int64_t n)
{
  std::vector<int32_t> data(static_cast<size_t>(n));
  for (int64_t i = 0; i < n; ++i) {
    int32_t cid = static_cast<int32_t>(i / cc::kChunkSize);
    int32_t pos = static_cast<int32_t>(i % cc::kChunkSize);
    switch (cid % 4) {
      case 0: data[i] = (pos & 1) ? 100 : 101; break;
      case 1: data[i] = static_cast<int32_t>(200 + (pos % 200)); break;
      case 2: data[i] = static_cast<int32_t>(-12345 + pos * 13); break;
      case 3: data[i] = static_cast<int32_t>(pos * 7 - 50000); break;
    }
  }
  return data;
}

// RLE-friendly fixture: spread of nruns_per_chunk to exercise the
// trivial (1 run), small (4 runs), and worst (1024 runs) cases.
std::vector<int32_t> synth_rle_data(int64_t n)
{
  std::vector<int32_t> data(static_cast<size_t>(n));
  for (int64_t i = 0; i < n; ++i) {
    int32_t cid = static_cast<int32_t>(i / cc::kChunkSize);
    int32_t pos = static_cast<int32_t>(i % cc::kChunkSize);
    switch (cid) {
      case 0: data[i] = 42; break;
      case 1: data[i] = pos / 256; break;
      case 2: data[i] = static_cast<int32_t>(1000 + pos); break;
      case 3: data[i] = (pos & 1) ? 50 : 51; break;
      default: {
        int32_t run_id = 0, cum = 0;
        const int32_t lens[5] = {17, 33, 80, 51, 44};
        while (cum + lens[run_id % 5] <= pos && run_id < 1000) {
          cum += lens[run_id % 5];
          ++run_id;
        }
        data[i] = 7 + run_id * 3;
        break;
      }
    }
  }
  return data;
}

// ---------------------------------------------------------------------
// Shape enumeration.
// ---------------------------------------------------------------------
using Tree = std::shared_ptr<jit::FusedTree>;

bool contains_rle(const jit::FusedTree& t)
{
  if (t.op == OpKind::Rle) return true;
  for (const auto& [_, c] : t.children) {
    if (contains_rle(*c)) return true;
  }
  return false;
}

// Pretty-print a tree like "Rle{runs=Bp, values=Delta(Bp)}" — used
// only as the per-shape diagnostic tag; the JIT cache keys on a hash
// of the rendered CUDA source, not this string.
std::string tag(const jit::FusedTree& t)
{
  auto short_name = [](OpKind k) -> const char* {
    switch (k) {
      case OpKind::Bitpack: return "Bp";
      case OpKind::For: return "For";
      case OpKind::Delta: return "Delta";
      case OpKind::Rle: return "Rle";
      case OpKind::Raw: return "Raw";
      case OpKind::Zigzag: return "Zigzag";
      default: return "?";
    }
  };
  if (t.is_leaf()) return short_name(t.op);
  std::ostringstream os;
  os << short_name(t.op);
  if (t.op == OpKind::Delta) {
    auto it = t.children.find("differences");
    os << "(" << (it != t.children.end() ? tag(*it->second) : "?") << ")";
  } else if (t.op == OpKind::For) {
    auto it = t.children.find("deltas");
    os << "(" << (it != t.children.end() ? tag(*it->second) : "?") << ")";
  } else if (t.op == OpKind::Zigzag) {
    auto it = t.children.find("zigzag");
    os << "(" << (it != t.children.end() ? tag(*it->second) : "?") << ")";
  } else if (t.op == OpKind::Rle) {
    auto r = t.children.find("runs");
    auto v = t.children.find("values");
    os << "{runs=" << (r != t.children.end() ? tag(*r->second) : "?")
       << ", values=" << (v != t.children.end() ? tag(*v->second) : "?") << "}";
  } else {
    os << "(...)";
  }
  return os.str();
}

// Build a tree of exact `depth` from a list of depth-(d-1) children.
// Each child becomes 5 new trees: Delta(c), For(c), Zigzag(c),
// Rle(Bp, c), and Rle(Raw, c).
// Raw terminal/passthrough forms are boundary cases below rather than recursive
// inputs, so this grammar remains bounded and every recursive child is decoded.
std::vector<Tree> compose_layer(const std::vector<Tree>& prev)
{
  std::vector<Tree> out;
  out.reserve(prev.size() * 5);
  for (const auto& c : prev) {
    out.push_back(jit::FusedTree::make(OpKind::Delta,
                                       {
                                         {"differences", c},
                                       }));
    out.push_back(jit::FusedTree::make(OpKind::For,
                                       {
                                         {"deltas", c},
                                       }));
    out.push_back(jit::FusedTree::make(OpKind::Zigzag,
                                       {
                                         {"zigzag", c},
                                       }));
    out.push_back(jit::FusedTree::make(OpKind::Rle,
                                       {
                                         {"runs", jit::FusedTree::make(OpKind::Bitpack)},
                                         {"values", c},
                                       }));
    out.push_back(jit::FusedTree::make(OpKind::Rle,
                                       {
                                         {"runs", jit::FusedTree::make(OpKind::Raw)},
                                         {"values", c},
                                       }));
  }
  return out;
}

std::vector<Tree> enumerate_shapes(int max_depth)
{
  std::vector<std::vector<Tree>> by_depth(max_depth + 1);
  by_depth[1].push_back(jit::FusedTree::make(OpKind::Bitpack));
  for (int d = 2; d <= max_depth; ++d) {
    by_depth[d] = compose_layer(by_depth[d - 1]);
  }
  std::vector<Tree> flat;
  for (int d = 1; d <= max_depth; ++d) {
    for (auto& t : by_depth[d])
      flat.push_back(t);
  }
  return flat;
}

// Shapes the recursive grammar does not generate:
//   * Zigzag may terminate as a leaf store.
//   * Delta and For may terminate through a Raw passthrough child.
//   * Rle may have Raw values, and may nest another Rle in its runs branch.
std::vector<Tree> extra_shapes()
{
  return {
    jit::FusedTree::make(OpKind::Zigzag),
    jit::FusedTree::make(OpKind::Delta,
                         {
                           {"differences", jit::FusedTree::make(OpKind::Raw)},
                         }),
    jit::FusedTree::make(OpKind::For,
                         {
                           {"deltas", jit::FusedTree::make(OpKind::Raw)},
                         }),
    jit::FusedTree::make(OpKind::Rle,
                         {
                           {"runs", jit::FusedTree::make(OpKind::Raw)},
                           {"values", jit::FusedTree::make(OpKind::Raw)},
                         }),
    jit::FusedTree::make(
      OpKind::Rle,
      {
        {"runs",
         jit::FusedTree::make(OpKind::Rle,
                              {
                                {"runs", jit::FusedTree::make(OpKind::Raw)},
                                {"values", jit::FusedTree::make(OpKind::Bitpack)},
                              })},
        {"values", jit::FusedTree::make(OpKind::Bitpack)},
      }),
  };
}

// ---------------------------------------------------------------------
// Per-shape runner — mirrors test_jit_roundtrip::run_shape with the
// addition that errors are reported but don't abort the sweep.
// ---------------------------------------------------------------------
int run_one_shape(const jit::FusedTree& tree,
                  const std::string& tag,
                  const std::vector<int32_t>& data,
                  int arch_cc)
{
  const int64_t n = static_cast<int64_t>(data.size());
  try {
    codegen_test::GpuEncoded encoded =
      codegen_test::gpu_encode_tree<int32_t>(tree, "int32_t", data.data(), n, arch_cc);
    auto recovered =
      codegen_test::jit_decode_tree<int32_t>(tree, "int32_t", n, encoded.buffers, encoded, arch_cc);
    if (!codegen_test::columns_equal(recovered, data)) {
      std::fprintf(stderr, "FAIL [%s] decode mismatch\n", tag.c_str());
      return 1;
    }
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "FAIL [%s] %s\n", tag.c_str(), e.what());
    return 1;
  }
}

int env_depth(int default_depth)
{
  if (const char* s = std::getenv("SIMPATICO_FUSED_SWEEP_DEPTH"); s != nullptr) {
    int v = std::atoi(s);
    if (v >= 1 && v <= 6) return v;  // cap at 6 — anything more is hours
    std::fprintf(stderr,
                 "warn: SIMPATICO_FUSED_SWEEP_DEPTH='%s' out of [1,6]; using default %d\n",
                 s,
                 default_depth);
  }
  return default_depth;
}

}  // namespace

int main()
{
  if (cudaSetDevice(0) != cudaSuccess) {
    std::fprintf(stderr, "FAIL: cudaSetDevice(0) failed\n");
    return 1;
  }
  // Single process, so clear at startup before any compile (see the sweep's
  // orchestrator for the same option).
  if (std::getenv("SIMPATICO_JIT_CACHE_CLEAR")) codegen::jit::clear_jit_disk_cache();
  const int arch      = jit::arch_cc_for_current_device();
  const int max_depth = env_depth(/*default=*/4);
  const int64_t n     = 4321;
  auto data_generic   = synth_data(n);
  auto data_rle       = synth_rle_data(n);

  auto shapes = enumerate_shapes(max_depth);
  for (auto& t : extra_shapes())
    shapes.push_back(t);
  std::printf("test_fused_operator_sweep: max_depth=%d shapes=%zu n=%lld\n",
              max_depth,
              shapes.size(),
              static_cast<long long>(n));

  int passed = 0;
  std::vector<std::string> failures;
  for (const auto& t : shapes) {
    const std::string s_tag = tag(*t);
    const auto& fixture     = contains_rle(*t) ? data_rle : data_generic;
    if (run_one_shape(*t, s_tag, fixture, arch) == 0) {
      std::printf("  %-50s OK\n", s_tag.c_str());
      ++passed;
    } else {
      failures.push_back(s_tag);
    }
  }

  std::printf("test_fused_operator_sweep: %d/%zu passed\n", passed, shapes.size());
  if (!failures.empty()) {
    std::fprintf(stderr, "test_fused_operator_sweep: %zu failures:\n", failures.size());
    for (const auto& f : failures) {
      std::fprintf(stderr, "  - %s\n", f.c_str());
    }
    return 1;
  }
  return 0;
}
