// Shape-parity sweep: encode -> decode -> equality across every
// decode-supported FusedTree shape up to a small depth bound.
//
// Why this exists
// ---------------
// The decode side supports {Bitpack, Delta, Rle, Raw (as Rle.runs)}
// composed freely; the GPU encode kernel (`gpu_encode_tree`) walks the
// same FusedTree IR.  Per-shape ctests cover hand-picked compositions
// but only those — adding a new op to one side without the other (or
// silently breaking composability for a particular nesting depth)
// goes unnoticed until a downstream user trips it.
//
// This test enumerates EVERY shape the IR can express up to a given
// depth and runs the full pipeline on each:
//
//     FusedTree -> gpu_encode_tree -> jit_decode_tree -> equality
//
// Any divergence between encode and decode coverage manifests as
// either a `render_type` failure, a JIT compile error, a bind-time
// dtype/size mismatch, or a decode-equality miss — all surfaced as
// per-shape failures with the offending tag.
//
// Enumeration
// -----------
// Each shape has depth d in [1, max_depth] where depth = longest
// root-to-leaf path.  Build by induction on d:
//
//   * d = 1: {Bitpack}
//   * d > 1: for every shape c of depth d-1, add Delta(values=c),
//            Rle(runs=Bitpack, values=c), Rle(runs=Raw, values=c).
//
// Counts up to d = 4: 1 + 3 + 9 + 27 = 40 shapes.  At ~400 ms cold
// JIT compile per shape, full sweep is ~15-20 s.  Set the env var
// SIMPATICO_PARITY_DEPTH to override (default 4).
//
// Fixture
// -------
// int32_t throughout (matches the existing JIT-roundtrip tests).
// Two synthetic columns: `synth_data` (generic, depth-no-Rle), and
// `synth_rle_data` (chunk-varied RLE patterns).  Shapes containing
// any Rle node use the RLE-friendly fixture.

#include "codegen/jit/fused_tree.hpp"
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
// fixtures so a parity failure here is directly comparable.
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

// The GPU encode kernel only emits the fixed-stride OverAllocate
// bitpack layout, so every Bitpack node must carry fixed_stride=true
// for *encode*.  The same tree object drives the JIT decode, but decode
// is Compact-only (drop-overalloc): the decode renderer ignores
// fixed_stride and the encode helper injects the real arithmetic
// bp_offsets for the Compact gather.
void set_fixed_stride(jit::FusedTree& t)
{
  if (t.op == OpKind::Bitpack) t.fixed_stride = true;
  for (auto& [_, c] : t.children)
    set_fixed_stride(*c);
}

// Pretty-print a tree like "Rle{runs=Bp, values=Delta(Bp)}" — used
// only as the per-shape diagnostic tag; the JIT cache keys on the
// rendered C++ type expression, not this string.
std::string tag(const jit::FusedTree& t)
{
  auto short_name = [](OpKind k) -> const char* {
    switch (k) {
      case OpKind::Bitpack: return "Bp";
      case OpKind::Delta: return "Delta";
      case OpKind::Rle: return "Rle";
      case OpKind::Raw: return "Raw";
      default: return "?";
    }
  };
  if (t.is_leaf()) return short_name(t.op);
  std::ostringstream os;
  os << short_name(t.op);
  if (t.op == OpKind::Delta) {
    auto it = t.children.find("differences");
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
// Each child becomes 3 new trees: Delta(c), Rle(Bp, c), Rle(Raw, c).
// Raw is only legal as a Rle.runs leaf; Bp is the other allowed runs
// child (other shapes go in values).
std::vector<Tree> compose_layer(const std::vector<Tree>& prev)
{
  std::vector<Tree> out;
  out.reserve(prev.size() * 3);
  for (const auto& c : prev) {
    out.push_back(jit::FusedTree::make(OpKind::Delta,
                                       {
                                         {"differences", c},
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

// Shapes the combinatorial sweep does not generate: it only nests in the
// `values` position with `runs` fixed to Bitpack/Raw. These cover an Rle nested
// in the `runs` position, the one composition outside that scheme.
std::vector<Tree> extra_shapes()
{
  return {
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
  if (const char* s = std::getenv("SIMPATICO_PARITY_DEPTH"); s != nullptr) {
    int v = std::atoi(s);
    if (v >= 1 && v <= 6) return v;  // cap at 6 — anything more is hours
    std::fprintf(stderr,
                 "warn: SIMPATICO_PARITY_DEPTH='%s' out of [1,6]; using default %d\n",
                 s,
                 default_depth);
  }
  return default_depth;
}

}  // namespace

int main()
{
  try {
    jit::ensure_cuda_context();
  } catch (const std::exception& e) {
    std::fprintf(stderr, "FAIL: ensure_cuda_context: %s\n", e.what());
    return 1;
  }
  const int arch      = detect_arch_cc();
  const int max_depth = env_depth(/*default=*/4);
  const int64_t n     = 4321;
  auto data_generic   = synth_data(n);
  auto data_rle       = synth_rle_data(n);

  auto shapes = enumerate_shapes(max_depth);
  for (auto& t : extra_shapes())
    shapes.push_back(t);
  std::printf("test_shape_parity: max_depth=%d shapes=%zu n=%lld\n",
              max_depth,
              shapes.size(),
              static_cast<long long>(n));

  int passed = 0;
  std::vector<std::string> failures;
  for (const auto& t : shapes) {
    set_fixed_stride(*t);
    const std::string s_tag = tag(*t);
    const auto& fixture     = contains_rle(*t) ? data_rle : data_generic;
    if (run_one_shape(*t, s_tag, fixture, arch) == 0) {
      std::printf("  %-50s OK\n", s_tag.c_str());
      ++passed;
    } else {
      failures.push_back(s_tag);
    }
  }

  std::printf("test_shape_parity: %d/%zu passed\n", passed, shapes.size());
  if (!failures.empty()) {
    std::fprintf(stderr, "test_shape_parity: %zu failures:\n", failures.size());
    for (const auto& f : failures) {
      std::fprintf(stderr, "  - %s\n", f.c_str());
    }
    return 1;
  }
  return 0;
}
