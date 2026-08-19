// Rendered decode signature contract — the emitted kernel parameter list and
// DecodeKernelSpec's machine-readable description of it must agree.
//
// Why this exists
// ===============
// The launcher pushes kernel arguments through cuLaunchKernel's untyped
// `void**`.  Nothing in the type system relates what the renderer DECLARES to
// what the launcher PUSHES, so a disagreement is not a compile error — it is
// silent argument misalignment (a pointer read as an int64, or a predicate
// bound where a mask pointer belongs).  The renderer therefore emits the
// declaration text and `spec.trailing` from one table, and the launcher binds
// by walking `spec.trailing`.  This test pins that agreement:
//
//   declared parameters == buffers.size() + 2 (out, n) + trailing.size()
//
// It parses the rendered signature rather than comparing against a golden
// string, so ordinary changes to kernel BODIES do not churn it — only a change
// to the signature contract can break it.
//
// Pure string generation: no GPU, no NVRTC, no encode fixtures.

#include "codegen/decode/jit/renderer.hpp"
#include "codegen/jit/fused_tree.hpp"

#include <cstdio>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace cdj = codegen::decode::jit;
namespace jit = codegen::jit;
using codegen::OpKind;

namespace {

int g_failures = 0;

#define CHECK(cond, ...)                          \
  do {                                            \
    if (!(cond)) {                                \
      std::fprintf(stderr, "FAIL: " __VA_ARGS__); \
      std::fprintf(stderr, "\n");                 \
      ++g_failures;                               \
    }                                             \
  } while (0)

// Count the parameters of the rendered `extern "C" __global__ void sym(...)`
// declaration: the text between the entry symbol's '(' and the matching ')'.
// Parameters are one per comma at depth 0; an empty list yields 0.
std::size_t count_declared_params(const std::string& src, const std::string& sym)
{
  const std::size_t at = src.find("void " + sym + "(");
  if (at == std::string::npos) return 0;
  std::size_t i     = src.find('(', at);
  int depth         = 0;
  std::size_t count = 1;  // no trailing comma after the last parameter
  bool any          = false;
  for (; i < src.size(); ++i) {
    const char c = src[i];
    if (c == '(') {
      ++depth;
    } else if (c == ')') {
      if (--depth == 0) break;
    } else if (c == ',' && depth == 1) {
      ++count;
    } else if (depth == 1 && !std::isspace(static_cast<unsigned char>(c))) {
      any = true;
    }
  }
  return any ? count : 0;
}

struct Shape {
  const char* name;
  std::shared_ptr<jit::FusedTree> tree;
};

bool check_variant(const Shape& shape, const std::string& dtype, cdj::DecodeShape variant)
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
{
  cdj::DecodeKernelSpec spec;
  try {
    spec = cdj::render(*shape.tree, dtype, /*num_chunks=*/7, variant);
  } catch (const cdj::RenderError&) {
    return true;  // shape legitimately unsupported for this variant
  }

  const std::size_t declared = count_declared_params(spec.source, spec.entry_symbol);
  const std::size_t expected = spec.buffers.size() + 2 /*out, n*/ + spec.trailing.size();
  CHECK(declared == expected,
        "[%s/%s/e%d.c%d] rendered signature declares %zu parameters but the spec describes %zu "
        "(buffers=%zu + out,n + trailing=%zu) — emission and binding have drifted, which "
        "cuLaunchKernel cannot detect",
        shape.name,
        dtype.c_str(),
        static_cast<int>(variant.enumerator),
        static_cast<int>(variant.consumer),
        declared,
        expected,
        spec.buffers.size(),
        spec.trailing.size());

  // Every trailing tag must be a real enumerator (the launcher indexes a table
  // by it) and must appear at most once per signature.
  std::vector<int> seen(static_cast<std::size_t>(cdj::TrailingParam::kCount), 0);
  for (const auto tag : spec.trailing) {
    const auto idx = static_cast<std::size_t>(tag);
    CHECK(idx < static_cast<std::size_t>(cdj::TrailingParam::kCount),
          "[%s/%s/e%d.c%d] trailing tag %zu out of range",
          shape.name,
          dtype.c_str(),
          static_cast<int>(variant.enumerator),
          static_cast<int>(variant.consumer),
          idx);
    if (idx < seen.size()) {
      CHECK(seen[idx] == 0,
            "[%s/%s/e%d.c%d] trailing tag %zu appears twice — the launcher would bind the same "
            "storage to two parameters",
            shape.name,
            dtype.c_str(),
            static_cast<int>(variant.enumerator),
            static_cast<int>(variant.consumer),
            idx);
      seen[idx] = 1;
    }
  }
  return true;
}

}  // namespace

int main()
{
  const std::vector<std::string> dtypes = {"int32_t", "int64_t", "int16_t", "int8_t"};
  const cdj::DecodeShape variants[]     = {
    cdj::kShapePlain,
    cdj::kShapeMaskOut,
    cdj::kShapeMaskConsume,
    cdj::kShapeDictGather,
    cdj::kShapeIndexConsume,
    cdj::kShapeStrSplitMeta,
  };

  const std::vector<Shape> shapes = {
    {"bitpack_leaf", jit::FusedTree::make(OpKind::Bitpack)},
    {"delta_bitpack",
     jit::FusedTree::make(OpKind::Delta, {{"differences", jit::FusedTree::make(OpKind::Bitpack)}})},
  };

  for (const auto& shape : shapes) {
    for (const auto& dtype : dtypes) {
      for (const auto variant : variants) {
        check_variant(shape, dtype, variant);
      }
    }
  }

  if (g_failures == 0) {
    std::printf(
      "PASS: rendered signature matches DecodeKernelSpec (params == buffers + 2 + "
      "trailing) for every shape/dtype/variant\n");
    std::printf("ALL PASS\n");
    return 0;
  }
  std::fprintf(stderr, "%d failure(s)\n", g_failures);
  return 1;
}
