// Meld's decoders against NVRTC's, bit for bit, over every closed-form tree of up to two
// transforms in every element width.

#include "codegen/bridge/meld_backend.hpp"
#include "codegen/jit/fused_tree.hpp"
#include "codegen/jit/kernel_cache.hpp"
#include "gpu_encode.hpp"
#include "jit_decode.hpp"

#include <cuda.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace cc  = codegen;
namespace jit = codegen::jit;
using cc::OpKind;
using Tree = std::shared_ptr<jit::FusedTree>;

namespace {

void expect(bool cond, std::string const& msg)
{
  if (!cond) throw std::runtime_error(msg);
}

std::string tag(jit::FusedTree const& t)
{
  switch (t.op) {
    case OpKind::Bitpack: return "Bp";
    case OpKind::Raw: return "Raw";
    case OpKind::For: return "For(" + tag(*t.children.at("deltas")) + ")";
    case OpKind::Zigzag: return "Zigzag(" + tag(*t.children.at("zigzag")) + ")";
    default: return "?";
  }
}

// Zigzag is never a leaf: fused trees always give it a child.
std::vector<Tree> closed_form_trees(int transforms)
{
  std::vector<Tree> layer = {jit::FusedTree::make(OpKind::Bitpack),
                             jit::FusedTree::make(OpKind::Raw)};
  std::vector<Tree> all   = layer;
  for (int depth = 0; depth < transforms; ++depth) {
    std::vector<Tree> next;
    for (auto const& child : layer) {
      next.push_back(jit::FusedTree::make(OpKind::For, {{"deltas", child}}));
      next.push_back(jit::FusedTree::make(OpKind::Zigzag, {{"zigzag", child}}));
    }
    all.insert(all.end(), next.begin(), next.end());
    layer = std::move(next);
  }
  return all;
}

template <typename T>
T from_bits(std::uint64_t low, std::uint64_t high)
{
  if constexpr (sizeof(T) == 16) {
    return static_cast<T>((static_cast<unsigned __int128>(high) << 64) | low);
  } else {
    (void)high;
    return static_cast<T>(low);
  }
}

template <typename T>
std::vector<std::vector<T>> fixtures(std::int64_t n)
{
  std::vector<T> small(n), full(n), extremes(n);
  std::uint64_t state = 0x9e3779b97f4a7c15ull;
  auto next           = [&] {
    state ^= state << 13;
    state ^= state >> 7;
    state ^= state << 17;
    return state;
  };
  constexpr std::int32_t range = sizeof(T) == 1 ? 50 : sizeof(T) == 2 ? 500 : 50000;
  T lowest, highest;
  if constexpr (sizeof(T) == 16) {
    lowest  = from_bits<T>(0, std::uint64_t{1} << 63);
    highest = from_bits<T>(~std::uint64_t{0}, ~std::uint64_t{0} >> 1);
  } else {
    lowest  = std::numeric_limits<T>::min();
    highest = std::numeric_limits<T>::max();
  }
  for (std::int64_t i = 0; i < n; ++i) {
    std::int32_t const pos  = static_cast<std::int32_t>(i % cc::kChunkSize);
    small[i]                = static_cast<T>(-(range / 10) + pos % (range / 2));
    std::uint64_t const low = next();
    full[i]                 = from_bits<T>(low, next());
    extremes[i]             = (i & 1) ? lowest : highest;
  }
  return {small, full, extremes};
}

template <typename T>
std::string hex(T value)
{
  static constexpr char digits[] = "0123456789abcdef";
  unsigned char bytes[sizeof(T)];
  std::memcpy(bytes, &value, sizeof(T));
  std::string text = "0x";
  for (std::size_t byte = sizeof(T); byte-- > 0;) {
    text += digits[bytes[byte] >> 4];
    text += digits[bytes[byte] & 15];
  }
  return text;
}

template <typename T>
std::vector<T> meld_decode_tree(jit::FusedTree const& tree,
                                char const* cxx,
                                std::int64_t n,
                                codegen_test::GpuEncoded& encoded)
{
  auto const decoder = simpatico::meld::plain_decode_kernel(tree, cxx);
  expect(decoder.has_value(), "Meld declined a closed-form tree");
  std::int32_t const num_chunks = cc::num_chunks_for(n);

  CUdeviceptr out   = encoded.alloc(static_cast<std::size_t>(n > 0 ? n : 1) * sizeof(T));
  std::int64_t rows = n;
  std::vector<CUdeviceptr> pointers;
  pointers.reserve(decoder->parameters.size());
  std::vector<void*> args;
  for (auto const& parameter : decoder->parameters) {
    switch (parameter.kind) {
      case simpatico::meld::Parameter::Kind::channel: {
        auto const key   = jit::buffer_key(parameter.node, parameter.field);
        auto const found = encoded.buffers.find(key);
        expect(found != encoded.buffers.end(), "no buffer '" + key + "' for Meld's decoder");
        pointers.push_back(reinterpret_cast<CUdeviceptr>(found->second.ptr));
        args.push_back(&pointers.back());
        break;
      }
      case simpatico::meld::Parameter::Kind::output: args.push_back(&out); break;
      case simpatico::meld::Parameter::Kind::row_count: args.push_back(&rows); break;
    }
  }
  unsigned const grid =
    static_cast<unsigned>((num_chunks + decoder->chunks_per_block - 1) / decoder->chunks_per_block);
  codegen_test::cu_check(cuLaunchKernel(decoder->kernel->func_for_current_device(),
                                        grid,
                                        1,
                                        1,
                                        static_cast<unsigned>(decoder->block_threads),
                                        1,
                                        1,
                                        static_cast<unsigned>(decoder->shared_bytes),
                                        nullptr,
                                        args.data(),
                                        nullptr),
                         "cuLaunchKernel(meld)");
  codegen_test::cu_check(cuCtxSynchronize(), "cuCtxSynchronize(meld)");
  std::vector<T> decoded(static_cast<std::size_t>(n));
  codegen_test::rt_check(cudaMemcpy(decoded.data(),
                                    reinterpret_cast<void const*>(out),
                                    decoded.size() * sizeof(T),
                                    cudaMemcpyDeviceToHost),
                         "memcpy meld D2H");
  return decoded;
}

template <typename T>
int sweep(char const* name, char const* cxx, int arch, std::vector<Tree> const& trees)
{
  int failures = 0, cases = 0;
  for (std::int64_t n : {std::int64_t{1}, std::int64_t{4321}}) {
    auto const data = fixtures<T>(n);
    for (std::size_t f = 0; f < data.size(); ++f) {
      for (auto const& tree : trees) {
        std::string const what = std::string(name) + "/" + tag(*tree) + " fixture " +
                                 std::to_string(f) + " n=" + std::to_string(n);
        ++cases;
        try {
          auto encoded = codegen_test::gpu_encode_tree<T>(*tree, cxx, data[f].data(), n, arch);
          auto const nvrtc =
            codegen_test::jit_decode_tree<T>(*tree, cxx, n, encoded.buffers, encoded, arch);
          auto const meld = meld_decode_tree<T>(*tree, cxx, n, encoded);
          if (std::memcmp(nvrtc.data(), meld.data(), meld.size() * sizeof(T)) != 0) {
            std::size_t row = 0;
            while (std::memcmp(&nvrtc[row], &meld[row], sizeof(T)) == 0)
              ++row;
            std::fprintf(stderr,
                         "FAIL [%s] Meld and NVRTC disagree from row %zu: nvrtc %s, meld %s\n",
                         what.c_str(),
                         row,
                         hex(nvrtc[row]).c_str(),
                         hex(meld[row]).c_str());
            ++failures;
          } else if (std::memcmp(meld.data(), data[f].data(), meld.size() * sizeof(T)) != 0) {
            std::fprintf(stderr, "FAIL [%s] both decode something else\n", what.c_str());
            ++failures;
          }
        } catch (std::exception const& e) {
          std::fprintf(stderr, "FAIL [%s] %s\n", what.c_str(), e.what());
          ++failures;
        }
      }
    }
  }
  std::printf("  [%-4s] %d/%d agree\n", name, cases - failures, cases);
  return failures;
}

void test_scheme_text()
{
  auto const tree =
    jit::FusedTree::make(OpKind::For, {{"deltas", jit::FusedTree::make(OpKind::Bitpack)}});
  auto const scheme = simpatico::meld::decode_scheme(*tree, "int64_t");
  expect(scheme.has_value(), "For(Bp) has no scheme");
  std::string const expected =
    "codec.scheme @fr_bp_int64_t : i64 chunk 1024 {\n"
    "^bb0(%arg0: !codec.strided<i64>):\n"
    "  %references, %deltas = codec.for %arg0 node 0 : !codec.strided<i64>\n"
    "  %chunk_min, %chunk_bits, %packed = codec.bitpack %deltas node 1 : !codec.strided<i64>\n"
    "  codec.channels %references, %chunk_min, %chunk_bits, %packed : !codec.per_chunk<i64>, "
    "!codec.per_chunk<i64>, !codec.per_chunk<i8>, !codec.words\n"
    "}\n";
  expect(*scheme == expected, "unexpected scheme for For(Bp):\n" + *scheme);
}

void test_declines_other_codecs()
{
  auto const delta =
    jit::FusedTree::make(OpKind::Delta, {{"differences", jit::FusedTree::make(OpKind::Bitpack)}});
  expect(!simpatico::meld::decode_scheme(*delta, "int32_t"), "Delta(Bp) has a scheme");
  auto const rle = jit::FusedTree::make(
    OpKind::Zigzag,
    {{"zigzag",
      jit::FusedTree::make(OpKind::Rle,
                           {{"runs", jit::FusedTree::make(OpKind::Bitpack)},
                            {"values", jit::FusedTree::make(OpKind::Bitpack)}})}});
  expect(!simpatico::meld::decode_scheme(*rle, "int32_t"), "Zigzag(Rle) has a scheme");
  expect(!simpatico::meld::decode_scheme(*jit::FusedTree::make(OpKind::Zigzag), "int32_t"),
         "a childless Zigzag has a scheme");
  expect(!simpatico::meld::decode_scheme(*jit::FusedTree::make(OpKind::Bitpack), "float"),
         "a float column has a scheme");
}

}  // namespace

int main()
{
  try {
    test_scheme_text();
    test_declines_other_codecs();
    if (cudaSetDevice(0) != cudaSuccess) {
      std::fprintf(stderr, "test_meld_decode: cudaSetDevice failed\n");
      return 1;
    }
    if (std::getenv("SIMPATICO_JIT_CACHE_CLEAR")) jit::clear_jit_disk_cache();
    int const arch   = jit::arch_cc_for_current_device();
    auto const trees = closed_form_trees(/*transforms=*/2);
    int failures     = 0;
    failures += sweep<std::int8_t>("i8", "int8_t", arch, trees);
    failures += sweep<std::int16_t>("i16", "int16_t", arch, trees);
    failures += sweep<std::int32_t>("i32", "int32_t", arch, trees);
    failures += sweep<std::int64_t>("i64", "int64_t", arch, trees);
    failures += sweep<__int128>("i128", "__int128", arch, trees);
    std::printf("test_meld_decode: %s\n", failures == 0 ? "OK" : "FAILED");
    return failures == 0 ? 0 : 1;
  } catch (std::exception const& e) {
    std::fprintf(stderr, "test_meld_decode: %s\n", e.what());
    return 1;
  }
}
