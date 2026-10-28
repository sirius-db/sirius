// SPDX-License-Identifier: Apache-2.0

#include "codegen/bridge/meld_backend.hpp"

#include "codegen/jit/kernel_cache.hpp"
#include "codegen/jit/render_util.hpp"
#include "codegen/util/cuda_check.hpp"
#include "jit/cache_identity.hpp"

#ifdef SIMPATICO_WITH_MELD
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/raw_ostream.h>
#include <meld/Codec/DecoderCompiler.h>
#include <meld/Dialect/Codec/IR/CodecDialect.h>
#include <meld/Dialect/Codec/IR/CodecOps.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OwningOpRef.h>
#include <mlir/IR/Verifier.h>
#endif

#include <cuda.h>
#include <cuda_runtime.h>

#include <unistd.h>

#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace simpatico::meld {
namespace {

namespace cg  = codegen;
namespace jit = codegen::jit;

#ifdef SIMPATICO_WITH_MELD

namespace codec = mlir::meld::codec;

std::optional<unsigned> element_bits(char const* cxx_dtype)
{
  if (cxx_dtype == nullptr) return std::nullopt;
  std::string_view const type{cxx_dtype};
  if (type == "int8_t") return 8;
  if (type == "int16_t") return 16;
  if (type == "int32_t") return 32;
  if (type == "int64_t") return 64;
  if (type == "__int128") return 128;
  return std::nullopt;
}

// Numbers nodes in DFS preorder, as the decode binder does.
class SchemeWriter {
 public:
  SchemeWriter(std::string const& name, unsigned bits)
  {
    context_.loadDialect<codec::CodecDialect>();
    auto const element = builder_.getIntegerType(bits);
    scheme_            = codec::SchemeOp::create(builder_, loc_, name, element, cg::kChunkSize);
    mlir::Type const column = codec::StridedType::get(&context_, element);
    builder_.createBlock(&scheme_->getBody(), {}, column, loc_);
  }

  mlir::Value column() { return scheme_->getColumn(); }

  // False where Meld does not decode `node`.
  bool write(jit::FusedTree const& node, mlir::Value input)
  {
    std::uint32_t const id = next_node_++;
    auto only_child        = [&](char const* port) -> jit::FusedTree const* {
      if (node.children.size() != 1) return nullptr;
      auto const found = node.children.find(port);
      return found == node.children.end() ? nullptr : found->second.get();
    };

    switch (node.op) {
      case cg::OpKind::Bitpack: {
        if (!node.is_leaf()) return false;
        auto bitpack = codec::BitpackOp::create(builder_, loc_, input, id);
        channels_.append(bitpack->result_begin(), bitpack->result_end());
        return true;
      }
      case cg::OpKind::Raw: {
        if (!node.is_leaf()) return false;
        channels_.push_back(codec::RawOp::create(builder_, loc_, input, id).getData());
        return true;
      }
      case cg::OpKind::For: {
        auto const* deltas_node = only_child("deltas");
        if (deltas_node == nullptr) return false;
        auto frame = codec::ForOp::create(builder_, loc_, input, id);
        channels_.push_back(frame.getReferences());
        return write(*deltas_node, frame.getDeltas());
      }
      case cg::OpKind::Zigzag: {
        auto const* zigzag_node = only_child("zigzag");
        if (zigzag_node == nullptr) return false;
        return write(*zigzag_node, codec::ZigzagOp::create(builder_, loc_, input, id).getZigzag());
      }
      default: return false;
    }
  }

  std::string finish()
  {
    codec::ChannelsOp::create(builder_, loc_, channels_);
    std::string diagnostics;
    mlir::ScopedDiagnosticHandler handler(&context_, [&](mlir::Diagnostic& diagnostic) {
      diagnostics += diagnostic.str() + "\n";
      return mlir::success();
    });
    if (mlir::failed(mlir::verify(*scheme_))) {
      throw std::logic_error("simpatico meld: the scheme written for a tree is not valid:\n" +
                             diagnostics);
    }
    std::string text;
    llvm::raw_string_ostream stream(text);
    scheme_->print(stream, mlir::OpPrintingFlags().assumeVerified());
    return text;
  }

 private:
  mlir::MLIRContext context_{mlir::MLIRContext::Threading::DISABLED};
  mlir::OpBuilder builder_{&context_};
  mlir::Location loc_ = builder_.getUnknownLoc();
  mlir::OwningOpRef<codec::SchemeOp> scheme_;
  llvm::SmallVector<mlir::Value> channels_;
  std::uint32_t next_node_ = 0;
};

// Printed at exit under SIMPATICO_JIT_STATS.
std::atomic<std::uint64_t> g_meld_compiles{0};
std::atomic<std::uint64_t> g_meld_compile_us{0};
std::atomic<std::uint64_t> g_meld_link_us{0};
std::atomic<std::uint64_t> g_meld_kernels{0};
struct MeldStatsReporter {
  ~MeldStatsReporter()
  {
    if (std::getenv("SIMPATICO_JIT_STATS") == nullptr) return;
    std::fprintf(stderr,
                 "[meld-stats pid=%d] kernels=%llu compiles=%llu compile_ms=%.0f link_ms=%.0f\n",
                 static_cast<int>(::getpid()),
                 static_cast<unsigned long long>(g_meld_kernels.load()),
                 static_cast<unsigned long long>(g_meld_compiles.load()),
                 g_meld_compile_us.load() / 1000.0,
                 g_meld_link_us.load() / 1000.0);
  }
};
MeldStatsReporter g_meld_stats_reporter;

std::uint64_t microseconds_since(std::chrono::steady_clock::time_point start)
{
  return static_cast<std::uint64_t>(
    std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - start)
      .count());
}

std::vector<char> link_ptx(std::string const& ptx)
{
  // The driver's linker needs a current context.
  CUcontext context = nullptr;
  if (cuCtxGetCurrent(&context) != CUDA_SUCCESS || context == nullptr) {
    simpatico::throw_if_cuda_error(cudaFree(nullptr), "simpatico meld: cudaFree(nullptr)");
  }

  std::array<char, 8192> log{};
  std::array<CUjit_option, 2> options = {CU_JIT_ERROR_LOG_BUFFER,
                                         CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES};
  std::array<void*, 2> values         = {log.data(), reinterpret_cast<void*>(log.size())};
  CUlinkState state                   = nullptr;
  simpatico::throw_if_cu_error(
    cuLinkCreate(static_cast<unsigned int>(options.size()), options.data(), values.data(), &state),
    "simpatico meld: cuLinkCreate");
  struct Destroy {
    CUlinkState state;
    ~Destroy() { cuLinkDestroy(state); }
  } destroy{state};

  if (cuLinkAddData(state,
                    CU_JIT_INPUT_PTX,
                    const_cast<char*>(ptx.c_str()),
                    ptx.size() + 1,
                    "meld.ptx",
                    0,
                    nullptr,
                    nullptr) != CUDA_SUCCESS) {
    throw std::runtime_error(std::string("simpatico meld: the driver rejected Meld's PTX: ") +
                             log.data());
  }
  void* cubin       = nullptr;
  std::size_t bytes = 0;
  if (cuLinkComplete(state, &cubin, &bytes) != CUDA_SUCCESS) {
    throw std::runtime_error(std::string("simpatico meld: linking Meld's PTX failed: ") +
                             log.data());
  }
  auto const* begin = static_cast<char const*>(cubin);
  return std::vector<char>(begin, begin + bytes);
}

jit::detail::Digest meld_environment()
{
  static jit::detail::Digest const identity = [] {
    int driver = 0;
    simpatico::throw_if_cu_error(cuDriverGetVersion(&driver), "simpatico meld: cuDriverGetVersion");
    std::string const driver_version             = std::to_string(driver);
    std::array<std::string_view, 2> const fields = {::meld::codec::compilerIdentity(),
                                                    driver_version};
    return jit::detail::framed_identity("simpatico-meld-environment-", fields);
  }();
  return identity;
}

// A decoder's cache key and launch parameters, without its code.
struct Decoder {
  std::string scheme;
  int arch = 0;
  jit::detail::Digest request{};
  std::string entry;
  std::vector<Parameter> parameters;
  int block_threads    = 0;
  int shared_bytes     = 0;
  int chunks_per_block = 1;
};

// Meld describes a decoder without compiling it.
Decoder describe(std::string scheme, int arch)
{
  std::string said;
  auto const description = ::meld::codec::describeDecoder(scheme, {}, &said);
  if (!description) {
    throw std::logic_error("simpatico meld: Meld rejects the scheme written for a tree:\n" + said +
                           "\n" + scheme);
  }
  Decoder decoder;
  std::string const architecture               = "sm_" + std::to_string(arch);
  std::array<std::string_view, 3> const fields = {scheme, description->entry, architecture};
  decoder.request          = jit::detail::framed_identity("simpatico-meld-request-", fields);
  decoder.scheme           = std::move(scheme);
  decoder.arch             = arch;
  decoder.entry            = description->entry;
  decoder.block_threads    = description->blockThreads;
  decoder.shared_bytes     = description->sharedBytes;
  decoder.chunks_per_block = description->chunksPerBlock;
  for (auto const& parameter : description->params) {
    using Kind = ::meld::codec::KernelParameter::Kind;
    Parameter bound;
    switch (parameter.kind) {
      case Kind::Channel:
        bound.kind  = Parameter::Kind::channel;
        bound.node  = parameter.node;
        bound.field = parameter.field;
        break;
      case Kind::Output: bound.kind = Parameter::Kind::output; break;
      case Kind::RowCount: bound.kind = Parameter::Kind::row_count; break;
    }
    decoder.parameters.push_back(std::move(bound));
  }
  return decoder;
}

jit::CompiledKernel compile(Decoder const& decoder)
{
  auto const start = std::chrono::steady_clock::now();
  std::string why;
  auto image = ::meld::codec::compileDecoder(decoder.scheme, {}, decoder.arch, &why);
  if (!image) {
    throw std::runtime_error("simpatico meld: Meld failed to compile a decoder:\n" + why + "\n" +
                             decoder.scheme);
  }
  g_meld_compile_us.fetch_add(microseconds_since(start), std::memory_order_relaxed);
  g_meld_compiles.fetch_add(1, std::memory_order_relaxed);
  auto const linked = std::chrono::steady_clock::now();
  auto cubin        = link_ptx(image->ptx);
  g_meld_link_us.fetch_add(microseconds_since(linked), std::memory_order_relaxed);
  return jit::load_kernel_from_cubin(std::move(cubin), image->entry, decoder.scheme);
}

std::mutex g_decoders_mu;
// The kernels themselves are the kernel cache's.
std::map<std::pair<int, std::string>, std::shared_ptr<Decoder const>> g_decoders;

#endif  // SIMPATICO_WITH_MELD

}  // namespace

Backend decode_backend()
{
  static Backend const backend = [] {
    char const* name = std::getenv("SIMPATICO_CODEGEN_BACKEND");
    if (name == nullptr || *name == '\0' || std::strcmp(name, "nvrtc") == 0) {
      return Backend::nvrtc;
    }
    Backend chosen;
    if (std::strcmp(name, "meld") == 0) {
      chosen = Backend::meld;
    } else if (std::strcmp(name, "shadow") == 0) {
      chosen = Backend::shadow;
    } else {
      throw std::invalid_argument(std::string("SIMPATICO_CODEGEN_BACKEND: unknown backend '") +
                                  name + "'; expected nvrtc, meld or shadow");
    }
#ifndef SIMPATICO_WITH_MELD
    throw std::invalid_argument(std::string("SIMPATICO_CODEGEN_BACKEND=") + name +
                                " needs Meld, which this build lacks (SIMPATICO_ENABLE_MELD)");
#endif
    return chosen;
  }();
  return backend;
}

std::optional<std::string> decode_scheme(jit::FusedTree const& tree, char const* cxx_dtype)
{
#ifdef SIMPATICO_WITH_MELD
  auto const bits = element_bits(cxx_dtype);
  if (!bits) return std::nullopt;
  SchemeWriter writer(jit::make_entry_symbol(tree, cxx_dtype, ""), *bits);
  if (!writer.write(tree, writer.column())) return std::nullopt;
  return writer.finish();
#else
  (void)tree;
  (void)cxx_dtype;
  return std::nullopt;
#endif
}

std::optional<DecodeKernel> plain_decode_kernel(jit::FusedTree const& tree, char const* cxx_dtype)
{
#ifdef SIMPATICO_WITH_MELD
  auto scheme = decode_scheme(tree, cxx_dtype);
  if (!scheme) return std::nullopt;
  int const arch = jit::arch_cc_for_current_device();
  auto key       = std::make_pair(arch, *scheme);

  std::shared_ptr<Decoder const> decoder;
  {
    std::lock_guard<std::mutex> lock(g_decoders_mu);
    if (auto found = g_decoders.find(key); found != g_decoders.end()) decoder = found->second;
  }
  if (!decoder) {
    auto described = std::make_shared<Decoder const>(describe(std::move(*scheme), arch));
    std::lock_guard<std::mutex> lock(g_decoders_mu);
    decoder = g_decoders.emplace(std::move(key), std::move(described)).first->second;
  }

  DecodeKernel launch;
  launch.kernel = jit::KernelCache::instance().get_or_build(
    decoder->request, meld_environment, decoder->entry, decoder->scheme, [&] {
      return compile(*decoder);
    });
  if (!launch.kernel || !launch.kernel->kern) {
    throw std::runtime_error("simpatico meld: null compiled kernel");
  }
  g_meld_kernels.fetch_add(1, std::memory_order_relaxed);
  launch.parameters       = decoder->parameters;
  launch.block_threads    = decoder->block_threads;
  launch.shared_bytes     = decoder->shared_bytes;
  launch.chunks_per_block = decoder->chunks_per_block;
  return launch;
#else
  (void)tree;
  (void)cxx_dtype;
  return std::nullopt;
#endif
}

}  // namespace simpatico::meld
