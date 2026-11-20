// Owned compiled-module and kernel handles for plain-CUDA nvrtc JIT (encode + decode renderers).
#pragma once

#include <cstdint>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

typedef struct CUlib_st* CUlibrary;
typedef struct CUfunc_st* CUfunction;
typedef struct CUkern_st* CUkernel;
typedef struct CUctx_st* CUcontext;

namespace codegen::jit {

struct CompileError : std::runtime_error {
  std::string log;
  std::string source;

  CompileError(std::string what_arg, std::string log_, std::string src_)
    : std::runtime_error(std::move(what_arg)), log(std::move(log_)), source(std::move(src_))
  {
  }
};

struct CompileOptions {
  int arch_cc = 0;  // must be set by caller via arch_cc_for_current_device()
  // Whether the source may #include the embedded project and CCCL headers. A source that includes
  // none of them should say so: handing the bundle to NVRTC costs every compilation a few
  // milliseconds, used or not. Such a source is compiled with NVRTC's --minimal option, so it must
  // not use the CUDA runtime API, the device runtime API or the texture and surface functions.
  bool embedded_headers = true;
  // Whether NVRTC may contract a floating-point multiplication and an addition or subtraction of
  // its product into one fused operation, as it does by default. The fused operation rounds once
  // where the two operations round twice, so a source whose floating-point results must not depend
  // on what the compiler chooses to fuse turns this off. A call of fma() is fused either way.
  bool fuse_multiply_add = true;
};

// Returns major*10+minor for the current CUDA device (e.g. 90 for H100).
// Throws std::runtime_error if the device cannot be queried.
int arch_cc_for_current_device();

class CompiledModule;

/**
 * @brief One kernel of a `CompiledModule`, obtained from `CompiledModule::kernel()`.
 */
class CompiledKernel {
 public:
  /** @brief Makes the constructor usable by `CompiledModule` alone, which creates every kernel. */
  class ConstructionKey {
    friend class CompiledModule;
    ConstructionKey() = default;
  };

  CompiledKernel(ConstructionKey, const CompiledModule& owner, CUkernel kernel) noexcept
    : kern_(kernel), module_(&owner)
  {
  }
  CompiledKernel(const CompiledKernel&)            = delete;
  CompiledKernel& operator=(const CompiledKernel&) = delete;

  // Device-independent kernel handle (valid on all devices sharing this library).
  // Use func_for_current_device() to get a context-specific CUfunction for launch.
  [[nodiscard]] CUkernel kern() const noexcept { return kern_; }

  // Returns the CUfunction for the CUDA context current on the calling thread,
  // deriving and caching it on first use in that context. A CUfunction can only be
  // launched from the context it was derived in, so the cache is keyed by context,
  // not device. Returns nullptr if kern is null or derivation fails.
  CUfunction func_for_current_device() const;

  /** @brief Returns the module this kernel belongs to. */
  [[nodiscard]] const CompiledModule& module() const noexcept { return *module_; }

 private:
  CUkernel kern_;
  const CompiledModule* module_;
  mutable std::mutex func_mu_;
  mutable std::unordered_map<CUcontext, CUfunction> func_per_ctx_;
};

/**
 * @brief One compiled source loaded into the CUDA driver, whose kernels are resolved by entry
 *        symbol.
 *
 * NVRTC compiles a source as a whole, so every `extern "C" __global__` function it defines is a
 * kernel of the same module. A `CompiledKernel` handle shares ownership of its module: the CUDA
 * library stays loaded until the module and every handle resolved from it have been released.
 */
class CompiledModule : public std::enable_shared_from_this<CompiledModule> {
 public:
  /**
   * @brief Loads an already-compiled @p cubin (skips nvrtc).
   *
   * @param cubin The image to load
   * @param rendered_source Stored for diagnostics only; it need not match the cubin
   * @throws std::runtime_error on driver-API failure so callers can fall back.
   */
  [[nodiscard]] static std::shared_ptr<const CompiledModule> load(std::vector<char> cubin,
                                                                  std::string rendered_source = {});

  /**
   * @brief Returns the kernel @p entry_symbol of this module. Thread-safe; every call for one
   *        symbol returns the same kernel.
   *
   * @throws std::runtime_error when the module has no kernel of that name.
   */
  [[nodiscard]] std::shared_ptr<const CompiledKernel> kernel(const std::string& entry_symbol) const;

  [[nodiscard]] CUlibrary library() const noexcept { return library_; }
  [[nodiscard]] const std::vector<char>& cubin() const noexcept { return cubin_; }
  [[nodiscard]] const std::string& rendered_source() const noexcept { return rendered_source_; }

  /** @brief Makes the constructor usable by `load()` alone, so a module is always shared. */
  class ConstructionKey {
    friend class CompiledModule;
    ConstructionKey() = default;
  };

  CompiledModule(ConstructionKey, std::vector<char> cubin, std::string rendered_source);
  ~CompiledModule();
  CompiledModule(const CompiledModule&)            = delete;
  CompiledModule& operator=(const CompiledModule&) = delete;

 private:
  std::vector<char> cubin_;
  std::string rendered_source_;
  CUlibrary library_ = nullptr;
  mutable std::mutex kernels_mu_;
  // Node-based: a kernel's address is stable for the life of the module.
  mutable std::unordered_map<std::string, CompiledKernel> kernels_;
};

// Compile a raw CUDA-C++ source string via nvrtc and load the resulting cubin.
// Throws CompileError on nvrtc rejection (with .log and .source) and
// std::runtime_error on driver-API failures.
std::shared_ptr<const CompiledKernel> compile_plain_kernel(const std::string& source,
                                                           const std::string& entry_symbol,
                                                           const CompileOptions& opts = {});

/**
 * @brief Cumulative cost of the NVRTC compilations and cubin loads this process has run, split by
 *        pipeline stage.
 *
 * Every counter only grows, so the cost of a piece of work is the difference between the snapshots
 * taken before and after it. A compilation that fails still adds the time of the stages it reached.
 */
struct CompileStats {
  std::uint64_t compiles{0};            ///< Compilations that produced a cubin
  std::uint64_t source_bytes{0};        ///< Kernel source bytes of those compilations
  std::uint64_t create_program_us{0};   ///< Microseconds in nvrtcCreateProgram
  std::uint64_t compile_program_us{0};  ///< Microseconds in nvrtcCompileProgram
  std::uint64_t get_cubin_us{0};        ///< Microseconds fetching the cubin and freeing the program
  std::uint64_t loads{0};               ///< Cubins loaded, whether just compiled or read from disk
  std::uint64_t load_us{0};             ///< Microseconds in cuLibraryLoadData
};

/** @brief Returns a snapshot of the process-wide `CompileStats`. Thread-safe. */
[[nodiscard]] CompileStats compile_stats() noexcept;

}  // namespace codegen::jit
