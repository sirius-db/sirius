// Single-flight contract of KernelCache::get_or_compile_plain:
//   1. Identical concurrent requests run one NVRTC compilation and share its kernel.
//   2. A compile failure reaches every requester of that compilation and is not remembered.
//   3. Different requests compile at the same time.
//   4. clear() during a compilation: the requesters of that compilation still get its kernel and
//   the kernel is not cached.
//   5. A request made after clear() does not join a compilation started before it.
//   6. Identical concurrent requests share one disk-cache load.
//   7. Concurrent requests for different kernels of one source share one compilation.
//   8. A source without the requested kernel is cached, and every request for that kernel fails.
//
// nvrtcCompileProgram and cuLibraryLoadData are replaced at link time (--wrap) by gates that count
// the calls and hold each one until the test lets it through. A held compilation or cubin load
// cannot finish before the other requesters have joined it, so no scenario depends on thread
// timing.

#include "codegen/jit/kernel_cache.hpp"

#include <cuda.h>
#include <cuda_runtime_api.h>

#include <nvrtc.h>

#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

namespace jit = codegen::jit;

namespace {

constexpr int requesters           = 8;
constexpr auto patience            = std::chrono::seconds(60);
constexpr const char entry[]       = "single_flight_test";
constexpr const char other_entry[] = "single_flight_other";

void require(bool condition, const char* message)
{
  if (!condition) throw std::runtime_error(message);
}

// Counts the calls of one wrapped function and, while closed, holds each of them.
class call_gate {
 public:
  void pass()
  {
    std::unique_lock lock(mutex_);
    ++entered_;
    changed_.notify_all();
    changed_.wait(lock, [&] { return open_; });
  }

  void set_open(bool open)
  {
    {
      const std::lock_guard lock(mutex_);
      open_ = open;
    }
    changed_.notify_all();
  }

  // Number of calls that have reached the gate since the process started.
  int entered() const
  {
    const std::lock_guard lock(mutex_);
    return entered_;
  }

  void wait_for_entered(int count, const char* message)
  {
    std::unique_lock lock(mutex_);
    require(changed_.wait_for(lock, patience, [&] { return entered_ >= count; }), message);
  }

 private:
  mutable std::mutex mutex_;
  std::condition_variable changed_;
  int entered_{0};
  bool open_{true};
};

call_gate& compile_gate()
{
  static call_gate instance;
  return instance;
}

// Every cubin load passes it, whether the cubin was just compiled or read from the disk cache.
call_gate& load_gate()
{
  static call_gate instance;
  return instance;
}

void open_gates()
{
  compile_gate().set_open(true);
  load_gate().set_open(true);
}

}  // namespace

extern "C" nvrtcResult __real_nvrtcCompileProgram(nvrtcProgram program,
                                                  int option_count,
                                                  const char* const* options);

extern "C" nvrtcResult __wrap_nvrtcCompileProgram(nvrtcProgram program,
                                                  int option_count,
                                                  const char* const* options)
{
  compile_gate().pass();
  return __real_nvrtcCompileProgram(program, option_count, options);
}

extern "C" CUresult __real_cuLibraryLoadData(CUlibrary* library,
                                             const void* code,
                                             CUjit_option* jit_options,
                                             void** jit_option_values,
                                             unsigned int jit_option_count,
                                             CUlibraryOption* library_options,
                                             void** library_option_values,
                                             unsigned int library_option_count);

extern "C" CUresult __wrap_cuLibraryLoadData(CUlibrary* library,
                                             const void* code,
                                             CUjit_option* jit_options,
                                             void** jit_option_values,
                                             unsigned int jit_option_count,
                                             CUlibraryOption* library_options,
                                             void** library_option_values,
                                             unsigned int library_option_count)
{
  load_gate().pass();
  return __real_cuLibraryLoadData(library,
                                  code,
                                  jit_options,
                                  jit_option_values,
                                  jit_option_count,
                                  library_options,
                                  library_option_values,
                                  library_option_count);
}

namespace {

using kernel_handle = std::shared_ptr<const jit::CompiledKernel>;

struct outcome {
  kernel_handle kernel;
  std::exception_ptr error;
};

// Requests one kernel of one source from several threads at once. Destruction opens the gates, so a
// failed expectation never leaves a requester waiting in one.
class concurrent_requests {
 public:
  concurrent_requests(int count,
                      std::string source,
                      const jit::CompileOptions& options,
                      std::string entry_symbol = entry)
    : source_(std::move(source)),
      entry_symbol_(std::move(entry_symbol)),
      outcomes_(static_cast<std::size_t>(count))
  {
    for (auto& result : outcomes_) {
      threads_.emplace_back([this, &result, options] {
        try {
          require(cudaSetDevice(0) == cudaSuccess, "requester cudaSetDevice failed");
          result.kernel =
            jit::KernelCache::instance().get_or_compile_plain(source_, entry_symbol_, options);
        } catch (...) {
          result.error = std::current_exception();
        }
      });
    }
  }

  ~concurrent_requests() { open_gates(); }

  const std::vector<outcome>& finish()
  {
    open_gates();
    for (auto& thread : threads_)
      thread.join();
    return outcomes_;
  }

 private:
  std::string source_;
  std::string entry_symbol_;
  std::vector<outcome> outcomes_;
  std::vector<std::jthread> threads_;
};

std::string kernel_definition(std::string_view name, std::string_view value)
{
  return "extern \"C\" __global__ void " + std::string(name) +
         "(int* out) { *out = " + std::string(value) + "; }\n";
}

// A source no other scenario requests, as long as each passes its own value.
std::string kernel_source(int value) { return kernel_definition(entry, std::to_string(value)); }

void wait_for_joined(std::uint64_t count, const char* message)
{
  const auto deadline = std::chrono::steady_clock::now() + patience;
  while (jit::KernelCache::instance().stats().joined < count) {
    require(std::chrono::steady_clock::now() < deadline, message);
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
}

// Starts `requesters` identical requests and returns once one of them holds the only call at the
// closed `held` gate and all the others wait for it.
std::unique_ptr<concurrent_requests> start_shared_work(call_gate& held,
                                                       const std::string& source,
                                                       const jit::CompileOptions& options)
{
  held.set_open(false);
  const auto calls  = held.entered();
  const auto joined = jit::KernelCache::instance().stats().joined;
  auto requests     = std::make_unique<concurrent_requests>(requesters, source, options);
  held.wait_for_entered(calls + 1, "no requester started the shared work");
  wait_for_joined(joined + requesters - 1, "requesters did not join the work under way");
  require(held.entered() == calls + 1, "identical requests did the work more than once");
  return requests;
}

std::unique_ptr<concurrent_requests> start_shared_compilation(const std::string& source,
                                                              const jit::CompileOptions& options)
{
  return start_shared_work(compile_gate(), source, options);
}

kernel_handle require_one_kernel(const std::vector<outcome>& outcomes)
{
  for (const auto& result : outcomes) {
    if (result.error) std::rethrow_exception(result.error);
    require(result.kernel != nullptr && result.kernel->kern() != nullptr,
            "requester got no kernel");
    require(result.kernel == outcomes.front().kernel, "requesters got different kernels");
  }
  return outcomes.front().kernel;
}

void identical_requests_compile_once(const jit::CompileOptions& options)
{
  auto& cache             = jit::KernelCache::instance();
  const auto source       = kernel_source(1);
  const auto compilations = compile_gate().entered();
  const auto compiled     = jit::compile_stats().compiles;
  const auto size         = cache.size();

  const auto requests = start_shared_compilation(source, options);
  const auto kernel   = require_one_kernel(requests->finish());

  require(compile_gate().entered() == compilations + 1,
          "a requester compiled after the shared result");
  require(jit::compile_stats().compiles == compiled + 1, "compile counter is not one");
  require(cache.size() == size + 1, "shared compilation did not add exactly one cache entry");
  require(cache.get_or_compile_plain(source, entry, options) == kernel,
          "shared kernel is not the cached kernel");
  require(compile_gate().entered() == compilations + 1, "cached kernel was recompiled");
}

void failure_reaches_every_requester_and_is_not_remembered(const jit::CompileOptions& options)
{
  auto& cache             = jit::KernelCache::instance();
  const auto source       = kernel_definition(entry, "undeclared_value");
  const auto compilations = compile_gate().entered();
  const auto size         = cache.size();

  const auto expect_compile_error = [](const auto& request) {
    try {
      request();
    } catch (const jit::CompileError& error) {
      require(error.log.find("undeclared_value") != std::string::npos,
              "compile error lacks the NVRTC log");
      return;
    }
    throw std::runtime_error("requester of a failing compilation did not get CompileError");
  };

  const auto requests = start_shared_compilation(source, options);
  for (const auto& result : requests->finish()) {
    expect_compile_error([&] {
      require(result.error != nullptr, "failing compilation returned a kernel");
      std::rethrow_exception(result.error);
    });
  }
  require(compile_gate().entered() == compilations + 1,
          "a failing request was compiled more than once");
  require(cache.size() == size, "failed compilation left a cache entry");

  expect_compile_error([&] { (void)cache.get_or_compile_plain(source, entry, options); });
  require(compile_gate().entered() == compilations + 2,
          "a failure was remembered instead of retried");
}

void different_requests_compile_concurrently(const jit::CompileOptions& options)
{
  compile_gate().set_open(false);
  const auto compilations = compile_gate().entered();
  concurrent_requests first(1, kernel_source(2), options);
  concurrent_requests second(1, kernel_source(3), options);
  compile_gate().wait_for_entered(compilations + 2,
                                  "different requests did not compile at the same time");
  const auto first_kernel  = require_one_kernel(first.finish());
  const auto second_kernel = require_one_kernel(second.finish());
  require(first_kernel != second_kernel, "different requests share a kernel");
}

void clear_keeps_a_compilation_under_way_out_of_the_cache(const jit::CompileOptions& options)
{
  auto& cache       = jit::KernelCache::instance();
  const auto source = kernel_source(4);

  const auto requests   = start_shared_compilation(source, options);
  const auto generation = cache.generation();
  cache.clear();
  require(cache.generation() == generation + 1, "clear() did not advance the generation");
  const auto detached = require_one_kernel(requests->finish());
  require(cache.size() == 0, "kernel compiled across clear() was added to the cache");

  // The next request starts over. The compilation above has published its cubin, so starting over
  // is a disk load here; either way it loads a module of its own.
  const auto loads = jit::compile_stats().loads;
  const auto fresh = cache.get_or_compile_plain(source, entry, options);
  require(fresh != detached, "request after clear() got the kernel compiled across it");
  require(jit::compile_stats().loads == loads + 1, "request after clear() did not start over");
  require(cache.size() == 1, "request after clear() was not cached");
}

void request_after_clear_does_not_join_an_earlier_compilation(const jit::CompileOptions& options)
{
  auto& cache             = jit::KernelCache::instance();
  const auto source       = kernel_source(5);
  const auto compilations = compile_gate().entered();

  const auto before = start_shared_compilation(source, options);
  cache.clear();
  const auto joined = cache.stats().joined;
  concurrent_requests after(1, source, options);
  compile_gate().wait_for_entered(compilations + 2,
                                  "request after clear() joined an earlier compilation");
  require(cache.stats().joined == joined, "request after clear() joined an earlier compilation");

  const auto detached = require_one_kernel(before->finish());
  const auto fresh    = require_one_kernel(after.finish());
  require(detached != fresh, "request after clear() got a kernel compiled before it");
}

void identical_requests_load_from_disk_once(const jit::CompileOptions& options)
{
  auto& cache       = jit::KernelCache::instance();
  const auto source = kernel_source(6);
  require(cache.get_or_compile_plain(source, entry, options) != nullptr, "seeding compile failed");
  cache.clear();

  const auto compilations = compile_gate().entered();
  const auto loads        = jit::compile_stats().loads;
  const auto disk_hits    = cache.stats().disk_hits;
  const auto requests     = start_shared_work(load_gate(), source, options);
  (void)require_one_kernel(requests->finish());

  require(compile_gate().entered() == compilations, "a kernel on disk was recompiled");
  require(cache.stats().disk_hits == disk_hits + 1, "identical requests read the disk repeatedly");
  require(jit::compile_stats().loads == loads + 1, "identical requests loaded more than one cubin");
}

void kernels_of_one_source_compile_together(const jit::CompileOptions& options)
{
  const auto source = kernel_source(7) + kernel_definition(other_entry, "8");
  compile_gate().set_open(false);
  const auto compilations = compile_gate().entered();
  const auto joined       = jit::KernelCache::instance().stats().joined;
  concurrent_requests first(requesters / 2, source, options, entry);
  concurrent_requests second(requesters / 2, source, options, other_entry);
  compile_gate().wait_for_entered(compilations + 1, "no requester started compiling");
  wait_for_joined(joined + requesters - 1, "requests for one source did not share a compilation");

  const auto first_kernel  = require_one_kernel(first.finish());
  const auto second_kernel = require_one_kernel(second.finish());
  require(compile_gate().entered() == compilations + 1,
          "kernels of one source compiled separately");
  require(first_kernel != second_kernel && first_kernel->kern() != second_kernel->kern(),
          "different entry symbols resolved to one kernel");
  require(&first_kernel->module() == &second_kernel->module(),
          "kernels of one source belong to different modules");
}

void missing_kernel_fails_every_request_of_a_cached_source(const jit::CompileOptions& options)
{
  auto& cache             = jit::KernelCache::instance();
  const auto source       = kernel_source(9);
  const auto compilations = compile_gate().entered();
  const auto size         = cache.size();

  const auto expect_no_such_kernel = [&] {
    try {
      (void)cache.get_or_compile_plain(source, other_entry, options);
    } catch (const jit::CompileError&) {
      throw;
    } catch (const std::runtime_error&) {
      return;
    }
    throw std::runtime_error("request for a kernel the source does not define returned one");
  };

  expect_no_such_kernel();
  require(compile_gate().entered() == compilations + 1, "source was not compiled once");
  require(cache.size() == size + 1, "source without the requested kernel was not cached");
  expect_no_such_kernel();
  require(compile_gate().entered() == compilations + 1,
          "source without the requested kernel was compiled again");
  require(cache.get_or_compile_plain(source, entry, options) != nullptr,
          "cached source lost the kernel it defines");
}

// Points the kernel disk cache, whose location is fixed at first use, at an empty directory.
class scratch_disk_cache {
 public:
  scratch_disk_cache()
  {
    auto pattern =
      (std::filesystem::temp_directory_path() / "simpatico-single-flight-XXXXXX").string();
    require(::mkdtemp(pattern.data()) != nullptr, "mkdtemp failed");
    path_ = pattern;
    require(::setenv("SIMPATICO_JIT_CACHE_DIR", path_.c_str(), 1) == 0, "setenv failed");
  }
  ~scratch_disk_cache()
  {
    std::error_code ignored;
    std::filesystem::remove_all(path_, ignored);
  }
  scratch_disk_cache(const scratch_disk_cache&)            = delete;
  scratch_disk_cache& operator=(const scratch_disk_cache&) = delete;

 private:
  std::filesystem::path path_;
};

}  // namespace

int main()
{
  try {
    require(cudaSetDevice(0) == cudaSuccess, "cudaSetDevice(0) failed");
    const scratch_disk_cache disk_cache;
    jit::CompileOptions options;
    options.arch_cc = jit::arch_cc_for_current_device();

    identical_requests_compile_once(options);
    failure_reaches_every_requester_and_is_not_remembered(options);
    different_requests_compile_concurrently(options);
    clear_keeps_a_compilation_under_way_out_of_the_cache(options);
    request_after_clear_does_not_join_an_earlier_compilation(options);
    identical_requests_load_from_disk_once(options);
    kernels_of_one_source_compile_together(options);
    missing_kernel_fails_every_request_of_a_cached_source(options);

    std::puts("test_jit_kernel_cache_single_flight: OK");
    return 0;
  } catch (const jit::CompileError& error) {
    std::fprintf(stderr, "FAIL: %s\n%s\n", error.what(), error.log.c_str());
  } catch (const std::exception& error) {
    std::fprintf(stderr, "FAIL: %s\n", error.what());
  }
  return 1;
}
