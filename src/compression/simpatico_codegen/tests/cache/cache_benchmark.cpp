// Diagnostic benchmark, deliberately excluded from pass/fail timing assertions.
#include "codegen/decode/jit/renderer.hpp"
#include "codegen/encode/jit/renderer.hpp"
#include "codegen/jit/kernel_cache.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <new>
#include <vector>

namespace {
thread_local bool measure_allocations    = false;
thread_local std::size_t allocations     = 0;
thread_local std::size_t allocated_bytes = 0;
}  // namespace

void* operator new(std::size_t bytes)
{
  if (measure_allocations) {
    ++allocations;
    allocated_bytes += bytes;
  }
  if (void* memory = std::malloc(bytes ? bytes : 1)) return memory;
  throw std::bad_alloc();
}
void operator delete(void* memory) noexcept { std::free(memory); }
void operator delete(void* memory, std::size_t) noexcept { std::free(memory); }
void* operator new[](std::size_t bytes) { return ::operator new(bytes); }
void operator delete[](void* memory) noexcept { ::operator delete(memory); }
void operator delete[](void* memory, std::size_t) noexcept { ::operator delete(memory); }

template <class Spec>
void benchmark(const char* name, const Spec& spec, codegen::jit::CompileOptions options)
{
  auto& cache           = codegen::jit::KernelCache::instance();
  const auto cold_start = std::chrono::steady_clock::now();
  auto* expected        = cache.get_or_compile_plain(spec.source, spec.entry_symbol, options);
  const auto cold_us =
    std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - cold_start)
      .count();
  constexpr int iterations = 5000;
  std::vector<double> times(iterations);
  allocations = allocated_bytes = 0;
  for (double& time : times) {
    const auto start    = std::chrono::steady_clock::now();
    measure_allocations = true;
    auto* actual        = cache.get_or_compile_plain(spec.source, spec.entry_symbol, options);
    measure_allocations = false;
    time =
      std::chrono::duration<double, std::nano>(std::chrono::steady_clock::now() - start).count();
    if (actual != expected) throw std::runtime_error("warm lookup returned a different kernel");
  }
  std::sort(times.begin(), times.end());
  std::printf(
    "%s source_bytes=%zu cold_us=%.0f p50_ns=%.0f p95_ns=%.0f allocations_per_hit=%.2f "
    "bytes_per_hit=%.2f\n",
    name,
    spec.source.size(),
    cold_us,
    times[iterations / 2],
    times[iterations * 95 / 100],
    double(allocations) / iterations,
    double(allocated_bytes) / iterations);
}

int main()
{
  try {
    codegen::jit::CompileOptions options{codegen::jit::arch_cc_for_current_device()};
    auto tree = codegen::jit::FusedTree::make(codegen::OpKind::Bitpack);
    benchmark("encode", codegen::encode::jit::render(*tree, "int32_t", 8), options);
    benchmark("decode", codegen::decode::jit::render(*tree, "int32_t", 8), options);
    return 0;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "%s\n", error.what());
    return 1;
  }
}
