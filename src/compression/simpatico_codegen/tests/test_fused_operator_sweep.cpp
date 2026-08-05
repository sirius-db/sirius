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
// Dtypes
// ------
// All four integer widths supported by the fused JIT path: int8_t, int16_t,
// int32_t, int64_t.  Each dtype runs an independent set of synthetic fixtures
// with values scaled to stay within the type's representable range.
//
// Parallelization
// ---------------
// NVRTC JIT-compiles one kernel per (shape, dtype) pair and that compilation
// serialises on an internal per-process lock — multiple host threads add lock
// contention without speedup.  The test uses the same posix_spawn multi-process
// approach as test_operator_sweep: the orchestrator (no SIMPATICO_FUSED_SWEEP_SHARD
// env var) spawns N workers, each processing 1/N of the (dtype, shape) work list
// in round-robin.  N defaults to hardware_concurrency; override with
// SIMPATICO_FUSED_SWEEP_WORKERS.

#include "codegen/jit/fused_tree.hpp"
#include "codegen/jit/kernel_cache.hpp"
#include "jit_decode.hpp"
#include "test_utils.hpp"

#include <cuda.h>

#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <memory>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

extern char** environ;

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
// Synthetic data — values scaled to fit within T's representable range
// so the same template works for all supported widths.
// ---------------------------------------------------------------------
template <typename T>
std::vector<T> synth_data(int64_t n)
{
  // kRange: the pattern amplitude, chosen so all generated values fit in T.
  // int8_t: [-128,127], int16_t: [-32768,32767], wider types: no constraint.
  constexpr int32_t kRange = sizeof(T) == 1 ? 50 : sizeof(T) == 2 ? 500 : 50000;
  std::vector<T> data(static_cast<size_t>(n));
  for (int64_t i = 0; i < n; ++i) {
    const int32_t cid = static_cast<int32_t>(i / cc::kChunkSize);
    const int32_t pos = static_cast<int32_t>(i % cc::kChunkSize);
    int32_t v;
    switch (cid % 4) {
      case 0: v = kRange / 50 + (pos & 1); break;              // near-constant (bits≈1)
      case 1: v = kRange / 5 + pos % (kRange / 5 + 1); break;  // small positive range
      case 2: v = -(kRange / 10) + pos % (kRange / 2); break;  // crosses zero
      case 3: v = (pos * 7) % kRange; break;                   // varied residuals
    }
    data[i] = static_cast<T>(v);
  }
  return data;
}

// RLE-friendly fixture: varied run lengths to exercise trivial (1 run),
// small (~4 runs), and worst (1024 runs) cases.
template <typename T>
std::vector<T> synth_rle_data(int64_t n)
{
  constexpr int32_t kMax = sizeof(T) == 1 ? 100 : sizeof(T) == 2 ? 1000 : 100000;
  std::vector<T> data(static_cast<size_t>(n));
  for (int64_t i = 0; i < n; ++i) {
    const int32_t cid = static_cast<int32_t>(i / cc::kChunkSize);
    const int32_t pos = static_cast<int32_t>(i % cc::kChunkSize);
    int32_t v;
    switch (cid) {
      case 0: v = 42 % kMax; break;
      case 1: v = (pos / 256) % kMax; break;
      case 2: v = (kMax / 50 + pos) % kMax; break;
      case 3: v = (pos & 1) ? 50 % kMax : 51 % kMax; break;
      default: {
        int32_t run_id = 0, cum = 0;
        const int32_t lens[5] = {17, 33, 80, 51, 44};
        while (cum + lens[run_id % 5] <= pos && run_id < 1000) {
          cum += lens[run_id % 5];
          ++run_id;
        }
        v = (7 + run_id * 3) % kMax;
        break;
      }
    }
    data[i] = static_cast<T>(v);
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

std::vector<Tree> compose_layer(const std::vector<Tree>& prev)
{
  std::vector<Tree> out;
  out.reserve(prev.size() * 5);
  for (const auto& c : prev) {
    out.push_back(jit::FusedTree::make(OpKind::Delta, {{"differences", c}}));
    out.push_back(jit::FusedTree::make(OpKind::For, {{"deltas", c}}));
    out.push_back(jit::FusedTree::make(OpKind::Zigzag, {{"zigzag", c}}));
    out.push_back(jit::FusedTree::make(
      OpKind::Rle, {{"runs", jit::FusedTree::make(OpKind::Bitpack)}, {"values", c}}));
    out.push_back(jit::FusedTree::make(
      OpKind::Rle, {{"runs", jit::FusedTree::make(OpKind::Raw)}, {"values", c}}));
  }
  return out;
}

std::vector<Tree> enumerate_shapes(int max_depth)
{
  std::vector<std::vector<Tree>> by_depth(max_depth + 1);
  by_depth[1].push_back(jit::FusedTree::make(OpKind::Bitpack));
  for (int d = 2; d <= max_depth; ++d)
    by_depth[d] = compose_layer(by_depth[d - 1]);
  std::vector<Tree> flat;
  for (int d = 1; d <= max_depth; ++d)
    for (auto& t : by_depth[d])
      flat.push_back(t);
  return flat;
}

std::vector<Tree> extra_shapes()
{
  return {
    jit::FusedTree::make(OpKind::Zigzag),
    jit::FusedTree::make(OpKind::Delta, {{"differences", jit::FusedTree::make(OpKind::Raw)}}),
    jit::FusedTree::make(OpKind::For, {{"deltas", jit::FusedTree::make(OpKind::Raw)}}),
    jit::FusedTree::make(
      OpKind::Rle,
      {{"runs", jit::FusedTree::make(OpKind::Raw)}, {"values", jit::FusedTree::make(OpKind::Raw)}}),
    jit::FusedTree::make(
      OpKind::Rle,
      {{"runs",
        jit::FusedTree::make(OpKind::Rle,
                             {{"runs", jit::FusedTree::make(OpKind::Raw)},
                              {"values", jit::FusedTree::make(OpKind::Bitpack)}})},
       {"values", jit::FusedTree::make(OpKind::Bitpack)}}),
  };
}

// ---------------------------------------------------------------------
// Per-shape runner — templated over element type.
// ---------------------------------------------------------------------
template <typename T>
int run_one_shape(const jit::FusedTree& tree,
                  const std::string& shape_tag,
                  const std::vector<T>& data,
                  int arch_cc,
                  const char* dtype_str)
{
  const int64_t n = static_cast<int64_t>(data.size());
  try {
    codegen_test::GpuEncoded encoded =
      codegen_test::gpu_encode_tree<T>(tree, dtype_str, data.data(), n, arch_cc);
    auto recovered =
      codegen_test::jit_decode_tree<T>(tree, dtype_str, n, encoded.buffers, encoded, arch_cc);
    if (recovered != data) {
      std::fprintf(stderr, "FAIL [%s] decode mismatch\n", shape_tag.c_str());
      return 1;
    }
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "FAIL [%s] %s\n", shape_tag.c_str(), e.what());
    return 1;
  }
}

// ---------------------------------------------------------------------
// Dtype registry — all element types the fused JIT path supports.
// cxx must match the renderer's lookup_dtype table (int8_t/int16_t/…).
// ---------------------------------------------------------------------
struct DtypeSpec {
  const char* name;
  const char* cxx;
};
constexpr std::array<DtypeSpec, 4> kDtypes = {
  {{"i8", "int8_t"}, {"i16", "int16_t"}, {"i32", "int32_t"}, {"i64", "int64_t"}}};

// Type-erased per-dtype runner (built at shard startup with pre-generated data).
// Captures the typed data vectors by shared_ptr so the std::function is copyable.
using ShapeRunner =
  std::function<int(const jit::FusedTree&, const std::string& shape_tag, bool use_rle)>;

template <typename T>
ShapeRunner make_dtype_runner(int64_t n, int arch, const char* cxx)
{
  auto gdata = std::make_shared<std::vector<T>>(synth_data<T>(n));
  auto rdata = std::make_shared<std::vector<T>>(synth_rle_data<T>(n));
  return [gdata, rdata, arch, cxx](
           const jit::FusedTree& t, const std::string& s_tag, bool use_rle) -> int {
    return run_one_shape<T>(t, s_tag, use_rle ? *rdata : *gdata, arch, cxx);
  };
}

// ---------------------------------------------------------------------
// Env helpers.
// ---------------------------------------------------------------------
int env_depth(int default_depth)
{
  if (const char* s = std::getenv("SIMPATICO_FUSED_SWEEP_DEPTH"); s != nullptr) {
    int v = std::atoi(s);
    if (v >= 1 && v <= 6) return v;
    std::fprintf(stderr,
                 "warn: SIMPATICO_FUSED_SWEEP_DEPTH='%s' out of [1,6]; using default %d\n",
                 s,
                 default_depth);
  }
  return default_depth;
}

unsigned env_workers(unsigned default_workers)
{
  if (const char* s = std::getenv("SIMPATICO_FUSED_SWEEP_WORKERS"); s != nullptr) {
    int v = std::atoi(s);
    if (v >= 1) return static_cast<unsigned>(v);
    std::fprintf(stderr,
                 "warn: SIMPATICO_FUSED_SWEEP_WORKERS='%s' must be >= 1; using default %u\n",
                 s,
                 default_workers);
  }
  return default_workers;
}

unsigned n_shards_for(std::size_t work_size)
{
  return std::min<unsigned>(env_workers(std::max(1u, std::thread::hardware_concurrency())),
                            static_cast<unsigned>(work_size));
}

// fd the child writes its result line to.
constexpr int kResultFd = 3;

// ---------------------------------------------------------------------
// Shard: process shard_idx % n_shards of all (dtype, shape) work items.
// Reports "<passed_i> <total_i> " for each dtype over kResultFd.
// ---------------------------------------------------------------------
int run_shard(unsigned shard_idx, unsigned n_shards)
{
  if (cudaSetDevice(0) != cudaSuccess) {
    std::fprintf(stderr, "test_fused_operator_sweep: cudaSetDevice failed\n");
    return 1;
  }
  if (std::getenv("SIMPATICO_JIT_CACHE_CLEAR")) codegen::jit::clear_jit_disk_cache();

  const int arch      = jit::arch_cc_for_current_device();
  const int max_depth = env_depth(/*default=*/4);
  const int64_t n     = 4321;

  auto shapes = enumerate_shapes(max_depth);
  for (auto& t : extra_shapes())
    shapes.push_back(t);
  const std::size_t n_shapes = shapes.size();
  const std::size_t total    = kDtypes.size() * n_shapes;

  // Build one type-erased runner per dtype (pre-generates data once per shard).
  std::array<ShapeRunner, 4> runners = {
    make_dtype_runner<int8_t>(n, arch, kDtypes[0].cxx),
    make_dtype_runner<int16_t>(n, arch, kDtypes[1].cxx),
    make_dtype_runner<int32_t>(n, arch, kDtypes[2].cxx),
    make_dtype_runner<int64_t>(n, arch, kDtypes[3].cxx),
  };

  std::array<int, 4> passed_per_dtype = {};
  std::array<int, 4> total_per_dtype  = {};

  for (std::size_t flat = shard_idx; flat < total; flat += n_shards) {
    const std::size_t di    = flat / n_shapes;
    const std::size_t si    = flat % n_shapes;
    const auto& t           = shapes[si];
    const bool use_rle      = contains_rle(*t);
    const std::string s_tag = std::string(kDtypes[di].name) + "/" + tag(*t);

    ++total_per_dtype[di];
    if (runners[di](*t, s_tag, use_rle) == 0) {
      std::printf("  %-60s OK\n", s_tag.c_str());
      ++passed_per_dtype[di];
    }
  }

  // Report results over the pipe.
  std::string report;
  for (std::size_t di = 0; di < kDtypes.size(); ++di)
    report +=
      std::to_string(passed_per_dtype[di]) + " " + std::to_string(total_per_dtype[di]) + " ";
  report += "\n";
  std::size_t written = 0;
  while (written < report.size()) {
    ssize_t rc = ::write(kResultFd, report.data() + written, report.size() - written);
    if (rc < 0) {
      std::fprintf(
        stderr, "test_fused_operator_sweep: write(kResultFd) failed: %s\n", std::strerror(errno));
      return 1;
    }
    written += static_cast<std::size_t>(rc);
  }

  for (std::size_t di = 0; di < kDtypes.size(); ++di)
    if (passed_per_dtype[di] != total_per_dtype[di]) return 1;
  return 0;
}

// ---------------------------------------------------------------------
// Orchestrator: never touches CUDA.  Spawns n_shards copies of this
// binary with SIMPATICO_FUSED_SWEEP_SHARD=i/n set, collects results.
// ---------------------------------------------------------------------
int run_orchestrator()
{
  auto shapes = enumerate_shapes(env_depth(/*default=*/4));
  for (auto& t : extra_shapes())
    shapes.push_back(t);
  const std::size_t total = kDtypes.size() * shapes.size();
  const unsigned n_shards = n_shards_for(total);

  if (std::getenv("SIMPATICO_JIT_CACHE_CLEAR")) codegen::jit::clear_jit_disk_cache();

  std::printf("test_fused_operator_sweep: max_depth=%d dtypes=%zu shapes=%zu total=%zu shards=%u\n",
              env_depth(/*default=*/4),
              kDtypes.size(),
              shapes.size(),
              total,
              n_shards);

  std::vector<pid_t> pids(n_shards);
  std::vector<int> read_fds(n_shards);

  for (unsigned i = 0; i < n_shards; ++i) {
    int fds[2];
    if (pipe(fds) != 0) {
      std::fprintf(stderr, "test_fused_operator_sweep: pipe() failed\n");
      return 1;
    }

    posix_spawn_file_actions_t fa;
    posix_spawn_file_actions_init(&fa);
    posix_spawn_file_actions_addclose(&fa, fds[0]);
    posix_spawn_file_actions_adddup2(&fa, fds[1], kResultFd);
    if (fds[1] != kResultFd) posix_spawn_file_actions_addclose(&fa, fds[1]);

    const std::string env_entry =
      "SIMPATICO_FUSED_SWEEP_SHARD=" + std::to_string(i) + "/" + std::to_string(n_shards);
    std::vector<std::string> env_storage;
    for (char** e = environ; *e != nullptr; ++e) {
      if (std::strncmp(*e, "SIMPATICO_FUSED_SWEEP_SHARD=", 28) != 0) env_storage.emplace_back(*e);
    }
    env_storage.push_back(env_entry);
    std::vector<char*> envp;
    envp.reserve(env_storage.size() + 1);
    for (auto& s : env_storage)
      envp.push_back(s.data());
    envp.push_back(nullptr);

    char exe[]         = "/proc/self/exe";
    char* argv_child[] = {exe, nullptr};

    pid_t pid = -1;
    int rc    = posix_spawn(&pid, exe, &fa, nullptr, argv_child, envp.data());
    posix_spawn_file_actions_destroy(&fa);
    close(fds[1]);

    if (rc != 0) {
      std::fprintf(
        stderr, "test_fused_operator_sweep: posix_spawn failed: %s\n", std::strerror(rc));
      close(fds[0]);
      return 1;
    }
    pids[i]     = pid;
    read_fds[i] = fds[0];
  }

  // Collect per-dtype passed/total from each shard.
  std::array<long long, 4> total_passed = {};
  std::array<long long, 4> total_total  = {};
  bool any_failed                       = false;

  for (unsigned i = 0; i < n_shards; ++i) {
    std::string buf;
    char chunk[4096];
    ssize_t n_read;
    while ((n_read = ::read(read_fds[i], chunk, sizeof(chunk))) > 0)
      buf.append(chunk, static_cast<std::size_t>(n_read));
    close(read_fds[i]);

    std::istringstream iss(buf);
    for (std::size_t di = 0; di < kDtypes.size(); ++di) {
      long long p = 0, t = 0;
      iss >> p >> t;
      total_passed[di] += p;
      total_total[di] += t;
    }

    int status = 0;
    waitpid(pids[i], &status, 0);
    if (!(WIFEXITED(status) && WEXITSTATUS(status) == 0)) {
      any_failed = true;
      std::fprintf(
        stderr, "test_fused_operator_sweep: shard %u/%u failed (see stderr above)\n", i, n_shards);
    }
  }

  long long grand_passed = 0, grand_total = 0;
  for (std::size_t di = 0; di < kDtypes.size(); ++di) {
    std::printf("  [%-4s] %lld/%lld passed\n", kDtypes[di].name, total_passed[di], total_total[di]);
    grand_passed += total_passed[di];
    grand_total += total_total[di];
  }
  std::printf("test_fused_operator_sweep: %lld/%lld passed\n", grand_passed, grand_total);
  return any_failed ? 1 : 0;
}

}  // namespace

int main()
{
  try {
    if (const char* shard_env = std::getenv("SIMPATICO_FUSED_SWEEP_SHARD")) {
      unsigned idx = 0, n = 1;
      if (std::sscanf(shard_env, "%u/%u", &idx, &n) != 2 || n == 0 || idx >= n) {
        std::fprintf(
          stderr, "test_fused_operator_sweep: bad SIMPATICO_FUSED_SWEEP_SHARD='%s'\n", shard_env);
        return 1;
      }
      return run_shard(idx, n);
    }
    return run_orchestrator();
  } catch (const std::exception& e) {
    std::fprintf(stderr, "test_fused_operator_sweep: FATAL: %s\n", e.what());
    return 1;
  }
}
