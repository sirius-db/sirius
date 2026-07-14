// SPDX-License-Identifier: Apache-2.0
//
// Operator sweep: generated compress -> decompress -> equality across every
// operator cascade the type system allows, over a set of typed input fixtures,
// up to a small depth bound.
//
// Why this exists
// ---------------
// test_fused_operator_sweep exhaustively covers the *fused* IR
// (Bitpack/Delta/Rle/Raw) on int32.  This test generalises that idea to EVERY
// operator in the catalog
// (fused + non-fused: dictionary, alp, alp_rd, ans, bitcomp, snappy, lz4,
// deflate, bitextract_*) and to EVERY input dtype class.
//
// The enumeration reuses the exact machinery the explorer uses
// (explore/operator_catalog.hpp):
//
//   * `all_compressor_names()` — the operator catalog.
//   * `try_operator(op, column, ...)` — apply one op; failure means the op is
//     not applicable to that column's dtype (dict on int, alp on int, …), so it
//     is simply skipped.  Applicability is therefore never hard-coded here — it
//     is derived from the operators themselves and cannot drift.
//   * `make_dsl_step(...)` — build the plan DSL line + the typed output paths.
//
// Type transitions fall out naturally: an op's `named_channels()` carry their
// own dtypes, so after `dictionary` (string -> int indices) or `alp`
// (float -> int) the sweep keeps chaining operators on the integer
// intermediate outputs, exactly as requested.
//
// Enumeration
// -----------
// DFS from each fixture's "input" channel.  At each channel try every operator;
// for each that succeeds, verify the accumulated cascade end-to-end, then (if
// the op is non-terminal and depth remains) recurse into each produced channel
// with its new dtype.  Terminal ops (snappy/lz4/deflate/ans/bitcomp) end a
// branch.  Only one channel is extended per chain (the others are stored as
// leaves); this keeps the sweep bounded while still exercising each
// (operator, dtype) transition.
//
// Depth defaults to 2 (fast enough for ctest); override with
// SIMPATICO_SWEEP_DEPTH (capped at 4).
//
// Parallelization
// ----------------
// The dominant cost is NVRTC JIT-compiling a kernel per distinct chain shape,
// and that compilation serializes on an internal, per-process lock — extra
// host threads in one process add lock-contention overhead instead of
// speedup (verified empirically: 16 threads was slower than 1). Separate OS
// processes each get their own NVRTC state and scale cleanly instead.
//
// With no SIMPATICO_SWEEP_SHARD env var, main() is the orchestrator: it
// posix_spawns N copies of this binary (re-executed via /proc/self/exe) with
// SIMPATICO_SWEEP_SHARD=<i>/<N> set, waits for them, and sums the
// per-fixture chains/passed counts each child reports over a pipe. Each
// child runs single-threaded over its 1-of-N slice of the (fixture, op)
// work list. N defaults to hardware_concurrency (compilation is the
// bottleneck and serializes per process); override with
// SIMPATICO_SWEEP_WORKERS. Failing-chain detail is printed by the child that
// hit it (stderr is inherited, not piped).

#include "api/simpatico_codegen.hpp"
#include "codegen/jit/kernel_cache.hpp"
#include "codegen/jit/nvrtc_compiler.hpp"
#include "explore/operator_catalog.hpp"
#include "test_utils.hpp"

#include <cudf/column/column_factories.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/mr/cuda_async_memory_resource.hpp>
#include <rmm/mr/per_device_resource.hpp>

#include <cuda_runtime.h>

#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>

#include <array>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <numeric>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

extern char** environ;

namespace {

using simpatico::all_compressor_names;
using simpatico::compress_with_plan;
using simpatico::decompress;
using simpatico::is_terminal_compressor;
using simpatico::make_dsl_step;
using simpatico::try_operator;

// ── RMM pool guard (mirrors bench/compress_with_plan_benchmark.cpp) ───────────
struct pool_mr_guard {
  rmm::mr::cuda_async_memory_resource mr{};
  rmm::device_async_resource_ref previous{rmm::mr::get_current_device_resource_ref()};
  bool installed = false;
  void install()
  {
    rmm::mr::set_current_device_resource_ref(mr);
    installed = true;
  }
  ~pool_mr_guard()
  {
    if (installed) rmm::mr::set_current_device_resource_ref(previous);
  }
};

// ── Fixtures ──────────────────────────────────────────────────────────────────

// A UINT8 "binary" column of pseudo-random-ish bytes.
std::unique_ptr<cudf::table> make_u8_table(int num_rows, int seed)
{
  std::vector<std::uint8_t> host(static_cast<std::size_t>(num_rows));
  for (int r = 0; r < num_rows; ++r)
    host[static_cast<std::size_t>(r)] = static_cast<std::uint8_t>((r * 31 + seed * 7) & 0xFF);
  auto col = cudf::make_numeric_column(
    cudf::data_type{cudf::type_id::UINT8}, num_rows, cudf::mask_state::UNALLOCATED);
  if (cudaMemcpy(col->mutable_view().head<std::uint8_t>(),
                 host.data(),
                 host.size(),
                 cudaMemcpyHostToDevice) != cudaSuccess)
    throw std::runtime_error("make_u8_table: cudaMemcpy failed");
  std::vector<std::unique_ptr<cudf::column>> cols;
  cols.push_back(std::move(col));
  return std::make_unique<cudf::table>(std::move(cols));
}

// Canonical fixture order, shared by the orchestrator (sizing/labeling only,
// no CUDA touched) and each shard (which actually builds them). MUST stay in
// lockstep with build_fixtures() below — the orchestrator sizes/labels work by
// index into this list, so a shorter list silently drops the trailing fixtures
// from the sweep.
constexpr std::array<char const*, 9> kFixtureNames = {
  "i32", "i64", "u32", "u64", "f32", "f64", "u8_binary", "date", "string"};

struct fixture {
  std::string name;
  std::unique_ptr<cudf::table> table;  // owns a numeric column (col 0)
  std::unique_ptr<cudf::column> str;   // owns the string column (if any)
  cudf::column_view view;
};

std::vector<fixture> build_fixtures(rmm::cuda_stream_view stream, int n)
{
  std::vector<fixture> fixtures;
  auto add_numeric = [&](std::string nm, std::unique_ptr<cudf::table> t) {
    fixture f;
    f.name  = std::move(nm);
    f.view  = t->view().column(0);
    f.table = std::move(t);
    fixtures.push_back(std::move(f));
  };
  add_numeric("i32", make_int32_table(1, n, 1));
  add_numeric("i64", make_int64_table(1, n, 2));
  add_numeric("u32", make_uint32_table(1, n, 7));
  add_numeric("u64", make_uint64_table(1, n, 8));
  add_numeric("f32", make_f32_table(1, n, 3));
  add_numeric("f64", make_f64_table(1, n, 4));
  add_numeric("u8_binary", make_u8_table(n, 5));
  add_numeric("date", make_chrono_table(cudf::type_id::TIMESTAMP_DAYS, n, 6));
  {
    fixture f;
    f.name = "string";
    f.str  = make_string_column(n, stream);
    f.view = f.str->view();
    fixtures.push_back(std::move(f));
  }
  return fixtures;
}

// ── Sweep state ────────────────────────────────────────────────────────────────

struct sweep_stats {
  long long chains = 0;
  long long passed = 0;
  std::vector<std::string> failures;
};

void verify_chain(cudf::column_view fixture_col,
                  std::string const& dsl,
                  std::string const& tag,
                  rmm::cuda_stream_view stream,
                  rmm::device_async_resource_ref mr,
                  sweep_stats& st)
{
  ++st.chains;
  try {
    cudf::table_view tv{{fixture_col}};
    auto ct  = compress_with_plan(tv, dsl, stream, mr);
    auto out = decompress(ct, stream, mr);
    stream.synchronize();
    bool ok = out && out->num_columns() == 1 &&
              columns_equal_any(fixture_col, out->view().column(0), stream);
    if (ok)
      ++st.passed;
    else
      st.failures.push_back(tag + "  [decode mismatch]\n    " + dsl);
  } catch (std::exception const& e) {
    st.failures.push_back(tag + "  [" + e.what() + "]\n    " + dsl);
  } catch (...) {
    st.failures.push_back(tag + "  [unknown exception]\n    " + dsl);
  }
}

void dfs(cudf::column_view fixture_col,
         std::string const& path,
         cudf::column_view cur,
         int depth,
         int max_depth,
         std::string const& dsl_prefix,
         std::string const& tag_prefix,
         rmm::cuda_stream_view stream,
         rmm::device_async_resource_ref mr,
         sweep_stats& st);

// Try one op at one node; on success, verify the chain and (if non-terminal
// and depth remains) recurse into each produced channel. This is the unit of
// work dispatched per (fixture, first-level op) by run_shard(), and is also
// what dfs() calls for every op at every node.
void try_op_and_recurse(std::string const& op,
                        cudf::column_view fixture_col,
                        std::string const& path,
                        cudf::column_view cur,
                        int depth,
                        int max_depth,
                        std::string const& dsl_prefix,
                        std::string const& tag_prefix,
                        rmm::cuda_stream_view stream,
                        rmm::device_async_resource_ref mr,
                        sweep_stats& st)
{
  auto trial = try_operator(op, cur, stream, mr);
  if (!trial.success) return;

  auto step       = make_dsl_step(path, op, trial.outputs);
  std::string dsl = dsl_prefix.empty() ? step.line : dsl_prefix + "\n" + step.line;
  std::string tag = tag_prefix.empty() ? (path + ":" + op) : tag_prefix + ">" + op;

  verify_chain(fixture_col, dsl, tag, stream, mr, st);

  if (is_terminal_compressor(op)) return;
  if (depth + 1 >= max_depth) return;

  // Recurse into each produced channel with its (possibly changed) dtype.
  for (std::size_t i = 0; i < trial.outputs.size(); ++i) {
    dfs(fixture_col,
        step.output_paths[i],
        trial.outputs[i].view,
        depth + 1,
        max_depth,
        dsl,
        tag,
        stream,
        mr,
        st);
  }
}

void dfs(cudf::column_view fixture_col,
         std::string const& path,
         cudf::column_view cur,
         int depth,
         int max_depth,
         std::string const& dsl_prefix,
         std::string const& tag_prefix,
         rmm::cuda_stream_view stream,
         rmm::device_async_resource_ref mr,
         sweep_stats& st)
{
  for (auto const& op : all_compressor_names()) {
    try_op_and_recurse(
      op, fixture_col, path, cur, depth, max_depth, dsl_prefix, tag_prefix, stream, mr, st);
  }
}

int env_depth(int default_depth)
{
  if (char const* s = std::getenv("SIMPATICO_SWEEP_DEPTH"); s != nullptr) {
    int v = std::atoi(s);
    if (v >= 1 && v <= 6) return v;
    std::fprintf(stderr,
                 "warn: SIMPATICO_SWEEP_DEPTH='%s' out of [1,6]; using default %d\n",
                 s,
                 default_depth);
  }
  return default_depth;
}

unsigned env_workers(unsigned default_workers)
{
  if (char const* s = std::getenv("SIMPATICO_SWEEP_WORKERS"); s != nullptr) {
    int v = std::atoi(s);
    if (v >= 1) return static_cast<unsigned>(v);
    std::fprintf(stderr,
                 "warn: SIMPATICO_SWEEP_WORKERS='%s' must be >= 1; using default %u\n",
                 s,
                 default_workers);
  }
  return default_workers;
}

// A unit of parallel work, split at the SECOND operator (one level deeper than
// (fixture, first-op)) so the heavy float + high-fan-out subtrees — e.g.
// (f32, alp), which alone dwarfs most units — spread across shards instead of
// pinning one shard on the tail.
//
//   op2 empty ("shallow"): apply op1 and verify the depth-1 chain [op1] only.
//   op2 set    ("deep")   : apply op1, then for each of op1's output channels
//                           apply op2 and recurse the full subtree beneath it.
//
// Union over a fixture's shallow unit + all its deep (op1, *) units == the
// old single (fixture, op1) subtree, so the set of verified chains (and thus
// the pass/fail counts) is unchanged — only the work distribution differs.
// Enumeration is pure host-side (catalog cross-product + is_terminal), so the
// orchestrator still sizes shards without touching the GPU.
struct work_item {
  std::size_t fixture_idx;
  std::string op1;
  std::string op2;  // empty => shallow (verify op1 only); else deep
};

std::vector<work_item> build_work()
{
  std::vector<work_item> work;
  for (std::size_t fi = 0; fi < kFixtureNames.size(); ++fi)
    for (auto const& op1 : all_compressor_names()) {
      work.push_back({fi, op1, ""});              // depth-1 chain [op1]
      if (is_terminal_compressor(op1)) continue;  // a terminal has no depth-2 subtree
      for (auto const& op2 : all_compressor_names())
        work.push_back({fi, op1, op2});  // [op1 -> channel -> op2 -> ...]
    }
  return work;
}

// Run one work unit. op1 is re-applied per unit (cheap, JIT-cached) to keep
// units self-contained across the shard processes.
void run_work_item(std::string const& op1,
                   std::string const& op2,
                   cudf::column_view fixture_col,
                   int max_depth,
                   rmm::cuda_stream_view stream,
                   rmm::device_async_resource_ref mr,
                   sweep_stats& st)
{
  auto trial1 = try_operator(op1, fixture_col, stream, mr);
  if (!trial1.success) return;
  auto step1 = make_dsl_step("input", op1, trial1.outputs);

  if (op2.empty()) {  // shallow: verify the depth-1 chain only
    verify_chain(fixture_col, step1.line, "input:" + op1, stream, mr, st);
    return;
  }

  // Deep: op1's own chain is owned by the shallow unit; here we only expand
  // op2 over each of op1's channels (mirrors the original depth-1 recursion,
  // one op2 per unit). Guards match try_op_and_recurse's own descent.
  if (is_terminal_compressor(op1)) return;
  if (1 >= max_depth) return;
  std::string const tag1 = "input:" + op1;
  for (std::size_t c = 0; c < trial1.outputs.size(); ++c) {
    try_op_and_recurse(op2,
                       fixture_col,
                       step1.output_paths[c],
                       trial1.outputs[c].view,
                       /*depth=*/1,
                       max_depth,
                       step1.line,
                       tag1,
                       stream,
                       mr,
                       st);
  }
}

unsigned n_shards_for(std::size_t work_size)
{
  // Cold runs are ~99% NVRTC compilation, which serializes per process, so
  // scale shards to the full CPU count. Overridable via SIMPATICO_SWEEP_WORKERS;
  // still capped by the number of work units.
  return std::min<unsigned>(env_workers(std::max(1u, std::thread::hardware_concurrency())),
                            static_cast<unsigned>(work_size));
}

// fd the child writes its "<chains> <passed> " *6 result line to; the
// orchestrator dup2s a fresh pipe onto this number in each child.
constexpr int kResultFd = 3;

// Single shard: build all fixtures, process every work_item where
// idx % n_shards == shard_idx, report per-fixture chains/passed over
// kResultFd. Failing chains are printed directly (stderr is inherited).
int run_shard(unsigned shard_idx, unsigned n_shards)
{
  if (cudaSetDevice(0) != cudaSuccess) {
    std::fprintf(stderr, "test_operator_sweep: cudaSetDevice failed\n");
    return 1;
  }
  pool_mr_guard mr_guard;
  mr_guard.install();

  auto stream         = cudf::get_default_stream();
  auto mr             = rmm::mr::get_current_device_resource_ref();
  int const max_depth = env_depth(/*default=*/3);
  int const n         = 4096;

  auto fixtures = build_fixtures(stream, n);
  auto work     = build_work();

  // Minimum applicability: the sweep treats any op failure as "inapplicable"
  // and skips silently, so a regression that breaks a whole (op, dtype) class
  // would otherwise just shrink coverage. Assert on shard 0 that the ops each
  // fixture class exists to exercise actually succeed on it.
  if (shard_idx == 0) {
    auto must_apply = [&](char const* fixture_name, std::initializer_list<char const*> ops) {
      for (auto const& f : fixtures) {
        if (f.name != fixture_name) continue;
        for (char const* op : ops) {
          auto trial = simpatico::try_operator(op, f.view, stream, mr);
          if (!trial.success) {
            throw std::runtime_error(std::string("minimum applicability: '") + op +
                                     "' failed on fixture '" + fixture_name +
                                     "': " + trial.error_message);
          }
        }
      }
    };
    must_apply("string", {"dictionary", "dictionary_fast", "str_split"});
    must_apply("u8_binary", {"delta", "rle", "for", "zigzag", "bitpack"});
    must_apply("date", {"delta", "rle", "for", "zigzag", "bitpack", "ans", "bitcomp"});
  }

  std::vector<sweep_stats> per_fixture(fixtures.size());
  for (std::size_t idx = shard_idx; idx < work.size(); idx += n_shards) {
    auto const& item = work[idx];
    auto const& f    = fixtures[item.fixture_idx];
    run_work_item(item.op1, item.op2, f.view, max_depth, stream, mr, per_fixture[item.fixture_idx]);
  }

  std::string report;
  for (auto const& s : per_fixture)
    report += std::to_string(s.chains) + " " + std::to_string(s.passed) + " ";
  report += "\n";
  std::size_t written = 0;
  while (written < report.size()) {
    ssize_t rc = ::write(kResultFd, report.data() + written, report.size() - written);
    if (rc < 0) {
      std::fprintf(
        stderr, "test_operator_sweep: write(kResultFd) failed: %s\n", std::strerror(errno));
      return 1;
    }
    written += static_cast<std::size_t>(rc);
  }

  bool any_failed = false;
  for (auto const& s : per_fixture) {
    if (s.failures.empty()) continue;
    any_failed = true;
    for (auto const& f : s.failures)
      std::fprintf(stderr, "  - %s\n", f.c_str());
  }
  return any_failed ? 1 : 0;
}

// Orchestrator: never touches CUDA. Spawns n_shards copies of this binary
// (re-executed via /proc/self/exe, inheriting the environment plus
// SIMPATICO_SWEEP_SHARD=<i>/<n_shards>), then collects each child's
// per-fixture counts over a pipe and its exit code.
int run_orchestrator()
{
  auto work               = build_work();
  unsigned const n_shards = n_shards_for(work.size());

  // Clear the persistent JIT cache before spawning shards (once, here in the
  // orchestrator -- doing it per-shard would race a shard that already wrote).
  // Forces a cold run for benchmarking the compile cost.
  if (std::getenv("SIMPATICO_JIT_CACHE_CLEAR")) codegen::jit::clear_jit_disk_cache();

  std::printf("test_operator_sweep: max_depth=%d fixtures=%zu ops=%zu shards=%u\n",
              env_depth(/*default=*/3),
              kFixtureNames.size(),
              all_compressor_names().size(),
              n_shards);

  std::vector<pid_t> pids(n_shards);
  std::vector<int> read_fds(n_shards);

  for (unsigned i = 0; i < n_shards; ++i) {
    int fds[2];
    if (pipe(fds) != 0) {
      std::fprintf(stderr, "test_operator_sweep: pipe() failed\n");
      return 1;
    }

    posix_spawn_file_actions_t fa;
    posix_spawn_file_actions_init(&fa);
    // Close the read end first: if it happens to already occupy kResultFd's
    // number, closing it *after* the dup2 below would silently undo the
    // dup2 and leave the child with no write fd at all.
    posix_spawn_file_actions_addclose(&fa, fds[0]);
    posix_spawn_file_actions_adddup2(&fa, fds[1], kResultFd);
    if (fds[1] != kResultFd) posix_spawn_file_actions_addclose(&fa, fds[1]);

    std::string const env_entry =
      "SIMPATICO_SWEEP_SHARD=" + std::to_string(i) + "/" + std::to_string(n_shards);
    std::vector<std::string> env_storage;
    for (char** e = environ; *e != nullptr; ++e) {
      if (std::strncmp(*e, "SIMPATICO_SWEEP_SHARD=", 22) != 0) env_storage.emplace_back(*e);
    }
    env_storage.push_back(env_entry);
    std::vector<char*> envp;
    envp.reserve(env_storage.size() + 1);
    for (auto& s : env_storage)
      envp.push_back(s.data());
    envp.push_back(nullptr);

    char exe[]         = "/proc/self/exe";
    char* argv_child[] = {exe, nullptr};

    pid_t pid    = -1;
    int const rc = posix_spawn(&pid, exe, &fa, nullptr, argv_child, envp.data());
    posix_spawn_file_actions_destroy(&fa);
    close(fds[1]);  // parent's copy; the child has its own via dup2

    if (rc != 0) {
      std::fprintf(stderr, "test_operator_sweep: posix_spawn failed: %s\n", std::strerror(rc));
      close(fds[0]);
      return 1;
    }
    pids[i]     = pid;
    read_fds[i] = fds[0];
  }

  std::vector<sweep_stats> total(kFixtureNames.size());
  bool any_failed = false;
  for (unsigned i = 0; i < n_shards; ++i) {
    std::string buf;
    char chunk[4096];
    ssize_t n_read;
    while ((n_read = ::read(read_fds[i], chunk, sizeof(chunk))) > 0)
      buf.append(chunk, static_cast<std::size_t>(n_read));
    close(read_fds[i]);

    std::istringstream iss(buf);
    for (auto& s : total) {
      long long c = 0, p = 0;
      iss >> c >> p;
      s.chains += c;
      s.passed += p;
    }

    int status = 0;
    waitpid(pids[i], &status, 0);
    if (!(WIFEXITED(status) && WEXITSTATUS(status) == 0)) {
      any_failed = true;
      std::fprintf(
        stderr, "test_operator_sweep: shard %u/%u failed (see output above)\n", i, n_shards);
    }
  }

  long long grand_chains = 0, grand_passed = 0;
  for (std::size_t fi = 0; fi < kFixtureNames.size(); ++fi) {
    std::printf(
      "  [%-9s] chains=%lld passed=%lld\n", kFixtureNames[fi], total[fi].chains, total[fi].passed);
    grand_chains += total[fi].chains;
    grand_passed += total[fi].passed;
  }
  std::printf("test_operator_sweep: %lld/%lld chains passed\n", grand_passed, grand_chains);

  return any_failed ? 1 : 0;
}

}  // namespace

int main()
{
  try {
    if (char const* shard_env = std::getenv("SIMPATICO_SWEEP_SHARD")) {
      unsigned idx = 0, n = 1;
      if (std::sscanf(shard_env, "%u/%u", &idx, &n) != 2 || n == 0 || idx >= n) {
        std::fprintf(stderr, "test_operator_sweep: bad SIMPATICO_SWEEP_SHARD='%s'\n", shard_env);
        return 1;
      }
      return run_shard(idx, n);
    }
    return run_orchestrator();
  } catch (std::exception const& e) {
    std::fprintf(stderr, "test_operator_sweep: FATAL: %s\n", e.what());
    return 1;
  }
}
