# Debugging Sirius

This document is a practical guide to debugging Sirius crashes and concurrency
bugs. It is aimed at developers who are relatively new to sanitizers and core
dumps. It covers:

- [When to reach for which tool](#choosing-a-tool)
- [Building and running with AddressSanitizer (ASan)](#addresssanitizer-asan)
- [Building and running with ThreadSanitizer (TSan)](#threadsanitizer-tsan)
- [The `tsan.supp` suppressions file](#the-tsansupp-suppressions-file)
- [Capturing and inspecting a core dump](#capturing-a-core-dump)

---

## Choosing a tool

Sirius is a heavily multi-threaded, GPU-accelerated engine, so the bugs you hit
tend to fall into a few buckets. Pick the tool that matches the symptom:

| Symptom | Best tool | Why |
|---------|-----------|-----|
| Segfault / `SIGSEGV`, use-after-free, heap/stack overflow, double free, memory leak | **ASan** | Reports the exact bad access with allocation + free stacks |
| Intermittent crash, corruption that "moves around" between runs, wrong results under load, a hang | **TSan** | Finds the underlying **data race** / lock-order bug that ASan can only see *after* it has already corrupted memory |
| A reproducible crash you want to inspect fully (all threads, all local variables, post-mortem) | **core dump + gdb** | Frozen snapshot of the whole process at the moment it died |
| A crash you can reproduce on demand and want to step through live | **run under gdb** | Stop at the fault, inspect, continue, set breakpoints |

Rules of thumb:

- **Start with ASan** for any segfault. It is fast to set up and usually points
  straight at the offending allocation.
- **Reach for TSan** when a bug is *non-deterministic* — it crashes in a
  different place each run, or only under concurrency. ASan tells you memory was
  corrupted; TSan tells you *which two threads* raced to corrupt it.
- **ASan and TSan cannot be combined** in one binary (the compiler rejects
  `-fsanitize=address,thread`). They are separate builds and separate runs.
- **Use a core dump / gdb** when you need the full machine state (e.g. "what were
  all 16 worker threads doing when it crashed?") rather than a single sanitizer
  report.

> **Heads-up:** GPU device-side errors (illegal address inside a CUDA kernel)
> are **not** caught by ASan/TSan — those only instrument host (CPU) code. For
> on-GPU faults use `compute-sanitizer ./your_binary ...` instead.

---

## AddressSanitizer (ASan)

ASan instruments host code to detect heap/stack buffer overflows,
use-after-free, use-after-return, double-free, and leaks.

### Build

```bash
make clang-asan -j12
```

This configures and builds the `clang-asan` preset (RelWithDebInfo + clang with
`-fsanitize=address`). Outputs land under `build/clang-asan/`; the unit-test
binary is:

```
build/clang-asan/extension/sirius/test/cpp/sirius_unittest
```

### Run

CUDA reserves huge virtual-address ranges that collide with ASan's shadow
memory, so a couple of options are **required** or ASan will fail at startup:

```bash
ASAN_OPTIONS="protect_shadow_gap=0:detect_leaks=0:halt_on_error=0:abort_on_error=1:external_symbolizer_path=/usr/bin/llvm-symbolizer-18" \
  ./build/clang-asan/extension/sirius/test/cpp/sirius_unittest "<catch2-test-filter>"
```

### What the options mean

| Option | Why |
|--------|-----|
| `protect_shadow_gap=0` | **Required with CUDA.** The driver maps VA ranges that overlap ASan's protected shadow gap; without this ASan aborts on startup. |
| `detect_leaks=0` | Silences the flood of "leaks" from cuDF/CUDA/RMM, which manage their own memory. Turn it back on only when hunting an actual leak. |
| `external_symbolizer_path=/usr/bin/llvm-symbolizer-18` | Produces **symbolized** stack traces. Without it you get raw `binary+0xADDR` lines (the pixi env has no symbolizer on `PATH`). |
| `halt_on_error=0` | Keep running after the first error so you can see whether there are earlier/related reports. |
| `abort_on_error=1` | Abort (rather than `_exit`) on the final error, which makes the failure loud. |

> If your report comes out as raw addresses anyway, you can symbolize them by
> hand: `addr2line -f -C -e build/clang-asan/.../sirius_unittest 0x2ff9e9e`.

---

## ThreadSanitizer (TSan)

TSan instruments host code to detect data races (two threads accessing the same
memory without synchronization, at least one writing) and lock-order
inversions. It is the right tool for the intermittent, "corruption moves around"
class of bug.

### Build

```bash
make clang-tsan -j12
```

Builds the `clang-tsan` preset (RelWithDebInfo + clang with
`-fsanitize=thread`). Test binary:

```
build/clang-tsan/extension/sirius/test/cpp/sirius_unittest
```

### Run

```bash
export TSAN_OPTIONS="external_symbolizer_path=/usr/bin/llvm-symbolizer-18:suppressions=$PWD/tsan.supp:ignore_noninstrumented_modules=1:halt_on_error=0:history_size=7:detect_deadlocks=0"

./build/clang-tsan/extension/sirius/test/cpp/sirius_unittest "<catch2-test-filter>"
```

### What the options mean

| Option | Why |
|--------|-----|
| `external_symbolizer_path=/usr/bin/llvm-symbolizer-18` | Symbolized stacks (same reason as ASan). |
| `suppressions=$PWD/tsan.supp` | Silence known false positives from uninstrumented libraries — see [below](#the-tsansupp-suppressions-file). |
| `ignore_noninstrumented_modules=1` | **The biggest noise reducer.** Drops races whose accesses are entirely inside uninstrumented libs (cuDF, CUDA, RMM), so only races touching Sirius's own code surface. |
| `halt_on_error=0` | Collect *all* races in one run instead of stopping at the first. |
| `history_size=7` | Max memory-access history (so TSan can show **both** stacks of a race — without enough history the second stack is truncated and the report is much less useful). |
| `detect_deadlocks=0` | TSan's deadlock detector has an internal limit of 128 simultaneously-held locks; Sirius's task storms can exceed that and trip a fatal `CHECK failed: sanitizer_deadlock_detector.h` abort. Disable it so race detection can continue. |

### A race may not fire every run

Unlike ASan (which catches a bad access whenever it happens), TSan only reports
a race if it actually observes the two conflicting accesses interleave during
that run. For an intermittent bug, **run it in a loop** until it trips:

```bash
for i in $(seq 1 20); do
  echo "=== run $i ==="
  ./build/clang-tsan/extension/sirius/test/cpp/sirius_unittest "<filter>" || break
done
```

### Reading a TSan report

A race report has three parts you care about:

1. **Write/Read of size N … by thread T1** — the first access + its stack.
2. **Previous write/read … by thread T2** — the conflicting access + its stack.
3. **Location is heap block … allocated by …** — *what* object is being raced on.

The fix is almost always: add a mutex/atomic around the shared field, or change
the lifecycle so the two accesses can't overlap.

---

## The `tsan.supp` suppressions file

`tsan.supp` lives at the **repository root**. It tells TSan which reports to
silence. Sirius statically links/loads libcudf, RMM, and the CUDA
runtime/driver, none of which are TSan-instrumented; without suppression they
generate false races that bury the real bug in Sirius code.

You opt in via `TSAN_OPTIONS=...:suppressions=$PWD/tsan.supp` (shown in the run
command above). It complements `ignore_noninstrumented_modules=1` — the option
handles most library noise, and the file mops up anything specific that slips
through.

### Pattern forms

```
race:<substring of a symbol or file>     # silence a data-race report
called_from_lib:<shared library name>    # silence anything called from that .so
deadlock:<substring>                     # silence a lock-order report
signal:<substring>                       # silence a signal-unsafe report
```

### Gotcha: `called_from_lib` names must be unambiguous

Each `called_from_lib` value is matched as a **substring** against every loaded
library, and TSan errors out if one entry matches more than one library. For
example, `libcuda` matches *both* `libcuda.so.1` and `libcudart.so.13`. Always
include the `.so` so the match is unique:

```
called_from_lib:libcuda.so     # matches libcuda.so.1 only
called_from_lib:libcudart.so   # matches libcudart.so.13 only
```

### Growing the file

When a new false positive appears from a library, add a line for it and re-run.
Keep suppressions as specific as possible so you don't accidentally hide a real
bug in Sirius code.

---

## Capturing a core dump

A **core dump** is a snapshot of the process's entire memory and register state
at the instant it crashed. Loaded into gdb, it lets you inspect every thread's
stack and every variable post-mortem — invaluable for a crash you can reproduce
but can't easily run live (e.g. it only happens on a long test, or on a remote
machine).

> ### ⚠️ Sirius installs a SIGSEGV handler that suppresses core dumps
>
> On extension load, Sirius installs a signal handler
> (`src/util/segfault_backtrace_handler.cpp`, called from
> `sirius_extension.cpp` `LoadInternal`). On a crash it prints a backtrace to
> **stderr** (and to `$SIRIUS_LOG_DIR/segfault_backtrace.txt` if
> `SIRIUS_LOG_DIR` is set), then calls **`_exit(1)`** — which terminates the
> process *without* producing a core file.
>
> So `ulimit -c unlimited` **alone will not give you a core for a Sirius
> segfault.** You have two ways around it:
>
> 1. **Disable the handler with an environment variable** (easiest, no rebuild):
>
>    ```bash
>    export DISABLE_SIRIUS_SIGNAL_HANDLER=true
>    ```
>
>    When this is set, Sirius skips installing its handler entirely, so the OS
>    default disposition is in effect and a core dump is written (subject to
>    `ulimit -c`). Accepted truthy values: `1`, `true`, `yes`, `on`
>    (case-insensitive). On startup Sirius logs a warning confirming the handler
>    is disabled.
> 2. **Run under gdb** — gdb intercepts the signal *before* the process handler
>    runs, so you can inspect the live crash without any change at all. See
>    [Running live under gdb](#running-live-under-gdb).
>
> For many cases the stderr backtrace is already enough — just build with
> symbols (below) so it is readable.

### 1. Build with symbols

A core (or a backtrace) is only useful if the binary has debug symbols. Use a
build that includes them:

```bash
make clang-relwithdebinfo -j12   # optimized + symbols (closest to release behavior)
# or
make clang-debug -j12            # unoptimized + full symbols (easiest to inspect)
```

Avoid plain `make` / `release` for crash investigation — those are stripped of
the line info you need.

### 2. Enable core dumps in your shell

```bash
ulimit -c unlimited                       # allow cores (per-shell)
export DISABLE_SIRIUS_SIGNAL_HANDLER=true # so Sirius doesn't _exit() before a core is written
```

`ulimit -c unlimited` is **per-shell** — set it in the same terminal you run the
program from. Check it with `ulimit -c` (should print `unlimited`).
`DISABLE_SIRIUS_SIGNAL_HANDLER` turns off Sirius's crash handler (see the
warning above) so the OS can actually produce the core.

#### Using `DISABLE_SIRIUS_SIGNAL_HANDLER`

This environment variable controls whether Sirius installs its crash backtrace
handler. By default the handler is installed and `_exit(1)`s on a crash (no
core); setting this variable skips installation so the OS default disposition
runs and a core is written.

```bash
# Enable core dumps (any of these truthy values works, case-insensitive):
export DISABLE_SIRIUS_SIGNAL_HANDLER=true   # or: 1, yes, on

# Turn it back off (restore the built-in backtrace handler):
unset DISABLE_SIRIUS_SIGNAL_HANDLER         # or: export DISABLE_SIRIUS_SIGNAL_HANDLER=0
```

Key points:

- **Set it before the process starts.** The variable is read once, when the
  Sirius extension is loaded (`LoadInternal`). Exporting it after the process is
  already running has no effect.
- **It applies to whatever process loads the extension** — the C++ unit-test
  binary (`sirius_unittest`), the DuckDB CLI, or a Python process that does
  `LOAD 'sirius.duckdb_extension'`. Export it in that process's environment.
- **Confirm it took effect:** on startup Sirius logs a warning —
  `Sirius crash backtrace handler DISABLED via DISABLE_SIRIUS_SIGNAL_HANDLER ...`.
  If you don't see it, the variable wasn't set in the right environment (or was
  set to a non-truthy value).
- **Only use it while debugging.** With the handler disabled you lose the
  automatic stderr / `segfault_backtrace.txt` backtrace; you rely on the core
  dump (or gdb) instead. Leave it unset for normal runs.

Examples:

```bash
# Unit test, one-off:
ulimit -c unlimited
DISABLE_SIRIUS_SIGNAL_HANDLER=true \
  ./build/clang-relwithdebinfo/extension/sirius/test/cpp/sirius_unittest "<filter>"

# Python (export in the same shell before launching python):
ulimit -c unlimited
export DISABLE_SIRIUS_SIGNAL_HANDLER=true
python my_repro.py
```

### 3. Find out where the core will go

The kernel decides the core's name and location via `core_pattern`:

```bash
cat /proc/sys/kernel/core_pattern
```

- If it prints something like `core` or `core.%p`, cores are written to the
  process's **current working directory**.
- If it **starts with a pipe** `|` (e.g. `|/usr/share/apport/apport ...` on
  Ubuntu, or `|/lib/systemd/systemd-coredump ...`), the core is handed to that
  program instead of landing in your directory.

**To get a plain core file in the current directory** (transient, until
reboot):

```bash
sudo sysctl -w kernel.core_pattern='core.%e.%p'
```

Format specifiers: `%e` = executable name, `%p` = PID, `%t` = timestamp,
`%h` = hostname. Example result: `core.sirius_unittest.31234`.

**If your system uses `systemd-coredump`** (the pipe case), you don't need to
change anything — just retrieve the core afterwards with `coredumpctl`:

```bash
coredumpctl list                 # find your crash
coredumpctl gdb  sirius_unittest # open the most recent core in gdb
coredumpctl dump sirius_unittest -o core.sirius   # or export it to a file
```

> **Disk space:** Sirius processes are large and map a lot of memory; cores can
> be **several GB** (5–7 GB is normal). Make sure you have room, and clean up
> old cores (`core.*`, or via `coredumpctl`) when done.

### 4. Load the core in gdb

```bash
gdb path/to/binary path/to/core
# e.g.
gdb build/clang-relwithdebinfo/extension/sirius/test/cpp/sirius_unittest core.sirius_unittest.31234
```

Then the essential commands:

```gdb
(gdb) bt                    # backtrace of the crashing thread
(gdb) thread apply all bt   # backtrace of EVERY thread — the most useful command for Sirius
(gdb) info threads          # list all threads and where they are
(gdb) thread 7              # switch to thread 7
(gdb) frame 3               # select stack frame 3 in the current thread
(gdb) print some_variable   # inspect a variable in the selected frame
(gdb) list                  # show source around the selected frame
```

For a multi-threaded engine, **`thread apply all bt` is usually the first thing
to run** — it shows what every worker, manager, and the main thread were doing,
which is exactly how you spot "a worker was still running task X while the main
thread tore down the query."

### Running live under gdb

Often easier than a core file, and it works **even with the Sirius signal
handler installed** because gdb sees the signal first:

```bash
gdb --args ./build/clang-relwithdebinfo/extension/sirius/test/cpp/sirius_unittest "<filter>"
(gdb) run
# ... reproduce the crash ...
# gdb stops at the faulting instruction:
(gdb) bt
(gdb) thread apply all bt
```

Useful extras:

```gdb
(gdb) set pagination off            # don't stop every screenful when dumping all threads
(gdb) handle SIGSEGV stop print     # ensure gdb stops on SIGSEGV (default, but explicit)
```

If you instead want the program's *own* handler to run (rare), use
`handle SIGSEGV nostop noprint pass`.

---

## See also

- `tsan.supp` (repo root) — the TSan suppressions file.
- `src/util/segfault_backtrace_handler.cpp` — the built-in crash backtrace handler.
- [Pipeline Execution](pipeline-execution.md) and [Multi-GPU Architecture](multi-gpu-architecture.md) — the threading model these bugs live in.
