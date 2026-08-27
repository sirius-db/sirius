---
name: build-sirius-starrocks
description: >
  Build the Sirius-as-StarRocks-CN stack on a fresh GPU box — UCX, nixl, libsirius, StarRocks FE,
  and the Rust CN. Use when standing the stack up from scratch, recovering a broken toolchain
  (g++/ld/nvml shims, CUDA 13, no-root conda deps), applying the StarRocks proto patch, or
  relocating pixi/cargo/Maven onto local NVMe.
---

This is the **build** skill. Cluster up/down and TPC-H sweeps are `tpch-cn-sweep` / `tpch-bench`.
Read [`bench/rtxpro6000-2gpu/BUILD-SIRIUS-STARROCKS.md`](../../../bench/rtxpro6000-2gpu/BUILD-SIRIUS-STARROCKS.md)
fully before running commands — it is the source of truth (verified 2026-08-19).

## Dependency order

Five artifacts. (1), (3) and (4) are independent and should run concurrently:

| # | Artifact | Command |
|---|---|---|
| 1 | UCX → `$TOOLS_DIR/ucx-install` | autotools, CUDA 13 toolkit (not `/usr/local/cuda` if that is older) |
| 2 | nixl → `$TOOLS_DIR/nvda_nixl` | meson; **build dir on local disk**, not NFS (clock skew) |
| 3 | libsirius | `pixi run make` at repo root |
| 4 | StarRocks FE | `pixi run -e fe fe-build` after the proto patch |
| 5 | `sirius-starrocks-cn` | `pixi run cn-build` (needs 1–4) |

Patch the StarRocks submodule **before** FE or CN: `pixi run --manifest-path experimental/starrocks/pixi.toml apply-starrocks-patches`. The submodule is dirty by design (`.gitmodules` `ignore = dirty`). **Never `git add` the submodule after patching.**

Also `git submodule update --init --recursive` at the repo root — `duckdb/`, `substrait`, `vcpkg` are required for cmake.

## Traps

- **CUDA 13 / driver r580+.** Older driver: put `/usr/local/cuda/compat` first on `LD_LIBRARY_PATH`.
- **Three shims** for `cargo build -p sirius-starrocks-cn` (must live outside `/tmp`): `g++` and `ld` → `/usr/bin/...` (pixi conda compilers mix sysroots); `libnvidia-ml.so` → the driver `.so.1`. PATH prefix must go **inside** `pixi run bash -c`, or pixi prepends conda `g++` and wins.
- **No root:** take UCX/nixl build deps from conda (§3 of the runbook), do not `apt-get`.
- **Small `/`:** symlink `.pixi`, `build`, cargo/Maven caches onto NVMe **before** first configure.
- **aarch64 `uv`:** nixl's python-wheel step needs a native `uv`; an x86-64 `~/.local/bin/uv` dies with `Exec format error`.
- Engine `.so` needs `NEEDED libnvidia-ml.so.1`; the unversioned symlink is link-time only.

## After the build

Smoke test is runbook §9. Then `tpch-cn-sweep` (portable) or `tpch-bench` (4× GB200 ops). CN tunables: [`experimental/starrocks/docs/TUNABLES.md`](../../../experimental/starrocks/docs/TUNABLES.md).
