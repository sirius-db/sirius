# Configuring the DISK tier (spilling) on the MIG SF500 box

Written 2026-09-09 after `R5-MCN16-2mig` failed q05/q08/q09/q21 with 31 engine log lines reading:

```
[downgrade] [HOST:0] memory pressure but no viable downgrade target (host full, no disk configured);
backing off until memory is released. Consider configuring a disk memory space to enable spilling.
```

This document is the answer to "what do I configure to enable spilling". Code anchors are from the
`wp0/exchange-telemetry` tree (engine `2e0cbf51` + telemetry). Nothing here needs an engine change.

## 1. Why the message appears

`downgrade_executor::has_viable_downgrade_target()` (`src/downgrade/downgrade_executor.cpp`) decides
whether a downgrade request can possibly free anything:

1. If a DISK tier exists it returns **true immediately** — DISK is treated as an effectively unbounded
   sink and is a valid target for any source tier.
2. With no DISK tier, only a **GPU** source has a lower tier (HOST). A **HOST** source has no target at
   all, so it must back off rather than re-fire.
3. For a GPU source with no disk, it probes each HOST space with a real chunk-sized
   `make_reservation_or_null`. Host full means no viable target.

On this box the HOST pool is 40 GiB per CN and the optimized (HOST-first) exchange fills it, so both
conditions fail and the executor backs off. Adding a disk tier makes case 1 fire and the HOST executor
starts spilling HOST -> DISK.

## 2. What already works (no code needed)

- **Converters are built in** (`cucascade/src/cudf/representation_converter_builtins.cpp`):
  `host_data_representation -> disk_data_representation`, `disk -> host`, and `gpu_table -> disk`.
- **The IO backend is automatic.** The disk `memory_space` constructor used by the reservation manager
  supplies `cucascade::make_pipeline_io_backend()` itself (`cucascade/src/memory/memory_space.cpp`).
  You never configure a backend; the overload that requires one is not the path taken.
- **Spill files** are created per batch under the mount directory by `generate_disk_file_path`, with
  4 KiB-aligned column offsets and no on-disk header (metadata stays in memory).
- **No DISK downgrade executor is created**, and that is correct: `sirius_context.cpp` creates
  executors for GPU and HOST only. DISK is a target, never a source.

## 3. The configuration

```yaml
sirius:
  memory:
    disk:
      capacity_bytes: "600GiB"
      downgrade_root_dirs: "/opt/dlami/nvme/spill/cn0"
```

Rules, from `disk_mem_config` in `src/sirius_config.cpp`:

| Key | Required | Default | Notes |
|---|---|---|---|
| `downgrade_root_dirs` | **yes** | empty | **A single path**, despite the plural name. It is stored as one `mounting_path_` and is never split on a separator. |
| `capacity_bytes` | **yes, non-zero** | 1 TiB | Accounting limit for the tier, not a filesystem quota. |
| `disk_id` | no | 0 | Use distinct ids per CN if you want the tiers accounted separately. |

`setup_configurator()` **returns early and silently** when `downgrade_root_dirs` is empty or
`capacity_bytes == 0` — a partially specified block gives you no disk tier and no warning. An empty
mount path passed further down throws `"Mount path must be provided for disk memory space"`.

Give each CN its **own directory**. Two CNs pointed at one directory is not a correctness problem
(file names are generated unique) but it makes per-CN accounting and cleanup ambiguous.

## 4. The launch path, and why the stock launcher cannot do it

`memory.disk` is reachable **only** through a full `--sirius-config` YAML. The CN's clap definition
(`experimental/starrocks/src/main.rs`) makes `--sirius-config` mutually exclusive with
`--gpu-memory-limit`, `--gpu-memory-fraction` and `--host-memory-limit`. The stock `cluster8.sh` launch
passes those flags, so it structurally cannot configure a disk tier.

That gap is what the WP1 P0 launcher fills (commit `wp1(launcher): CN_SIRIUS_CONFIG_DIR ...`, cherry-picked
onto `wp0/exchange-telemetry` as `cd239575`):

- `experimental/starrocks/benchmarks/gen-cn-config.sh` writes one full YAML per CN, in the same shape the
  CN derives from the flags (topology, `memory.gpu.usage_limit_bytes` + `reservation_limit_fraction: 1.0`,
  `memory.host.capacity_bytes`, executor `cpu_affinity`, telemetry dir), plus the optional `memory.disk`
  block and optional pinned `operator_params`.
- `cluster8.sh` gained `CN_SIRIUS_CONFIG_DIR=<dir>`: CN *i* launches with `--sirius-config <dir>/cn<i>.yaml`
  instead of the carve-out flags.

Generate and launch:

```bash
DISK_ROOT=/opt/dlami/nvme/spill DISK_BYTES=600GiB \
NUM_CNS=2 GPU_MEM=36GiB HOST_MEM=40GiB OUT_DIR=/tmp/cn-cfg \
  bash experimental/starrocks/benchmarks/gen-cn-config.sh
```

`gen-cn-config.sh` creates `$DISK_ROOT/cn<i>` for you and refuses if only one of `DISK_ROOT`/`DISK_BYTES`
is set. Then pass `CN_SIRIUS_CONFIG_DIR=/tmp/cn-cfg` to the arm; `capture-arm.sh` forwards it.

Omit `BATCH_BYTES` to keep engine-default `operator_params` (`min(device/40, pool/40)`), which is what a
flag launch produces — that keeps a DISK arm one-variable against a flag-launched reference.

## 5. Sizing on this box

`/opt/dlami/nvme` is 1.7 TB with **1.5 TB free** (135 GB is the SF500 dataset). 600 GiB per CN for two
CNs fits with room to spare. The instance store is **wiped on stop/start**, which is fine for spill files
but means `$DISK_ROOT` must be recreated after a restart — `gen-cn-config.sh` does that on each run.

## 6. What to expect, and how to read the result

With the tier configured, the `no viable downgrade target` line should disappear and be replaced by
`[downgrade]` request lines whose `to_disk:` counters are non-zero. The instrumented build reports, per
request: `candidates / converted / skipped_subscribed / skipped_not_idle / lock_failed /
already_in_target / reserve_failed / no_target_space`, so a request that still frees nothing says why.

Read a DISK arm honestly: spilling is a **host-capacity substitute for the 375 GiB of RAM this box does
not have**, not a fix for the exchange working-set problem. It trades a hard failure for disk I/O. If
q05/q08/q09/q21 pass only with DISK on, the correct conclusion is "this box needs spilling to run them at
SF500", and the working-set problem is still WP1's and WP3's to solve.
