# CN tunables

Environment variables the Sirius StarRocks CN reads. Most operators only need a
worked config (`configs/gb200-4gpu/engine-a.env`, or the launcher in
`bench/rtxpro6000-2gpu/`) and should not set these by hand.

## How they work

Transport knobs live in one registry, [`src/tunable.rs`](../src/tunable.rs).
They are resolved once at bring-up: a typo or out-of-range value **fails CN
startup** (it is never clamped or silently ignored). The CN then logs the
resolved set — that line is what the process actually got, not what the
launcher echoed.

Unset means the compiled default. Empty string is treated as unset.

Everything else below is read outside that registry and follows its own rules.

## Transport (validated registry)

| Knob | Role |
|---|---|
| `SIRIUS_CN_RPC_TIMEOUT_SECS` | How long a CN waits for a peer RPC (lease, metadata). Raise this before treating a large-SF timeout as a query bug — a busy peer can sit behind this bound. |
| `SIRIUS_CN_NIXL_XFER_TIMEOUT_SECS` | How long one nixl WRITE may take. Distinguishes a stuck fabric from a busy peer. |
| `SIRIUS_CN_NIXL_CANARY_BYTES` / `_FLOOR_GBPS` | First-contact bandwidth probe. A slow link is refused so a silent staged-copy fallback cannot look like a healthy transfer. `0` on the floor disables the check. |
| `SIRIUS_CN_NIXL_WARMUP_TIMEOUT_SECS` / `_EXPECT_PEERS` | Bring-up session warmup. The timeout is a budget, not a hard fail; expect-peers ends the loop early once that many peers are up. |

Related, but not in the registry: `SIRIUS_CN_NIXL_WARMUP` (off switch) and
`SIRIUS_CN_NIXL_WARMUP_PEERS` (explicit peer list).

## Exchange staging

`SIRIUS_EXCHANGE_STAGING_BYTES` sizes each CN's GPU staging arena. Unset means
**no arena**: the CN boots and serves local work, then every remote exchange
fails. There is no engine default — launchers pick a size per box.

## Engine-side

| Knob | Role |
|---|---|
| `SIRIUS_CN_USE_SIRIUS_DATASOURCE` | Scan backend. Default is the uring path; `false` selects kvikio/cudf. |
| `SIRIUS_CN_CPU_AFFINITY` | Pin engine thread pools to a cpulist, or `off` to leave them free. Unset discovers the GPU's socket from sysfs. |
| `SIRIUS_QUERY_WATCHDOG_SECS` | Kill a wedged statement so it does not poison the CN. `0` / unset is off. |

GPU and host memory carve-outs are CLI flags (`--gpu-memory-limit`,
`--host-memory-limit`), not env vars — they become the derived Sirius YAML.

## Debug

`SIRIUS_CN_DUMP_FRAGMENTS` writes received fragments and translated plans.
`SIRIUS_CN_TRANSLATE_ONLY` stops after translation.
