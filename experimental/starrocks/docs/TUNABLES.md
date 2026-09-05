# CN tunables

Environment variables in the Sirius StarRocks CN's validated transport
registry. Most operators should not need to set these by hand.

## How they work

Transport and dispatch knobs live in one registry, [`src/tunable.rs`](../src/tunable.rs).
They are resolved once at bring-up: a typo or out-of-range value **fails CN
startup** (it is never clamped or silently ignored). The CN then logs the
resolved set. That line is what the process actually got, not what the
launcher echoed.

Unset means the compiled default. Empty string is treated as unset.

Everything else below is read outside that registry and follows its own rules,
except where a row says otherwise (`SIRIUS_CN_FRAGMENT_FUSION` is a registry
knob listed under "Dispatch").

## Transport (validated registry)

| Knob | Role |
|---|---|
| `SIRIUS_CN_RPC_TIMEOUT_SECS` | How long a CN waits for a peer RPC (lease, metadata). Raise this before treating a large-SF timeout as a query bug, since a busy peer can sit behind this bound. |
| `SIRIUS_CN_NIXL_XFER_TIMEOUT_SECS` | How long one nixl WRITE may take. Distinguishes a stuck fabric from a busy peer. |
| `SIRIUS_CN_NIXL_CANARY_BYTES` / `_FLOOR_GBPS` | First-contact bandwidth probe. A slow link is refused so a silent staged-copy fallback cannot look like a healthy transfer. `0` on the floor disables the check. |
| `SIRIUS_CN_NIXL_WARMUP_TIMEOUT_SECS` / `_EXPECT_PEERS` | Bring-up session warmup. The timeout is a budget, not a hard fail; expect-peers ends the loop early once that many peers are up. |
| `SIRIUS_CN_NIXL_WARMUP` | Warmup kill switch, `on` by default. `off` returns to lazy sessions: the first cross-node query after bring-up pays first contact, and on a cold cluster that is the first-contact deadlock the warmup exists to prevent. |
| `SIRIUS_CN_NIXL_WARMUP_PEERS` | Explicit `host:port,host:port` warmup peer list, skipping FE discovery. A malformed entry fails startup instead of leaving that peer cold. |

## Exchange staging

`SIRIUS_EXCHANGE_STAGING_BYTES` sizes each CN's GPU staging arena. Unset means
**no arena**: the CN boots and serves local work, then every remote exchange
fails. There is no engine default — launchers pick a size per box.

The arena only holds frames in flight. An inbound frame is copied out of its
lease into ordinary pool memory the moment its `transmit_packed` arrives (the
engine's inbound store, on the RPC thread) and the lease is released on that
RPC; the receiver fragment later takes the batch by ticket. So the arena has to
cover the frames a peer can have in the air at once (one lease per batch per
sender drain), not a shuffle's whole inbound share, and the shuffle inputs of a
query count against `--gpu-memory-limit`, where the pool's admission and
downgrade already apply. A frame whose receiver never runs (a failed or
cancelled query) is dropped from the store when the CN releases the receiver's
inputs; the CN logs the store's outstanding count with the arena's at quiesce.

## Dispatch

| Knob | Role |
|---|---|
| `SIRIUS_CN_FRAGMENT_FUSION` | Which same-node senders are spliced into their receiver's plan instead of running and parking their rows. `leaf` (default): a leaf fragment (file scans only) whose `HASH_PARTITIONED` stream sink has exactly one destination, on this CN, into a plain exchange that expects one sender and does not feed an aggregation — the shuffle shape that parks a fact table whole at 1 CN. `leaf-any`: every single-destination local leaf whatever its partition type (broadcast dimension tables too; the engine then plans them from footer estimates instead of exact parked counts). `off`: every sender runs and parks, the pre-fusion path, without a rebuild. Validated at bring-up in the registry above (any other value fails CN startup) and logged as `fusion_mode=` in the `resolved CN transport tunables` line. |
| `SIRIUS_CN_ASYNC_SENDER_DISPATCH` | `1`, `true` or `on` queues sender-only fragments (no exchange input, no RESULT_SINK) on the dispatch worker so their `exec_plan_fragment` RPC returns before the scan runs. The FE deploys the first fragment instance of every node in one wave and waits for those RPCs, so an inline sender holds every other node's second-wave instance back for its whole scan (q06 at 4 CNs, SF1000: 1.22 s inline, 0.85 s queued). Off by default: a queued sender's failure only reaches result instances reserved on this node. Read outside the registry. |

A fused sender has no `fragment run started` line of its own. The CN logs
`fused sender fragment into its local receiver` per absorbed sender,
`fused deferred sender plans into receiver` (with `fused=`) per receiver that
absorbed some, and `fragment fusion skipped` (with `reason=`) per sender that
was offered and declined. Fusion is decided when the sender arrives, on the
inline, batch and queued (`SIRIUS_CN_ASYNC_SENDER_DISPATCH`) paths alike; a
fused leaf never reaches the dispatch queue.
