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
| `SIRIUS_CN_RPC_TIMEOUT_SECS` | Deadline for peer control RPCs and optimized credit/export retries. Distinguish this deadline from an engine watchdog or a WRITE timeout when classifying a failure. |
| `SIRIUS_CN_NIXL_XFER_TIMEOUT_SECS` | How long one nixl WRITE may take. Distinguishes a stuck fabric from a busy peer. |
| `SIRIUS_CN_NIXL_CANARY_BYTES` / `_FLOOR_GBPS` | First-contact bandwidth probe. A slow link is refused so a silent staged-copy fallback cannot look like a healthy transfer. `0` on the floor disables the check. |
| `SIRIUS_CN_NIXL_WARMUP_TIMEOUT_SECS` / `_EXPECT_PEERS` | Bring-up session warmup. The timeout is a budget, not a hard fail; expect-peers ends the loop early once that many peers are up. |
| `SIRIUS_CN_NIXL_WARMUP` | Proactive discovery switch, `on` by default. With `off`, the first query pays peer setup. The baseline lazy path can enter reciprocal first-contact waits; optimized mode uses the asynchronous coordinator for both proactive and query-driven setup. |
| `SIRIUS_CN_NIXL_WARMUP_PEERS` | Explicit `host:port,host:port` warmup peer list, skipping FE discovery. A malformed entry fails startup instead of leaving that peer cold. |
| `SIRIUS_CN_NIXL_TRANSFER_WINDOW` | Optimized mode: active frames per peer, default `1`, valid range `1..8`. Byte limits still apply independently. |
| `SIRIUS_CN_NIXL_TRANSFER_PEERS` | Optimized mode: maximum peer control workers, default `32`, valid range `1..128`. One stalled peer cannot occupy another peer's control worker. |

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
query count against `--gpu-memory-limit`. The baseline store retains these GPU
batches until the receiver is ready; allocating from the pool alone does not
make an unregistered exchange batch discoverable by downgrade. A frame whose receiver never runs (a failed or
cancelled query) is dropped from the store when the CN releases the receiver's
inputs: when the rendezvous refuses the frame, when the receiver is retired,
or in the engine's sweep after a failed run. A replayed frame (brpc
reconnect-retry) is recognised by sequence number before its lease is touched.

## Optimized exchange protocol (opt-in)

Set `SIRIUS_EXCHANGE_OPTIMIZED=1` on every CN before startup to enable the new
owned ingress, independent export, dispatch overlap, and fair transport paths.
Exactly `1` enables this mode. Deploy homogeneous CN binaries and negotiate
lease protocol version `1` and a nonzero process epoch before admitting a peer.
Changed wire fields are carried by the checked-in `nixl-exchange-proto.patch`.
An old peer cannot silently enter the optimized ownership protocol.

The receive ledger reserves the exact packed payload in the configured host
pool **before** granting an arena range. The receive copy uses this allocation;
it does not first expand the entire table in the GPU pool. Temporary host
pressure returns `retryable_unavailable` without allocating an arena range.
An unsupported payload size or unavailable evacuation tier is a concrete
capacity error. Pool allocation rounding remains charged to the pool.

The current fixed admission limits are:

| Resource | Bound |
|---|---|
| Receive arena bytes, all peers | Half the arena, rounded down to 256 bytes |
| Receive bytes per sender process epoch | Half the receive budget |
| One receive payload | One quarter of the arena, rounded down to 256 bytes |
| Receive jobs | 64 per CN and 32 per sender process epoch |
| Source staging | Half the arena, including packing slack and alignment |
| Transport streams | 128 per CN and 64 per query |
| Receive lease replay records | 262,144 identities per CN session |

The source half and receive half are accounting partitions of the same arena,
not additional GPU allocations. The host evacuation reserve is charged to the
existing host pool. Increasing a transfer handle window never authorizes more
bytes than these budgets. Oversized batches fail explicitly; splitting them
is a separate implementation path.

Each grant has a receiver epoch, allocation token, and immutable sender/query/
receiver/exchange/sequence identity. A reconnect repeats that identity and
allocates once. Publication acknowledges completion only after ingress has
finished all reads of the arena. A replay during copying returns
`retryable_pending`; a completed replay returns the original result without
touching a potentially reused offset. Data and EOS may complete out of order,
but the receiver is dispatched only after every admitted copy completes, and
batch order follows sequence order. The EOS execution barrier remains.

Abort operation `1` promises that no WRITE can still access the destination.
It can use the stable identity without a token when every grant reply was lost;
an abort that arrives before its grant leaves a tombstone. Operation `2`, FE
cancellation, and unresolved transfer failure quarantine bytes instead of
reusing them on a timer. In-progress ingress retains ownership until its GPU
reads finish. Completed cancellation permits the next query; unresolved
quarantine may require restarting the CN to recover all capacity.

Replay records are retained for the CN session. Hitting the explicit record
bound rejects new admission and requires session retirement/restart; automatic
session garbage collection is not implemented. Rollback requires draining or
stopping the current CNs before starting the old mode. Do not change the mode
of an active allocation.

A peer restart at the same endpoint changes its process epoch. The cached
session fails closed on that mismatch; automatic re-handshake and retirement
of old NIXL registrations are not implemented. Restart the participating CNs
before resuming traffic. This is a recovery limit, not evidence that path 02's
in-place peer-restart acceptance scenario has passed.

Diagnostics include `owned receive credit granted`, `owned receive credit
returned after ingress completion`, `owned ingress publication completed`, and
quarantine warnings with token and byte counts. The first two expose live/peak
arena bytes; ingress completion exposes elapsed microseconds. Use these with
engine pool occupancy and spill/reload measurements, rather than treating
successful RPCs as proof of bounded total memory.

These per-frame lines use DEBUG logging. At INFO, `owned ingress query retirement
accounting` reports completed frames/payload bytes, requested bytes, live/peak
arena bytes, and remaining copying/quarantined bytes for the retiring query.
Completed payload bytes are separate from requested bytes so aborted grants
cannot inflate a throughput measurement.

The initial tests are in `exchange_protocol.rs`, `local_exchange.rs`, and
`compute_node_service.rs`: grant replay/conflicts, abort-before-grant, epoch
mismatch, evacuation pressure, byte/job bounds, copy-pending duplicate
publication, EOS before copy completion, and terminal replay after dispatch.
Implementation and benchmark evidence are tracked separately in `STATUS.md`;
these interface descriptions do not claim that the full performance-plan
acceptance matrix has passed.

## Dispatch

| Knob | Role |
|---|---|
| `SIRIUS_CN_FRAGMENT_FUSION` | Which same-node senders are spliced into their receiver's plan instead of running and parking their rows. `leaf` (default): a leaf fragment (file scans only) whose `HASH_PARTITIONED` stream sink has exactly one destination, on this CN, into a plain exchange that expects one sender and does not feed an aggregation — the shuffle shape that parks a fact table whole at 1 CN. `leaf-any`: every single-destination local leaf whatever its partition type (broadcast dimension tables too; the engine then plans them from footer estimates instead of exact parked counts). `off`: every sender runs and parks, the pre-fusion path, without a rebuild. Validated at bring-up in the registry above (any other value fails CN startup) and logged as `fusion_mode=` in the `resolved CN transport tunables` line. |
| `SIRIUS_CN_ASYNC_SENDER_DISPATCH` | `1`, `true` or `on` queues sender-only fragments (no exchange input, no RESULT_SINK) on the dispatch worker so their `exec_plan_fragment` RPC returns before the scan runs. The FE deploys the first fragment instance of every node in one wave and waits for those RPCs, so an inline sender holds every other node's second-wave instance back for its whole scan (q06 at 4 CNs, SF1000: 1.22 s inline, 0.85 s queued). Off by default, independently of `SIRIUS_EXCHANGE_OPTIMIZED`. Production CNs report queued execution and drain failures to the FE through `FrontendService.reportExecStatus`, including when the result fragment is on another CN. Test/library embeddings without a `FragmentFailureReporter` have only the local result-store error path. Read outside the registry. |

A fused sender has no `fragment run started` line of its own. The CN logs
`fused sender fragment into its local receiver` per absorbed sender,
`fused deferred sender plans into receiver` (with `fused=`) per receiver that
absorbed some, and `fragment fusion skipped` (with `reason=`) per sender that
was offered and declined. Fusion is decided when the sender arrives, on the
inline, batch and queued (`SIRIUS_CN_ASYNC_SENDER_DISPATCH`) paths alike; a
fused leaf never reaches the dispatch queue.
