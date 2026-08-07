> we already have all the capabilities on stream source in sirius, no additional operator is needed.

Agreed that no new capability is invented.  But this new operator  also convers cases for a streaming source as one stream with an expected *set* of senders, where fan-in is the general case. 

> you can send end_stream, wait_for_data and data batches to scan operators through split_connector?

Only one of the three:

| Capability | `split_connector` today | What a stream needs |
|---|---|---|
| `end_stream` | ✅ but `close()` is a single-producer bool | N-sender EOS, deduped by sender identity |
| `data batches` | ⚠️ private `std::deque` — invisible to the memory manager | batches can queue for unbounded time → must stay **spillable** in a registered repository |
| `wait_for_data` | ❌ doesn't exist — the only wait is the **blocking** `get_next_split()`, on the task-creator thread (`task_creator.cpp:377`) | non-blocking three-way hint + `on_data` re-nomination (a source that answers WAITING is dropped by the scheduler for good) |


Notes: Blocking is fine for a scan (see sirius_gpu_scan_operator and split_connector) — waits are short and locally bounded. A stream (see sirius_physical_streaming_sink + stream_lifecycle) is paced by a **remote sender**: a blocked pull pins engine machinery on data that may arrive minutes
later, or never.

 
Blocking is fine for a scan — waits are short and locally bounded. A stream is paced by a
**remote sender**: a blocked pull pins engine machinery on data that may arrive minutes
later, or never.

Also worth clarifying the topology — `STREAMING_SOURCE` does **not** replace scans. In the
demo query:

- **Fragment 1 (sender):** `GPU_SCAN(lineitem) → filter → project → STREAMING_SINK` —
  starts with the normal GPU scan; its `split_connector` is untouched by this stack.
- **Fragment 2 (receiver):** `STREAMING_SOURCE → aggregate → RESULT_COLLECTOR` — the source
  appears only at the exchange boundary, a leaf with **no table to scan**.

> do you think we need to add this or just a rewiring of split_connector would be adequate?

"Rewiring" concretely means adding to the connector: sender-identity EOS,
repository-backed storage, a non-blocking classify, and re-arm hooks — i.e.,
re-implementing `stream_lifecycle` inside `scan_manager`, while also rewriting the
synchronization under the working scan path (its wait-then-pop is atomic today; a storage
seam breaks that). We explored that design in detail before rejecting it: the result is a
class where the scan uses only the blocking half and the stream only the non-blocking
half — a union, not shared logic.

So: keep `split_connector` as the scan's tool, keep the source as thin glue over the
shared primitives. Both stay simple, and the scan path carries zero risk.

---

Thanks for the suggestions — they pushed me to dig into how the scan works internally, and
that was genuinely useful. Based on what I learned, I'm now working on a follow-up
improvement: a small `batch_stream` abstraction (one `stream_lifecycle` bound to one
repository) so the source, the sink, and the upcoming stream session share the same
lifecycle/repository pairing — including a proper producer-error plane — instead of each
hand-writing it. I'll share the plan when it's ready.
