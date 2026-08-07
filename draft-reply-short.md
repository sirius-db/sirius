> we already have all the capabilities on stream source in sirius, no additional operator is needed.

Agreed — no new capability is invented: the operator is thin glue over the existing `shared_data_repository` ([#1276](https://github.com/sirius-db/sirius/issues/1276): the repository *is* the queue, spillable, no bounded channel). The only thing it adds is what a repository can't say — fan-in end-of-stream across an expected **set** of senders ([#836](https://github.com/sirius-db/sirius/issues/836)).

> you can send end_stream, wait_for_data and data batches to scan operators through split_connector?

For the special case of exactly one sender — probably, yes. But an exchange input is a **fan-in** in general: N remote senders into one stream, where end-of-stream means *every expected sender* has closed, deduped by identity — so a duplicate close from one sender can't end the stream early and silently drop the others' rows. `split_connector` models the opposite shape: one local producer, one `close()` bool, no notion of who is sending.

Also worth clarifying the topology — `STREAMING_SOURCE` does **not** replace scans. In the demo query, for example:

- **Fragment 1 (sender):** `GPU_SCAN(lineitem) → filter → project → STREAMING_SINK` —
  starts with the normal GPU scan; its `split_connector` is untouched by this stack.
- **Fragment 2 (receiver):** `STREAMING_SOURCE → aggregate → RESULT_COLLECTOR` — the source
  appears only at the exchange boundary, a leaf with **no table to scan**.

> do you think we need to add this or just a rewiring of split_connector would be adequate?

We need to add it. Rewiring means giving `split_connector` everything it lacks above — the sender-set EOS, plus a non-blocking "wait vs. over" answer with a re-arm hook, because a stream must never block on a remote sender (the connector's only wait blocks the task-creator thread, which is fine for a scan's short local waits). That is the streaming source rebuilt inside `scan_manager`, with the working scan path rewritten underneath it — adding the operator is the smaller and safer change, and the scan path stays untouched.

Thanks for pushing on this — it led to a real follow-up, now on the branch: `exec::batch_stream`, one primitive (repository + stream state + a producer-error plane) shared by the source, the sink (#1321), and the upcoming session, so none of them hand-writes the pairing.
