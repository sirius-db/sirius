# REST redirect policy

This document describes the behavior-preserving extraction of the Sirius REST
implementation referenced by commit `4422f3eb`. It covers the existing libcurl
adapter and its four policy families; it does not enable delegation by default
or implement a cuCascade transport port.

Configuration lives under `scan_manager.rest`. See the
[configuration reference](../super-sirius/configuration.md#scan_managerrest--rest--s3-backend-iorestconfighpp).
The implementation is in [redirect_policy.hpp](../../src/io/rest/redirect_policy.hpp),
[redirect_policy.cpp](../../src/io/rest/redirect_policy.cpp) and
[rest_reactor.cpp](../../src/io/rest/rest_reactor.cpp).

## Defaults and scope

| Setting | Default | Meaning |
|---------|---------|---------|
| `cluster_redirect.enabled` | `false` | Object HEAD, suffix GET and ranged GET may delegate only when enabled |
| `cluster_redirect.origin` | empty | Explicit gateway `host[:port]`; required when enabled |
| `cluster_redirect.allowed_targets` | empty list | Explicit target authorities; at least one when enabled |
| `cluster_redirect.retention` | `expanded` | Requested connection-cache mode; disabled delegation still uses L=C |
| `cluster_redirect.route_reuse` | `off` | `same_open` requires qualified backend replay semantics |
| `request_log` | `false` | Opt-in structural transfer and redirect-decision records |

Origin/target matching compares authority spelling case-insensitively, with
literal ports: it does not equate an omitted default port with an explicit one.
TLS verification and CA policy still come from `scan_manager.object_store`.
These settings belong to the live reactor context; changing a configuration file
does not migrate grants or alter the identity of an in-flight operation.

## Policy and adapter responsibilities

| Family | Extracted responsibility | Adapter responsibility retained |
|--------|--------------------------|---------------------------------|
| `redirect_admission` | Pure delegation decision and refusal reason | Actual response source/provenance, decision counters, headers and wire submission |
| `redirect_routes` | Per-open snapshot/owner state, conditional invalidation, request-state selection gate | Strong-validator check, selection placement, validation before seed, recovery classification and scheduling |
| `redirect_retention_policy` | Checked connection-retention arithmetic and requested-mode name | Applying curl options; per-host C remains unchanged |
| `redirect_records` | Borrowed transfer-record representation and fixed field order | Clock, URL sanitization, curl-result normalization, connect counters and emission |

The adapter also owns request/token/staging lifetimes, body/range/ETag checks,
retry budgets, authorization, cancellation, CUDA copy completion and drain.
Policy helpers do not submit requests, own transport resources or replace the
integration tests.

## Admission: one delegated hop

The decision uses the URL of the request that actually received the response,
not merely the object's logical S3 URL. Delegation requires all of:

1. Enabled policy, HTTP 307, and no prior delegated URL for this attempt.
2. A request sent to the configured origin authority.
3. A nonempty absolute HTTP(S) Location naming an allowed target authority.
4. No HTTPS-to-HTTP downgrade, userinfo, fragment, whitespace, control byte or DEL.

The adapter uses its own received Location verbatim, preserving path and query,
including empty segments, escaped bytes and legitimate gateway rewrites. It
repeats method, Range and If-Match but does not invoke the provider authorizer
or forward its credential headers to the target. The Location itself may carry
the gateway's grant; admission does not establish cross-range replay permission.

Fresh-delegated 307 and target 401/403 are terminal, including truncated error
bodies. Other 3xx responses remain terminal outside the 307 decision counters.
LIST retains its existing error/retry path; warm-up remains best effort. Neither
enters delegation or its counters. Cached-route refusal is the separately scoped
recovery transition below, not a second target-to-target hop.

## Route state and event boundaries

Each open owns one `route_cache`. The immutable string's shared-pointer identity
is its publication identity: equal URL bytes in separate allocations are
separate publications. An attempt retains the exact snapshot it selected.
Conditional invalidation clears only that publication if still current; a later
publication survives an older attempt's failure.

The route is visible only to the same live context owner. A foreign, expired or
ownerless context cannot read it. An invisible occupied slot is not automatically
reclaimed or overwritten. Separate opens never share a grant, including opens
with identical object names or ETags. The route does not replace the open's
expected object identity.

### Validated publication

The adapter can optionally store a nonempty admitted GET Location of at most
2048 bytes after its existing successful network/probe validation, with reuse
enabled and a strong validator on the REST open. Longer Locations may still be
used for one delegation but are not retained. An occupied slot is not replaced.
Optional seed allocation/locking or seed-diagnostic failure cannot turn a
validated read into a failure.

Fresh HEAD and suffix discovery always start at the origin. HEAD never publishes
a GET grant. A successfully validated delegated suffix GET can seed the newly
opened object, as can a validated fresh-delegated worker GET. This is the existing
behavior; suffix GET must not be described as categorically forbidden to seed.

### First physical submission

Selection occurs after the cancellation check and after the slot owns the
request/token, inside the existing setup/completion error boundary, before
transport setup. It requires enabled + `same_open`, a REST open with a strong
validator, no delegated URL, no previous refresh, and both transport and auth
attempt counters equal to zero. The chosen context-visible snapshot is retained
by the request and the route-hit counter advances at selection.

A delegated, refreshed or retried request cannot reconsider the cache merely
because another request header is being built. Selection counters are not proof
of wire submission: setup can still fail. The adapter's original cleanup and
exactly-once completion paths remain responsible for that case.

## Cached-route recovery and error priority

| Outcome on a cached attempt | Hint and retry behavior |
|----------------------------|-------------------------|
| `CURLE_OUT_OF_MEMORY` or `CURLE_ABORTED_BY_CALLBACK`, even with HTTP status | Not classified as cached-route refusal; preserve local error handling |
| HTTP 307, 401, 403, 404 or 503, subject to the protected local outcomes above | Conditionally invalidate the selected snapshot; at most one origin re-resolution if budget permits |
| No-response `CURLE_COULDNT_CONNECT` or `CURLE_COULDNT_RESOLVE_HOST` | Same conditional invalidation/re-resolution path |
| HTTP 412 on a conditional read | Terminal `object_changed_error`; no clearing, fallback or identity rebinding |
| ETag/Content-Range failure, cancellation or other transport error | Preserve the adapter's existing terminal or same-route retry behavior; do not expand the refusal set |

Invalidation precedes the budget check: an exhausted request may invalidate its
own hint without earning another request. Refresh increments the transport
attempt counter, marks the request refreshed and returns to the origin. The
origin and its fresh delegated hop share that attempt index; delegation and
refresh never reset either counter.

For a cache-started operation, transport and authorization retries share the
existing total attempt ceiling, while auth retries also retain their separate
ceiling. `max_retry_attempts=10` and `max_auth_retry_attempts=3` remain the defaults.
The expected identity stays with the operation and it completes once. Recovery
classification and error-precedence execution remain in the adapter.

## Connection retention

Let C be `max_connections`, T the number of configured `allowed_targets` entries
and L the per-worker connection-cache allowance:

```text
L = C                         if disabled, legacy, or C >= 4096
L = min(4096, C * (1 + T))     otherwise (enabled/expanded and 0 < C < 4096)
```

The helper checks the cap before addition/multiplication, avoiding overflow. Its
synthetic C=0 result is zero; the reactor still rejects C=0. The per-host curl
limit stays C. The requested mode is reported separately from effective L:
requesting expanded does not override the enabled guard or the C>=4096 branch.

W workers have aggregate configured allowance W×L. This is neither a measured
socket count nor a per-authority reservation, and is not a bound on active,
warm-up or synchronous connections. Eviction may still happen.

## Transfer records

With `request_log=true`, executed HEAD, suffix GET, ranged GET and LIST attempts
emit the existing structural record in this order:

```text
rest_reactor: transfer method={} authority={} range={} status={} curl={} bytes={} attempt={} delegated={} elapsed_ms={} route={} connects={}
```

The record borrows views for synchronous emission; it owns no transport resource.
`attempt` is the sum of earlier transport and auth attempts; a delegated hop
keeps that index. `route` is `origin`, `fresh` or `cached`. `curl` is `ok`, the
adapter-normalized curl error text, or `aborted-local` during active drain.

Authority sanitization admits only a valid hostname, IPv4 or bracketed IPv6
literal and optional port 1–65535, strips userinfo and uses `-` for malformed
input. An absent Range is `-`; Range otherwise comes from numeric request
structure. No raw path/query, object name, arbitrary headers, provider credential
or grant is added to these measurement records.

`connects` is the current completed transfer's successful new-connection count.
Zero alone is not proof of reuse. `-` means unknown, including a no-response
error, an unavailable counter or active drain; response-bearing failures can
still have known counts. Drain status is curl's current HTTP-code snapshot
(zero if absent), not a claim of a completed response.

The active-attempt guard prevents duplicate terminal records during completion
and drain. Drain records submitted attempts with received status/bytes and
`aborted-local`, with best-effort diagnostics preserving cleanup. Work never
submitted emits no transfer record. Connection warm-up is excluded. A transfer
may separately cause a redirect-decision record. The enabled-policy construction
record reports requested retention mode, effective maxconnects and per-host C.
When logging is off these measurement records are absent; this does not suppress
unrelated existing diagnostics.

## AIStore topology and qualification

The intended root-mode path is client → AIStore proxy → HTTP 307 → owning target.
The target may cold-fill from its backing S3 store. The client follows the
received Location, including `/s3/` and gateway query fields such as `pid`/`utm`;
these fields alone are not authorization for later byte ranges.

Before real `same_open` use, qualify the exact deployed backend, build,
configuration, object/capture and replay domain using the
[Step R controller](../../test/io/redirect-policy/README.md#local-step-r-domain-and-reproduction).
The local controller exercises cross-range and suffix-to-bounded GET, including
an actual replay after at least 600 monotonic seconds. Its permission is scoped
to a live controller/deployment session; an archived PASS is not reusable after
restart or a bound input changes. This is a deployment prerequisite enforced by
the test workflow, not a new production runtime configuration guard. Synthetic
fixtures model their own allowed/refused domains and cannot qualify a deployment.

The supplied maintainer notes describe AIStore 5.1 `a3769a15`: empty-segment keys
could resolve to a sibling (a rejection fix was reported), S3-backed LIST ETags
were empty, bucket discovery could fail for a read-only role, and blob-download
headers applied to plain cold GET rather than range reads. These are historical
compatibility observations, not guarantees about another deployment. The notes
also describe 5.2 chunked storage; non-resident range semantics were not qualified.
Do not normalize object keys or invent validators to hide these limitations.

The October 20 maintainer measurement of the combined pre-extraction feature
(three targets, resident data, 100 Gb/s client, TPC-H SF3000, 22 queries) reported
457.640 s via AIStore versus 449.267 s direct S3 for cold input cache (+1.864%),
and 420.358 s versus 427.890 s lukewarm (-1.760%). These are supplied historical
parity results, not measurements of the extracted binary or a new speed-up claim.

## Evidence and rollback

Keep the inherited `[rest]` 132 root cases and the original redirect/request-log,
c3, AC9 and classifier assertions. Policy-unit tests supplement them. Serial
F0/AC1 (one worker, C=1, 32 delegated reads) requires expanded new connections ≤2
and legacy ≥16, with warm-up separate. Exact AC9 uses one worker and 16 physical
operations: C=2 gives 2 origin / 14 cached / 1 seed; C=16 gives 16 / 0 / 1.
Q64/256 three-target stress is separate coverage, not that serial oracle.

Matched receipts compare operation/wire-leg status, method, range, outcome,
bytes, attempt, delegation/route, connections and terminal result. Only declared
volatile timing/run fields are normalized; each actual Location is preserved.
The [D1 observer](../../test/io/redirect-policy/d1/README.md) adds bounded
single-worker joint publication receipts; it does not replace adapter tests or
prove arbitrary concurrent schedules. A characterization report that permits
UNSUPPORTED/NOT_TESTED is not a required-feature qualification: mandatory rows
must exist and PASS, and all unknown/missing/timeout outcomes must fail it.

For configuration rollback, construct new contexts with `route_reuse=off`, then
use `retention=legacy` if a comparison is needed, then `enabled=false` to stop
delegation. Existing in-flight operations retain their identity/context.
For code rollback, revert the relevant independent extraction commit in reverse
dependency order; `4422f3eb` remains the behavior reference. Do not reset a shared
checkout or alter archived evidence. Review completion and any remaining
qualification gaps belong in the delivery checklist; this document is not itself
a declaration that every acceptance gate has passed.
