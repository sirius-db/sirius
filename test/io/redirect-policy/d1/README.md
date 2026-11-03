# D1: joint route-publication receipts

Test tooling for the S2 design §7 evidence gap. This does not change production
policy or logging. The runner accepts **already built**, frozen Linux/CUDA
reference and candidate checkouts and uses their own compile/link commands.
It creates a new output directory and never overwrites the accepted core/test
binaries. Run inside the same frozen Pixi environment used to build the engines.

```sh
pixi run --as-is --offline -e default --manifest-path /absolute/reference/pixi.toml \
  python3 /absolute/candidate/test/io/redirect-policy/d1/build_run.py \
  --reference /absolute/reference --candidate /absolute/candidate \
  --out /absolute/new-run-directory
```

Both paths and the output should be absolute. Required existing tools: Python
3.9+, GNU ar, Ninja, and the configured C++ toolchain. No dependency update or
Internet access is required. This is explicitly a single-controller test tool;
do not concurrently run it against checkouts being built or edited.

The command generates instrumented copies of `rest_reactor.cpp`, compiles each
against its own headers, copies its core archive and replaces only the reactor
member. `archive_audit.py` verifies all other member contents are unchanged.
The test TU is the unmodified integration fixture plus `fixture_extension.cpp`.
No production TU, header, CMake target or accepted binary is rewritten. Every
source replacement must match exactly once; inverse replacements must recover
the original bytes. Inputs, tools, commands, outputs and hashes are retained.
The runner then executes the nine D1 sections, the original 132 `[rest]` roots,
paired qualification, and both engines' receipt-negative tests. A failed build,
test, source audit or qualification makes the command fail.

## What is observed

* Actual op/object/context pointers are checked in-process. The fixture allocates
  unique ranges within a held open; retries must retain the same physical op.
  Exported physical/open/context IDs are local tokens, not pointer addresses.
* Actual retained `shared_ptr` objects define publication tokens. URL equality
  never defines identity. The stale scenario explicitly proves P1 and P2 have
  the same URL but different snapshot identities.
* Before/after selection inputs, selected/current publications, actual counters,
  successful multi submission, curl completion/status/received bytes, original
  recovery classifier, conditional invalidation and validated publication are
  joined with the actual future result and exactly-once completion assertion.
* The HTTP server independently records role, GET, unique Range and If-Match.
  No production request ID or sensitive grant is logged.

The bounded observer is single-writer, no-throw and fail-closed. It takes extra
read-only route snapshots using the existing lock and retains their strings
until serialization after worker shutdown/join. Thus it changes observation
cost and snapshot destruction time; it is not a timing/performance instrument.
It does not retain context owners or alter production lock implementations.
Only one reactor worker is active. The fixture's HTTP response gate creates the
specified overlap of two real cached operations; it never manually seeds or
invalidates a live route. This is not proof for arbitrary multiworker scheduling.

## Scenarios and decisions

Six qualifying scenarios: reuse, stale P1/P2, cached 412, reuse off, disabled,
and weak validator. The stale scenario withholds a nonempty 404 body until the
other operation has refreshed and published P2. A zero-length body would not
hold the first curl operation in flight.

Three deliberate observer failures: dropped event, capacity exhaustion, and an
injected `bad_alloc` inside observation. Actual reads must still complete, but
these receipts **cannot qualify**. The overall test succeeds only when their
sticky failure flags are set and the validator rejects them. They are not
functional PASS samples. This tests a caught exception, not a claim of actual
system memory exhaustion.

`validate.py` verifies exact event structure, operation/publication association,
selection gates and counters, wire legs, classifier, invalidation, final result
and completion. Before grouping by operation, it tracks the live publication and
all published snapshot identities in original observer order. Only a successful
seed may publish a new identity, and invalidation clears only the selected
current identity. Seed, hit and refresh counters must increase at their exact
observed boundaries, including the immediate refresh continuation in this
single-worker fixture; unchanged and failed transitions must not increment. It
then compares the two valid engines' per-op receipts exactly.
It is a finite fixture oracle, not a general concurrent trace comparator.
Historical S2 owner-lifetime/allocation/AC9 evidence remains separate.

```sh
python3 validate.py /run/reference /run/candidate --out /run/recheck.json
python3 validate.py --single /run/reference/reuse.jsonl --scenario reuse
python3 test_validate.py /run/reference
```

The single-receipt command returns nonzero for observer failures, wrong
associations, missing events or other invalid evidence. Negative tests cover
29 original mutated receipts and five original CLI failures per engine, plus
eight causal mutations (unpublished identity, missing seed/refresh increments,
early/late increments and an increment on a failed seed). Each new mutation
must fail as a single receipt, a one-sided pair and a common-error pair, for 24
additional nonzero CLI checks per engine. Normal receipts and a normal pair
remain positive controls. Generated sources and raw failed attempts should be
preserved alongside successful runs. An implementation self-check does not close D1 without independent review.
