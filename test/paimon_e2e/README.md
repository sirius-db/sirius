# Paimon E2E test runner

This first R5 increment provides a bounded runner and offline adversarial tests.
The actual C++ Paimon SQL tests and query-associated route checker are not yet
delivered. No Paimon GPU qualification is claimed by this directory's Python
tests or its CPU-only workflow. Acceptance status remains in the sole
[version-bound mapping](../../docs/paimon-e2e-acceptance.md).

Run the launcher protocol tests in their own locked CPU environment:

```sh
pixi run --locked --manifest-path test/paimon_e2e/pixi.toml test
```

The child fixtures write Catch2 XML and exercise the real launcher CLI. They
prove decisions about process status, discovery, hashes and complete reports;
they do not simulate native-reader correctness or establish GPU execution.

Once a qualified test binary contains both dedicated tests, run:

```sh
pixi run --locked --manifest-path test/paimon_e2e/pixi.toml python \
  test/paimon_e2e/run_e2e.py --binary /absolute/path/sirius_unittest \
  --sha256 <qualified-binary-sha256> --output /absolute/path/new-attempts
```

The fixed selector is `[paimon_e2e]`. Both exact case names are required:
`Paimon E2E append snapshot GPU` and
`Paimon E2E forced CPU rejects GPU qualification`. They must implement the
ordinary Paimon SELECT and forced-CPU counterexample described by Q01/Q02, using
independent expected values/types. C++ tests own the actual source/route checker;
this launcher verifies their structured result and exit status. Until those
cases exist, missing discovery must fail. Do not add passing placeholders.

Each invocation creates a fresh `run-*` directory and keeps the exact argv,
binary hash, terminal status, stdout/stderr and XML, including failure/timeout
output. No retries overwrite a previous attempt. Keep the executable and its path
stable throughout the attempt in the frozen deployment. The launcher checks the
path's content hash before and after execution. It requires a zero C++ exit status
and complete successful XML with both case identities, no skips or expected
failures, and a positive assertion count. It reads the binary and XML as regular
files. The XML hash covers the same bytes it verifies.

On POSIX, a private session leader holds the process group identity until the
launcher signals that group. This happens on both success and failure, including
timeout and interrupt. The launcher records the C++ exit status separately from
the session leader's cleanup status. A refused C++ launch retains the OS error
in `stderr.log` and leaves the C++ exit status unset. Cleanup is scoped to the
launched group. Qualification of the full owned process tree, including children that
leave that group, requires the separately coordinated outer controller.

After replacing `report.json` atomically, the launcher emits one JSON line with
event `inner_report_persisted`, the attempt directory, the report state and the
report's SHA-256. Use the launcher exit status together with this record and the
report. An interrupt during publication can still make the launcher exit nonzero.

A runner PASS means this particular binary passed the selected pair. A caller
provided binary hash does not independently prove how the binary was built.
Final qualification also requires reviewed source/build receipts, host/provider/
SDK/module identities, original corpus/oracle hashes, the full applicable matrix,
actual CI discovery/execution and merged-main validation. Production resource
admission and source-query association remain owner blockers, not launcher
exceptions. `.github/workflows/paimon-e2e-harness.yml` runs only cheap CPU launcher
tests; it schedules no GPU job and cannot close #2213 or #137.

## D9 offline checker preparation

`check_offline_evidence.py` exercises fail-closed causal checks through a real
CLI, reusing the conformance runner's typed multiset comparison. Its expected
and trace JSON are a **private synthetic normalized test model**, not Quent wire,
a production collector format or actual runtime evidence. The fixture's native
ownership/view tokens model required relationships; they provide no independent
proof of an actual SDK read. No production ID or observer API is introduced.
Supply independently authored oracle values that are valid for their declared
DuckDB types. The comparison checks type labels and normalized value equality.

The model joins run/runtime/process/statement and artifact/input/oracle identity,
execution window, one nonempty source's contract and deduplicated owned units,
isolated import deltas, the mapped GPU scan's Computing and successful
Finalizing, fully consumed results and same-runtime cleanup/liveness. Positive
aggregate import deltas establish only conditional single-source existence,
never a per-unit successful-import ledger. It does not cover LIMIT/multiple
scans, empty tables or all M1/M2 behavior. Execution `finalize_generation=0` and
retained nonzero plan generations are both legal and separate from nonzero Arrow
materialization generation.
The fixture's batch identity denotes one immutable batch and must map to one
unit; every listed native batch must have validation. These are private model
relationships, not a claim that production has such a receipt ledger. OOM retry
can count the same unit repeatedly; the CLI accepts larger positive deltas. A
reused finalized plan can retain a nonzero registry generation without a fresh
window entry. The model's execution window/UUID association is independent of
registry creation; its source query identity denotes the current scan manager
query token. Real collection must obtain that association and actual contract/
scan mapping. Zero registry generation, a fresh registry window and aggregate
count equality with deduplicated validation are not requirements.

All tests, including the complete synthetic positive and correct-row forced-CPU
negative, run in the locked CPU command above. Every successful offline decision
is `CONSISTENT_OFFLINE_MODEL`, with `actual_gpu_qualified: false`. Failure returns
nonzero and writes a fixed diagnostic without row payloads. Input bytes and
checker hashes accompany the decision; output must be a new file. Original
inputs remain caller-retained artifacts.

Actual collection is pending C/B: the current generic driver's post-production
input stamp does not establish factory-to-native-read-to-owned-batch origin,
and validation precedes import. Existing hooks plus whole-runtime isolation
require a bounded causal sufficiency proof before they can qualify execution.
No real C++ pair, P1/P2 runtime or GPU PASS follows from these offline tests.
