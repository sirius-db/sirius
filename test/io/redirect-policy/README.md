# Redirect policy prerequisite tools (E2/E3)

These are **test-only tools** for the existing Sirius REST implementation. No
production policy extraction is included. The reference engine is commit
`4422f3eb66c2f2d220af1804c3022c5c6555f0df`; subsequent candidates must have their own
engine build and manifest. Use only disposable loopback fixtures and public test
credentials. This is a functional baseline, not a performance benchmark.

## Review map

| File | Responsibility |
| --- | --- |
| `probe.cpp` | Real `rest_ioctx`, HEAD then serial ranged host reads, byte verification and JSONL records |
| `build_probe.py` | Compile/link the probe against an explicitly selected existing engine build; capture hashes |
| `qualify_probe.py` | Independent loopback origin/target fixture, baseline records, mutations and real fault replies |
| `compare_runs.py` | Validate two receipts and compare serial engine records plus wire observations |
| `qualify_comparator.py` | Comparator CLI negative controls and 32-read connection baseline |
| `step_r.py` | Session-scoped intent, capture, aged replay and fail-closed local permission |
| `test_step_r.py` | Storage, generation, timing, tampering, restart and consumer failure tests |
| `qualify_local.py` | Deploy an isolated AIStore/backend, run real Step R, consume permission with Sirius and clean up |

## Build and qualification

Prerequisites: Linux, Python 3 with boto3 for the deployment harness, the selected
Sirius locked build environment, an already built `s3_throughput_test` target and
its `compile_commands.json`. E3 also needs the separately reviewed cuCascade
cache harness with Linux pidfd cleanup, and verified AIStore/SeaweedFS binaries.

Run the build helper **inside the selected engine's activated Pixi environment**.
It derives flags from that build's generated Ninja commands without invoking a
shell. It compiles this probe only; it does not rebuild the engine or prove that
an old archive matches current sources. Review the engine's build receipt and
rebuild it for each production change before using a candidate manifest.

```sh
export TOOLS=/absolute/path/to/this/test/io/redirect-policy
export ENGINE=/absolute/path/to/selected/sirius
export RUN=/absolute/path/to/new/run-directory
# From ENGINE, using its locked environment:
pixi run python "$TOOLS/build_probe.py" --engine-root "$ENGINE" --output-dir "$RUN/build"
python3 "$TOOLS/qualify_probe.py" --binary "$RUN/build/sirius_rest_probe" \
  --build-manifest "$RUN/build/build-manifest.json" --run-root "$RUN/qualification"
python3 "$TOOLS/qualify_comparator.py" --binary "$RUN/build/sirius_rest_probe" \
  --manifest "$RUN/build/build-manifest.json" --qualification "$RUN/qualification" \
  --run-root "$RUN/comparator"
python3 "$TOOLS/test_step_r.py"
```

`qualify_probe.py` deliberately qualifies the fixed reference engine. To compare
a later candidate, generate the same workload with `run_probe`, save its own
build manifest and use the generic comparison CLI:

```sh
python3 "$TOOLS/compare_runs.py" --baseline /reference/run/off-a \
  --candidate /candidate/run/off-a --baseline-manifest /reference/build/build-manifest.json \
  --candidate-manifest /candidate/build/build-manifest.json
```

No production CMake target is added. The helper is specific to the generated
Sirius build layout and fails if the expected linker command is absent.

## Frozen serial contract

CLI options are unique `--name value` pairs. Required: endpoint, bucket, key,
fixture and out. Optional: target, reuse (`off`/`same_open`), retention
(`legacy`/`expanded`), connections (1–64), workers (must be 1), reads (1–256),
footer-first and request-log (true/false). Defaults are expanded, off, one
connection/worker, four reads, footer-first false and request-log true. Only
loopback HTTP origin and a single loopback target authority are supported.

The fixture is 64 bytes to 4 MiB. Reads are at most 64 KiB; the first can be a
tail-range read after the ordinary HEAD; this is not cold suffix metadata discovery. Later offsets are deterministic. Each returned byte is checked
against the fixture. The engine uses a five-second request timeout and two-second stall limit,
retry budget 2, auth refresh budget 1, 1 ms backoff, no jitter and no Retry-After.
See `probe.cpp` for the complete effective configuration.

JSONL schema 2 has one config record (`client=sirius-rest-probe-v2`), a begin/end
pair for HEAD (op 0) and each read (op 1..N), native log records with the current
serial op, and exactly one final complete record after shutdown. A PASS requires
all operations and bytes to match. Operation IDs are wrapper metadata, not a new
production request-ID API. Native transfer/decision messages retain attempt,
route, outcome, budget and connection fields for exact comparison.

The wire receipt records role, method, path/query, Range, If-Match, status,
Location, response bytes, connection identity and whether provider credentials
were present. It never records credential values. Each target URL must match an
observed origin Location verbatim; target requests must carry no provider
credentials. The comparator binds binary and fixture digests and invocation
configuration before comparing records.

Normalization is limited to native `elapsed_ms`, declared ephemeral endpoint
authorities and per-role connection identity numbering. Non-REST diagnostic
messages remain in raw evidence but are outside policy comparison. All remaining
serial REST records and normalized wire rows must match exactly. Missing/extra
legs, attempts, bytes, terminal status, client/config, validator, credentials and
Location changes are rejected. Nonzero exits, including timeout/skip receipts,
are rejected; the corresponding negative controls mutate receipts, not actual
OS timeouts. Real fault fixtures separately exercise second-hop 307, 403, 412 and
wrong bytes.

**Coverage limit:** no concurrent partial-order comparator is implemented here.
Existing unchanged `[rest]`, `[rest_cluster_redirect]`, `[rest_request_log]`,
`[rest_c3]` and `[rest_c3_ac9]` tests remain mandatory per production commit,
including asynchronous ownership and exact AC9 counts. The serial probe cannot
substitute for them. Run them with the default log environment: forcing
`SIRIUS_TEST_REST_REQUEST_LOG=1` globally contradicts the existing default-off
test. Dedicated tests set their own logging configuration.

## Local Step R domain and reproduction

```sh
python3 "$TOOLS/qualify_local.py" --run-root "$RUN/step-r" \
  --harness-dir /absolute/path/to/cucascade/scripts/cache-harness \
  --binary "$RUN/build/sirius_rest_probe" --build-manifest "$RUN/build/build-manifest.json" \
  --ais-source /absolute/path/to/aistore --aisnode /absolute/path/to/aisnode \
  --aisnode-sha256 VERIFIED_AIS_SHA256 --weed /absolute/path/to/weed \
  --weed-sha256 VERIFIED_WEED_SHA256
```

The controller uses local ports 19630..19637 and 19730..19735 and creates a new
private run directory. Existing listeners are not killed. A disposable 3 MiB+31
byte object exercises first, second and last MiB ranges against the same GET
Location, fresh captures as references, and suffix-capture-to-bounded replay.
The selected original Location is replayed after **at least 600 actual monotonic
seconds**. The successful replay must retain status 206, Content-Range, length
and identical bytes. HEAD-to-GET grants are not claimed.

Permission is limited to one serial controller instance and its live loopback
deployment, fixture, GET/range domain, binary/config/controller hashes, process
identities and boot ID. It expires after a conservative local 3600-second cap;
this cap is not a backend capability TTL. Any mismatch or failure revokes it.
The controller immediately consumes permission with real Sirius `same_open`
reads and verifies that a cached route was exercised, then stops its services.

SQLite stores monotonically increasing attempt generations. Intent commits
before the control-storage probe. PASS is read back before use. Every consumer
checks the latest generation, exact binding and capture digests. Registry and
control paths can share a disk: this is not an independently durable distributed
registry. The compensating rule is an irreversible in-memory veto after failure
and **no resumption of any persisted PASS by a new controller instance**.
Consequently storage failures, a newer attempt, process restart or cleanup cannot
silently restore an older grant. Concurrent or cross-machine consumers are not
supported and require a separate design review.

Unit tests simulate timing boundaries; they do not replace the real ten-minute
run. Keep raw failures and successful evidence separate. `location.secret`,
capture files and generated service configs are local evidence, not reusable
permission or public review attachments. No production auth, performance,
multi-target stress or general cross-open reuse qualification is claimed.

## Review repairs R1/R2

Schema2 binds all invocation options (including defaults), bucket/key, fixture/output paths and actual fixture size. The comparator rejects duplicate/unknown arguments and checks the origin object path against the declared bucket/key. Only fixture/output paths, declared ephemeral authorities, native elapsed time and normalized connection IDs are volatile across runs; contents remain hash-bound. Schema1 artifacts must be regenerated, not silently accepted.

Expected faults require exit1, one complete failure, a matching failing operation/end record, exact wire status/body hash and native transfer sequence. The probe reports stable test-only categories: object_changed uses the actual engine exception type; HTTP307/403 categories match exact current engine runtime_error text and object; fixture_mismatch is a dedicated probe exception. Unknown errors remain unexpected_error. The validators reject crashes, missing/duplicate records, unrelated failures and undelivered injections. Engine wording changes must be reviewed, not broadened into arbitrary failures.

Run `python3 test_review_controls.py --qualification RUN/qualification --manifest RUN/build/build-manifest.json --output RUN/review-controls.json` after generating fresh receipts. This exercises the independently reported cases plus operation/transfer/body and parser controls.
