# Paimon CPU conformance corpus

These six small tables are real Paimon writer output. Hand-authored SQL answers
cover append duplicates, primary-key update/delete/reinsert histories, historical
reads, NULL/decimal/string/date values, empty tables, and multiple partitions/files.
The suite compares values and SQL types, then checks connection liveness. It does
not implement or demonstrate a GPU Paimon scan.

## Pinned source build

The optional native suite builds `paimon.duckdb_extension` from source before
reading the corpus. It needs no hosted Paimon binary or community installation.
`source-build.json` pins the extension, native reader and DuckDB source archives
by commit, size and SHA256, together with the build options. Native dependencies
use the upstream bundled versions and checksums. The recipe supports DuckDB
v1.5.6 on Linux x86_64; other versions/platforms require a reviewed recipe update.

Build through the repository's locked Pixi environment. The local-filesystem
recipe disables OSS, S3 and REST and produces a loadable extension for the Sirius
CLI. It does not add a Paimon dependency to ordinary Sirius builds. Cold builds
compile native dependencies including Arrow and Parquet and take longer than
reading this small corpus.

A build receipt records the source recipe, builder and Pixi lock hashes, compiler
versions, target executable identity/hash, and output size/hash. The runner
requires that receipt by default and rejects stale inputs, a changed executable,
or changed extension bytes. This is provenance from a trusted local/CI builder,
not a signed attestation or a promise of byte-identical rebuilds. A successful
build does not qualify reader correctness; conformance must pass separately.

`qualified-artifacts.json` retains historical community-binary identities for
explicit `--registry` investigations. Those results do not qualify a newly built
reader. Qualification covers the committed fixtures, not general Paimon support.
Sirius GPU regressions exercise the surrounding engine; Paimon reads use the CPU.
Dated validation results and delivery status belong in the pull request.

Known limit: [duckdb-paimon #95](https://github.com/polardb/duckdb-paimon/issues/95)
reports stale same-key updates with a Flink-written composite primary key, subset
bucket key, two buckets, and overlapping level-0 files. These one-bucket,
one-column-key fixtures do not establish correctness for that workload. This
corpus also excludes concurrent writers, schema evolution, deletion vectors,
other merge engines, remote storage, and statement-bound snapshot guarantees.

## Routine checks and reading

The standard-library harness tests require Python 3.11 or later; CI uses Python
3.12. They need no native build, extension, GPU, external network, or corpus
regeneration. Download tests use a temporary loopback HTTP server:

```bash
pixi run python3 -m unittest discover \
  -s test/cpp/integration/data/paimon_conformance -p 'test_*.py' -v
```

Run repository commands from its root using its Pixi environment. On a Mac without
the Linux CUDA environment, the same unittest command can run through an isolated
CPU-only Pixi environment containing Python 3.12. The fake CLI tests check harness
decisions; they do not qualify native Paimon behavior.

Build the test dependency into a **new** directory (Python 3.12 is recommended):

```bash
pixi run --locked python3 test/cpp/integration/data/paimon_conformance/build_extension.py \
  --duckdb build/release/duckdb --output build/paimon-conformance/source-build --jobs 4
```

The builder refuses an existing output directory, including a failed build. Use
another directory for a retry; logs from the failed attempt remain available.
Source downloads retry transport failures up to three times; checksum mismatches
fail immediately. The optional `--source-cache DIR` accepts previously downloaded
archives named `<sha256>.tar.gz`, verifying their length and checksum before use.
Native dependency archives can use the same naming convention; only entries
whose hashes appear in the pinned native `third_party/versions.txt` are reused.
A missing cache entry downloads normally; a corrupt entry fails. The default
needs no cache. This cache contains source archives, never prebuilt libraries.

Then read the committed corpus using the same Sirius executable:

```bash
pixi run --locked python3 test/cpp/integration/data/paimon_conformance/run_conformance.py \
  --duckdb build/release/duckdb \
  --paimon-extension build/paimon-conformance/source-build/paimon.duckdb_extension
```

The default receipt is `build-receipt.json` beside the extension; use
`--build-receipt` if it was relocated separately. Keep the receipt and extension
together when moving them. Historical byte qualification instead requires an
explicit `--registry` and its exact binary; there is no automatic fallback.

The runner never downloads or installs extensions. It checks exact warehouse
inventory/hashes, recorded oracle consistency, and actual executable version and
platform before loading the verified build output. A full run plans 53 cases and
starts 54 processes: one identity probe, 50 ordinary cases, and three smoke cases.
Each ordinary/smoke query uses one process for DESCRIBE, rows, and liveness.
Source-built extensions are unsigned: after receipt validation, their case
processes use DuckDB's `-unsigned` option. The identity probe and historical
registry mode keep signature enforcement enabled.

Runtime-disabled and explicit-CPU modes use the same Sirius executable. Normal
initialization, transparent fallback, and rejection checks require a working GPU
environment even though Paimon reads use the CPU. An unavailable GPU is a pending
test prerequisite, not permission to claim these cases passed.

`--case append_b` selects ordinary cases in both modes; the three smoke checks
still run. `--timeout` bounds each child; `--output` places reports outside the
corpus. Reports distinguish PASS, FAIL, and NOT RUN and preserve case tracebacks
and child stdout/stderr, including partial timeout output. Shared setup failure
leaves cases NOT RUN and the command fails. Case-local failure does not suppress
later independent cases. Interrupts still terminate the run.

The optional `Paimon conformance` workflow accepts a completed `Test` run whose
source head matches the dispatched revision. The CUDA 13 artifact must contain
`build/release/sirius-build-sha.txt`, written from the build checkout before
packaging. Old artifacts without this marker are rejected.

For PR runs, that checkout is the synthetic PR-plus-base merge at build time,
not the PR head alone. `build-provenance.json`, the setup log, and the workflow
summary explicitly distinguish the source head and actual build checkout SHA.
For manual/merge-group builds, the recorded checkout must equal the dispatched
revision. Workflow paths with or without an `@ref` suffix are accepted.

This workflow does not rebuild or rerun the ordinary test matrix. It waits until
the selected Test run completes, which can mean waiting for its full 90-minute
job budget even when the build artifact is already available. Artifacts expire
after one day; missing/expired builds fail explicitly.

GitHub requires the workflow file to exist on the repository's default branch
before [`workflow_dispatch`](https://docs.github.com/en/actions/reference/workflows-and-actions/workflow-syntax#onworkflow_dispatch)
can trigger it. A newly introduced workflow therefore cannot be dispatched in
the base repository merely by publishing a PR ref; its first run requires the
workflow to reach the default branch. Dispatch also requires a branch/tag in the
repository where it runs. To validate a fork PR revision in the base repository,
a maintainer must make that revision available on a base-repository ref. A fork
can dispatch only when its default branch contains the workflow and its required
runners are available.

Build/case diagnostics, receipts and nested CMake logs are uploaded even on failure. Normal PR/merge-group
checks run the CPU harness in its own required job, independently of lint and
thread-sweep checks, with a Paimon-specific summary. Workflow validation requires
a completed real run with retained reports; local checks alone do not establish
CI delivery.

## Explicit regeneration

Regeneration changes fixture data only when recipes change. A new reader binary
requires requalification, not automatic regeneration. Never replace expected
answers with output from the reader being tested.

Use Linux, a Python 3.12 environment, `pypaimon==2.0.0`, `pyarrow==19.0.1`, and the
exact native revision below. The example creates a separate virtual environment
from the activated Python; verify its version first. PyPaimon writes five table
families. The small native helper is needed only for the DELETE history because
the selected Python writer does not emit DELETE row kinds.

```bash
pixi run python3 -c 'import sys; assert sys.version_info[:2] == (3, 12), "Use Python 3.12 for regeneration"'
pixi run python3 -m venv build/paimon-generator
pixi run build/paimon-generator/bin/python -m pip install pypaimon==2.0.0 pyarrow==19.0.1
git clone https://github.com/apache/paimon-cpp.git build/paimon-generator-source
git -C build/paimon-generator-source checkout 53f9c86d45aabb0a6f1a379271da07d7a9f27a3d
pixi run cmake -S build/paimon-generator-source -B build/paimon-generator-native -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DPAIMON_DEPENDENCY_SOURCE=BUNDLED \
  -DPAIMON_BUILD_TESTS=OFF -DPAIMON_BUILD_SHARED=OFF -DPAIMON_BUILD_STATIC=ON \
  -DPAIMON_ENABLE_AVRO=ON -DPAIMON_ENABLE_OSS=OFF -DPAIMON_ENABLE_S3=OFF \
  -DPAIMON_ENABLE_REST=OFF \
  -DCMAKE_PROJECT_paimon_INCLUDE="$PWD/test/cpp/integration/data/paimon_conformance/generator/enable.cmake"
pixi run cmake --build build/paimon-generator-native --parallel 4 --target paimon_delete_writer
pixi run build/paimon-generator/bin/python \
  test/cpp/integration/data/paimon_conformance/gen_corpus.py \
  build/paimon-corpus-candidate \
  --delete-writer build/paimon-generator-native/release/paimon_delete_writer
```

Use a new output directory; the generator refuses to overwrite one and retains
partial output on failure. It reads complete histories independently with
PyPaimon, records effective persisted options and structured partition values,
and emits canonical JSON. Raw warehouse files must never be reformatted.
`partition.legacy-name=false` is intentional for the DATE partition fixture.

Run conformance against `build/paimon-corpus-candidate` as the positional corpus
argument, then a relocated copy with the original path unavailable and networking
disabled after provisioning. Review data, metadata, independent observations, and
hashes before replacing committed files. Fresh UUIDs/timestamps may differ;
formatting the generated metadata must not cause further changes.

The format-2 metadata repair preserved the existing warehouse and original
PyPaimon observations. Added oracle identity associations were reconstructed from
the committed files/cases, not a new independent read. `metadata_revision` records
that distinction. Runtime comparisons detect drift, but coordinated edits of
both expected answers and recorded observations still require independent review.

## Updating the source recipe

Review the exact extension/native/DuckDB commits, source archive checksums,
upstream dependency pins and patches, compiler environment, and build options.
Update the recipe deliberately; never select the newest upstream branch during
a test run. A DuckDB upgrade requires a compatible recipe and fresh validation.

Build in a clean directory without prebuilt dependency caches. Verify source
retrieval from an empty archive cache, run native conformance, then repeat against
a relocated corpus with networking disabled after building. Run the appropriate
Sirius GPU fallback/regressions with the newly built extension. Investigate
mismatches against authored answers and independent reader evidence. Record the
actual build receipt and test results; do not copy a historical binary's passing
status onto a new build. No permanent binary publication is required.
