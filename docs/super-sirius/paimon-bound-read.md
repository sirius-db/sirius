# Paimon bound-read consumer

This is a descriptor-consumer prerequisite for [#2212](https://github.com/sirius-db/sirius/issues/2212).
It does not register a production connector, create native plans/readers, import batches,
or authorize CPU replay. Normal Paimon SQL behavior is unchanged.

`planner::paimon_read_provider::from_loaded()` obtains an independent reference by running
the already-loaded Paimon initializer in a private CPU-only catalog. It never trusts the
caller's mutable `paimon_scan` registration. The initializer must belong to the exact
loaded module; a symbol resolved from a dependency is refused. The acquired module
reference stays alive until all trusted function definitions and retained descriptors
are released. Function-info destructors run before module release.

`retain()` compares the complete semantic callback/signature/settings surface, using the
same comparison as the existing scan registry, and the exact shared function-info
singleton **before** interpreting the public wrapper. It then checks descriptor ABI and
required callback provenance, asks the producer to accept the live binding, and retains
and validates its description. Failed and partial outputs are released using the
producer's allocator. Provider diagnostic buffers are not copied into errors.

The descriptor owns its borrowed view and retains the provider independently of the
original binding, provider wrapper, connection, and database. Moving an owner invalidates
access through the moved-from wrapper. View options can contain credentials: do not
log, serialize, or expose them. Keep the descriptor alive while using its view.

## Current scope

| Property | Contract |
|---|---|
| Provider discovery | Linux dynamic extension loaded by explicit path; normal DuckDB ABI checking required. Installed-name, built-in, and custom-loader providers are refused. |
| Descriptor ABI | Exact V1, or the unchanged descriptor prefix of exact V2. V2 acceptance does not acquire or validate native-reader callbacks. |
| Retained identity | Borrowed warehouse/database/table/location/branch/schema JSON, stable field IDs, schema/snapshot IDs and effective options from the provider. No rebind or metadata/planning call. |
| Empty-at-bind | `has_snapshot == 0` and snapshot ID zero remain empty after subsequent writes. |
| Candidate correspondence | Not implemented. These descriptors are inputs to future scan identity capture/comparison; this component does not prove logical/physical correspondence. |
| Snapshot availability | Retention is not a lease. This component does not read files or qualify expired-snapshot behavior. |
| Native planning/preparation/GPU/replay | Not enabled or qualified by this component. No global policy changes. |

The copied C declarations come from duckdb-paimon `9e89afe799e88d36214be9026cb60add10c538f9`
and the public wrapper definition from `80b391069a3192c56d8e52ae9a51d0d1dbb4074e`,
with Sirius formatting. The wrapper definition and V1 layout agree at these revisions.
Neither header brings paimon-cpp or Arrow C++ dependencies into Sirius.

The real DSO qualification pairing is Sirius DuckDB
`88ecd0a49d7d3d838e002b2fbc0ce91ccf74110e` with duckdb-paimon
`80b391069a3192c56d8e52ae9a51d0d1dbb4074e`. This historical producer does not implement
`FunctionData::Copy()`; its explicit exception is not copied-plan support. Current
duckdb-paimon `9e89afe` uses a different DuckDB pin and is not qualified with this host.
Do not disable extension ABI checking to bridge that difference.

Before production registration, #2212 still needs retained-identity correspondence across
eligible plans/prepared executions/multiple scans, native-reader storage classification,
capability and residual-filter checks, and the existing whole-plan replay admission.
#2211 must supply bounded preparation; the current experimental producer's refusal of
all finite planning budgets does not satisfy that requirement. Append-only GPU execution
and primary-key visibility qualification remain separate work under #2213.

## Tests

Normal Sirius unit tests include `[paimon_binding]`: missing-provider/disabled-ABI checks,
semantic callback comparison, a small independent hidden DSO with malformed ABI and foreign
callback responses, incomplete outputs, partial-error cleanup, and move ownership. The
fixture models descriptor responses only; it is not a Paimon correctness oracle.

`sirius_paimon_binding_test` is a separate CPU-only host linked with DuckDB and Catch2,
without Sirius or Paimon/Arrow native libraries. Build and run its default tests with:

```sh
pixi run cmake --build build/release --target sirius_paimon_binding_test
pixi run build/release/test/cpp/sirius_paimon_binding_test '[paimon_binding]'
```

The hidden `[paimon_binding_dynamic]` cases require an externally built, matching real
Paimon DSO. Missing configuration fails; it is not an empty or skipped qualification:

```sh
SIRIUS_PAIMON_TEST_EXTENSION=/absolute/path/paimon.duckdb_extension \
  pixi run build/release/test/cpp/sirius_paimon_binding_test \
  '[paimon_binding_dynamic]' --reporter xml --out /absolute/path/paimon-binding.xml
```

These cases exercise two `paimon_scan` overloads and attached-table bindings, retained
snapshots after later writes, empty-at-bind, foreign info/callbacks, catalog replacement,
acquisition before/after provider load, and ownership after database teardown. They are
CPU descriptor tests, including when run on a GPU host. A passing default fixture suite
does not substitute for this real DSO qualification or establish GPU support.

The separate hidden `[paimon_binding_coload]` case requires the matching real Paimon
DSO plus a Sirius extension built from the same source revision as the test host.
It exercises both extension load orders and all three binding entry points with
the Sirius context active, while Paimon queries use CPU execution. It requires a
GPU for Sirius context initialization; it does not qualify native Paimon GPU reads:

```sh
SIRIUS_PAIMON_TEST_EXTENSION=/absolute/path/paimon.duckdb_extension \
SIRIUS_EXTENSION_PATH=/absolute/path/sirius.duckdb_extension \
SIRIUS_CONFIG_FILE="$PWD/test/cpp/config/data/configurator.yaml" \
  pixi run build/release/test/cpp/sirius_paimon_binding_test \
  '[paimon_binding_coload]' --reporter xml --out /absolute/path/paimon-coload.xml
```
