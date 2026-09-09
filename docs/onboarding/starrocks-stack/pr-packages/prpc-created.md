# Created Draft PR: concurrent PRPC dispatch

- PR: [#1739](https://github.com/sirius-db/sirius/pull/1739)
- Title: `fix(cn): serve concurrent PRPC requests on one connection`
- Base / head: `sirius-db:sirius/dev` ← `aocsa:codex/cn-concurrent-prpc`
- State: Draft; labels `starrocks`, `rust`, `perf`
- Extracted source commit: `5e71059c` → `a9869aee`; stable patch-id `69e9aca3` matches exactly
- Scope: one file, `experimental/starrocks/src/brpc.rs` (+135/−17)

## Exact PR body

## Summary

Serve PRPC requests received on one TCP connection concurrently. The server now splits the socket, clones its Tower service per request, and serializes only response writes. Responses may therefore arrive in completion order while their correlation ids continue to pair them with requests.

This prevents a long `fetch_data` poll or blocking `exec_plan_fragment` from holding a later `cancel_plan_fragment` behind it on the FE's multiplexed CN connection. The included regression test sends a slow `fetch_data` and a fast `cancel_plan_fragment` on one connection and proves that the fast response arrives first.

## Scope and prerequisites

This is one self-contained file and one preserved source commit. It does not require the peer PRPC client to compile or test. The motivating FE-to-CN exchange/cancellation path still needs the checked-in exchange RPC interface from #1707 (or its landed equivalent) and the later distributed CN runtime; this change is the server-side concurrency primitive that path uses.

## Validation

- `pixi run cargo fmt --manifest-path experimental/starrocks/Cargo.toml --check`
- Verified the extracted commit's stable patch-id exactly matches source commit `5e71059c`.
- `git diff --check upstream/dev...HEAD`

The Cargo test suite was not run: this standalone clone intentionally does not initialize the StarRocks submodule, and the extraction does not warrant a GPU build or rebuild.
