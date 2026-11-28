# Sirius Quent browser analyzer

This crate exposes the Sirius analyzer (`SiriusUiAnalyzer`) to a static Quent
viewer through WASM, and converts Sirius telemetry into the `.quent-browser`
file that analyzer opens. It follows the same contract as the cudf-polars
browser crate.

## Bundles

A bundle is a `QNTBR001` envelope holding metadata and a postcard-encoded list
of Sirius events. The analyzer rejects a bundle unless:

- `model_name` is `sirius`;
- `schema_digest` equals the digest of this build's
  [`model.yaml`](../model.yaml) (`sha256:<hex>`, exported by
  `sirius-telemetry-store` as `SCHEMA_DIGEST`);
- `codec` is `postcard`;
- the events contain exactly one engine, and its ID equals the metadata's
  `engine_id`.

Hosts select the analyzer from `(model_name, schema_digest, format_version)`.
Changing `model.yaml` changes the digest, so bundles written by older schemas
need the analyzer revision that was built from them. The archival Quent ZIP is
a separate artifact from the bundle.

## Generate a bundle

The `generate` feature adds native filesystem loading and the
`sirius-quent-browser-bundle` binary. From the repository root:

```bash
pixi run -e browser quent-browser-bundle <telemetry_dir_or_zip> trace.quent-browser
```

This runs `cargo run --release -p sirius-telemetry-browser --features generate
--bin sirius-quent-browser-bundle -- <input> --output <output>` in the `browser`
pixi environment.

It prints the model name, schema digest, format version, engine ID, and the
number of contexts and events. Every context directory under the input is
included. `analyzer_revision` in the bundle metadata is the Sirius commit
this binary was built from.

## Build the analyzer

```bash
pixi run -e browser build-quent-browser build/quent-browser
```

The task builds the crate for `wasm32-unknown-unknown` and runs `wasm-bindgen`
([`scripts/build_quent_browser.sh`](../../../../scripts/build_quent_browser.sh)).
The `wasm-bindgen` CLI must match the `wasm-bindgen` crate version in
[`Cargo.lock`](../../../Cargo.lock); the script fails if the `wasm-bindgen-cli`
pinned in the `browser` feature of `pixi.toml` drifts from it. Update both
together.

The generated JavaScript exports Quent's standardized `BrowserServer`, so no
Sirius-specific Web Worker is needed. Host both generated files on the same
origin as the Quent UI and select them with the viewer URL:

```text
?module=/analyzers/sirius/sirius_telemetry_browser.js&data=https://traces.example/run.quent-browser#/
```

## Features

| Feature | Effect |
|---------|--------|
| `generate` | Native bundle generation and the CLI. Off for the WASM build. |

`sirius-telemetry-store` and `sirius-telemetry-analyzer` have a default-on
`filesystem` feature for loading collector output; this crate disables it so
both compile for `wasm32-unknown-unknown`.
