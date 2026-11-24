use std::path::Path;

use quent_store_build::{Options, generate};
use sha2::{Digest, Sha256};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    quent_build_info::emit_source();

    let model = Path::new(env!("CARGO_MANIFEST_DIR")).join("../model.yaml");
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed={}", model.display());

    // `sha256:<hex>` digest of the schema, matching the form Quent's simulator and
    // cudf-polars use to key browser bundles to their analyzer.
    let digest = Sha256::digest(std::fs::read(&model)?);
    let digest = digest
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<String>();
    println!("cargo:rustc-env=SIRIUS_SCHEMA_DIGEST=sha256:{digest}");

    let parsed = quent_yaml::parse_from_file(&model)?;
    for warning in &parsed.warnings {
        println!("cargo:warning={warning}");
    }

    // YAML validation already checks metadata that code generation does not validate.
    generate(
        &parsed.schema,
        &Options {
            filesystem: cfg!(feature = "filesystem"),
            ..Options::default()
        },
    )?;

    Ok(())
}
