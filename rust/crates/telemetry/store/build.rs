use std::path::Path;

use quent_store_build::{Options, generate};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    quent_build_info::emit_source();

    let model = Path::new(env!("CARGO_MANIFEST_DIR")).join("../model.yaml");
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed={}", model.display());

    let parsed = quent_yaml::parse_from_file(&model)?;
    for warning in &parsed.warnings {
        println!("cargo:warning={warning}");
    }

    // YAML validation already checks metadata that code generation does not validate.
    generate(
        &parsed.schema,
        &Options {
            combined_event: true,
            ..Options::default()
        },
    )?;

    Ok(())
}
