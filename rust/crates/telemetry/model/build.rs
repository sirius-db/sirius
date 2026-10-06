// Captures this crate's git into QUENT_SOURCE_* so the model.qmi sidecar records
// Sirius as the model source instead of falling back to quent's build info.
use std::path::Path;

use quent_instrumentation_build::{Options, generate};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model = Path::new(env!("CARGO_MANIFEST_DIR")).join("../model.yaml");
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed={}", model.display());

    let parsed = quent_yaml::parse_from_file(&model)?;
    for warning in &parsed.warnings {
        println!("cargo:warning={warning}");
    }

    generate(
        &parsed.schema,
        &Options {
            serde: true,
            umbrella_event: true,
            analyzer_package: Some("sirius-telemetry-analyzer".to_owned()),
            collector_sink: true,
            ..Options::default()
        },
    )?;

    Ok(())
}
