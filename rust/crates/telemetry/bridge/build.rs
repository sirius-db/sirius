use std::{collections::HashSet, path::Path};

fn copy_headers(source: &Path, destination: &Path) -> std::io::Result<()> {
    std::fs::create_dir_all(destination)?;
    let expected = std::fs::read_dir(source)?
        .map(|entry| entry.map(|entry| entry.file_name()))
        .collect::<Result<HashSet<_>, _>>()?;
    for entry in std::fs::read_dir(destination)? {
        let entry = entry?;
        if !expected.contains(&entry.file_name()) {
            if entry.file_type()?.is_dir() {
                std::fs::remove_dir_all(entry.path())?;
            } else {
                std::fs::remove_file(entry.path())?;
            }
        }
    }
    for entry in std::fs::read_dir(source)? {
        let entry = entry?;
        let target = destination.join(entry.file_name());
        if entry.file_type()?.is_dir() {
            copy_headers(&entry.path(), &target)?;
        } else {
            std::fs::copy(entry.path(), target)?;
        }
    }
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model = Path::new(env!("CARGO_MANIFEST_DIR")).join("../model.yaml");
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed={}", model.display());
    println!("cargo:rerun-if-env-changed=SIRIUS_TELEMETRY_BRIDGE_INCLUDE_DIR");
    let schema = quent_yaml::parse_from_file(model)?.schema;
    let options = quent_schema_codegen_cpp::Options {
        crate_name: "telemetry-bridge".to_owned(),
        instrumentation_path: "instrumentation_model".to_owned(),
        exporters: quent_schema_codegen_cpp::Exporters {
            ndjson: true,
            msgpack: true,
            postcard: true,
            ..Default::default()
        },
        ..Default::default()
    };
    let files = quent_schema_codegen_cpp::emit(&schema, &options)?;
    let bridges = quent_schema_codegen_cpp::write_bridge_files(&files, &options)?;

    let mut build = cxx_build::bridges(bridges);
    let include_dir = quent_schema_codegen_cpp::stage_cxx_headers(&options)?;
    build
        .include(&include_dir)
        .std("c++20")
        .compile("telemetry_bridge");
    // Only CMake builds need the headers outside OUT_DIR. Copy just the trees C++ includes;
    // the rest of include_dir mirrors absolute OUT_DIR paths and is not public API.
    if let Some(public_include) = std::env::var_os("SIRIUS_TELEMETRY_BRIDGE_INCLUDE_DIR") {
        let public_include = Path::new(&public_include);
        for subtree in ["rust", "telemetry-bridge/gen"] {
            copy_headers(&include_dir.join(subtree), &public_include.join(subtree))?;
        }
        println!("cargo:rerun-if-changed={}", public_include.display());
    }
    Ok(())
}
