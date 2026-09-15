use std::{
    collections::HashSet,
    path::{Path, PathBuf},
};

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

    let parsed = quent_yaml::parse_from_file(&model)?;
    for warning in &parsed.warnings {
        println!("cargo:warning={warning}");
    }

    let options = quent_schema_codegen_cpp::Options {
        crate_name: env!("CARGO_PKG_NAME").to_owned(),
        instrumentation_path: "sirius_telemetry_instrumentation".to_owned(),
        exporters: quent_schema_codegen_cpp::Exporters::all(),
        ..Default::default()
    };
    let files = quent_schema_codegen_cpp::emit(&parsed.schema, &options)?;
    let bridges = quent_schema_codegen_cpp::write_bridge_files(&files, &options)?;
    let mut build = cxx_build::bridges(bridges);
    let include_dir = quent_schema_codegen_cpp::stage_cxx_headers(&options)?;
    build
        .include(&include_dir)
        .std("c++20")
        .compile("telemetry_bridge");

    let source_include = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("include");
    copy_headers(&include_dir, &source_include)?;
    println!("cargo:include={}", include_dir.display());

    Ok(())
}
