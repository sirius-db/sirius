use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let gen_dir = out_dir.join("thrift_gen");
    fs::create_dir_all(&gen_dir).unwrap();

    // Path to vendored Doris thrift files (git submodule)
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let thrift_dir = manifest_dir
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("thirdparty/apache-doris/gensrc/thrift");

    assert!(
        thrift_dir.exists(),
        "Doris thrift directory not found at {}. Did you initialize the git submodule?",
        thrift_dir.display()
    );

    // Find all .thrift files
    let thrift_files: Vec<PathBuf> = fs::read_dir(&thrift_dir)
        .unwrap()
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().map_or(false, |ext| ext == "thrift"))
        .collect();

    // Generate Rust code for each thrift file
    for thrift_file in &thrift_files {
        println!("cargo:rerun-if-changed={}", thrift_file.display());

        let output = Command::new("thrift")
            .args(["--gen", "rs", "-I"])
            .arg(&thrift_dir)
            .arg("-out")
            .arg(&gen_dir)
            .arg(thrift_file)
            .output()
            .expect("Failed to run thrift compiler. Is `thrift` in PATH?");

        if !output.status.success() {
            panic!(
                "thrift codegen failed for {}:\n{}",
                thrift_file.display(),
                String::from_utf8_lossy(&output.stderr)
            );
        }
    }

    // Post-process: convert #![...] inner attributes to #[...] outer attributes
    // (inner attributes are only valid at crate root, not in modules)
    for entry in fs::read_dir(&gen_dir).unwrap() {
        let path = entry.unwrap().path();
        if path.extension().map_or(true, |ext| ext != "rs") {
            continue;
        }
        let content = fs::read_to_string(&path).unwrap();
        let fixed = content
            .replace("#![allow(", "#[allow(")
            .replace("#![cfg_attr(", "#[cfg_attr(");
        fs::write(&path, fixed).unwrap();
    }

    // Generate a mod.rs that includes all generated modules
    let mut mod_content = String::from(
        "// Auto-generated module declarations for Doris Thrift bindings.\n\
         // Do not edit manually.\n\n",
    );
    let mut mod_names: Vec<String> = Vec::new();
    for entry in fs::read_dir(&gen_dir).unwrap() {
        let path = entry.unwrap().path();
        if path.extension().map_or(true, |ext| ext != "rs") {
            continue;
        }
        let stem = path.file_stem().unwrap().to_str().unwrap().to_string();
        mod_names.push(stem);
    }
    mod_names.sort();
    for name in &mod_names {
        mod_content.push_str(&format!(
            "#[path = \"{gen_dir}/{name}.rs\"]\npub mod {name};\n",
            gen_dir = gen_dir.display(),
            name = name,
        ));
    }
    let mod_file = out_dir.join("thrift_mods.rs");
    fs::write(&mod_file, mod_content).unwrap();
}
