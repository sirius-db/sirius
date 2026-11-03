//! Link the public Sirius C ABI without compiling native code.

use std::path::{Path, PathBuf};

#[path = "bindgen/mod.rs"]
mod bindings;

fn main() {
    let manifest = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    println!("cargo:rerun-if-env-changed=SIRIUS_BINDGEN");
    if let Ok(mode) = std::env::var("SIRIUS_BINDGEN") {
        bindings::run(&manifest, &mode);
    }
    println!("cargo:rerun-if-env-changed=DOCS_RS");
    // Rustdoc needs only declarations; Miri tests provide Rust ABI stand-ins.
    println!("cargo:rerun-if-env-changed=CARGO_CFG_MIRI");
    if std::env::var_os("DOCS_RS").is_some() || std::env::var_os("CARGO_CFG_MIRI").is_some() {
        return;
    }

    let repo = manifest
        .join("../../..")
        .canonicalize()
        .expect("resolve repo root from CARGO_MANIFEST_DIR");
    let build_dir = std::env::var_os("SIRIUS_BUILD_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| repo.join("build/release"));
    let static_link = std::env::var_os("CARGO_FEATURE_STATIC").is_some();
    // Link the single Sirius artifact.
    let lib_dir = resolve_lib_dir(&build_dir, static_link);
    println!("cargo:rustc-link-search=native={}", lib_dir.display());
    if static_link {
        println!("cargo:rustc-link-lib=static=sirius");
        // The static engine contains C++ code even when consumers use only the C ABI.
        println!("cargo:rustc-link-lib=stdc++");
    } else {
        // The shared lib pulls its own deps in via DT_NEEDED; make them findable
        // at link time by searching the conda/CUDA lib dirs.
        if let Some(conda) = std::env::var_os("CONDA_PREFIX").map(PathBuf::from) {
            for dir in conda_lib_dirs(&conda) {
                println!("cargo:rustc-link-search=native={}", dir.display());
            }
        }
        println!("cargo:rustc-link-lib=dylib=sirius");
    }

    let artifact = lib_dir.join(if static_link {
        "libsirius.a"
    } else {
        "libsirius.so"
    });
    println!("cargo:rerun-if-changed={}", artifact.display());
    println!("cargo:rerun-if-env-changed=SIRIUS_BUILD_DIR");
    println!("cargo:rerun-if-env-changed=CONDA_PREFIX");
}

/// Directory holding the Sirius artifact to link (`libsirius.{a,so}`).
///
/// Panics with actionable guidance if nothing is found — the common cause is
/// forgetting to build Sirius first.
fn resolve_lib_dir(build_dir: &Path, static_link: bool) -> PathBuf {
    let candidates = [build_dir.to_path_buf(), build_dir.join("extension/sirius")];
    let file = if static_link {
        "libsirius.a"
    } else {
        "libsirius.so"
    };

    if let Some(dir) = candidates.iter().find(|d| d.join(file).is_file()) {
        return dir.clone();
    }

    let looked_for = if static_link {
        "libsirius.a (a bundled static archive)".to_string()
    } else {
        file.to_string()
    };
    panic!(
        "could not find {looked_for} under {} — build Sirius first (`pixi run make`), or point \
         SIRIUS_BUILD_DIR at the build tree",
        build_dir.display()
    );
}

/// conda env lib dirs (`lib`, `targets/<arch>/lib[/stubs]`) for the shared lib's
/// transitive deps.
fn conda_lib_dirs(conda: &Path) -> Vec<PathBuf> {
    let mut dirs = vec![conda.join("lib")];
    if let Ok(entries) = std::fs::read_dir(conda.join("targets")) {
        for entry in entries.flatten() {
            let lib = entry.path().join("lib");
            dirs.push(lib.join("stubs"));
            dirs.push(lib);
        }
    }
    dirs.retain(|d| d.is_dir());
    dirs
}
