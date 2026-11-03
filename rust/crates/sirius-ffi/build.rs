//! Compile the C++ FFI bridge and link the Sirius library.

use std::path::{Path, PathBuf};

fn main() {
    println!("cargo:rerun-if-env-changed=DOCS_RS");
    // Rustdoc needs the Rust declarations, not native compilation or linking.
    if std::env::var_os("DOCS_RS").is_some() {
        return;
    }

    let manifest = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    let repo = manifest
        .join("../../..")
        .canonicalize()
        .expect("resolve repo root from CARGO_MANIFEST_DIR");
    cxx_build::bridge("src/bridge.rs")
        .file("src/shim.cpp")
        .std("c++20")
        .include(repo.join("include"))
        .compile("sirius_ffi");
    for path in ["src/bridge.rs", "src/shim.hpp", "src/shim.cpp"] {
        println!("cargo:rerun-if-changed={path}");
    }
    println!(
        "cargo:rerun-if-changed={}",
        repo.join("include/sirius/ffi.hpp").display()
    );
    let build_dir = std::env::var_os("SIRIUS_BUILD_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| repo.join("build/release"));
    let static_link = std::env::var_os("CARGO_FEATURE_STATIC").is_some();
    // Link the single Sirius artifact.
    let lib_dir = resolve_lib_dir(&build_dir, static_link);
    println!("cargo:rustc-link-search=native={}", lib_dir.display());
    if static_link {
        println!("cargo:rustc-link-lib=static=sirius");
        // The static engine requires the C++ runtime.
        println!("cargo:rustc-link-lib=stdc++");
        println!("cargo:rustc-link-arg=-Wl,--allow-multiple-definition");
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
