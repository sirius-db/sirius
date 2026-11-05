//! Compile the existing C++ FFI bridge; sirius-sys supplies native library linking.

fn main() {
    println!("cargo:rerun-if-env-changed=DOCS_RS");
    if std::env::var_os("DOCS_RS").is_some() {
        return;
    }
    let repo = std::path::PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap())
        .join("../../..")
        .canonicalize()
        .expect("resolve repository root");
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
    if std::env::var_os("CARGO_FEATURE_STATIC").is_some() {
        println!("cargo:rustc-link-arg=-Wl,--allow-multiple-definition");
    }
}
