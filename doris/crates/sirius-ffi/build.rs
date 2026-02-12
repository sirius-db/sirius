fn main() {
    println!("cargo:rerun-if-changed=build.rs");

    #[cfg(feature = "sirius-engine")]
    {
        build_bridge();
    }
}

#[cfg(feature = "sirius-engine")]
fn build_bridge() {
    use std::path::PathBuf;

    let manifest_dir = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    let doris_dir = manifest_dir.parent().unwrap().parent().unwrap(); // doris/
    let sirius_root = doris_dir.parent().unwrap(); // sirius repo root
    let bridge_dir = doris_dir.join("bridge");

    // --- 1. Build the C++ bridge static library via CMake ---
    let dst = cmake::Config::new(&bridge_dir)
        .define("SIRIUS_ROOT", sirius_root.to_str().unwrap())
        .build_target("doris_bridge")
        .build();

    println!(
        "cargo:rustc-link-search=native={}",
        dst.join("build").display()
    );
    println!("cargo:rustc-link-lib=static=doris_bridge");

    // --- 2. DuckDB static library ---
    let duckdb_lib_dir = sirius_root.join("build/release/src");
    if duckdb_lib_dir.exists() {
        println!(
            "cargo:rustc-link-search=native={}",
            duckdb_lib_dir.display()
        );
        println!("cargo:rustc-link-lib=static=duckdb_static");
    }

    // --- 3. Arrow C++ library (from vcpkg) ---
    let arrow_lib_dir = sirius_root.join("vcpkg/packages/arrow_x64-linux/lib");
    if arrow_lib_dir.exists() {
        println!(
            "cargo:rustc-link-search=native={}",
            arrow_lib_dir.display()
        );
        println!("cargo:rustc-link-lib=dylib=arrow");
    }

    // --- 4. System libraries ---
    println!("cargo:rustc-link-lib=dylib=stdc++");

    // --- 5. Generate cxx bridge glue ---
    let include_dir = bridge_dir.join("include");

    cxx_build::bridge("src/lib.rs")
        .include(&include_dir)
        .std("c++20")
        .compile("sirius_ffi_cxx");

    println!("cargo:rerun-if-changed=src/lib.rs");
    println!(
        "cargo:rerun-if-changed={}",
        bridge_dir.join("src/doris_bridge.cpp").display()
    );
    println!(
        "cargo:rerun-if-changed={}",
        bridge_dir.join("include/doris_bridge.hpp").display()
    );
}
