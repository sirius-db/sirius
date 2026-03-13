fn main() {
    // Link against libcuda (CUDA driver API) for GPU memory allocation.
    // On NixOS, libcuda.so lives in /run/opengl-driver/lib/.
    println!("cargo:rustc-link-search=native=/run/opengl-driver/lib");

    // CUDA toolkit stubs (conda env provides libcuda.so stub for linking)
    if let Ok(prefix) = std::env::var("CONDA_PREFIX") {
        println!(
            "cargo:rustc-link-search=native={}/targets/x86_64-linux/lib/stubs",
            prefix
        );
    }

    println!("cargo:rustc-link-lib=dylib=cuda");
}
