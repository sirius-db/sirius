fn main() {
    // Link against libcuda (CUDA driver API) for GPU memory allocation.
    // On NixOS, libcuda.so lives in /run/opengl-driver/lib/.
    println!("cargo:rustc-link-search=native=/run/opengl-driver/lib");

    // CUDA toolkit stubs (conda env provides libcuda.so stub for linking).
    // sbsa-linux is the CUDA target dir for aarch64 SBSA (data center ARM, e.g. Grace Hopper).
    // x86_64-linux is used for standard x86_64 systems.
    if let Ok(prefix) = std::env::var("CONDA_PREFIX") {
        let cuda_target = if cfg!(target_arch = "aarch64") {
            "sbsa-linux"
        } else {
            "x86_64-linux"
        };
        println!(
            "cargo:rustc-link-search=native={}/targets/{}/lib/stubs",
            prefix, cuda_target
        );
    }

    println!("cargo:rustc-link-lib=dylib=cuda");
}
