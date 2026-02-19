fn main() {
    // Link against libcuda (CUDA driver API) for GPU memory allocation.
    // On NixOS, libcuda.so lives in /run/opengl-driver/lib/.
    println!("cargo:rustc-link-search=native=/run/opengl-driver/lib");
    println!("cargo:rustc-link-lib=dylib=cuda");
}
