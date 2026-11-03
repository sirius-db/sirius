//! Maintainer-only generation of the checked-in C declarations.

use std::{fs, path::Path};

pub fn run(manifest: &Path) {
    let repo = manifest.join("../../..");
    let output = manifest.join("src/generated.rs");
    for path in [
        repo.join("pixi.lock"),
        repo.join("include/sirius-c"),
        output.clone(),
    ] {
        println!("cargo:rerun-if-changed={}", path.display());
    }
    println!("cargo:rerun-if-changed=bindgen/wrapper.h");
    println!("cargo:rerun-if-env-changed=LIBCLANG_PATH");
    let builder = bindgen::Builder::default()
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .header(manifest.join("bindgen/wrapper.h").to_str().unwrap())
        .clang_arg(format!("-I{}", repo.join("include").display()))
        .clang_args(["-xc", "-std=c99", "-ffreestanding"])
        .allowlist_function("sirius_.*")
        .allowlist_type("sirius_.*")
        .allowlist_var("SIRIUS_.*")
        // Preserve the handwritten opaque handles' thread-safety and pinning markers.
        .blocklist_type("sirius_(context_config|context_config_builder|error)")
        .raw_line(
            "use super::{sirius_context_config, sirius_context_config_builder, sirius_error};",
        )
        .derive_copy(false)
        .derive_debug(false)
        .merge_extern_blocks(true)
        .use_core()
        .rust_target(bindgen::RustTarget::stable(85, 0).unwrap())
        .rust_edition(bindgen::RustEdition::Edition2024)
        .formatter(bindgen::Formatter::Prettyplease);
    let generate = |target: &str| {
        builder
            .clone()
            .clang_arg(format!("--target={target}"))
            .generate()
            .expect("generate C ABI bindings")
            .to_string()
    };
    let contents = generate("x86_64-unknown-linux-gnu");
    assert_eq!(
        contents,
        generate("aarch64-unknown-linux-gnu"),
        "Linux amd64 and arm64 declarations differ"
    );
    let compact: String = contents.chars().filter(|c| !c.is_whitespace()).collect();
    assert!(
        !compact.contains("_unused:[u8;0]"),
        "new opaque handle: add its markers in c_api.rs and blocklist it here"
    );
    let contents = contents
        .replace("///<", "///")
        .replace("@code{.c}", "```c")
        .replace("@endcode", "```");
    if fs::read_to_string(&output).ok().as_deref() != Some(&contents) {
        fs::write(output, contents).expect("write generated bindings");
    }
}
