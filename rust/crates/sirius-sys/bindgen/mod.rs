//! Opt-in maintenance of the checked-in C declarations.

use std::{env, fs, path::Path, process::Command};

const TARGETS: [&str; 3] = [
    "x86_64-unknown-linux-gnu",
    "aarch64-unknown-linux-gnu",
    "aarch64-apple-darwin",
];

pub fn run(manifest: &Path, mode: &str) {
    assert!(
        matches!(mode, "write" | "check"),
        "SIRIUS_BINDGEN must be write or check"
    );
    let repo = manifest.join("../../..");
    let headers = repo.join("include/sirius-c");
    let output = manifest.join("src/generated.rs");
    println!(
        "cargo:rerun-if-changed={}",
        repo.join("pixi.lock").display()
    );
    println!("cargo:rerun-if-changed={}", headers.display());
    println!("cargo:rerun-if-changed={}", output.display());
    println!("cargo:rerun-if-changed=bindgen/wrapper.h");
    println!("cargo:rerun-if-env-changed=CONDA_PREFIX");

    let prefix =
        env::var_os("CONDA_PREFIX").expect("run binding maintenance through pixi run -e bindings");
    let prefix = Path::new(&prefix);
    let contents = generate(&repo, prefix, TARGETS[0]);
    for target in &TARGETS[1..] {
        assert_eq!(
            generate(&repo, prefix, target),
            contents,
            "bindings differ for {target}; target-specific output needs review"
        );
    }
    let current = match fs::read_to_string(&output) {
        Ok(contents) => contents,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => String::new(),
        Err(error) => panic!("read {}: {error}", output.display()),
    };
    if mode == "check" {
        assert!(
            current == contents,
            "Rust bindings are stale; run pixi run --locked -e bindings generate-bindings"
        );
    } else if current != contents {
        fs::write(output, contents).expect("write generated bindings");
    }
}

fn generate(repo: &Path, prefix: &Path, target: &str) -> String {
    let mut command = Command::new(prefix.join("bin/bindgen"));
    // Do not allow shell overrides to replace the pinned toolchain or headers.
    for (key, _) in env::vars_os() {
        if key
            .to_string_lossy()
            .starts_with("BINDGEN_EXTRA_CLANG_ARGS")
        {
            command.env_remove(key);
        }
    }
    for key in [
        "CPATH",
        "C_INCLUDE_PATH",
        "CPLUS_INCLUDE_PATH",
        "OBJC_INCLUDE_PATH",
    ] {
        command.env_remove(key);
    }
    let result = command
        .current_dir(repo)
        .env("LIBCLANG_PATH", prefix.join("lib"))
        .env("CLANG_PATH", prefix.join("bin/clang"))
        .args([
            "rust/crates/sirius-sys/bindgen/wrapper.h",
            "--allowlist-function",
            "sirius_.*",
            "--allowlist-type",
            "sirius_.*",
            "--allowlist-var",
            "SIRIUS_.*",
            // Opaque markers remain handwritten in c_api.rs.
            "--blocklist-type",
            "sirius_(context_config|context_config_builder|error)",
            "--raw-line",
            "use super::{sirius_context_config, sirius_context_config_builder, sirius_error};",
            "--no-derive-copy",
            "--no-derive-debug",
            "--merge-extern-blocks",
            "--use-core",
            "--rust-target",
            "1.85",
            "--rust-edition",
            "2024",
            "--formatter",
            "prettyplease",
            "--",
            "-Iinclude",
            "-xc",
            "-std=c99",
            // Use Clang's integer/pointer headers, not the host libc sysroot.
            "-ffreestanding",
        ])
        .arg(format!("--target={target}"))
        .output()
        .expect("run pinned bindgen from the bindings Pixi environment");
    assert!(
        result.status.success(),
        "bindgen failed: {}",
        String::from_utf8_lossy(&result.stderr)
    );
    if !result.stderr.is_empty() {
        eprint!("{}", String::from_utf8_lossy(&result.stderr));
    }
    let contents = String::from_utf8(result.stdout).expect("bindgen emits UTF-8");
    let compact: String = contents.chars().filter(|c| !c.is_whitespace()).collect();
    assert!(
        !compact.contains("_unused:[u8;0]"),
        "new opaque handle: add its markers in c_api.rs and blocklist it here"
    );
    contents
        .replace("///<", "///")
        .replace("@code{.c}", "```c")
        .replace("@endcode", "```")
}
