use std::{
    env, fs,
    io::{self, ErrorKind},
    path::{Path, PathBuf},
    process::Command,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR")?);
    let workspace_dir = manifest_dir
        .parent()
        .ok_or_else(|| io::Error::other("cn crate has workspace parent"))?
        .to_path_buf();
    let thrift_dir = manifest_dir.join("../starrocks/gensrc/thrift");
    let heartbeat_thrift = thrift_dir.join("HeartbeatService.thrift");
    let out_dir = PathBuf::from(env::var("OUT_DIR")?);

    for file in [
        "HeartbeatService.thrift",
        "Status.thrift",
        "StatusCode.thrift",
        "Types.thrift",
    ] {
        println!("cargo:rerun-if-changed={}", thrift_dir.join(file).display());
    }
    println!("cargo:rerun-if-env-changed=THRIFT");

    let mut candidates = Vec::new();
    if let Ok(thrift) = env::var("THRIFT") {
        candidates.push(PathBuf::from(thrift));
    }
    candidates.push(PathBuf::from("thrift"));
    candidates.push(workspace_dir.join(".pixi/envs/cn/bin/thrift"));
    candidates.push(workspace_dir.join(".pixi/envs/fe/bin/thrift"));

    for thrift in candidates {
        match run_thrift(&thrift, &thrift_dir, &out_dir, &heartbeat_thrift) {
            Ok(()) => return Ok(()),
            Err(err) if err.kind() == ErrorKind::NotFound => continue,
            Err(err) => {
                return Err(io::Error::other(format!(
                    "failed to run thrift compiler `{}`: {err}",
                    thrift.display()
                ))
                .into());
            }
        }
    }

    Err(io::Error::new(
        ErrorKind::NotFound,
        "failed to find thrift compiler; install thrift-compiler or run through `pixi run -e cn`",
    )
    .into())
}

fn run_thrift(
    thrift: &Path,
    thrift_dir: &Path,
    out_dir: &Path,
    heartbeat_thrift: &Path,
) -> std::io::Result<()> {
    let status = Command::new(thrift)
        .arg("-r")
        .arg("--gen")
        .arg("rs")
        .arg("-I")
        .arg(thrift_dir)
        .arg("-out")
        .arg(out_dir)
        .arg(heartbeat_thrift)
        .status()?;

    if !status.success() {
        return Err(io::Error::other(format!(
            "thrift compiler `{}` failed for {} with status {status}",
            thrift.display(),
            heartbeat_thrift.display()
        )));
    }

    write_generated_module_index(out_dir)?;
    Ok(())
}

fn write_generated_module_index(out_dir: &Path) -> std::io::Result<()> {
    let mut modules = Vec::new();
    for entry in fs::read_dir(out_dir)? {
        let path = entry?.path();
        if path.extension().and_then(|extension| extension.to_str()) != Some("rs")
            || path.file_name().and_then(|name| name.to_str()) == Some("generated.rs")
        {
            continue;
        }

        let module = path
            .file_stem()
            .and_then(|stem| stem.to_str())
            .ok_or_else(|| {
                io::Error::other(format!("invalid Rust module path {}", path.display()))
            })?
            .to_string();
        modules.push((module, path));
    }
    modules.sort_by(|left, right| left.0.cmp(&right.0));

    if modules.is_empty() {
        return Err(io::Error::other(format!(
            "thrift compiler did not generate Rust modules in {}",
            out_dir.display()
        )));
    }

    let mut source = String::new();
    for (module, path) in modules {
        source.push_str("#[allow(dead_code, unused_imports, unused_extern_crates)]\n");
        source.push_str(&format!(
            "#[path = {:?}]\npub mod {module};\n\n",
            path.display().to_string()
        ));
    }

    fs::write(out_dir.join("generated.rs"), source)
}
