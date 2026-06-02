use std::env;
use std::ffi::OsStr;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let workspace_dir = manifest_dir
        .ancestors()
        .nth(2)
        .expect("starrocks-thrift lives under experimental/starrocks/crates")
        .to_path_buf();
    let thrift_dir = workspace_dir.join("starrocks/gensrc/thrift");
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let gen_dir = out_dir.join("thrift_gen");

    if gen_dir.exists() {
        fs::remove_dir_all(&gen_dir).expect("failed to clean thrift output directory");
    }
    fs::create_dir_all(&gen_dir).expect("failed to create thrift output directory");

    println!("cargo:rerun-if-env-changed=THRIFT");
    println!("cargo:rerun-if-changed={}", thrift_dir.display());

    let thrift = find_thrift(&workspace_dir).expect(
        "could not find thrift compiler; set THRIFT or install the experimental/starrocks pixi cn env",
    );

    let thrift_files =
        collect_thrift_files(&thrift_dir).expect("failed to list StarRocks thrift files");
    for thrift_file in &thrift_files {
        println!("cargo:rerun-if-changed={}", thrift_file.display());
        let status = Command::new(&thrift)
            .arg("--gen")
            .arg("rs")
            .arg("-I")
            .arg(&thrift_dir)
            .arg("-out")
            .arg(&gen_dir)
            .arg(thrift_file)
            .status()
            .unwrap_or_else(|err| {
                panic!(
                    "failed to run thrift compiler at {}: {err}",
                    thrift.display()
                )
            });
        assert!(
            status.success(),
            "thrift compiler failed for {}",
            thrift_file.display()
        );
    }

    rewrite_inner_attributes(&gen_dir).expect("failed to normalize generated thrift attributes");
    write_module_index(&gen_dir, &out_dir.join("thrift_mods.rs"))
        .expect("failed to write generated thrift module index");
}

fn find_thrift(workspace_dir: &Path) -> Option<PathBuf> {
    if let Ok(path) = env::var("THRIFT") {
        return Some(PathBuf::from(path));
    }

    [
        workspace_dir.join(".pixi/envs/cn/bin/thrift"),
        workspace_dir.join(".pixi/envs/fe/bin/thrift"),
    ]
    .into_iter()
    .find(|candidate| candidate.exists())
    .or_else(|| find_in_path("thrift"))
}

fn find_in_path(binary: &str) -> Option<PathBuf> {
    env::var_os("PATH").and_then(|path| {
        env::split_paths(&path)
            .map(|dir| dir.join(binary))
            .find(|candidate| candidate.exists())
    })
}

fn collect_thrift_files(thrift_dir: &Path) -> io::Result<Vec<PathBuf>> {
    let mut files = fs::read_dir(thrift_dir)?
        .map(|entry| entry.map(|entry| entry.path()))
        .collect::<io::Result<Vec<_>>>()?;
    files.retain(|path| path.extension() == Some(OsStr::new("thrift")));
    files.sort();
    Ok(files)
}

fn rewrite_inner_attributes(gen_dir: &Path) -> io::Result<()> {
    for entry in fs::read_dir(gen_dir)? {
        let path = entry?.path();
        if path.extension() != Some(OsStr::new("rs")) {
            continue;
        }
        let source = fs::read_to_string(&path)?;
        let source = source
            .replace("#![allow(", "#[allow(")
            .replace(
                "#![cfg_attr(rustfmt, rustfmt_skip)]",
                "#[allow(clippy::deprecated_cfg_attr, clippy::empty_line_after_outer_attr)]\n\
                 #[cfg_attr(rustfmt, rustfmt_skip)]",
            )
            .replace("#![cfg_attr(", "#[cfg_attr(");
        fs::write(path, source)?;
    }
    Ok(())
}

fn write_module_index(gen_dir: &Path, output: &Path) -> io::Result<()> {
    let mut modules = fs::read_dir(gen_dir)?
        .map(|entry| entry.map(|entry| entry.path()))
        .collect::<io::Result<Vec<_>>>()?;
    modules.retain(|path| path.extension() == Some(OsStr::new("rs")));
    modules.sort();

    let mut index = String::new();
    for module in modules {
        let name = module
            .file_stem()
            .and_then(OsStr::to_str)
            .expect("generated thrift module has a UTF-8 stem");
        index.push_str(&format!(
            "#[allow(dead_code, unused_imports, unused_extern_crates)]\n\
             #[allow(unreachable_patterns, unused_variables)]\n\
             #[allow(non_camel_case_types, non_snake_case, non_upper_case_globals)]\n\
             #[path = \"{}\"]\n\
             pub mod {name};\n",
            module.display()
        ));
    }

    fs::write(output, index)
}
