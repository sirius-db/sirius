//! Build script that generates Rust bindings from the Apache Doris Thrift IDL.
//!
//! It runs the Thrift compiler over `doris/gensrc/thrift` (the pinned Doris
//! submodule), normalizes the generated attributes, and writes a module index
//! that `src/lib.rs` includes.

use std::env;
use std::ffi::OsStr;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::process::Command;

/// Generates Rust modules for every Doris Thrift file into `OUT_DIR`.
fn main() {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let workspace_dir = manifest_dir
        .ancestors()
        .nth(2)
        .expect("doris-thrift lives under experimental/doris/crates")
        .to_path_buf();
    let thrift_dir = workspace_dir.join("doris/gensrc/thrift");
    assert!(
        thrift_dir.join("PaloInternalService.thrift").exists(),
        "Doris thrift IDL not found at {}; initialize the submodule: \
         git submodule update --init --depth=1 experimental/doris/doris",
        thrift_dir.display()
    );
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let gen_dir = out_dir.join("thrift_gen");

    if gen_dir.exists() {
        fs::remove_dir_all(&gen_dir).expect("failed to clean thrift output directory");
    }
    fs::create_dir_all(&gen_dir).expect("failed to create thrift output directory");

    println!("cargo:rerun-if-env-changed=THRIFT");
    println!("cargo:rerun-if-changed={}", thrift_dir.display());

    let thrift = find_thrift(&workspace_dir).expect(
        "could not find thrift compiler; set THRIFT or install the experimental/doris pixi be env",
    );

    let thrift_files =
        collect_thrift_files(&thrift_dir).expect("failed to list Doris thrift files");
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
    derive_default_where_possible(&gen_dir).expect("failed to add Default derives");
    name_enum_values(&gen_dir).expect("failed to add enum Debug/Display impls");
    write_module_index(&gen_dir, &out_dir.join("thrift_mods.rs"))
        .expect("failed to write generated thrift module index");
}

/// Adds `Default` to the derive list of every generated struct whose fields all implement
/// `Default`, and to every generated enum newtype (`pub struct TX(pub i32)`).
///
/// The generator only derives `Default` for structs made of optional fields; a struct with a
/// required field gets a positional `new(..)` taking every field (58 arguments for
/// `TPlanNode`). Fixtures written against `new(..)` are unreadable and break on every IDL
/// change, so this pass lets tests write `TPlanNode { node_id, .., ..Default::default() }`.
/// A struct is left alone when a required field is a thrift union (generated as a Rust enum
/// with data-carrying variants, which cannot derive `Default`) or transitively contains one.
fn derive_default_where_possible(gen_dir: &Path) -> io::Result<()> {
    let files = {
        let mut files = fs::read_dir(gen_dir)?
            .map(|entry| entry.map(|entry| entry.path()))
            .collect::<io::Result<Vec<_>>>()?;
        files.retain(|path| path.extension() == Some(OsStr::new("rs")));
        files.sort();
        files
    };
    let sources = files
        .iter()
        .map(|path| fs::read_to_string(path).map(|source| (path.clone(), source)))
        .collect::<io::Result<Vec<_>>>()?;

    let mut model = GeneratedTypes::default();
    for (_, source) in &sources {
        model.scan(source);
    }
    let capable = model.default_capable_structs();

    for (path, source) in &sources {
        let rewritten = add_default_derives(source, &capable, &model.newtypes);
        if rewritten != *source {
            fs::write(path, rewritten)?;
        }
    }
    Ok(())
}

/// Type-level facts scraped from the generated sources by line shape.
#[derive(Default)]
struct GeneratedTypes {
    /// `pub type A = B;` aliases, module qualifiers stripped.
    typedefs: std::collections::HashMap<String, String>,
    /// `pub struct X(pub i32);` enum newtypes.
    newtypes: std::collections::HashSet<String>,
    /// `pub enum X {` thrift unions.
    unions: std::collections::HashSet<String>,
    /// `pub struct X {` → its field types (module qualifiers stripped).
    structs: std::collections::HashMap<String, Vec<String>>,
}

impl GeneratedTypes {
    /// Scrapes typedefs, enum newtypes, unions, and struct field types from one generated file.
    fn scan(&mut self, source: &str) {
        let mut current: Option<String> = None;
        for line in source.lines() {
            if let Some(rest) = line.strip_prefix("pub type ")
                && let Some((name, target)) = rest.split_once(" = ")
            {
                self.typedefs.insert(
                    name.trim().to_string(),
                    strip_qualifiers(target.trim().trim_end_matches(';')),
                );
            } else if let Some(rest) = line.strip_prefix("pub struct ")
                && let Some(name) = rest.strip_suffix("(pub i32);")
            {
                self.newtypes.insert(name.trim().to_string());
            } else if let Some(rest) = line.strip_prefix("pub enum ")
                && let Some(name) = rest.strip_suffix(" {")
            {
                self.unions.insert(name.trim().to_string());
            } else if let Some(rest) = line.strip_prefix("pub struct ")
                && let Some(name) = rest.strip_suffix(" {")
            {
                current = Some(name.trim().to_string());
                self.structs.insert(name.trim().to_string(), Vec::new());
            } else if line == "}" {
                current = None;
            } else if let Some(name) = &current
                && let Some(rest) = line.strip_prefix("  pub ")
                && let Some((_, ty)) = rest.split_once(": ")
            {
                self.structs
                    .get_mut(name)
                    .expect("struct registered when its header was seen")
                    .push(strip_qualifiers(ty.trim_end_matches(',')));
            }
        }
    }

    /// Fixpoint: start optimistic, then drop every struct with a field that cannot be
    /// `Default` (a union, or a struct already dropped) until nothing changes.
    fn default_capable_structs(&self) -> std::collections::HashSet<String> {
        let mut capable: std::collections::HashSet<String> = self.structs.keys().cloned().collect();
        loop {
            let dropped: Vec<String> = capable
                .iter()
                .filter(|name| {
                    !self.structs[*name]
                        .iter()
                        .all(|ty| self.type_is_default(ty, &capable))
                })
                .cloned()
                .collect();
            if dropped.is_empty() {
                return capable;
            }
            for name in dropped {
                capable.remove(&name);
            }
        }
    }

    /// Whether a (qualifier-stripped) field type implements `Default` given the current set of
    /// capable structs.
    fn type_is_default(&self, ty: &str, capable: &std::collections::HashSet<String>) -> bool {
        if ty.starts_with("Option<")
            || ty.starts_with("Vec<")
            || ty.starts_with("BTreeMap<")
            || ty.starts_with("BTreeSet<")
        {
            return true;
        }
        if let Some(inner) = ty
            .strip_prefix("Box<")
            .and_then(|rest| rest.strip_suffix('>'))
        {
            return self.type_is_default(inner, capable);
        }
        match ty {
            "bool" | "i8" | "i16" | "i32" | "i64" | "f64" | "String" | "OrderedFloat<f64>" => true,
            _ if self.newtypes.contains(ty) => true,
            _ if self.unions.contains(ty) => false,
            _ if capable.contains(ty) => true,
            _ => match self.typedefs.get(ty) {
                Some(target) => self.type_is_default(target, capable),
                None => false,
            },
        }
    }
}

/// Replaces the derived `Debug` of every enum newtype (`pub struct TX(pub i32)`) with one that
/// prints the enum member's name, and adds a matching `Display`.
///
/// The generator models a thrift enum as a newtype over `i32` with one associated const per
/// member, so `{:?}` prints `TPlanNodeType(29)`; fragment dumps and translator errors are
/// unreadable that way. Unknown values still print as `TPlanNodeType(29)`.
fn name_enum_values(gen_dir: &Path) -> io::Result<()> {
    for entry in fs::read_dir(gen_dir)? {
        let path = entry?.path();
        if path.extension() != Some(OsStr::new("rs")) {
            continue;
        }
        let source = fs::read_to_string(&path)?;
        let lines: Vec<&str> = source.lines().collect();

        // Pass 1: collect `pub const NAME: TX = TX(N);` members per enum newtype.
        let mut members: std::collections::BTreeMap<String, Vec<(String, String)>> =
            std::collections::BTreeMap::new();
        let mut current: Option<String> = None;
        for line in &lines {
            if let Some(rest) = line.strip_prefix("pub struct ")
                && let Some(name) = rest.strip_suffix("(pub i32);")
            {
                members.entry(name.trim().to_string()).or_default();
            } else if let Some(rest) = line.strip_prefix("impl ")
                && let Some(name) = rest.strip_suffix(" {")
                && members.contains_key(name.trim())
            {
                current = Some(name.trim().to_string());
            } else if *line == "}" {
                current = None;
            } else if let Some(name) = &current
                && let Some(rest) = line.trim_start().strip_prefix("pub const ")
                && let Some((member, value)) = rest.split_once(": ")
                && member != "ENUM_VALUES"
                && let Some(value) = value
                    .trim_end_matches(';')
                    .rsplit_once('(')
                    .and_then(|(_, tail)| tail.strip_suffix(')'))
            {
                members
                    .get_mut(name)
                    .expect("enum registered")
                    .push((member.to_string(), value.to_string()));
            }
        }
        if members.is_empty() {
            continue;
        }

        // Pass 2: drop `Debug` from each newtype's derive list and append the impls.
        let mut out = String::with_capacity(source.len() + 4096);
        for (idx, line) in lines.iter().enumerate() {
            let is_newtype_derive = lines.get(idx + 1).is_some_and(|next| {
                next.strip_prefix("pub struct ")
                    .and_then(|rest| rest.strip_suffix("(pub i32);"))
                    .is_some_and(|name| members.contains_key(name.trim()))
            });
            if is_newtype_derive && line.starts_with("#[derive(") {
                out.push_str(&line.replace("Debug, ", ""));
            } else {
                out.push_str(line);
            }
            out.push('\n');
        }
        for (name, values) in &members {
            let mut arms = String::new();
            let mut seen = std::collections::HashSet::new();
            for (member, value) in values {
                // Thrift allows aliases (two members, one value); the first name wins.
                if seen.insert(value.clone()) {
                    arms.push_str(&format!("      {value} => f.write_str(\"{member}\"),\n"));
                }
            }
            out.push_str(&format!(
                "\nimpl ::std::fmt::Debug for {name} {{\n\
                 \x20 fn fmt(&self, f: &mut ::std::fmt::Formatter<'_>) -> ::std::fmt::Result {{\n\
                 \x20   match self.0 {{\n{arms}\
                 \x20     other => write!(f, \"{name}({{other}})\"),\n\
                 \x20   }}\n\
                 \x20 }}\n\
                 }}\n\
                 \n\
                 impl ::std::fmt::Display for {name} {{\n\
                 \x20 fn fmt(&self, f: &mut ::std::fmt::Formatter<'_>) -> ::std::fmt::Result {{\n\
                 \x20   ::std::fmt::Debug::fmt(self, f)\n\
                 \x20 }}\n\
                 }}\n"
            ));
        }
        fs::write(&path, out)?;
    }
    Ok(())
}

/// Removes `module::` qualifiers anywhere in a type string (`Vec<types::TTupleId>` → `Vec<TTupleId>`).
fn strip_qualifiers(ty: &str) -> String {
    let mut out = String::with_capacity(ty.len());
    let mut ident = String::new();
    let mut chars = ty.chars().peekable();
    while let Some(c) = chars.next() {
        if c.is_alphanumeric() || c == '_' {
            ident.push(c);
        } else if c == ':' && chars.peek() == Some(&':') {
            chars.next();
            ident.clear();
        } else {
            out.push_str(&ident);
            ident.clear();
            out.push(c);
        }
    }
    out.push_str(&ident);
    out
}

/// Inserts `Default` into the derive line preceding each capable struct / enum newtype.
fn add_default_derives(
    source: &str,
    capable: &std::collections::HashSet<String>,
    newtypes: &std::collections::HashSet<String>,
) -> String {
    let lines: Vec<&str> = source.lines().collect();
    let mut out = String::with_capacity(source.len() + 1024);
    for (idx, line) in lines.iter().enumerate() {
        let wants_default = lines.get(idx + 1).is_some_and(|next| {
            next.strip_prefix("pub struct ").is_some_and(|rest| {
                rest.strip_suffix(" {")
                    .is_some_and(|name| capable.contains(name.trim()))
                    || rest
                        .strip_suffix("(pub i32);")
                        .is_some_and(|name| newtypes.contains(name.trim()))
            })
        });
        if wants_default
            && let Some(derives) = line.strip_prefix("#[derive(")
            && !derives.contains("Default")
        {
            out.push_str("#[derive(Default, ");
            out.push_str(derives);
        } else {
            out.push_str(line);
        }
        out.push('\n');
    }
    out
}

/// Locates the Thrift compiler via `$THRIFT`, the pixi envs, then `$PATH`.
fn find_thrift(workspace_dir: &Path) -> Option<PathBuf> {
    if let Ok(path) = env::var("THRIFT") {
        return Some(PathBuf::from(path));
    }

    [
        workspace_dir.join(".pixi/envs/be/bin/thrift"),
        workspace_dir.join(".pixi/envs/default/bin/thrift"),
    ]
    .into_iter()
    .find(|candidate| candidate.exists())
    .or_else(|| find_in_path("thrift"))
}

/// Returns the first directory in `$PATH` containing `binary`.
fn find_in_path(binary: &str) -> Option<PathBuf> {
    env::var_os("PATH").and_then(|path| {
        env::split_paths(&path)
            .map(|dir| dir.join(binary))
            .find(|candidate| candidate.exists())
    })
}

/// Returns the sorted set of `*.thrift` files in `thrift_dir`.
fn collect_thrift_files(thrift_dir: &Path) -> io::Result<Vec<PathBuf>> {
    let mut files = fs::read_dir(thrift_dir)?
        .map(|entry| entry.map(|entry| entry.path()))
        .collect::<io::Result<Vec<_>>>()?;
    files.retain(|path| path.extension() == Some(OsStr::new("thrift")));
    files.sort();
    Ok(files)
}

/// Rewrites the generated files' inner attributes (`#![..]`) into outer ones so
/// each can be included as a module via `#[path]`.
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

/// Writes a module index declaring every generated file as a `pub mod`, with the
/// lint allowances the generated code needs.
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
        // `unreachable_code`: the generator emits `unimplemented!()` for complex constants
        // (`MetricDefs.thrift`'s const maps), which trips the lint on every following entry.
        index.push_str(&format!(
            "#[allow(dead_code, unused_imports, unused_extern_crates)]\n\
             #[allow(unreachable_patterns, unreachable_code, unused_variables)]\n\
             #[allow(non_camel_case_types, non_snake_case, non_upper_case_globals)]\n\
             #[path = \"{}\"]\n\
             pub mod {name};\n",
            module.display()
        ));
    }

    fs::write(output, index)
}
