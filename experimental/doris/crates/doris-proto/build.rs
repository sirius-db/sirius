//! Build script that generates prost messages and the tonic `PBackendService`
//! server trait from the Apache Doris protobuf IDL.
//!
//! Doris FE talks to a BE's `brpc_port` with grpc-java (gRPC over h2c), so a
//! plain tonic server implementing `PBackendService` is wire-compatible; no
//! baidu_std framing is involved on the FE→BE path.

use std::env;
use std::path::{Path, PathBuf};

/// `internal_service.proto` and its transitive imports (in `doris/gensrc/proto`).
const PROTOS: &[&str] = &[
    "internal_service.proto",
    "data.proto",
    "descriptors.proto",
    "types.proto",
    "olap_common.proto",
    "olap_file.proto",
    "runtime_profile.proto",
    "segment_v2.proto",
];

fn main() {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let workspace_dir = manifest_dir
        .ancestors()
        .nth(2)
        .expect("doris-proto lives under experimental/doris/crates")
        .to_path_buf();
    let proto_dir = workspace_dir.join("doris/gensrc/proto");
    assert!(
        proto_dir.join("internal_service.proto").exists(),
        "Doris proto IDL not found at {}; initialize the submodule: \
         git submodule update --init --depth=1 experimental/doris/doris",
        proto_dir.display()
    );

    println!("cargo:rerun-if-env-changed=PROTOC");
    if env::var_os("PROTOC").is_none()
        && let Some(protoc) = find_protoc(&workspace_dir)
    {
        // prost-build does not bundle protoc; point it at the pixi env's copy when the build
        // is not running inside `pixi run`.
        unsafe { env::set_var("PROTOC", protoc) };
    }

    let protos: Vec<PathBuf> = PROTOS.iter().map(|name| proto_dir.join(name)).collect();
    for proto in &protos {
        println!("cargo:rerun-if-changed={}", proto.display());
    }

    tonic_prost_build::configure()
        .build_server(true)
        .build_client(true)
        .compile_protos(&protos, std::slice::from_ref(&proto_dir))
        .expect("failed to compile Doris proto files");
}

/// Locates `protoc` in the pixi envs when `$PROTOC` is unset and it is not on `$PATH`.
fn find_protoc(workspace_dir: &Path) -> Option<PathBuf> {
    if find_in_path("protoc").is_some() {
        return None;
    }
    [
        workspace_dir.join(".pixi/envs/be/bin/protoc"),
        workspace_dir.join(".pixi/envs/default/bin/protoc"),
    ]
    .into_iter()
    .find(|candidate| candidate.exists())
}

/// Returns the first directory in `$PATH` containing `binary`.
fn find_in_path(binary: &str) -> Option<PathBuf> {
    env::var_os("PATH").and_then(|path| {
        env::split_paths(&path)
            .map(|dir| dir.join(binary))
            .find(|candidate| candidate.exists())
    })
}
