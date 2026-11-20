//! Build script that generates prost messages and the tonic `PBackendService`
//! server trait from the Apache Doris protobuf IDL.
//!
//! Doris FE talks to a BE's `brpc_port` with grpc-java (gRPC over h2c), so a
//! plain tonic server implementing `PBackendService` is wire-compatible; no
//! baidu_std framing is involved on the FE→BE path.

use std::env;
use std::path::PathBuf;

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

    let protos: Vec<PathBuf> = PROTOS.iter().map(|name| proto_dir.join(name)).collect();
    for proto in &protos {
        println!("cargo:rerun-if-changed={}", proto.display());
    }

    let descriptors = protox::compile(&protos, std::slice::from_ref(&proto_dir))
        .expect("failed to compile Doris proto files");
    for proto in &descriptors.file {
        if let Some(name) = proto.name.as_deref() {
            println!("cargo:rerun-if-changed={}", proto_dir.join(name).display());
        }
    }

    tonic_prost_build::configure()
        .build_server(true)
        .build_client(true)
        .compile_fds(descriptors)
        .expect("failed to generate Rust code for Doris protos");
}
