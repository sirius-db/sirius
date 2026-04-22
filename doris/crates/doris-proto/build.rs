use std::env;
use std::path::PathBuf;

fn main() {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let proto_dir = manifest_dir
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("thirdparty/apache-doris/gensrc/proto");

    assert!(
        proto_dir.exists(),
        "Doris proto directory not found at {}. Did you initialize the git submodule?",
        proto_dir.display()
    );

    // All proto files needed for PBackendService
    let protos = [
        "internal_service.proto",
        "data.proto",
        "descriptors.proto",
        "types.proto",
        "olap_common.proto",
        "olap_file.proto",
        "runtime_profile.proto",
        "segment_v2.proto",
    ];

    let proto_paths: Vec<PathBuf> = protos.iter().map(|p| proto_dir.join(p)).collect();

    for p in &proto_paths {
        println!("cargo:rerun-if-changed={}", p.display());
    }

    tonic_build::configure()
        .build_server(true)
        .build_client(false)
        .compile_protos(
            &proto_paths.iter().map(|p| p.as_path()).collect::<Vec<_>>(),
            &[proto_dir.as_path()],
        )
        .expect("Failed to compile Doris proto files");

    // bRPC baidu_std protocol metadata (for inter-BE exchange)
    let brpc_proto_dir = manifest_dir.join("proto");
    let brpc_proto = brpc_proto_dir.join("brpc_meta.proto");
    println!("cargo:rerun-if-changed={}", brpc_proto.display());

    prost_build::Config::new()
        .compile_protos(&[brpc_proto.as_path()], &[brpc_proto_dir.as_path()])
        .expect("Failed to compile bRPC meta proto");

    // NIXL GPU-direct exchange protocol (custom proto for Sirius)
    let nixl_proto = brpc_proto_dir.join("nixl_exchange.proto");
    println!("cargo:rerun-if-changed={}", nixl_proto.display());

    prost_build::Config::new()
        .compile_protos(&[nixl_proto.as_path()], &[brpc_proto_dir.as_path()])
        .expect("Failed to compile NIXL exchange proto");

    // NIXL metadata exchange service (gRPC service definition)
    let nixl_service_proto = brpc_proto_dir.join("nixl_service.proto");
    println!("cargo:rerun-if-changed={}", nixl_service_proto.display());

    tonic_build::configure()
        .build_server(true)
        .build_client(true)
        .compile_protos(&[nixl_service_proto.as_path()], &[brpc_proto_dir.as_path()])
        .expect("Failed to compile NIXL service proto");
}
