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
}
