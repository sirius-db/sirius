/// Generated metadata messages from the Apache BRPC protobuf definitions.
#[allow(clippy::all, dead_code)]
pub mod brpc {
    include!(concat!(env!("OUT_DIR"), "/brpc.rs"));

    /// Generated messages for the Baidu PRPC/BRPC frame envelope.
    #[allow(clippy::all, dead_code)]
    pub mod policy {
        include!(concat!(env!("OUT_DIR"), "/brpc.policy.rs"));
    }
}

/// Generated StarRocks protobuf messages plus the generated BRPC service facade.
#[allow(clippy::all, dead_code)]
pub mod starrocks {
    include!(concat!(env!("OUT_DIR"), "/starrocks.rs"));
}
