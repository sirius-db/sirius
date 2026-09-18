//! Sirius as an Apache Doris backend: a Rust process that speaks the Doris FE→BE
//! protocol and runs the plan fragments it receives on the embedded Sirius engine.
//!
//! The FE is the unmodified official Doris release; this crate is the whole BE side
//! (`ALTER SYSTEM ADD BACKEND` self-registration, thrift `HeartbeatService` /
//! `BackendService` on the heartbeat and BE ports, and the gRPC `PBackendService`
//! on the brpc port). It follows `experimental/starrocks` structurally: a synchronous
//! [`FragmentExecutor`] seam lets the whole protocol shell build and test without an
//! engine (`--no-default-features`), which is what CI and a Mac run.

use std::{fmt, str::FromStr};

use doris_thrift::status::{TStatus, TStatusCode};

mod backend_service;
#[cfg(feature = "sirius-engine")]
mod engine;
mod file_schema;
mod fragment_executor;
mod node;
mod params;
mod result_encoder;
mod result_store;

pub use backend_service::{SiriusBackendService, serve_backend_service};
#[cfg(feature = "sirius-engine")]
pub use engine::SiriusEngine;
pub use fragment_executor::{FragmentExecutor, FragmentResult, StubExecutor};
pub use node::{
    BackendHeartbeatHandler, BackendServer, BackendServiceHandler, HeartbeatServer,
    HeartbeatStateSnapshot, SharedHeartbeatState, ThriftServer, ThriftServerShutdown,
    register_node, start_backend_server, start_heartbeat_server,
};
pub use params::{DispatchedFragment, FragmentBatch, decode_fragment_params_list};

/// Non-empty host name or IP literal.
#[derive(Clone, Eq, PartialEq, Hash)]
pub struct Host(String);

impl Host {
    pub fn new(value: impl Into<String>) -> std::result::Result<Self, HostParseError> {
        let value = value.into();
        if value.trim().is_empty() {
            Err(HostParseError)
        } else {
            Ok(Self(value))
        }
    }

    pub fn local() -> Self {
        Self("127.0.0.1".to_string())
    }

    pub fn unspecified() -> Self {
        Self("0.0.0.0".to_string())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Debug for Host {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.debug_tuple("Host").field(&self.0).finish()
    }
}

impl fmt::Display for Host {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl FromStr for Host {
    type Err = HostParseError;

    fn from_str(value: &str) -> std::result::Result<Self, Self::Err> {
        Self::new(value)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, thiserror::Error)]
#[error("host must not be empty")]
pub struct HostParseError;

/// String whose `Debug` output is redacted (FE password, FE-issued tokens).
#[derive(Clone, Default, Eq, PartialEq)]
pub struct SecretString(String);

impl SecretString {
    pub fn new(value: impl Into<String>) -> Self {
        Self(value.into())
    }

    pub fn expose_secret(&self) -> &str {
        &self.0
    }
}

impl fmt::Debug for SecretString {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("SecretString(<redacted>)")
    }
}

impl FromStr for SecretString {
    type Err = String;

    fn from_str(value: &str) -> std::result::Result<Self, Self::Err> {
        Ok(Self::new(value))
    }
}

/// Listener and advertised-identity settings of this backend.
///
/// The port defaults are the stock Doris BE ports so a stock `fe.conf` and the usual
/// `mysql`/`curl` incantations work unchanged.
#[derive(Clone, Debug, clap::Args)]
pub struct BackendConfig {
    /// Address the thrift/gRPC listeners bind to.
    #[arg(long, default_value = "0.0.0.0")]
    pub bind_host: Host,
    /// Host the FE registers and heartbeats this backend under (`ALTER SYSTEM ADD BACKEND
    /// "<advertise_host>:<heartbeat_port>"`); must be reachable from the FE.
    #[arg(long, default_value = "127.0.0.1")]
    pub advertise_host: Host,
    /// Thrift `HeartbeatService` port (Doris default 9050).
    #[arg(long, default_value_t = 9050)]
    pub heartbeat_port: u16,
    /// Thrift `BackendService` port, advertised as `be_port` (Doris default 9060).
    #[arg(long = "be-port", default_value_t = 9060)]
    pub be_port: u16,
    /// Advertised HTTP port (Doris default 8040). Nothing listens there yet.
    #[arg(long, default_value_t = 8040)]
    pub http_port: u16,
    /// gRPC `PBackendService` port, advertised as `brpc_port` (Doris default 8060).
    #[arg(long, default_value_t = 8060)]
    pub brpc_port: u16,
    /// Version string reported in heartbeats (FE displays it, never compares it).
    #[arg(skip = default_backend_version())]
    pub version: String,
}

impl Default for BackendConfig {
    fn default() -> Self {
        Self {
            bind_host: Host::unspecified(),
            advertise_host: Host::local(),
            heartbeat_port: 9050,
            be_port: 9060,
            http_port: 8040,
            brpc_port: 8060,
            version: default_backend_version(),
        }
    }
}

fn default_backend_version() -> String {
    format!("{}/{}", env!("CARGO_PKG_NAME"), env!("CARGO_PKG_VERSION"))
}

/// Doris FE connection settings (MySQL protocol on `query_port`, for registration).
#[derive(Clone, Debug, clap::Args)]
pub struct FeConfig {
    #[arg(long = "fe-host", default_value = "127.0.0.1")]
    pub host: Host,
    #[arg(long = "fe-query-port", default_value_t = 9030)]
    pub query_port: u16,
    #[arg(long = "fe-user", default_value = "root")]
    pub user: String,
    #[arg(long = "fe-password", default_value = "")]
    pub password: SecretString,
}

impl Default for FeConfig {
    fn default() -> Self {
        Self {
            host: Host::local(),
            query_port: 9030,
            user: "root".to_string(),
            password: SecretString::default(),
        }
    }
}

/// Doris success status helper.
pub(crate) fn ok_status() -> TStatus {
    TStatus::new(TStatusCode::OK, None)
}

/// Doris unsupported status helper that names the RPC for operator diagnostics.
pub(crate) fn not_implemented_status(rpc: &str) -> TStatus {
    TStatus::new(
        TStatusCode::NOT_IMPLEMENTED_ERROR,
        Some(vec![format!("{rpc} is not implemented")]),
    )
}

/// Generic failure status helper.
pub(crate) fn error_status(message: String) -> TStatus {
    TStatus::new(TStatusCode::INTERNAL_ERROR, Some(vec![message]))
}
