//! Blocking PRPC client for CN→CN calls.
//!
//! The PRPC framing is symmetric, so the client is small: connect to the peer, write a request
//! frame, and match the reply by correlation id. One request is in flight at a time — the nixl
//! transport thread is the only caller — so a plain cached [`TcpStream`] per client is enough.
//! brpc-level failures (unknown method, undecodable body) surface as `Err` with the peer's error
//! code; method-level StarRocks failures stay in the response body's `StatusPB` for the caller.

// The nixl transport tier is the only production caller; the client still compiles (and its
// tests run) in every build so CI catches breakage without a libnixl install.
#![cfg_attr(not(feature = "nixl-transport"), allow(dead_code))]

use std::io::Write;
use std::net::TcpStream;
use std::time::Duration;

use tracing::{info, warn};

use crate::proto::starrocks::p_internal_service_brpc::SERVICE_NAME;
use crate::prpc;

/// Bound on connect and on waiting for a reply. Generous because a peer's `request_staging_lease`
/// queues behind whatever fragment its engine thread is currently running; a peer that exceeds
/// this is treated as wedged and the query fails loudly.
///
/// Tunable, and worth tuning at scale — see [`SIRIUS_CN_RPC_TIMEOUT_SECS`](crate::tunable).
/// Read per dial rather than cached in the client: `Tunables::get` is an atomic load plus a
/// small clone, dials are rare, and reading through keeps a single source of truth.
fn reply_timeout() -> Duration {
    crate::tunable::Tunables::get().rpc_timeout
}

/// One cached connection to a peer CN's brpc port.
#[derive(Debug)]
pub(crate) struct PrpcClient {
    /// Peer endpoint, `host:brpc_port`.
    peer: String,
    /// Cached connection; dropped on any transport failure and re-dialed on the next call.
    connection: Option<TcpStream>,
    /// Correlation id of the next request frame.
    next_correlation_id: i64,
}

/// Transport failures (retryable by reconnecting) kept apart from brpc-level rejections
/// (retrying those would just repeat the rejection).
enum CallError {
    Transport(String),
    Rpc(String),
}

impl PrpcClient {
    /// Builds a client for `host:brpc_port`. Dialing happens lazily on the first call.
    pub(crate) fn new(host: &str, brpc_port: u16) -> Self {
        Self {
            peer: format!("{host}:{brpc_port}"),
            connection: None,
            next_correlation_id: 1,
        }
    }

    /// The peer endpoint this client dials.
    pub(crate) fn peer(&self) -> &str {
        &self.peer
    }

    /// Sends one framed `PInternalService` request and returns the peer's response body and
    /// attachment. A transport failure on a previously-used connection is retried once over a
    /// fresh connection — the peer may simply have closed an idle socket; a duplicate delivery
    /// of an already-processed frame is idempotent receiver-side by design (sequence numbers).
    pub(crate) fn call(
        &mut self,
        method_name: &str,
        body: Vec<u8>,
        attachment: Vec<u8>,
    ) -> Result<prpc::Response, String> {
        let had_cached_connection = self.connection.is_some();
        match self.try_call(method_name, body.clone(), attachment.clone()) {
            Ok(response) => Ok(response),
            Err(CallError::Rpc(err)) => Err(format!("{method_name} to {}: {err}", self.peer)),
            Err(CallError::Transport(err)) if had_cached_connection => {
                warn!(
                    peer = %self.peer,
                    method = method_name,
                    error = %err,
                    "cached PRPC connection failed; retrying once over a fresh connection"
                );
                self.connection = None;
                match self.try_call(method_name, body, attachment) {
                    Ok(response) => Ok(response),
                    Err(CallError::Rpc(err)) | Err(CallError::Transport(err)) => {
                        self.connection = None;
                        Err(format!(
                            "{method_name} to {} (after reconnect): {err}",
                            self.peer
                        ))
                    }
                }
            }
            Err(CallError::Transport(err)) => {
                self.connection = None;
                Err(format!("{method_name} to {}: {err}", self.peer))
            }
        }
    }

    /// One attempt: ensure a connection, write the frame, read the matching reply.
    fn try_call(
        &mut self,
        method_name: &str,
        body: Vec<u8>,
        attachment: Vec<u8>,
    ) -> Result<prpc::Response, CallError> {
        let correlation_id = self.next_correlation_id;
        self.next_correlation_id += 1;

        let frame = prpc::Frame::for_request(
            SERVICE_NAME,
            method_name,
            body,
            attachment,
            Some(correlation_id),
        );
        let bytes = frame.encode();

        let stream = self.stream()?;
        stream
            .write_all(&bytes)
            .map_err(|err| CallError::Transport(format!("failed to write request frame: {err}")))?;
        stream
            .flush()
            .map_err(|err| CallError::Transport(format!("failed to flush request frame: {err}")))?;

        let reply = prpc::Frame::read(stream)
            .map_err(|err| CallError::Transport(format!("failed to read reply frame: {err}")))?
            .ok_or_else(|| {
                CallError::Transport("peer closed the connection before replying".to_string())
            })?;
        // One request in flight per connection, so the reply must correlate with it — anything
        // else means the framing lost sync and the connection cannot be trusted.
        if reply.correlation_id() != Some(correlation_id) {
            return Err(CallError::Transport(format!(
                "reply correlation id {:?} does not match request {correlation_id}",
                reply.correlation_id()
            )));
        }
        reply.into_response().map_err(|err| {
            CallError::Rpc(format!("peer returned brpc error {}: {err}", err.code()))
        })
    }

    /// The cached connection, dialing the peer when there is none.
    fn stream(&mut self) -> Result<&mut TcpStream, CallError> {
        if self.connection.is_none() {
            let address = self
                .peer
                .parse()
                .map_err(|err| CallError::Transport(format!("invalid peer address: {err}")))?;
            let timeout = reply_timeout();
            let stream = TcpStream::connect_timeout(&address, timeout)
                .map_err(|err| CallError::Transport(format!("failed to connect: {err}")))?;
            // Bound every read/write: a wedged peer must fail the query, not hang the transport
            // thread forever. NODELAY because request frames are small and latency-sensitive.
            stream.set_read_timeout(Some(timeout)).map_err(|err| {
                CallError::Transport(format!("failed to set read timeout: {err}"))
            })?;
            stream.set_write_timeout(Some(timeout)).map_err(|err| {
                CallError::Transport(format!("failed to set write timeout: {err}"))
            })?;
            stream
                .set_nodelay(true)
                .map_err(|err| CallError::Transport(format!("failed to set nodelay: {err}")))?;
            info!(peer = %self.peer, "connected PRPC client");
            self.connection = Some(stream);
        }
        Ok(self.connection.as_mut().expect("connection set above"))
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::thread;

    use prost::Message;
    use tokio_util::sync::CancellationToken;

    use starrocks_thrift::status_code::TStatusCode;

    use super::*;
    use crate::BrpcServer;
    use crate::compute_node_service::ExchangeIdentity;
    use crate::fragment_executor::StubExecutor;
    use crate::proto::starrocks::p_internal_service_brpc::methods;
    use crate::proto::starrocks::{PFetchDataRequest, PFetchDataResult, PUniqueId};

    /// Serves the CN's real BRPC dispatch (stub executor) on a loopback port for the client
    /// tests; returns the port and a guard that stops the server on drop.
    fn serve_stub_cn() -> Option<(u16, ServerGuard)> {
        let listener = match BrpcServer::bind("127.0.0.1", 0) {
            Ok(listener) => listener,
            // Sandboxed environments may deny binding; skip like the brpc server tests do.
            Err(err)
                if err.chain().any(|cause| {
                    cause
                        .downcast_ref::<std::io::Error>()
                        .is_some_and(|err| err.kind() == std::io::ErrorKind::PermissionDenied)
                }) =>
            {
                return None;
            }
            Err(err) => panic!("{err:?}"),
        };
        let port = listener.local_addr().unwrap().port();
        let shutdown = CancellationToken::new();
        let server_shutdown = shutdown.clone();
        let join = thread::spawn(move || {
            let runtime = tokio::runtime::Builder::new_current_thread()
                .enable_io()
                .build()
                .unwrap();
            runtime.block_on(
                BrpcServer::with_executor(
                    Arc::new(StubExecutor),
                    ExchangeIdentity::new("127.0.0.1", port),
                    None,
                    None,
                )
                .serve_with_listener_shutdown(listener, server_shutdown.cancelled_owned()),
            )
        });
        Some((
            port,
            ServerGuard {
                shutdown,
                join: Some(join),
            },
        ))
    }

    struct ServerGuard {
        shutdown: CancellationToken,
        join: Option<thread::JoinHandle<anyhow::Result<()>>>,
    }

    impl Drop for ServerGuard {
        fn drop(&mut self) {
            self.shutdown.cancel();
            if let Some(join) = self.join.take() {
                join.join().unwrap().unwrap();
            }
        }
    }

    /// Round trip against the CN's own server dispatch: the request body decodes server-side,
    /// the handler runs, and the reply body decodes client-side with the correlation matched.
    #[test]
    fn client_round_trips_a_method_call_against_the_real_dispatch() {
        let Some((port, _guard)) = serve_stub_cn() else {
            return;
        };
        let mut client = PrpcClient::new("127.0.0.1", port);

        let body = PFetchDataRequest {
            finst_id: PUniqueId { hi: 9, lo: 9 },
        }
        .encode_to_vec();
        let response = client
            .call(methods::FETCH_DATA, body, Vec::new())
            .expect("fetch_data round trip");

        // No result is buffered for that id, so the *method* fails in StatusPB terms while the
        // PRPC exchange itself succeeded — exactly the split the client must preserve.
        let result = PFetchDataResult::decode(response.body.as_slice()).unwrap();
        assert_ne!(result.status.status_code, TStatusCode::OK.0);
        assert!(
            result.status.error_msgs[0].contains("no buffered result"),
            "{:?}",
            result.status.error_msgs
        );
        assert!(response.attachment.is_empty());

        // A second call reuses the cached connection (new correlation id).
        let body = PFetchDataRequest {
            finst_id: PUniqueId { hi: 9, lo: 10 },
        }
        .encode_to_vec();
        client
            .call(methods::FETCH_DATA, body, Vec::new())
            .expect("second call over the cached connection");
    }

    /// A brpc-level error frame (unknown method) surfaces as `Err`, not as a decodable body.
    #[test]
    fn client_surfaces_brpc_error_frames_as_errors() {
        let Some((port, _guard)) = serve_stub_cn() else {
            return;
        };
        let mut client = PrpcClient::new("127.0.0.1", port);

        let err = client
            .call("no_such_method", Vec::new(), Vec::new())
            .expect_err("unknown method must be a brpc error");
        assert!(err.contains("not found"), "{err}");
        assert!(
            err.contains("1002"),
            "brpc ENOMETHOD code should surface: {err}"
        );
    }

    /// A connection the peer closed between calls is re-dialed transparently.
    #[test]
    fn client_reconnects_after_the_peer_drops_the_connection() {
        let Some((port, guard)) = serve_stub_cn() else {
            return;
        };
        let mut client = PrpcClient::new("127.0.0.1", port);
        let body = PFetchDataRequest {
            finst_id: PUniqueId { hi: 1, lo: 1 },
        }
        .encode_to_vec();
        client
            .call(methods::FETCH_DATA, body.clone(), Vec::new())
            .expect("first call");

        // Restart the server on the same port: the cached connection is now dead.
        drop(guard);
        let listener = match BrpcServer::bind("127.0.0.1", port) {
            Ok(listener) => listener,
            // The freed port can be re-claimed by another process in between; don't flake.
            Err(_) => return,
        };
        let shutdown = CancellationToken::new();
        let server_shutdown = shutdown.clone();
        let join = thread::spawn(move || {
            let runtime = tokio::runtime::Builder::new_current_thread()
                .enable_io()
                .build()
                .unwrap();
            runtime.block_on(
                BrpcServer::with_executor(
                    Arc::new(StubExecutor),
                    ExchangeIdentity::new("127.0.0.1", port),
                    None,
                    None,
                )
                .serve_with_listener_shutdown(listener, server_shutdown.cancelled_owned()),
            )
        });
        let _guard = ServerGuard {
            shutdown,
            join: Some(join),
        };

        client
            .call(methods::FETCH_DATA, body, Vec::new())
            .expect("call after peer restart must reconnect");
    }
}
