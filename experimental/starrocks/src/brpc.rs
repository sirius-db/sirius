use std::{future::Future, net::TcpListener as StdTcpListener, pin::Pin};

use crate::{
    compute_node_service::SiriusComputeNodeService,
    proto::starrocks::p_internal_service_brpc::PInternalServiceRouter, prpc,
};
use anyhow::{Context, Result};
use tokio::net::{TcpListener, TcpStream};
use tower::{Service, ServiceExt};
use tracing::{info, warn};

/// BRPC service runner for StarRocks PInternalService.
pub struct BrpcServer {
    /// Generic Tower service runner hidden behind the public default service.
    inner: BrpcServiceServer<PInternalServiceRouter<SiriusComputeNodeService>>,
}

/// Generic BRPC service runner over StarRocks PRPC frames.
struct BrpcServiceServer<S> {
    /// Tower service invoked for each decoded PRPC request.
    service: S,
}

impl BrpcServer {
    /// Builds a BRPC server for Sirius compute-node RPCs over StarRocks PInternalService.
    pub fn new() -> Self {
        let service = PInternalServiceRouter::new(SiriusComputeNodeService::new());
        Self {
            inner: BrpcServiceServer::with_service(service),
        }
    }

    /// Binds a std listener that can be moved into the runtime that serves it.
    pub fn bind(bind_host: &str, brpc_port: u16) -> Result<StdTcpListener> {
        let listen_addr = format!("{bind_host}:{brpc_port}");
        let listener = StdTcpListener::bind(&listen_addr)
            .with_context(|| format!("failed to bind BRPC server at {listen_addr}"))?;
        listener
            .set_nonblocking(true)
            .context("failed to put BRPC listener in nonblocking mode")?;
        Ok(listener)
    }

    /// Serves BRPC requests on `bind_host:brpc_port` until the task is cancelled.
    pub async fn serve(self, bind_host: &str, brpc_port: u16) -> Result<()> {
        self.inner.serve(bind_host, brpc_port).await
    }

    /// Serves BRPC requests on `bind_host:brpc_port` until `signal` resolves.
    pub async fn serve_with_shutdown<F>(
        self,
        bind_host: &str,
        brpc_port: u16,
        signal: F,
    ) -> Result<()>
    where
        F: Future<Output = ()>,
    {
        self.inner
            .serve_with_shutdown(bind_host, brpc_port, signal)
            .await
    }

    /// Serves BRPC requests from an existing listener until the task is cancelled.
    pub async fn serve_with_listener(self, listener: StdTcpListener) -> Result<()> {
        self.inner.serve_with_listener(listener).await
    }

    /// Serves BRPC requests from an existing listener until `signal` resolves.
    pub async fn serve_with_listener_shutdown<F>(
        self,
        listener: StdTcpListener,
        signal: F,
    ) -> Result<()>
    where
        F: Future<Output = ()>,
    {
        self.inner
            .serve_with_listener_shutdown(listener, signal)
            .await
    }
}

impl Default for BrpcServer {
    fn default() -> Self {
        Self::new()
    }
}

impl<S> BrpcServiceServer<S> {
    /// Builds a BRPC server with an injected Tower service.
    fn with_service(service: S) -> Self {
        Self { service }
    }
}

impl<S> BrpcServiceServer<S>
where
    S: Service<prpc::Request, Response = prpc::Response, Error = prpc::Error>,
{
    /// Serves BRPC requests on `bind_host:brpc_port` until the task is cancelled.
    pub async fn serve(self, bind_host: &str, brpc_port: u16) -> Result<()> {
        self.serve_with_shutdown(bind_host, brpc_port, std::future::pending::<()>())
            .await
    }

    /// Serves BRPC requests on `bind_host:brpc_port` until `signal` resolves.
    pub async fn serve_with_shutdown<F>(
        self,
        bind_host: &str,
        brpc_port: u16,
        signal: F,
    ) -> Result<()>
    where
        F: Future<Output = ()>,
    {
        let listener = BrpcServer::bind(bind_host, brpc_port)?;
        self.serve_with_listener_shutdown(listener, signal).await
    }

    /// Serves BRPC requests from an existing listener until the task is cancelled.
    pub async fn serve_with_listener(self, listener: StdTcpListener) -> Result<()> {
        self.serve_with_listener_shutdown(listener, std::future::pending::<()>())
            .await
    }

    /// Serves BRPC requests from an existing listener until `signal` resolves.
    pub async fn serve_with_listener_shutdown<F>(
        self,
        listener: StdTcpListener,
        signal: F,
    ) -> Result<()>
    where
        F: Future<Output = ()>,
    {
        let local_addr = listener
            .local_addr()
            .context("failed to read BRPC server address")?;
        info!(address = %local_addr, "starting BRPC server");
        let listener =
            TcpListener::from_std(listener).context("failed to create async BRPC listener")?;
        let mut service = self.service;
        tokio::pin!(signal);

        loop {
            tokio::select! {
                _ = signal.as_mut() => break,
                accepted = listener.accept() => {
                    match accepted {
                        Ok((stream, _addr)) => {
                            if Self::handle_connection(&mut service, stream, signal.as_mut()).await? == ConnectionExit::Shutdown {
                                break;
                            }
                        }
                        Err(err) => warn!(error = %err, "failed to accept BRPC connection"),
                    }
                },
            }
        }

        info!("BRPC server stopped");
        Ok(())
    }

    /// Reads PRPC frames from one connection and writes one response per request.
    async fn handle_connection(
        service: &mut S,
        mut stream: TcpStream,
        mut signal: Pin<&mut impl Future<Output = ()>>,
    ) -> Result<ConnectionExit>
    where
        S: Service<prpc::Request, Response = prpc::Response, Error = prpc::Error>,
    {
        loop {
            let frame = tokio::select! {
                _ = signal.as_mut() => return Ok(ConnectionExit::Shutdown),
                frame = prpc::Frame::read_async(&mut stream) => frame?,
            };
            let Some(frame) = frame else {
                return Ok(ConnectionExit::Closed);
            };

            let response = match frame.request() {
                Ok(request) => {
                    tokio::select! {
                        _ = signal.as_mut() => return Ok(ConnectionExit::Shutdown),
                        response = Self::call_service(service, request) => response,
                    }
                }
                Err(err) => Err(err),
            };
            let response_frame = frame.into_response_frame(response);
            tokio::select! {
                _ = signal.as_mut() => return Ok(ConnectionExit::Shutdown),
                result = response_frame.write_async(&mut stream) => result?,
            }
        }
    }

    /// Awaits Tower readiness and dispatches one decoded PRPC request.
    async fn call_service(
        service: &mut S,
        request: prpc::Request,
    ) -> std::result::Result<prpc::Response, prpc::Error>
    where
        S: Service<prpc::Request, Response = prpc::Response, Error = prpc::Error>,
    {
        service.ready().await?.call(request).await
    }
}

/// Outcome of processing one accepted BRPC connection.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ConnectionExit {
    /// The peer closed the connection normally.
    Closed,
    /// The server shutdown signal resolved while the connection was active.
    Shutdown,
}

#[cfg(test)]
mod tests {
    use std::{
        future::{Ready, ready},
        net::TcpStream as StdTcpStream,
        task::{Context, Poll},
        thread,
    };

    use super::*;
    use tokio_util::sync::CancellationToken;

    /// Verifies BRPC shutdown cancels an accepted connection waiting for PRPC input.
    #[test]
    fn brpc_server_shutdown_cancels_active_connection() {
        let listener = match BrpcServer::bind("127.0.0.1", 0) {
            Ok(listener) => listener,
            Err(err) if is_permission_denied(&err) => return,
            Err(err) => panic!("{err:?}"),
        };
        let addr = listener.local_addr().unwrap();
        let shutdown = CancellationToken::new();
        let server_shutdown = shutdown.clone();
        let join = thread::spawn(move || {
            let runtime = tokio::runtime::Builder::new_current_thread()
                .enable_io()
                .build()
                .unwrap();
            runtime.block_on(
                BrpcServiceServer::with_service(EmptyService)
                    .serve_with_listener_shutdown(listener, server_shutdown.cancelled_owned()),
            )
        });
        let stream = StdTcpStream::connect(addr).unwrap();

        shutdown.cancel();
        join.join().unwrap().unwrap();
        drop(stream);
    }

    /// Minimal Tower service used only to exercise transport shutdown.
    #[derive(Clone)]
    struct EmptyService;

    impl Service<prpc::Request> for EmptyService {
        type Response = prpc::Response;
        type Error = prpc::Error;
        type Future = Ready<std::result::Result<Self::Response, Self::Error>>;

        fn poll_ready(
            &mut self,
            _context: &mut Context<'_>,
        ) -> Poll<std::result::Result<(), Self::Error>> {
            Poll::Ready(Ok(()))
        }

        fn call(&mut self, _request: prpc::Request) -> Self::Future {
            ready(Ok(prpc::Response::new(Vec::new())))
        }
    }

    /// Detects sandboxed environments where binding a local listener is denied.
    fn is_permission_denied(err: &anyhow::Error) -> bool {
        err.chain().any(|cause| {
            cause
                .downcast_ref::<std::io::Error>()
                .is_some_and(|err| err.kind() == std::io::ErrorKind::PermissionDenied)
        })
    }
}
