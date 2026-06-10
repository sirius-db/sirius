use std::{
    mem::ManuallyDrop,
    net::{SocketAddr, TcpListener as StdTcpListener},
    thread::{self, JoinHandle},
};

use crate::{
    internal_service::PlanFragmentTranslatorService,
    proto::starrocks::p_internal_service_brpc::PInternalServiceRouter, prpc,
};
use anyhow::{Context, Result, anyhow};
use tokio::net::{TcpListener, TcpStream};
use tokio_util::sync::CancellationToken;
use tower::{Service, ServiceExt};
use tracing::{info, warn};

/// Handle for the blocking BRPC listener thread.
pub struct BrpcServer {
    join_handle: JoinHandle<Result<()>>,
    shutdown: BrpcServerShutdown,
    local_addr: SocketAddr,
}

impl BrpcServer {
    /// Returns a cloneable shutdown handle for coordinating process shutdown.
    pub fn shutdown_handle(&self) -> BrpcServerShutdown {
        self.shutdown.clone()
    }

    /// Requests listener and active-connection shutdown.
    pub fn shutdown(&self) {
        self.shutdown.shutdown();
    }

    /// Returns the address the BRPC listener bound to.
    pub fn local_addr(&self) -> SocketAddr {
        self.local_addr
    }

    /// Waits for the listener thread to exit.
    pub fn join(self) -> Result<()> {
        let this = ManuallyDrop::new(self);
        // SAFETY: `join` consumes `self`, wraps it in `ManuallyDrop` so `Drop`
        // will not run, and reads each non-Copy field exactly once.
        let join_handle = unsafe { std::ptr::read(&this.join_handle) };
        // SAFETY: this is the matching single read of the shutdown token so it
        // is dropped normally without requesting shutdown.
        let shutdown = unsafe { std::ptr::read(&this.shutdown) };
        drop(shutdown);

        join_handle
            .join()
            .map_err(|panic| anyhow!("BRPC server thread panicked: {panic:?}"))?
    }
}

impl Drop for BrpcServer {
    fn drop(&mut self) {
        self.shutdown();
    }
}

/// Cloneable signal used to stop the BRPC listener and current connection.
#[derive(Clone)]
pub struct BrpcServerShutdown {
    token: CancellationToken,
}

impl BrpcServerShutdown {
    /// Creates a shutdown handle shared by the listener and connection loop.
    fn new() -> Self {
        Self {
            token: CancellationToken::new(),
        }
    }

    /// Requests shutdown for the listener and any active connection.
    pub fn shutdown(&self) {
        self.token.cancel();
    }
}

/// Starts the StarRocks BRPC server using the generated PInternalService router.
pub fn start_brpc_server(bind_host: &str, brpc_port: u16) -> Result<BrpcServer> {
    let service = PInternalServiceRouter::new(PlanFragmentTranslatorService::new());
    start_brpc_server_with_service(bind_host, brpc_port, service)
}

/// Starts a BRPC server with an injected Tower service, primarily for tests and future layering.
fn start_brpc_server_with_service<S>(
    bind_host: &str,
    brpc_port: u16,
    service: S,
) -> Result<BrpcServer>
where
    S: Service<prpc::Request, Response = prpc::Response, Error = prpc::Error> + Send + 'static,
{
    let listen_addr = format!("{bind_host}:{brpc_port}");
    let listener = StdTcpListener::bind(&listen_addr)
        .with_context(|| format!("failed to bind BRPC server at {listen_addr}"))?;
    listener
        .set_nonblocking(true)
        .context("failed to put BRPC listener in nonblocking mode")?;
    let local_addr = listener
        .local_addr()
        .context("failed to read BRPC server address")?;
    let shutdown = BrpcServerShutdown::new();
    let server_shutdown = shutdown.clone();

    info!(address = %local_addr, "starting BRPC server");
    let join_handle = thread::spawn(move || run_brpc_server(listener, server_shutdown, service));

    Ok(BrpcServer {
        join_handle,
        shutdown,
        local_addr,
    })
}

/// Runs the async listener runtime on the dedicated BRPC thread.
fn run_brpc_server<S>(
    listener: StdTcpListener,
    shutdown: BrpcServerShutdown,
    service: S,
) -> Result<()>
where
    S: Service<prpc::Request, Response = prpc::Response, Error = prpc::Error>,
{
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_io()
        .build()
        .context("failed to create BRPC service runtime")?;
    runtime.block_on(run_brpc_server_async(listener, shutdown.token, service))
}

/// Runs the async BRPC accept loop until the shutdown token is cancelled.
async fn run_brpc_server_async<S>(
    listener: StdTcpListener,
    shutdown: CancellationToken,
    mut service: S,
) -> Result<()>
where
    S: Service<prpc::Request, Response = prpc::Response, Error = prpc::Error>,
{
    let listener =
        TcpListener::from_std(listener).context("failed to create async BRPC listener")?;

    loop {
        tokio::select! {
            _ = shutdown.cancelled() => break,
            accepted = listener.accept() => {
                match accepted {
                    Ok((stream, _addr)) => {
                        handle_brpc_connection(&mut service, stream, shutdown.clone()).await?;
                    }
                    Err(_) if shutdown.is_cancelled() => break,
                    Err(err) => warn!(error = %err, "failed to accept BRPC connection"),
                }
            },
        }
    }

    info!("BRPC server stopped");
    Ok(())
}

/// Reads PRPC frames from one connection and writes one response per request.
async fn handle_brpc_connection<S>(
    service: &mut S,
    mut stream: TcpStream,
    shutdown: CancellationToken,
) -> Result<()>
where
    S: Service<prpc::Request, Response = prpc::Response, Error = prpc::Error>,
{
    loop {
        let frame = tokio::select! {
            _ = shutdown.cancelled() => return Ok(()),
            frame = prpc::read_frame_async(&mut stream) => frame?,
        };
        let Some(frame) = frame else {
            return Ok(());
        };

        let response = match frame.request() {
            Ok(request) => {
                tokio::select! {
                    _ = shutdown.cancelled() => return Ok(()),
                    response = call_service(service, request) => response,
                }
            }
            Err(err) => Err(err),
        };
        let response_frame = frame.into_response_frame(response);
        tokio::select! {
            _ = shutdown.cancelled() => return Ok(()),
            result = prpc::write_frame_async(&mut stream, &response_frame) => result?,
        }
    }
}

/// Awaits Tower readiness and dispatches one decoded PRPC request.
async fn call_service<S>(
    service: &mut S,
    request: prpc::Request,
) -> std::result::Result<prpc::Response, prpc::Error>
where
    S: Service<prpc::Request, Response = prpc::Response, Error = prpc::Error>,
{
    service.ready().await?.call(request).await
}

#[cfg(test)]
mod tests {
    use std::{
        future::{Ready, ready},
        net::TcpStream as StdTcpStream,
        task::{Context, Poll},
    };

    use super::*;

    /// Verifies BRPC shutdown cancels an accepted connection waiting for PRPC input.
    #[test]
    fn brpc_server_shutdown_cancels_active_connection() {
        let server = match start_brpc_server_with_service("127.0.0.1", 0, EmptyService) {
            Ok(server) => server,
            Err(err) if is_permission_denied(&err) => return,
            Err(err) => panic!("{err:?}"),
        };
        let stream = StdTcpStream::connect(server.local_addr()).unwrap();

        server.shutdown();
        server.join().unwrap();
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
