use std::{
    future::Future,
    net::{SocketAddr, TcpListener as StdTcpListener},
    sync::Arc,
};

use crate::{
    compute_node_service::{ExchangeIdentity, FragmentFailureReporter, SiriusComputeNodeService},
    fragment_executor::FragmentExecutor,
    nixl_transport::NixlTransport,
    proto::starrocks::p_internal_service_brpc::PInternalServiceRouter,
    prpc,
};
use anyhow::{Context, Result};
use tokio::{
    net::{TcpListener, TcpStream},
    sync::Mutex,
    task::JoinSet,
};
use tokio_util::sync::CancellationToken;
use tower::{Service, ServiceExt};
use tracing::{info, instrument, warn};

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
    /// Builds a BRPC server that dispatches fragments to `executor` (the GPU-backed
    /// `SiriusEngine`, or a stub). `identity` is the exchange endpoint this CN advertises,
    /// used to tell local data-stream destinations from remote CNs; `transport` carries
    /// exchanges to remote CNs over nixl (`None` keeps remote destinations a loud error).
    pub fn with_executor(
        executor: Arc<dyn FragmentExecutor>,
        identity: ExchangeIdentity,
        transport: Option<NixlTransport>,
        failure_reporter: Option<Arc<dyn FragmentFailureReporter>>,
    ) -> Self {
        let service =
            PInternalServiceRouter::new(SiriusComputeNodeService::with_transport_and_reporter(
                executor,
                identity,
                transport,
                failure_reporter,
            ));
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

impl<S> BrpcServiceServer<S> {
    /// Builds a BRPC server with an injected Tower service.
    fn with_service(service: S) -> Self {
        Self { service }
    }
}

impl<S> BrpcServiceServer<S>
where
    S: Service<prpc::Request, Response = prpc::Response, Error = prpc::Error>
        + Clone
        + Send
        + 'static,
    S::Future: Send,
{
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
        let service = self.service;
        // An internal token fans the external shutdown signal out to every spawned connection.
        let shutdown = CancellationToken::new();
        let mut connections = JoinSet::new();
        tokio::pin!(signal);

        loop {
            tokio::select! {
                _ = signal.as_mut() => break,
                accepted = listener.accept() => {
                    match accepted {
                        // Spawn per connection (tonic-style) so a slow or long-lived peer cannot
                        // block the accept loop, and so one connection's failure stays isolated.
                        Ok((stream, peer)) => {
                            connections.spawn(Self::handle_connection(
                                service.clone(),
                                stream,
                                peer,
                                shutdown.clone(),
                            ));
                        }
                        Err(err) => warn!(error = %err, "failed to accept BRPC connection"),
                    }
                },
                // Reap finished connection tasks as they complete so the set does not grow with
                // the total number of historical connections on a long-running server.
                Some(_joined) = connections.join_next(), if !connections.is_empty() => {}
            }
        }

        // Ask in-flight connections to stop, then drain them for a graceful shutdown.
        shutdown.cancel();
        while connections.join_next().await.is_some() {}

        info!("BRPC server stopped");
        Ok(())
    }

    /// Reads PRPC frames from one connection and writes one response per request. Read, decode,
    /// and write failures are logged and close only this connection, never the whole server.
    ///
    /// Requests are served concurrently: the FE multiplexes every RPC to this CN over one
    /// connection (correlation ids pair responses with requests), and a `fetch_data` long-poll
    /// or a blocking `exec_plan_fragment` served in line held every request queued behind it.
    /// The FE's `cancel_plan_fragment` for a query the CN had just reported failed then timed
    /// out behind the result fragment's own pending fetch, the fetch never learned of the cancel,
    /// and the client waited out the query timeout (MEASURED 2026-09-05, TPC-H q18 at 2 CNs:
    /// eight `RpcTimerTask` timeouts on the one FE channel, 4 min 20 s until `getNext` saw
    /// CANCELLED). Responses go out in completion order under one writer lock.
    #[instrument(skip_all, fields(peer = %peer))]
    async fn handle_connection(
        service: S,
        stream: TcpStream,
        peer: SocketAddr,
        shutdown: CancellationToken,
    ) {
        let (mut reader, writer) = stream.into_split();
        let writer = Arc::new(Mutex::new(writer));
        let mut in_flight: JoinSet<()> = JoinSet::new();
        loop {
            let frame = tokio::select! {
                _ = shutdown.cancelled() => return,
                frame = prpc::Frame::read_async(&mut reader) => frame,
                // Reap finished requests as they complete so the set does not grow with the
                // connection's lifetime.
                Some(_joined) = in_flight.join_next(), if !in_flight.is_empty() => continue,
            };
            let frame = match frame {
                Ok(Some(frame)) => frame,
                // A clean connection close ends the read side; requests already in flight
                // still finish and answer (the peer may only have half-closed).
                Ok(None) => break,
                // A malformed/unsupported frame or transport error closes only this connection.
                Err(err) => {
                    warn!(error = %err, "failed to read PRPC frame; closing connection");
                    break;
                }
            };

            let correlation_id = frame.correlation_id();
            let request = frame.into_request();
            let mut service = service.clone();
            let writer = Arc::clone(&writer);
            let shutdown = shutdown.clone();
            in_flight.spawn(async move {
                let response = match request {
                    Ok(request) => {
                        tokio::select! {
                            _ = shutdown.cancelled() => return,
                            response = Self::call_service(&mut service, request) => response,
                        }
                    }
                    Err(err) => Err(err),
                };
                let response_frame = prpc::Frame::response_frame(correlation_id, response);
                let mut writer = writer.lock().await;
                tokio::select! {
                    _ = shutdown.cancelled() => {}
                    result = response_frame.write_async(&mut *writer) => {
                        if let Err(err) = result {
                            warn!(error = %err, "failed to write PRPC response");
                        }
                    }
                }
            });
        }
        // Let in-flight requests finish (their responses may still be deliverable) unless the
        // server is shutting down.
        loop {
            tokio::select! {
                _ = shutdown.cancelled() => return,
                joined = in_flight.join_next() => {
                    if joined.is_none() {
                        return;
                    }
                }
            }
        }
    }

    /// Awaits Tower readiness and dispatches one decoded PRPC request.
    async fn call_service(
        service: &mut S,
        request: prpc::Request,
    ) -> std::result::Result<prpc::Response, prpc::Error> {
        service.ready().await?.call(request).await
    }
}

#[cfg(test)]
mod tests {
    use std::{
        future::{Ready, ready},
        io::{Read, Write},
        net::TcpStream as StdTcpStream,
        task::{Context, Poll},
        thread,
    };

    use super::*;
    use tokio_util::sync::CancellationToken;

    /// A malformed frame from one peer must close only that connection; the server keeps
    /// accepting and serving later connections instead of crashing the whole process.
    #[test]
    fn brpc_server_isolates_bad_frame_and_keeps_serving() {
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

        // First connection sends garbage; the server should drop just this connection. The read
        // returns once the server closes the socket (EOF, or a reset because unread bytes remain),
        // confirming the server handled the bad frame and closed it rather than dying.
        let mut bad = StdTcpStream::connect(addr).unwrap();
        bad.write_all(b"not a valid prpc frame").unwrap();
        let mut drained = Vec::new();
        let _ = bad.read_to_end(&mut drained);

        // A second connection with a well-formed frame still gets a response.
        let request = prpc::Frame::for_request(
            "PInternalService",
            "exec_plan_fragment",
            b"body".to_vec(),
            Vec::new(),
            Some(7),
        );
        let mut good = StdTcpStream::connect(addr).unwrap();
        good.write_all(&request.encode()).unwrap();
        let response = prpc::Frame::read(&mut good).unwrap();
        assert!(
            response.is_some(),
            "server should still serve a second connection after a bad frame"
        );

        shutdown.cancel();
        join.join().unwrap().unwrap();
    }

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

    /// The FE multiplexes its RPCs to one CN over one connection: a slow request (a `fetch_data`
    /// long-poll, here a sleep) must not hold the answer to a later one (the `cancel_plan_fragment`
    /// that would end it). The fast request's response arrives first, then the slow one's.
    #[test]
    fn slow_request_does_not_block_later_requests_on_the_same_connection() {
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
                .enable_all()
                .build()
                .unwrap();
            runtime.block_on(
                BrpcServiceServer::with_service(SlowFetchService)
                    .serve_with_listener_shutdown(listener, server_shutdown.cancelled_owned()),
            )
        });

        let mut stream = StdTcpStream::connect(addr).unwrap();
        let slow = prpc::Frame::for_request(
            "PInternalService",
            "fetch_data",
            b"body".to_vec(),
            Vec::new(),
            Some(1),
        );
        let fast = prpc::Frame::for_request(
            "PInternalService",
            "cancel_plan_fragment",
            b"body".to_vec(),
            Vec::new(),
            Some(2),
        );
        stream.write_all(&slow.encode()).unwrap();
        stream.write_all(&fast.encode()).unwrap();
        let first = prpc::Frame::read(&mut stream).unwrap().expect("a response");
        assert_eq!(
            first.correlation_id(),
            Some(2),
            "the fast request is answered while the slow one is still being served"
        );
        let second = prpc::Frame::read(&mut stream).unwrap().expect("a response");
        assert_eq!(second.correlation_id(), Some(1));

        shutdown.cancel();
        join.join().unwrap().unwrap();
    }

    /// Answers `fetch_data` after a delay and everything else at once.
    #[derive(Clone)]
    struct SlowFetchService;

    impl Service<prpc::Request> for SlowFetchService {
        type Response = prpc::Response;
        type Error = prpc::Error;
        type Future = std::pin::Pin<
            Box<dyn Future<Output = std::result::Result<Self::Response, Self::Error>> + Send>,
        >;

        fn poll_ready(
            &mut self,
            _context: &mut Context<'_>,
        ) -> Poll<std::result::Result<(), Self::Error>> {
            Poll::Ready(Ok(()))
        }

        fn call(&mut self, request: prpc::Request) -> Self::Future {
            let slow = request.method_name == "fetch_data";
            Box::pin(async move {
                if slow {
                    tokio::time::sleep(std::time::Duration::from_millis(1500)).await;
                }
                Ok(prpc::Response::with_attachment(Vec::new(), Vec::new()))
            })
        }
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
            ready(Ok(prpc::Response::with_attachment(Vec::new(), Vec::new())))
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
