use std::{
    net::{IpAddr, Ipv4Addr, Ipv6Addr, Shutdown, SocketAddr, TcpListener, TcpStream},
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
    },
    thread::{self, JoinHandle},
};

use crate::{
    internal_service::PlanFragmentTranslatorService,
    proto::starrocks::p_internal_service_brpc::PInternalServiceRouter, prpc,
};
use anyhow::{Context, Result, anyhow};
use tokio::runtime::Runtime;
use tower::{Service, ServiceExt};
use tracing::{info, warn};

pub struct BrpcServer {
    join_handle: Option<JoinHandle<Result<()>>>,
    shutdown: BrpcServerShutdown,
    local_addr: SocketAddr,
}

impl BrpcServer {
    pub fn shutdown_handle(&self) -> BrpcServerShutdown {
        self.shutdown.clone()
    }

    pub fn shutdown(&self) {
        self.shutdown.shutdown();
    }

    pub fn local_addr(&self) -> SocketAddr {
        self.local_addr
    }

    pub fn join(mut self) -> Result<()> {
        let join_handle = self
            .join_handle
            .take()
            .context("BRPC server join handle was already consumed")?;
        join_handle
            .join()
            .map_err(|panic| anyhow!("BRPC server thread panicked: {panic:?}"))?
    }
}

impl Drop for BrpcServer {
    fn drop(&mut self) {
        if self.join_handle.is_some() {
            self.shutdown();
        }
    }
}

#[derive(Clone)]
pub struct BrpcServerShutdown(Arc<BrpcServerShutdownState>);

impl BrpcServerShutdown {
    fn new(wake_addr: SocketAddr) -> Self {
        Self(Arc::new(BrpcServerShutdownState {
            requested: AtomicBool::new(false),
            active_connection: Arc::new(Mutex::new(None)),
            wake_addr,
        }))
    }

    pub fn shutdown(&self) {
        if self.0.requested.swap(true, Ordering::SeqCst) {
            return;
        }

        self.close_active_connection();
        let _ = TcpStream::connect(self.0.wake_addr);
    }

    fn is_requested(&self) -> bool {
        self.0.requested.load(Ordering::SeqCst)
    }

    fn track_connection(&self, stream: &TcpStream) -> Result<BrpcActiveConnectionGuard> {
        let shutdown_stream = stream
            .try_clone()
            .context("failed to clone BRPC client connection")?;

        let mut active_connection = self
            .0
            .active_connection
            .lock()
            .map_err(|_| anyhow!("active BRPC connection mutex poisoned"))?;
        *active_connection = Some(shutdown_stream);

        Ok(BrpcActiveConnectionGuard {
            active_connection: self.0.active_connection.clone(),
        })
    }

    fn close_active_connection(&self) {
        if let Ok(active_connection) = self.0.active_connection.lock()
            && let Some(connection) = active_connection.as_ref()
        {
            let _ = connection.shutdown(Shutdown::Both);
        }
    }
}

struct BrpcServerShutdownState {
    requested: AtomicBool,
    active_connection: Arc<Mutex<Option<TcpStream>>>,
    wake_addr: SocketAddr,
}

struct BrpcActiveConnectionGuard {
    active_connection: Arc<Mutex<Option<TcpStream>>>,
}

impl Drop for BrpcActiveConnectionGuard {
    fn drop(&mut self) {
        if let Ok(mut active_connection) = self.active_connection.lock() {
            *active_connection = None;
        }
    }
}

pub fn start_brpc_server(bind_host: &str, brpc_port: u16) -> Result<BrpcServer> {
    let service = PInternalServiceRouter::new(PlanFragmentTranslatorService::new());
    start_brpc_server_with_service(bind_host, brpc_port, service)
}

fn start_brpc_server_with_service<S>(
    bind_host: &str,
    brpc_port: u16,
    service: S,
) -> Result<BrpcServer>
where
    S: Service<prpc::Request, Response = prpc::Response, Error = prpc::Error> + Send + 'static,
    S::Future: Send,
{
    let listen_addr = format!("{bind_host}:{brpc_port}");
    let listener = TcpListener::bind(&listen_addr)
        .with_context(|| format!("failed to bind BRPC server at {listen_addr}"))?;
    let local_addr = listener
        .local_addr()
        .context("failed to read BRPC server address")?;
    let shutdown = BrpcServerShutdown::new(listener_wake_addr(local_addr));
    let server_shutdown = shutdown.clone();

    info!(address = %local_addr, "starting BRPC server");
    let join_handle = thread::spawn(move || run_brpc_server(listener, server_shutdown, service));

    Ok(BrpcServer {
        join_handle: Some(join_handle),
        shutdown,
        local_addr,
    })
}

fn run_brpc_server<S>(
    listener: TcpListener,
    shutdown: BrpcServerShutdown,
    mut service: S,
) -> Result<()>
where
    S: Service<prpc::Request, Response = prpc::Response, Error = prpc::Error>,
{
    let runtime = tokio::runtime::Builder::new_current_thread()
        .build()
        .context("failed to create BRPC service runtime")?;

    for stream in listener.incoming() {
        if shutdown.is_requested() {
            break;
        }

        match stream {
            Ok(stream) => {
                if shutdown.is_requested() {
                    break;
                }
                let active_connection = shutdown.track_connection(&stream)?;
                handle_brpc_connection(&mut service, &runtime, stream, active_connection)?;
                if shutdown.is_requested() {
                    break;
                }
            }
            Err(_) if shutdown.is_requested() => break,
            Err(err) => warn!(error = %err, "failed to accept BRPC connection"),
        }
    }

    shutdown.close_active_connection();
    info!("BRPC server stopped");
    Ok(())
}

fn handle_brpc_connection<S>(
    service: &mut S,
    runtime: &Runtime,
    mut stream: TcpStream,
    active_connection: BrpcActiveConnectionGuard,
) -> Result<()>
where
    S: Service<prpc::Request, Response = prpc::Response, Error = prpc::Error>,
{
    let _active_connection = active_connection;

    loop {
        let Some(frame) = prpc::read_frame(&mut stream)? else {
            return Ok(());
        };
        let response = match frame.request() {
            Ok(request) => runtime.block_on(call_service(service, request)),
            Err(err) => Err(err),
        };
        let response_frame = frame.into_response_frame(response);
        prpc::write_frame(&mut stream, &response_frame)?;
    }
}

async fn call_service<S>(
    service: &mut S,
    request: prpc::Request,
) -> std::result::Result<prpc::Response, prpc::Error>
where
    S: Service<prpc::Request, Response = prpc::Response, Error = prpc::Error>,
{
    service.ready().await?.call(request).await
}

fn listener_wake_addr(local_addr: SocketAddr) -> SocketAddr {
    match local_addr.ip() {
        IpAddr::V4(ip) if ip.is_unspecified() => {
            SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), local_addr.port())
        }
        IpAddr::V6(ip) if ip.is_unspecified() => {
            SocketAddr::new(IpAddr::V6(Ipv6Addr::LOCALHOST), local_addr.port())
        }
        _ => local_addr,
    }
}
