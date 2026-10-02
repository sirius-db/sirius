//! The NIXL exchange transport.
//!
//! The engine's GPU pool is one `cudaMalloc` slab, registered once with NIXL as VRAM. To ship a
//! parked batch, the sender exports it, asks the receiver (Alloc) for matching buffers from the
//! receiver's own pool, WRITEs every buffer straight into them, and announces the receiver's token
//! with a Packed frame. No staging copy on either side.
//!
//! One thread owns the [`Agent`] (nixl-sys documents a multithreading deadlock caveat), so every
//! WRITE goes through it. What peers ask of this CN (Md, Alloc, Release) never touches that thread:
//! it reads the cached metadata or calls the [`DirectExchange`] on the brpc thread.

use std::collections::{HashMap, VecDeque};
use std::net::SocketAddr;
use std::sync::Arc;
use std::sync::mpsc::{Receiver, Sender, channel};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use nixl_sys::{
    Agent, MemType, MemoryRegion, NixlDescriptor, OptArgs, RegistrationHandle, XferDescList,
    XferOp, XferRequest, XferStatus,
};
use prost::Message;
use sirius::DirectExchange;
use tracing::{info, warn};

use crate::fragment_executor::{ExportedBatch, FragmentExecutor, SenderSlot};
use crate::nixl_chunk::{AllocReply, NixlEndpoint, NixlEnvelope, control_params, packed_params};
use crate::proto::starrocks::{
    PTransmitChunkParams, PTransmitChunkResult,
    p_internal_service_brpc::{SERVICE_NAME, methods},
};
use crate::prpc;

/// Appended to bring-up errors, whose usual cause is the environment.
const ENV_HINT: &str = "source experimental/starrocks/scripts/cn-env.sh (NIXL_PREFIX, \
                        NIXL_PLUGIN_DIR, LD_LIBRARY_PATH) and set \
                        UCX_TLS=cuda_copy,cuda_ipc,tcp,self";

const RPC_TIMEOUT: Duration = Duration::from_secs(60);

/// Handle to the transport thread.
#[derive(Debug)]
pub struct NixlTransport {
    /// Taken on drop to end the thread's loop.
    requests: Option<Sender<SendRequest>>,
    thread: Option<JoinHandle<()>>,
    local_md: Vec<u8>,
    exchange: Arc<DirectExchange>,
}

struct SendRequest {
    peer: SocketAddr,
    slot: SenderSlot,
    names: Vec<String>,
    executor: Arc<dyn FragmentExecutor>,
    respond: Sender<Result<(), String>>,
}

impl NixlTransport {
    /// Brings up an agent named `agent_name` with the UCX backend on its own thread and registers
    /// `exchange`'s slab with it, so a broken NIXL install fails here, before any query.
    pub fn start(agent_name: String, exchange: Arc<DirectExchange>) -> Result<Self, String> {
        let (request_tx, request_rx) = channel();
        let (ready_tx, ready_rx) = channel();
        let thread_exchange = Arc::clone(&exchange);
        let thread = std::thread::Builder::new()
            .name("nixl-transport".to_string())
            .spawn(
                move || match Transport::bring_up(&agent_name, thread_exchange) {
                    Ok(mut transport) => {
                        if ready_tx.send(Ok(transport.local_md.clone())).is_ok() {
                            transport.serve(request_rx);
                        }
                    }
                    Err(err) => {
                        let _ = ready_tx.send(Err(err));
                    }
                },
            )
            .map_err(|err| format!("failed to spawn the nixl-transport thread: {err}"))?;
        let local_md = ready_rx
            .recv()
            .map_err(|_| "the nixl-transport thread exited during bring-up".to_string())??;
        Ok(Self {
            requests: Some(request_tx),
            thread: Some(thread),
            local_md,
            exchange,
        })
    }
}

impl Drop for NixlTransport {
    fn drop(&mut self) {
        // Joined so the slab is deregistered before the engine that owns it goes.
        self.requests.take();
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

impl NixlEndpoint for NixlTransport {
    fn local_md(&self) -> Vec<u8> {
        self.local_md.clone()
    }

    fn allocate(&self, layout: &[u8]) -> Result<AllocReply, String> {
        let (token, buffers) = self
            .exchange
            .allocate(layout)
            .map_err(|err| format!("failed to allocate receive buffers: {err}"))?;
        Ok(AllocReply {
            token,
            device: self.exchange.region().0,
            buffers,
        })
    }

    fn release(&self, token: u64) {
        release(&self.exchange, token);
    }

    fn send(
        &self,
        peer: SocketAddr,
        slot: SenderSlot,
        names: Vec<String>,
        executor: Arc<dyn FragmentExecutor>,
    ) -> Result<(), String> {
        let (respond, response) = channel();
        self.requests
            .as_ref()
            .ok_or_else(|| "the nixl transport is shutting down".to_string())?
            .send(SendRequest {
                peer,
                slot,
                names,
                executor,
                respond,
            })
            .map_err(|_| "the nixl-transport thread is not running".to_string())?;
        response
            .recv()
            .map_err(|_| "the nixl-transport thread dropped the hop".to_string())?
    }
}

/// The slab as a NIXL memory region.
#[derive(Debug)]
struct SlabRegion {
    base: usize,
    len: usize,
    device: u64,
}

impl MemoryRegion for SlabRegion {
    unsafe fn as_ptr(&self) -> *const u8 {
        self.base as *const u8
    }

    fn size(&self) -> usize {
        self.len
    }
}

impl NixlDescriptor for SlabRegion {
    fn mem_type(&self) -> MemType {
        MemType::Vram
    }

    fn device_id(&self) -> u64 {
        self.device
    }
}

/// Transport-thread state.
struct Transport {
    // Fields drop in declaration order: the slab is deregistered before `exchange`, which may
    // hold the last reference keeping it allocated.
    _registration: RegistrationHandle,
    exchange: Arc<DirectExchange>,
    agent: Agent,
    device: u64,
    local_md: Vec<u8>,
    /// Remote agent name per peer, once its metadata is loaded.
    peers: HashMap<SocketAddr, String>,
}

/// One posted WRITE, completed oldest first.
struct Write {
    /// `None` once it finished.
    request: Option<XferRequest>,
    /// The sender's batch, released once the WRITE succeeds.
    local: u64,
    /// The receiver's buffers, announced once the WRITE succeeds.
    remote: u64,
    rows: u64,
    bytes: u64,
    posted: Instant,
}

#[derive(Default)]
struct HopTotals {
    frames: u64,
    rows: u64,
    bytes: u64,
    write_us: u64,
}

impl Transport {
    fn bring_up(agent_name: &str, exchange: Arc<DirectExchange>) -> Result<Self, String> {
        check_single_visible_device(std::env::var("CUDA_VISIBLE_DEVICES").ok().as_deref())?;
        let agent = Agent::new(agent_name).map_err(|err| {
            format!("failed to create nixl agent '{agent_name}': {err} ({ENV_HINT})")
        })?;
        let (_, params) = agent
            .get_plugin_params("UCX")
            .map_err(|err| format!("nixl UCX plugin unavailable: {err} ({ENV_HINT})"))?;
        let backend = agent
            .create_backend("UCX", &params)
            .map_err(|err| format!("failed to create the nixl UCX backend: {err} ({ENV_HINT})"))?;
        let mut options =
            OptArgs::new().map_err(|err| format!("failed to create nixl options: {err}"))?;
        options
            .add_backend(&backend)
            .map_err(|err| format!("failed to select the nixl UCX backend: {err}"))?;
        let (device, base, len) = exchange.region();
        let slab = SlabRegion {
            base,
            len: len as usize,
            device: device as u64,
        };
        let registration = agent
            .register_memory(&slab, Some(&options))
            .map_err(|err| {
                format!(
                    "failed to register the {len}-byte GPU slab with nixl: {err}; UCX_TLS must \
                 include cuda_copy ({ENV_HINT})"
                )
            })?;
        let local_md = agent
            .get_local_md()
            .map_err(|err| format!("failed to serialize nixl agent metadata: {err}"))?;
        info!(
            agent = agent_name,
            device, base, len, "nixl transport ready; GPU slab registered"
        );
        Ok(Self {
            _registration: registration,
            exchange,
            agent,
            device: device as u64,
            local_md,
            peers: HashMap::new(),
        })
    }

    fn serve(&mut self, requests: Receiver<SendRequest>) {
        for request in requests {
            let result = self.send(&request);
            let _ = request.respond.send(result);
        }
        info!("nixl-transport thread shutting down");
    }

    /// Ships every batch parked under the request's slot, keeping a window of WRITEs in flight.
    /// Announces leave in order from their own thread, so a WRITE never waits on the previous
    /// batch's announce round trip.
    fn send(&mut self, request: &SendRequest) -> Result<(), String> {
        let started = Instant::now();
        let peer = request.peer;
        let remote = self.remote_agent(peer)?;
        let window = env_u64("SIRIUS_CN_NIXL_WINDOW").map_or(4, |n| n as usize);
        let mut totals = HopTotals::default();
        std::thread::scope(|scope| {
            let (announce, frames) = channel::<(u64, u64)>();
            let (slot, names) = (request.slot, &request.names);
            let announcer = scope.spawn(move || announce_all(peer, slot, names, frames));
            let mut inflight = VecDeque::new();
            let pumped = self.pump(
                request,
                &remote,
                window,
                &mut inflight,
                &announce,
                &mut totals,
            );
            // Wait out every WRITE still posted: a finished one frees the sender's batch, and the
            // receiver's buffers it filled are never announced, so they are released.
            for mut write in inflight {
                if self.complete(&remote, &mut write).is_ok() {
                    let _ = call(peer, control_params(), &NixlEnvelope::Release(write.remote));
                }
            }
            if pumped.is_ok() {
                let _ = announce.send((0, 0));
            }
            drop(announce);
            let announced = announcer
                .join()
                .unwrap_or_else(|_| Err("the announce thread panicked".to_string()));
            pumped.and(announced)
        })?;
        info!(
            peer = %peer,
            dest_stream = request.slot.node_id,
            sender_id = request.slot.sender_id,
            frames = totals.frames,
            bytes = totals.bytes,
            rows = totals.rows,
            window,
            write_us = totals.write_us,
            span_us = started.elapsed().as_micros() as u64,
            "shipping packed exchange hop"
        );
        Ok(())
    }

    fn pump(
        &self,
        request: &SendRequest,
        remote: &str,
        window: usize,
        inflight: &mut VecDeque<Write>,
        announce: &Sender<(u64, u64)>,
        totals: &mut HopTotals,
    ) -> Result<(), String> {
        let mut drained = false;
        loop {
            while !drained && inflight.len() < window {
                match request.executor.export_direct_next(request.slot)? {
                    Some(batch) => inflight.push_back(self.post(request.peer, remote, batch)?),
                    None => drained = true,
                }
            }
            let Some(mut write) = inflight.pop_front() else {
                return Ok(());
            };
            self.complete(remote, &mut write)?;
            totals.frames += 1;
            totals.rows += write.rows;
            totals.bytes += write.bytes;
            totals.write_us += write.posted.elapsed().as_micros() as u64;
            let _ = announce.send((write.remote, write.rows));
        }
    }

    /// Allocates the receiver's buffers for `batch` and posts one WRITE of all of them.
    fn post(&self, peer: SocketAddr, remote: &str, batch: ExportedBatch) -> Result<Write, String> {
        let reply = call(peer, control_params(), &NixlEnvelope::Alloc(batch.layout))
            .and_then(|reply| AllocReply::decode(&reply))
            .inspect_err(|_| self.release(batch.token))?;
        let request = self
            .create_write(remote, &batch.src, &reply)
            .inspect_err(|_| {
                self.release(batch.token);
                let _ = call(peer, control_params(), &NixlEnvelope::Release(reply.token));
            })?;
        let posted = Instant::now();
        let in_progress = match self.agent.post_xfer_req(&request, None) {
            Ok(in_progress) => in_progress,
            Err(err) => {
                // The NIC may have started; leak the request and both batches as on a timeout.
                std::mem::forget(request);
                return Err(format!(
                    "failed to post a nixl WRITE to agent '{remote}': {err}"
                ));
            }
        };
        Ok(Write {
            request: in_progress.then_some(request),
            local: batch.token,
            remote: reply.token,
            rows: batch.rows,
            bytes: batch.src.iter().map(|(_, len)| len).sum(),
            posted,
        })
    }

    /// One WRITE of every `src` buffer into the matching buffer of `dst`, not yet posted.
    fn create_write(
        &self,
        remote: &str,
        src: &[(u64, u64)],
        dst: &AllocReply,
    ) -> Result<XferRequest, String> {
        if src.len() != dst.buffers.len() || src.iter().zip(&dst.buffers).any(|(s, d)| s.1 != d.1) {
            return Err(format!(
                "nixl agent '{remote}' allocated buffers {:?} for a batch of {src:?}",
                dst.buffers
            ));
        }
        let descriptors = || {
            XferDescList::new(MemType::Vram)
                .map_err(|err| format!("failed to create a nixl descriptor list: {err}"))
        };
        let (mut local, mut target) = (descriptors()?, descriptors()?);
        for (&(from, len), &(to, _)) in src.iter().zip(&dst.buffers) {
            local.add_desc(from as usize, len as usize, self.device);
            target.add_desc(to as usize, len as usize, dst.device as u64);
        }
        self.agent
            .create_xfer_req(XferOp::Write, &local, &target, remote, None)
            .map_err(|err| format!("failed to create a nixl WRITE to agent '{remote}': {err}"))
    }

    /// Waits for `write` to finish, then frees the sender's batch. A WRITE that fails or times out
    /// may still be in flight, so its request is leaked rather than released (dropping it would
    /// release it) and both batches stay held.
    fn complete(&self, remote: &str, write: &mut Write) -> Result<(), String> {
        let timeout = env_u64("SIRIUS_CN_NIXL_XFER_TIMEOUT_SECS")
            .map_or(Duration::from_secs(30), Duration::from_secs);
        while let Some(request) = &write.request {
            match self.agent.get_xfer_status(request) {
                Ok(XferStatus::Success) => write.request = None,
                Ok(XferStatus::InProgress) if write.posted.elapsed() < timeout => {
                    std::thread::yield_now()
                }
                status => {
                    std::mem::forget(write.request.take());
                    let bytes = write.bytes;
                    return Err(match status {
                        Ok(_) => format!(
                            "a {bytes}-byte nixl WRITE to agent '{remote}' did not finish within \
                             {timeout:?} (SIRIUS_CN_NIXL_XFER_TIMEOUT_SECS)"
                        ),
                        Err(err) => {
                            format!("a {bytes}-byte nixl WRITE to agent '{remote}' failed: {err}")
                        }
                    });
                }
            }
        }
        self.release(write.local);
        Ok(())
    }

    /// The peer's agent name, loading its metadata on first contact.
    fn remote_agent(&mut self, peer: SocketAddr) -> Result<String, String> {
        if let Some(name) = self.peers.get(&peer) {
            return Ok(name.clone());
        }
        let peer_md = call(
            peer,
            control_params(),
            &NixlEnvelope::Md(self.local_md.clone()),
        )?;
        let name = self
            .agent
            .load_remote_md(&peer_md)
            .map_err(|err| format!("failed to load the nixl metadata of {peer}: {err}"))?;
        self.peers.insert(peer, name.clone());
        Ok(name)
    }

    fn release(&self, token: u64) {
        release(&self.exchange, token);
    }
}

/// Sends `(token, rows)` frames in order, then EOS (token 0). After the first failure the rest
/// are not announced, so their receive buffers are released instead.
fn announce_all(
    peer: SocketAddr,
    slot: SenderSlot,
    names: &[String],
    frames: Receiver<(u64, u64)>,
) -> Result<(), String> {
    let mut result = Ok(());
    for (seq, (token, rows)) in frames.into_iter().enumerate() {
        if result.is_err() {
            if token != 0 {
                let _ = call(peer, control_params(), &NixlEnvelope::Release(token));
            }
            continue;
        }
        let envelope = NixlEnvelope::Packed {
            token,
            rows,
            names: names.to_vec(),
        };
        result = call(
            peer,
            packed_params(Some(slot), seq as i64, token == 0),
            &envelope,
        )
        .map(drop);
    }
    result
}

/// One `transmit_chunk` round trip to `peer`, returning the reply attachment.
fn call(
    peer: SocketAddr,
    params: PTransmitChunkParams,
    envelope: &NixlEnvelope,
) -> Result<Vec<u8>, String> {
    let (body, attachment) = prpc::call_blocking(
        peer,
        SERVICE_NAME,
        methods::TRANSMIT_CHUNK,
        params.encode_to_vec(),
        envelope.encode(),
        RPC_TIMEOUT,
    )
    .map_err(|err| format!("transmit_chunk to {peer}: {err:#}"))?;
    let status = PTransmitChunkResult::decode(body.as_slice())
        .map_err(|err| format!("transmit_chunk reply from {peer}: {err}"))?
        .status
        .ok_or_else(|| format!("transmit_chunk reply from {peer} carries no status"))?;
    if status.status_code != 0 {
        return Err(format!(
            "{peer} refused transmit_chunk: {}",
            status.error_msgs.join("; ")
        ));
    }
    Ok(attachment)
}

fn release(exchange: &DirectExchange, token: u64) {
    if let Err(err) = exchange.release(token) {
        warn!(token, error = %err, "failed to release a direct-exchange token");
    }
}

/// A positive integer from the environment.
fn env_u64(name: &str) -> Option<u64> {
    std::env::var(name).ok()?.parse().ok().filter(|&n| n > 0)
}

/// One CN per GPU: cross-process `cuda_ipc` is only proven with each CN seeing its own device.
fn check_single_visible_device(visible: Option<&str>) -> Result<(), String> {
    let Some(visible) = visible else {
        return Ok(());
    };
    let devices = visible.split(',').filter(|d| !d.trim().is_empty()).count();
    if devices > 1 {
        return Err(format!(
            "CUDA_VISIBLE_DEVICES={visible:?} names {devices} devices; pin each CN to one GPU"
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::check_single_visible_device;

    #[test]
    fn one_cn_sees_one_gpu() {
        assert!(check_single_visible_device(None).is_ok());
        assert!(check_single_visible_device(Some(" 3, ")).is_ok());
        assert!(check_single_visible_device(Some("0,1")).is_err());
    }
}
