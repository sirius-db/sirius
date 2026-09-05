//! Bounded asynchronous control/export continuations around one NIXL agent owner.
//!
//! PRPC and pack workers never touch the agent. The owner only imports metadata, posts WRITEs
//! and polls their handles; even cold reciprocal metadata requests remain serviceable. Per-peer
//! control queues isolate slow destinations, FIFO byte permits bound packing before allocation,
//! and per-stream publication gates keep EOS behind every earlier frame.

use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::mpsc::{Receiver, SyncSender, TryRecvError, sync_channel};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use nixl_sys::{MemType, XferDescList, XferOp, XferRequest, XferStatus};
use prost::Message;
use tokio::sync::{OnceCell, OwnedSemaphorePermit, Semaphore, oneshot};
use tracing::{info, warn};

use super::agent_tier::{TransportState, rpc_exchange_md};
use super::fair::{FairPeers, PublicationOrder, Worker};
use super::{MdReply, RemoteSendSpec, TransportRequest};
use crate::exchange_protocol::{PROTOCOL_VERSION, process_epoch};
use crate::fragment_executor::{FragmentExecutor, StagedBatch};
use crate::proto::starrocks::p_internal_service_brpc::methods;
use crate::proto::starrocks::{
    PExchangeLeaseIdentity, PStagingLeaseRequest, PStagingLeaseResult, PTransmitPackedParams,
    PTransmitPackedResult, PUniqueId, StatusPb,
};
use crate::prpc_client::PrpcClient;
use crate::tunable::Tunables;

const PACK_SLACK: u64 = 8 << 20;
const BYTE_UNIT: u64 = 1024;
const MAX_STREAMS: usize = 128;
const PER_QUERY_STREAMS: usize = 64;
const RETRY_PAUSE: Duration = Duration::from_millis(2);

fn units(bytes: u64) -> u32 {
    bytes
        .div_ceil(BYTE_UNIT)
        .try_into()
        .expect("staging capacity validated at startup")
}

async fn acquire_bytes(
    pool: &Arc<Semaphore>,
    count: u32,
    what: &str,
) -> Result<OwnedSemaphorePermit, String> {
    match tokio::time::timeout(
        Tunables::get().rpc_timeout,
        Arc::clone(pool).acquire_many_owned(count),
    )
    .await
    {
        Ok(Ok(permit)) => Ok(permit),
        Ok(Err(_)) => Err(format!("{what} admission closed")),
        Err(_) => Err(format!(
            "{what} capacity wait exceeded {:?}",
            Tunables::get().rpc_timeout
        )),
    }
}

struct Session {
    agent_name: String,
    epoch: u64,
    max_payload: u64,
}

struct Peer {
    key: String,
    control: Worker<PrpcClient>,
    session: OnceCell<Session>,
    frames: Arc<Semaphore>,
}

struct Shared {
    commands: SyncSender<AgentCommand>,
    executor: Arc<dyn FragmentExecutor>,
    packing: Worker<Arc<dyn FragmentExecutor>>,
    /// Fragment retirement still needs the engine owner. Keep it off the buffer-only packing
    /// queue, or a completed sender could block unrelated exports behind another engine run.
    cleanup: Worker<Arc<dyn FragmentExecutor>>,
    agent_name: String,
    local_md: Vec<u8>,
    staging_base: u64,
    pack_reservation: u64,
    tx_bytes: Arc<Semaphore>,
    peers: Mutex<HashMap<String, Arc<Peer>>>,
    query_slots: Mutex<HashMap<String, Arc<Semaphore>>>,
    next_request: AtomicU64,
    closing: AtomicBool,
    frame_tasks: AtomicUsize,
}

struct FrameTaskGuard(Arc<Shared>);

impl Drop for FrameTaskGuard {
    fn drop(&mut self) {
        self.0.frame_tasks.fetch_sub(1, Ordering::AcqRel);
    }
}

struct QueryGuard {
    shared: Arc<Shared>,
    key: String,
    pool: Arc<Semaphore>,
    permit: Option<OwnedSemaphorePermit>,
}

impl Drop for QueryGuard {
    fn drop(&mut self) {
        self.permit.take();
        let mut queries = self
            .shared
            .query_slots
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        if Arc::strong_count(&self.pool) == 2 && self.pool.available_permits() == PER_QUERY_STREAMS
        {
            queries.remove(&self.key);
        }
    }
}

struct WriteError {
    message: String,
    uncertain: bool,
}

enum AgentCommand {
    Import {
        peer: MdReply,
        reply: oneshot::Sender<Result<(), String>>,
    },
    Write {
        peer: String,
        local: u64,
        remote: u64,
        bytes: u64,
        reply: oneshot::Sender<Result<Duration, WriteError>>,
    },
}

/// Descriptors and handle remain on the agent owner, including when a timeout makes their
/// quiescence unknown. The binding's destructor is not treated as an abort/completion fence.
struct ActiveWrite {
    request: XferRequest,
    // Built with add_desc(address, length), not borrowed NixlDescriptor references. Their
    // Rust lists own the values; SourceLease and the receiver ledger own the pointed-to bytes.
    _local: XferDescList<'static>,
    _remote: XferDescList<'static>,
    peer: String,
    bytes: u64,
    posted: Instant,
    reply: Option<oneshot::Sender<Result<Duration, WriteError>>>,
}

/// Releasing the source is legal only before posting or after confirmed DONE. Unwinding while
/// a WRITE is uncertain retains its arena lease, byte reservation and executor for process life.
struct SourceLease {
    executor: Arc<dyn FragmentExecutor>,
    offset: u64,
    length: u64,
    reservation: Option<OwnedSemaphorePermit>,
    uncertain: bool,
}

impl SourceLease {
    fn release(&mut self) -> Result<(), String> {
        if self.uncertain {
            return Err("cannot release an uncertain WRITE source".to_string());
        }
        if self.length > 0 {
            self.executor.staging_release(self.offset)?;
            self.length = 0;
        }
        self.reservation.take();
        Ok(())
    }
}

impl Drop for SourceLease {
    fn drop(&mut self) {
        if self.uncertain {
            warn!(
                offset = self.offset,
                bytes = self.length,
                "quarantining uncertain WRITE source until CN restart"
            );
            if let Some(permit) = self.reservation.take() {
                permit.forget();
            }
            std::mem::forget(Arc::clone(&self.executor));
        } else if let Err(err) = self.release() {
            warn!(error = %err, "failed to release completed WRITE source");
            // A failed release cannot be counted as reusable bytes.
            if let Some(permit) = self.reservation.take() {
                permit.forget();
            }
        }
    }
}

fn check_status(what: &str, status: &StatusPb) -> Result<(), String> {
    if status.status_code == 0 {
        Ok(())
    } else {
        Err(format!(
            "{what} failed with status {}: {}",
            status.status_code,
            status.error_msgs.join("; ")
        ))
    }
}

impl Shared {
    fn peer(&self, host: &str, port: u16) -> Result<Arc<Peer>, String> {
        let key = format!("{host}:{port}");
        let mut peers = self.peers.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(peer) = peers.get(&key) {
            return Ok(Arc::clone(peer));
        }
        let config = Tunables::get();
        if peers.len() >= config.transfer_peers {
            return Err(format!(
                "optimized peer admission full: {} peers",
                config.transfer_peers
            ));
        }
        let peer = Arc::new(Peer {
            key: key.clone(),
            control: Worker::start("nixl-peer-control", PrpcClient::new(host, port), 32)?,
            session: OnceCell::new(),
            frames: Arc::new(Semaphore::new(config.transfer_window)),
        });
        peers.insert(key, Arc::clone(&peer));
        Ok(peer)
    }

    async fn import(&self, peer: MdReply) -> Result<(), String> {
        let (reply, answer) = oneshot::channel();
        self.commands
            .try_send(AgentCommand::Import { peer, reply })
            .map_err(|err| format!("agent import admission failed: {err}"))?;
        answer
            .await
            .map_err(|_| "agent owner dropped import response".to_string())?
    }

    async fn write(
        &self,
        peer: &str,
        source: &mut SourceLease,
        remote: u64,
        bytes: u64,
    ) -> Result<Duration, String> {
        let (reply, answer) = oneshot::channel();
        self.commands
            .try_send(AgentCommand::Write {
                peer: peer.to_string(),
                local: self.staging_base + source.offset,
                remote,
                bytes,
                reply,
            })
            .map_err(|err| format!("agent WRITE admission failed: {err}"))?;
        source.uncertain = true;
        match answer.await {
            Ok(Ok(elapsed)) => {
                source.uncertain = false;
                Ok(elapsed)
            }
            Ok(Err(error)) => {
                source.uncertain = error.uncertain;
                Err(error.message)
            }
            Err(_) => Err("agent owner dropped WRITE response; source is quarantined".to_string()),
        }
    }

    fn identity(&self, spec: Option<&RemoteSendSpec>, seq: i64) -> PExchangeLeaseIdentity {
        PExchangeLeaseIdentity {
            sender_epoch: process_epoch(),
            request_id: self.next_request.fetch_add(1, Ordering::Relaxed),
            query_id: spec.and_then(|s| s.query_id).map(|id| {
                let (hi, lo) = id.as_halves();
                PUniqueId { hi, lo }
            }),
            finst_id: spec.map(|s| {
                let (hi, lo) = s.slot.fragment_instance_id.as_halves();
                PUniqueId { hi, lo }
            }),
            node_id: spec.map(|s| s.slot.node_id),
            sender_id: spec.map(|s| s.slot.sender_id),
            seq: spec.map(|_| seq),
            canary: Some(spec.is_none()),
        }
    }

    async fn session<'a>(&self, peer: &'a Peer) -> Result<&'a Session, String> {
        peer.session.get_or_try_init(|| async {
            let start = Instant::now();
            let name = self.agent_name.clone();
            let md = self.local_md.clone();
            let remote = peer.control.call(move |client| rpc_exchange_md(client, &name, &md)).await?;
            if remote.lease_protocol != Some(PROTOCOL_VERSION) || remote.process_epoch.unwrap_or(0) == 0 {
                return Err(format!("peer {} does not support owned exchange protocol {PROTOCOL_VERSION}", peer.key));
            }
            let session = Session { agent_name: remote.agent_name.clone(), epoch: remote.process_epoch.unwrap(), max_payload: self.pack_reservation - PACK_SLACK };
            self.import(remote).await?;
            let bytes = Tunables::get().canary_bytes;
            if bytes > session.max_payload { return Err("canary exceeds negotiated maximum packed frame".to_string()); }
            let permit = acquire_bytes(&self.tx_bytes, units(bytes), "canary TX bytes").await?;
            let offset = self.executor.staging_lease(bytes)?;
            let mut source = SourceLease { executor: Arc::clone(&self.executor), offset, length: bytes, reservation: Some(permit), uncertain: false };
            let identity = self.identity(None, 0);
            let grant = match grant(peer, &session, &identity, bytes).await {
                Ok(grant) => grant,
                Err(err) => {
                    let _ = release_remote(peer, &session, &identity, None, bytes, false).await;
                    return Err(err);
                }
            };
            let result = async {
                self.write(&session.agent_name, &mut source, grant.remote_addr.unwrap(), bytes.min(1 << 20)).await?;
                let elapsed = self.write(&session.agent_name, &mut source, grant.remote_addr.unwrap(), bytes).await?;
                publish(peer, &session, PTransmitPackedParams {
                    canary: Some(true), offset: grant.offset, length: Some(bytes),
                    receiver_epoch: Some(session.epoch), lease_token: grant.lease_token,
                    identity: Some(identity.clone()), ..Default::default()
                }, Vec::new()).await?;
                let gbps = bytes as f64 / elapsed.as_secs_f64() / 1e9;
                let floor = Tunables::get().canary_floor_gbps;
                info!(peer = %peer.key, gbps, bytes, floor_gbps = floor, "nixl bandwidth canary");
                if floor > 0.0 && gbps < floor {
                    return Err(format!("nixl link to {} measured {gbps:.2} GB/s below {floor} GB/s floor", peer.key));
                }
                Ok(())
            }.await;
            if result.is_err() {
                let _ = release_remote(peer, &session, &identity, grant.lease_token, bytes, source.uncertain).await;
            }
            result?;
            source.release()?;
            info!(peer = %peer.key, epoch = session.epoch, setup_ms = start.elapsed().as_millis(), "optimized nixl peer session ready");
            Ok(session)
        }).await
    }
}

async fn grant(
    peer: &Peer,
    session: &Session,
    identity: &PExchangeLeaseIdentity,
    length: u64,
) -> Result<PStagingLeaseResult, String> {
    let started = Instant::now();
    loop {
        let body = PStagingLeaseRequest {
            length,
            identity: Some(identity.clone()),
            receiver_epoch: Some(session.epoch),
            lease_token: None,
            operation: Some(0),
        }
        .encode_to_vec();
        let response = peer
            .control
            .call(move |client| {
                let response =
                    client.call_idempotent(methods::REQUEST_STAGING_LEASE, body, Vec::new())?;
                PStagingLeaseResult::decode(response.body.as_slice())
                    .map_err(|e| format!("invalid owned grant reply: {e}"))
            })
            .await?;
        if response.retryable_unavailable == Some(true) {
            if started.elapsed() >= Tunables::get().rpc_timeout {
                return Err(format!(
                    "receive credit wait at {} exceeded {:?}",
                    peer.key,
                    Tunables::get().rpc_timeout
                ));
            }
            tokio::time::sleep(RETRY_PAUSE).await;
            continue;
        }
        check_status("owned staging grant", &response.status)?;
        if response.receiver_epoch != Some(session.epoch)
            || response.lease_token.unwrap_or(0) == 0
            || response.remote_addr.unwrap_or(0) == 0
            || response.offset.is_none()
        {
            return Err("owned staging grant has stale epoch or missing address/token".to_string());
        }
        if response.max_batch_bytes.is_some_and(|max| length > max) {
            return Err("owned staging grant exceeds peer maximum frame".to_string());
        }
        return Ok(response);
    }
}

async fn publish(
    peer: &Peer,
    session: &Session,
    params: PTransmitPackedParams,
    metadata: Vec<u8>,
) -> Result<(), String> {
    if params.receiver_epoch != Some(session.epoch) {
        return Err("publication epoch mismatch".to_string());
    }
    let start = Instant::now();
    loop {
        let body = params.encode_to_vec();
        let metadata = metadata.clone();
        let result = peer
            .control
            .call(move |client| {
                let response = client.call_idempotent(methods::TRANSMIT_PACKED, body, metadata)?;
                PTransmitPackedResult::decode(response.body.as_slice())
                    .map_err(|e| format!("invalid owned publication reply: {e}"))
            })
            .await?;
        if result.retryable_pending == Some(true) {
            if start.elapsed() >= Tunables::get().rpc_timeout {
                return Err(format!(
                    "ingress publication wait at {} exceeded {:?}",
                    peer.key,
                    Tunables::get().rpc_timeout
                ));
            }
            tokio::time::sleep(RETRY_PAUSE).await;
            continue;
        }
        return check_status("owned publication", &result.status);
    }
}

async fn release_remote(
    peer: &Peer,
    session: &Session,
    identity: &PExchangeLeaseIdentity,
    token: Option<u64>,
    length: u64,
    uncertain: bool,
) -> Result<(), String> {
    let body = PStagingLeaseRequest {
        length,
        identity: Some(identity.clone()),
        receiver_epoch: Some(session.epoch),
        lease_token: token,
        operation: Some(if uncertain { 2 } else { 1 }),
    }
    .encode_to_vec();
    let result = peer
        .control
        .call(move |client| {
            let response =
                client.call_idempotent(methods::REQUEST_STAGING_LEASE, body, Vec::new())?;
            PStagingLeaseResult::decode(response.body.as_slice())
                .map_err(|e| format!("invalid owned release reply: {e}"))
        })
        .await?;
    if result.retryable_unavailable == Some(true) {
        return Err("remote release retained an active ingress reader".to_string());
    }
    check_status("owned staging release", &result.status)
}

async fn send_frame(
    shared: Arc<Shared>,
    peer: Arc<Peer>,
    spec: RemoteSendSpec,
    seq: i64,
    batch: StagedBatch,
    mut source: SourceLease,
    _window: OwnedSemaphorePermit,
    order: Arc<PublicationOrder>,
) -> Result<(), String> {
    let session = shared.session(&peer).await?;
    let identity = shared.identity(Some(&spec), seq);
    let mut remote = None;
    let result = async {
        if order.failed.load(Ordering::Acquire) {
            return Err("sender already failed".to_string());
        }
        let (offset, token) = if batch.len > 0 {
            let lease = grant(&peer, session, &identity, batch.len).await?;
            remote = Some(lease.clone());
            shared
                .write(
                    &session.agent_name,
                    &mut source,
                    lease.remote_addr.unwrap(),
                    batch.len,
                )
                .await?;
            (lease.offset, lease.lease_token)
        } else {
            (Some(0), None)
        };
        // Transfer completion, publication acknowledgement, and receive reuse are separate.
        source.release()?;
        order.wait(seq).await?;
        publish(
            &peer,
            session,
            PTransmitPackedParams {
                finst_id: identity.finst_id,
                node_id: identity.node_id,
                sender_id: identity.sender_id,
                eos: Some(false),
                seq: Some(seq),
                offset,
                length: Some(batch.len),
                column_names: spec.names.clone(),
                rows: batch.rows,
                canary: Some(false),
                receiver_epoch: Some(session.epoch),
                lease_token: token,
                identity: Some(identity.clone()),
            },
            batch.metadata,
        )
        .await?;
        order.advance(seq).await;
        Ok(())
    }
    .await;
    if let Err(ref error) = result {
        order.fail();
        if batch.len > 0 {
            let token = remote.as_ref().and_then(|lease| lease.lease_token);
            if let Err(cleanup) = release_remote(
                &peer,
                session,
                &identity,
                token,
                batch.len,
                source.uncertain,
            )
            .await
            {
                warn!(peer = %peer.key, error = %error, cleanup = %cleanup, "owned frame cleanup retained receiver allocation");
            }
        }
    }
    result
}

async fn send_stream(shared: Arc<Shared>, spec: RemoteSendSpec) -> Result<(), String> {
    let peer = shared.peer(&spec.host, spec.brpc_port)?;
    shared.session(&peer).await?;
    let query = spec.query_id.map_or_else(
        || format!("receiver:{}", spec.slot.fragment_instance_id),
        |id| id.to_string(),
    );
    let query_limit = {
        let mut queries = shared.query_slots.lock().unwrap_or_else(|e| e.into_inner());
        Arc::clone(
            queries
                .entry(query.clone())
                .or_insert_with(|| Arc::new(Semaphore::new(PER_QUERY_STREAMS))),
        )
    };
    let query_permit = acquire_bytes(&query_limit, 1, "query stream").await?;
    let _query_slot = QueryGuard {
        shared: Arc::clone(&shared),
        key: query,
        pool: query_limit,
        permit: Some(query_permit),
    };
    let order = Arc::new(PublicationOrder::default());
    let mut pending = VecDeque::new();
    let mut seq = 0;
    let mut bytes = 0;
    let mut failure = None;
    let window = Tunables::get().transfer_window;
    loop {
        if shared.closing.load(Ordering::Acquire) {
            failure = Some("transport is shutting down".to_string());
            order.fail();
            break;
        }
        if pending.len() >= window {
            let result = pending.pop_front().unwrap();
            match result.await {
                Ok(Ok(())) => {}
                Ok(Err(err)) => {
                    failure = Some(err);
                    order.fail();
                    break;
                }
                Err(err) => {
                    failure = Some(format!("frame task failed: {err}"));
                    order.fail();
                    break;
                }
            }
        }
        let frame_permit = match acquire_bytes(&peer.frames, 1, "peer frame window").await {
            Ok(permit) => permit,
            Err(err) => {
                failure = Some(err);
                order.fail();
                break;
            }
        };
        let mut reservation = match acquire_bytes(
            &shared.tx_bytes,
            units(shared.pack_reservation),
            "TX pack bytes",
        )
        .await
        {
            Ok(permit) => permit,
            Err(err) => {
                failure = Some(err);
                order.fail();
                break;
            }
        };
        let slot = spec.slot;
        let pack_started = Instant::now();
        let packed = loop {
            let attempt = shared
                .packing
                .call(move |executor| executor.export_packed_next(slot))
                .await;
            if attempt
                .as_ref()
                .err()
                .is_some_and(|err| err.contains("EXPORT_CAPACITY_UNAVAILABLE"))
                && pack_started.elapsed() < Tunables::get().rpc_timeout
                && !order.failed.load(Ordering::Acquire)
                && !shared.closing.load(Ordering::Acquire)
            {
                tokio::time::sleep(RETRY_PAUSE).await;
                continue;
            }
            break attempt;
        };
        let batch = match packed {
            Ok(Some(batch)) => batch,
            Ok(None) => break,
            Err(err) => {
                failure = Some(err);
                order.fail();
                break;
            }
        };
        let allocated = if batch.len == 0 {
            0
        } else {
            batch.len + PACK_SLACK
        };
        if allocated > shared.pack_reservation {
            if batch.len > 0 {
                let _ = shared.executor.staging_release(batch.offset);
            }
            failure = Some(format!(
                "packed frame allocation {allocated} exceeds reserved {} bytes",
                shared.pack_reservation
            ));
            order.fail();
            break;
        }
        let spare = reservation.num_permits() - units(allocated) as usize;
        if spare > 0 {
            drop(reservation.split(spare));
        }
        let source = SourceLease {
            executor: Arc::clone(&shared.executor),
            offset: batch.offset,
            length: batch.len,
            reservation: Some(reservation),
            uncertain: false,
        };
        bytes += batch.len;
        info!(peer = %peer.key, seq, bytes = batch.len, pack_wait_us = pack_started.elapsed().as_micros(), window, "optimized exchange frame packed");
        shared.frame_tasks.fetch_add(1, Ordering::AcqRel);
        let frame_guard = FrameTaskGuard(Arc::clone(&shared));
        let frame = send_frame(
            Arc::clone(&shared),
            Arc::clone(&peer),
            spec.clone(),
            seq,
            batch,
            source,
            frame_permit,
            Arc::clone(&order),
        );
        pending.push_back(tokio::spawn(async move {
            let _guard = frame_guard;
            frame.await
        }));
        seq += 1;
    }
    // Never detach frames: their source leases and WRITE readers outlive a cancelled stream.
    while let Some(task) = pending.pop_front() {
        match task.await {
            Ok(Ok(())) => {}
            Ok(Err(err)) => {
                if failure.is_none() {
                    failure = Some(err);
                }
                order.fail();
            }
            Err(err) => {
                if failure.is_none() {
                    failure = Some(format!("frame task failed: {err}"));
                }
                order.fail();
            }
        }
    }
    if let Some(err) = failure {
        return Err(err);
    }
    let session = shared.session(&peer).await?;
    let identity = shared.identity(Some(&spec), seq);
    publish(
        &peer,
        session,
        PTransmitPackedParams {
            finst_id: identity.finst_id,
            node_id: identity.node_id,
            sender_id: identity.sender_id,
            eos: Some(true),
            seq: Some(seq),
            column_names: spec.names.clone(),
            receiver_epoch: Some(session.epoch),
            identity: Some(identity),
            ..Default::default()
        },
        Vec::new(),
    )
    .await?;
    let slot = spec.slot;
    shared
        .cleanup
        .call(move |executor| executor.drop_parked(slot))
        .await?;
    info!(stream_id = spec.slot.node_id, sender_id = spec.slot.sender_id, dest = %peer.key, batches = seq, bytes, window, "transmitted batches via nixl");
    Ok(())
}

fn post(
    state: &TransportState,
    command: AgentCommand,
    active: &mut Vec<ActiveWrite>,
    quarantined: &mut Vec<ActiveWrite>,
) {
    match command {
        AgentCommand::Import { peer, reply } => {
            let result = state
                .agent
                .load_remote_md(&peer.metadata)
                .map_err(|err| format!("cannot import peer {}: {err}", peer.agent_name))
                .and_then(|name| {
                    if name == peer.agent_name {
                        Ok(())
                    } else {
                        Err("imported peer name differs from handshake".to_string())
                    }
                });
            let _ = reply.send(result);
        }
        AgentCommand::Write {
            peer,
            local,
            remote,
            bytes,
            reply,
        } => {
            let prepared = (|| {
                let mut local_desc =
                    XferDescList::new(MemType::Vram).map_err(|err| err.to_string())?;
                local_desc.add_desc(local as usize, bytes as usize, 0);
                let mut remote_desc =
                    XferDescList::new(MemType::Vram).map_err(|err| err.to_string())?;
                remote_desc.add_desc(remote as usize, bytes as usize, 0);
                let request = state
                    .agent
                    .create_xfer_req(XferOp::Write, &local_desc, &remote_desc, &peer, None)
                    .map_err(|err| err.to_string())?;
                Ok::<_, String>((request, local_desc, remote_desc))
            })();
            let (request, local_desc, remote_desc) = match prepared {
                Ok(prepared) => prepared,
                Err(message) => {
                    let _ = reply.send(Err(WriteError {
                        message,
                        uncertain: false,
                    }));
                    return;
                }
            };
            let mut write = ActiveWrite {
                request,
                _local: local_desc,
                _remote: remote_desc,
                peer,
                bytes,
                posted: Instant::now(),
                reply: Some(reply),
            };
            match state.agent.post_xfer_req(&write.request, None) {
                Ok(true) => active.push(write),
                Ok(false) => {
                    let _ = write.reply.take().unwrap().send(Ok(write.posted.elapsed()));
                }
                Err(err) => {
                    let message = format!(
                        "posting {}-byte WRITE failed: {err}; transfer quiescence unknown",
                        write.bytes
                    );
                    let _ = write.reply.take().unwrap().send(Err(WriteError {
                        message,
                        uncertain: true,
                    }));
                    quarantined.push(write);
                }
            }
        }
    }
}

pub(super) fn run(mut state: TransportState, requests: Receiver<TransportRequest>) {
    let capacity = match state.executor.staging_info() {
        Ok((_, capacity)) => capacity,
        Err(err) => {
            warn!(error = %err, "optimized transport staging unavailable");
            return;
        }
    };
    let pack_reservation = (capacity / 4 / BYTE_UNIT) * BYTE_UNIT;
    if pack_reservation <= PACK_SLACK || capacity / 2 / BYTE_UNIT > u32::MAX as u64 {
        warn!(
            capacity,
            "optimized staging capacity outside supported byte-credit range"
        );
        return;
    }
    let runtime = match tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
    {
        Ok(runtime) => runtime,
        Err(err) => {
            warn!(error = %err, "cannot start transport continuation runtime");
            return;
        }
    };
    let (commands, incoming) = sync_channel(256);
    let packing = match Worker::start("nixl-pack-export", Arc::clone(&state.executor), 256) {
        Ok(worker) => worker,
        Err(err) => {
            warn!(error = %err);
            return;
        }
    };
    let cleanup = match Worker::start(
        "nixl-output-retire",
        Arc::clone(&state.executor),
        MAX_STREAMS,
    ) {
        Ok(worker) => worker,
        Err(err) => {
            warn!(error = %err, "cannot start output retirement worker");
            return;
        }
    };
    let shared = Arc::new(Shared {
        commands,
        executor: Arc::clone(&state.executor),
        packing,
        cleanup,
        agent_name: state.agent_name.clone(),
        local_md: state.local_md.clone(),
        staging_base: state.staging_base,
        pack_reservation,
        tx_bytes: Arc::new(Semaphore::new((capacity / 2 / BYTE_UNIT) as usize)),
        peers: Mutex::new(HashMap::new()),
        query_slots: Mutex::new(HashMap::new()),
        next_request: AtomicU64::new(1),
        closing: AtomicBool::new(false),
        frame_tasks: AtomicUsize::new(0),
    });
    info!(
        tx_bytes = capacity / 2,
        max_pack_allocation = pack_reservation,
        window = Tunables::get().transfer_window,
        "optimized fair transport ready"
    );
    let mut tasks = Vec::new();
    let mut active = Vec::new();
    let mut quarantined = Vec::new();
    let mut closed = false;
    let mut fair_writes = FairPeers::new();
    loop {
        tasks.retain(|task: &tokio::task::JoinHandle<()>| !task.is_finished());
        for _ in 0..128 {
            match requests.try_recv() {
                Ok(TransportRequest::ExchangeMd {
                    peer_agent_name,
                    peer_metadata,
                    respond,
                }) => {
                    let _ = respond.send(state.exchange_md(&peer_agent_name, &peer_metadata));
                }
                Ok(TransportRequest::SendFragment { spec, respond }) => {
                    if tasks.len() >= MAX_STREAMS || closed {
                        let _ =
                            respond
                                .send(Err("optimized transport stream admission limit reached"
                                    .to_string()));
                        continue;
                    }
                    let shared = Arc::clone(&shared);
                    tasks.push(runtime.spawn(async move {
                        let slot = spec.slot;
                        let result = send_stream(Arc::clone(&shared), spec).await;
                        if result.is_err() {
                            let _ = shared
                                .cleanup
                                .call(move |executor| executor.drop_parked(slot))
                                .await;
                        }
                        let _ = respond.send(result);
                    }));
                }
                Ok(TransportRequest::WarmSession {
                    host,
                    brpc_port,
                    respond,
                    ..
                }) => {
                    if tasks.len() >= MAX_STREAMS || closed {
                        let _ = respond
                            .send(Err("optimized session admission limit reached".to_string()));
                        continue;
                    }
                    let shared = Arc::clone(&shared);
                    tasks.push(runtime.spawn(async move {
                        let result = match shared.peer(&host, brpc_port) {
                            Ok(peer) => shared.session(&peer).await.map(|_| ()),
                            Err(err) => Err(err),
                        };
                        let _ = respond.send(result);
                    }));
                }
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => {
                    closed = true;
                    shared.closing.store(true, Ordering::Release);
                    shared.tx_bytes.close();
                    for peer in shared
                        .peers
                        .lock()
                        .unwrap_or_else(|e| e.into_inner())
                        .values()
                    {
                        peer.frames.close();
                    }
                    for query in shared
                        .query_slots
                        .lock()
                        .unwrap_or_else(|e| e.into_inner())
                        .values()
                    {
                        query.close();
                    }
                    break;
                }
            }
        }
        for _ in 0..256 {
            match incoming.try_recv() {
                Ok(command) => post(&state, command, &mut active, &mut quarantined),
                Err(_) => break,
            }
        }
        for write in &active {
            fair_writes.insert(write.peer.clone());
        }
        for peer in fair_writes.pass() {
            let mut index = 0;
            while index < active.len() {
                if active[index].peer != peer {
                    index += 1;
                    continue;
                }
                let result = if active[index].posted.elapsed() > Tunables::get().xfer_timeout {
                    Some(Err(format!(
                        "{}-byte WRITE to {peer} exceeded {:?}",
                        active[index].bytes,
                        Tunables::get().xfer_timeout
                    )))
                } else {
                    match state.agent.get_xfer_status(&active[index].request) {
                        Ok(XferStatus::Success) => Some(Ok(active[index].posted.elapsed())),
                        Ok(XferStatus::InProgress) => None,
                        Err(err) => Some(Err(format!("polling WRITE to {peer} failed: {err}"))),
                    }
                };
                let Some(result) = result else {
                    index += 1;
                    continue;
                };
                let mut write = active.swap_remove(index);
                match result {
                    Ok(elapsed) => {
                        let _ = write.reply.take().unwrap().send(Ok(elapsed));
                    }
                    Err(message) => {
                        warn!(peer = %write.peer, bytes = write.bytes, error = %message, "quarantining uncertain NIXL request");
                        let _ = write.reply.take().unwrap().send(Err(WriteError {
                            message,
                            uncertain: true,
                        }));
                        quarantined.push(write);
                    }
                }
            }
        }
        if closed
            && tasks.is_empty()
            && active.is_empty()
            && shared.frame_tasks.load(Ordering::Acquire) == 0
        {
            break;
        }
        if active.is_empty() {
            std::thread::sleep(Duration::from_micros(100));
        } else {
            std::thread::yield_now();
        }
    }
    drop(runtime);
    drop(shared);
    if !quarantined.is_empty() {
        let bytes: u64 = quarantined.iter().map(|write| write.bytes).sum();
        warn!(
            bytes,
            handles = quarantined.len(),
            "retaining uncertain requests, registrations and engine until process exit"
        );
        std::mem::forget(quarantined);
        std::mem::forget(state);
    }
}
