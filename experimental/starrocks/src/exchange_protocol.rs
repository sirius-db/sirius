//! Owned receive grants for the optimized exchange protocol.
//!
//! A grant reserves both an arena range and its exact packed host evacuation allocation. The
//! latter is owned until publication consumes it. Neither a timeout nor FE cancellation proves
//! a remote WRITE has stopped: those paths quarantine grants, and only a sender's explicit
//! quiescence acknowledgement can reclaim them. Completed identities are retained, not evicted
//! into a possible replay. Once the bounded ledger fills, new admission fails explicitly.

use std::collections::{HashMap, HashSet};
use std::hash::{Hash, Hasher};
use std::sync::{Mutex, OnceLock};
use std::time::{SystemTime, UNIX_EPOCH};

use crate::fragment_executor::FragmentExecutor;
use crate::proto::starrocks::{
    PExchangeLeaseIdentity, PStagingLeaseRequest, PTransmitPackedParams,
};
use crate::result_store::FragmentInstanceId;

pub(crate) const PROTOCOL_VERSION: u32 = 1;
const MAX_RECORDS: usize = 262_144;
const MAX_METADATA: usize = 1 << 20;

pub(crate) fn optimized_enabled() -> bool {
    std::env::var("SIRIUS_EXCHANGE_OPTIMIZED").as_deref() == Ok("1")
}

/// Shared by the service and transport within one CN, changed by every process restart.
pub(crate) fn process_epoch() -> u64 {
    static EPOCH: OnceLock<u64> = OnceLock::new();
    *EPOCH.get_or_init(|| {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system clock predates the Unix epoch")
            .as_nanos();
        ((nanos as u64) ^ ((std::process::id() as u64) << 32)).max(1)
    })
}

#[derive(Clone, Debug)]
pub(crate) struct Grant {
    pub(crate) token: u64,
    pub(crate) offset: u64,
    pub(crate) address: u64,
    pub(crate) max_batch: u64,
}

#[derive(Debug)]
pub(crate) enum Admission {
    Granted(Grant),
    Unavailable { max_batch: u64 },
    Released,
}

#[derive(Debug)]
pub(crate) enum Publication {
    Fresh {
        reservation: u64,
        offset: u64,
        retired: bool,
    },
    Pending,
    Complete(Result<(), String>),
}

#[derive(Debug)]
enum Phase {
    Granted { reservation: u64 },
    Copying,
    Quarantined { reservation: u64 },
    Complete(Result<(), String>),
}

#[derive(Debug)]
struct Record {
    identity: PExchangeLeaseIdentity,
    length: u64,
    charged: u64,
    grant: Grant,
    publication: Option<u64>,
    phase: Phase,
}

#[derive(Debug, Default)]
struct State {
    records: HashMap<(u64, u64), Record>,
    tokens: HashMap<u64, (u64, u64)>,
    live: u64,
    peak: u64,
    peer_live: HashMap<u64, u64>,
    next_token: u64,
    active_frames: usize,
    peer_frames: HashMap<u64, usize>,
    retired_queries: HashSet<FragmentInstanceId>,
}

#[derive(Debug, Default)]
pub(crate) struct ReceiveLedger {
    state: Mutex<State>,
}

fn identity_key(identity: &PExchangeLeaseIdentity) -> Result<(u64, u64), String> {
    if identity.sender_epoch == 0 || identity.request_id == 0 {
        return Err("owned lease requires nonzero sender epoch and request identity".to_string());
    }
    if !identity.canary.unwrap_or(false)
        && (identity.query_id.is_none()
            || identity.finst_id.is_none()
            || identity.node_id.is_none()
            || identity.sender_id.is_none_or(|sender| sender < 0)
            || identity.seq.is_none_or(|seq| seq < 0))
    {
        return Err(
            "owned lease is missing its query, receiver, exchange, sender or sequence".to_string(),
        );
    }
    Ok((identity.sender_epoch, identity.request_id))
}

pub(crate) fn validate_frame_identity(params: &PTransmitPackedParams) -> Result<(), String> {
    validate_epoch(params.receiver_epoch)?;
    let identity = params
        .identity
        .as_ref()
        .ok_or("owned publication requires an identity")?;
    identity_key(identity)?;
    if identity.canary.unwrap_or(false) != params.canary.unwrap_or(false)
        || (!identity.canary.unwrap_or(false)
            && (identity.finst_id != params.finst_id
                || identity.node_id != params.node_id
                || identity.sender_id != params.sender_id
                || identity.seq != params.seq))
    {
        return Err("owned publication disagrees with its granted frame identity".to_string());
    }
    Ok(())
}

pub(crate) fn validate_epoch(epoch: Option<u64>) -> Result<(), String> {
    if epoch != Some(process_epoch()) {
        return Err(format!(
            "owned lease receiver epoch mismatch: received {epoch:?}, current {}",
            process_epoch()
        ));
    }
    Ok(())
}

impl ReceiveLedger {
    pub(crate) fn request(
        &self,
        executor: &dyn FragmentExecutor,
        request: &PStagingLeaseRequest,
    ) -> Result<Admission, String> {
        validate_epoch(request.receiver_epoch)?;
        let identity = request
            .identity
            .as_ref()
            .ok_or("owned lease identity is required")?;
        let key = identity_key(identity)?;
        let (base, capacity) = executor.staging_info()?;
        // Export has a separate half-arena budget, including its pack slack. A receive grant
        // never consumes that progress reserve, even if every peer is sending simultaneously.
        let receive_limit = (capacity / 2) & !255;
        let max_batch = (receive_limit / 2) & !255;
        let mut state = self.state.lock().unwrap_or_else(|p| p.into_inner());
        let operation = request.operation.unwrap_or(0);
        if operation != 0 {
            if operation != 1 && operation != 2 {
                return Err(format!("unknown owned lease operation {operation}"));
            }
            let Some(record) = state.records.get_mut(&key) else {
                // A grant request may have failed before allocation, or its reply may have
                // been lost after allocation. Remember an abort that outran its delayed grant
                // too: otherwise the late request could allocate after we acknowledged cleanup.
                if state.records.len() >= MAX_RECORDS {
                    return Err(
                        "owned lease replay ledger cannot admit another abort tombstone"
                            .to_string(),
                    );
                }
                state.next_token += 1;
                let token = state.next_token;
                state.records.insert(
                    key,
                    Record {
                        identity: identity.clone(),
                        length: request.length,
                        charged: 0,
                        grant: Grant {
                            token,
                            offset: 0,
                            address: 0,
                            max_batch,
                        },
                        publication: None,
                        phase: Phase::Complete(Err(
                            "owned lease was aborted before grant admission".to_string(),
                        )),
                    },
                );
                return Ok(Admission::Released);
            };
            Self::check_record(record, identity, request.length)?;
            if request
                .lease_token
                .is_some_and(|token| token != record.grant.token)
            {
                return Err("owned lease token does not match its identity".to_string());
            }
            let token = record.grant.token;
            match operation {
                1 => {
                    let reservation = match record.phase {
                        Phase::Granted { reservation } | Phase::Quarantined { reservation } => {
                            reservation
                        }
                        Phase::Copying => return Ok(Admission::Unavailable { max_batch }),
                        Phase::Complete(_) => return Ok(Admission::Released),
                    };
                    // Operation 1 promises no transfer can still WRITE this range. A timeout
                    // is never translated into this operation by the sender.
                    executor.cancel_ingress(reservation)?;
                    executor.staging_release(record.grant.offset)?;
                    record.phase = Phase::Complete(Err(
                        "owned lease was aborted before publication".to_string(),
                    ));
                    let charged = record.charged;
                    Self::uncharge(&mut state, key.0, charged);
                    Ok(Admission::Released)
                }
                2 => {
                    if let Phase::Granted { reservation } = record.phase {
                        record.phase = Phase::Quarantined { reservation };
                    }
                    tracing::warn!(
                        token,
                        bytes = record.charged,
                        "quarantined owned receive grant; remote WRITE quiescence unproven"
                    );
                    Ok(Admission::Released)
                }
                other => Err(format!("unknown owned lease operation {other}")),
            }
        } else {
            if let Some(record) = state.records.get(&key) {
                Self::check_record(record, identity, request.length)?;
                // Returning the original token after completion is safe: a sender replaying a
                // lost grant reply has not posted its WRITE yet; reuse of a completed request
                // identity for a new WRITE is explicitly forbidden instead of returning an old
                // address that may have been allocated to another frame.
                return match record.phase {
                    Phase::Granted { .. } => Ok(Admission::Granted(record.grant.clone())),
                    Phase::Copying => Ok(Admission::Unavailable { max_batch }),
                    Phase::Quarantined { .. } => Err("owned lease is quarantined pending transfer quiescence".to_string()),
                    Phase::Complete(_) => Err("owned lease request identity is already completed; refusing address replay".to_string()),
                };
            }
            if request.length == 0 || request.length > max_batch {
                return Err(format!(
                    "owned ingress capacity: batch {} bytes exceeds supported nonzero maximum {max_batch} bytes (arena {capacity})",
                    request.length
                ));
            }
            if identity
                .query_id
                .as_ref()
                .map(FragmentInstanceId::from)
                .is_some_and(|query| state.retired_queries.contains(&query))
            {
                return Err("owned lease belongs to a retired query".to_string());
            }
            if state.records.len() >= MAX_RECORDS || state.retired_queries.len() >= MAX_RECORDS {
                return Err(format!(
                    "owned lease replay ledger reached its {MAX_RECORDS}-identity bound; retire this CN session before further admission"
                ));
            }
            let charged = request
                .length
                .checked_add(255)
                .ok_or("owned lease length overflow")?
                & !255;
            let peer_live = *state.peer_live.get(&key.0).unwrap_or(&0);
            if state.live + charged > receive_limit
                || peer_live + charged > receive_limit / 2
                || state.active_frames >= 64
                || *state.peer_frames.get(&key.0).unwrap_or(&0) >= 32
            {
                return Ok(Admission::Unavailable { max_batch });
            }
            // Fail with a concrete evacuation-capacity error; do not wait while holding the
            // admission mutex, and never grant GPU memory without its host reservation.
            let reservation = match executor.reserve_ingress(request.length) {
                Ok(reservation) => reservation,
                Err(error) if error.contains("INGRESS_CAPACITY_UNAVAILABLE") => {
                    return Ok(Admission::Unavailable { max_batch });
                }
                Err(error) => {
                    return Err(format!(
                        "owned ingress evacuation reservation of {} bytes failed: {error}",
                        request.length
                    ));
                }
            };
            let offset = match executor.staging_lease(request.length) {
                Ok(offset) => offset,
                Err(error) => {
                    executor.cancel_ingress(reservation)?;
                    return Err(error);
                }
            };
            state.next_token += 1;
            let grant = Grant {
                token: state.next_token,
                offset,
                address: base + offset,
                max_batch,
            };
            state.live += charged;
            state.active_frames += 1;
            *state.peer_frames.entry(key.0).or_default() += 1;
            state.peak = state.peak.max(state.live);
            state.peer_live.insert(key.0, peer_live + charged);
            state.tokens.insert(grant.token, key);
            state.records.insert(
                key,
                Record {
                    identity: identity.clone(),
                    length: request.length,
                    charged,
                    grant: grant.clone(),
                    publication: None,
                    phase: Phase::Granted { reservation },
                },
            );
            tracing::debug!(
                token = grant.token,
                sender_epoch = key.0,
                request_id = key.1,
                query_id = ?identity.query_id,
                receiver = ?identity.finst_id,
                bytes = charged,
                live_bytes = state.live,
                peak_bytes = state.peak,
                "owned receive credit granted"
            );
            Ok(Admission::Granted(grant))
        }
    }

    fn check_record(
        record: &Record,
        identity: &PExchangeLeaseIdentity,
        length: u64,
    ) -> Result<(), String> {
        if record.identity != *identity || record.length != length {
            return Err("owned lease identity replay changed immutable request fields".to_string());
        }
        Ok(())
    }

    pub(crate) fn begin_publication(
        &self,
        params: &PTransmitPackedParams,
        metadata: &[u8],
    ) -> Result<Publication, String> {
        validate_epoch(params.receiver_epoch)?;
        let identity = params
            .identity
            .as_ref()
            .ok_or("owned publication requires an identity")?;
        let key = identity_key(identity)?;
        let token = params
            .lease_token
            .ok_or("owned payload publication requires a lease token")?;
        if metadata.len() > MAX_METADATA {
            return Err(format!(
                "owned publication metadata exceeds {MAX_METADATA} bytes"
            ));
        }
        if identity.canary.unwrap_or(false) != params.canary.unwrap_or(false)
            || (!identity.canary.unwrap_or(false)
                && (identity.finst_id != params.finst_id
                    || identity.node_id != params.node_id
                    || identity.sender_id != params.sender_id
                    || identity.seq != params.seq))
        {
            return Err("owned publication disagrees with its granted frame identity".to_string());
        }
        // Retain a bounded fingerprint to detect conflicting publication retries, including
        // empty frames, names, cardinality, and EOS. Payload identity is the immutable grant.
        let mut hash = std::collections::hash_map::DefaultHasher::new();
        metadata.hash(&mut hash);
        params.column_names.hash(&mut hash);
        params.rows.hash(&mut hash);
        params.eos.hash(&mut hash);
        let fingerprint = hash.finish();
        let mut state = self.state.lock().unwrap_or_else(|p| p.into_inner());
        let retired = identity
            .query_id
            .as_ref()
            .map(FragmentInstanceId::from)
            .is_some_and(|query| state.retired_queries.contains(&query));
        let record = state
            .records
            .get_mut(&key)
            .ok_or("owned publication has no granted identity")?;
        Self::check_record(record, identity, params.length.unwrap_or(0))?;
        if record.grant.token != token || params.offset != Some(record.grant.offset) {
            return Err("owned publication token or offset disagrees with its grant".to_string());
        }
        if record
            .publication
            .is_some_and(|previous| previous != fingerprint)
        {
            return Err("owned publication replay changed metadata or frame fields".to_string());
        }
        record.publication = Some(fingerprint);
        match &record.phase {
            Phase::Granted { reservation } | Phase::Quarantined { reservation } => {
                let reservation = *reservation;
                // Publication certifies WRITE completion. Copying now owns the reservation;
                // neither cancellation nor an abort can return these bytes while GPU reads run.
                record.phase = Phase::Copying;
                Ok(Publication::Fresh {
                    reservation,
                    offset: record.grant.offset,
                    retired,
                })
            }
            Phase::Copying => Ok(Publication::Pending),
            Phase::Complete(result) => Ok(Publication::Complete(result.clone())),
        }
    }

    /// Called only after the GPU copy has finished and the arena range was released.
    pub(crate) fn complete(&self, token: u64, result: Result<(), String>) {
        let mut state = self.state.lock().unwrap_or_else(|p| p.into_inner());
        let key = *state
            .tokens
            .get(&token)
            .expect("publication token was admitted");
        let record = state
            .records
            .get_mut(&key)
            .expect("publication record retained");
        assert!(matches!(record.phase, Phase::Copying));
        record.phase = Phase::Complete(result);
        let charged = record.charged;
        Self::uncharge(&mut state, key.0, charged);
        tracing::debug!(
            token,
            returned_bytes = charged,
            live_bytes = state.live,
            peak_bytes = state.peak,
            "owned receive credit returned after ingress completion"
        );
    }

    fn uncharge(state: &mut State, peer: u64, charged: u64) {
        state.live -= charged;
        state.active_frames -= 1;
        *state.peer_frames.get_mut(&peer).expect("admitted peer") -= 1;
        *state.peer_live.get_mut(&peer).expect("admitted peer") -= charged;
    }

    pub(crate) fn retire_query(&self, query_id: FragmentInstanceId) {
        let mut state = self.state.lock().unwrap_or_else(|p| p.into_inner());
        if state.retired_queries.len() < MAX_RECORDS {
            state.retired_queries.insert(query_id);
        }
        let mut frames = 0_u64;
        let mut payload_bytes = 0_u64;
        let mut quarantined_bytes = 0_u64;
        let mut copying_bytes = 0_u64;
        let mut completed_frames = 0_u64;
        let mut completed_payload_bytes = 0_u64;
        for record in state.records.values_mut() {
            if record
                .identity
                .query_id
                .as_ref()
                .map(FragmentInstanceId::from)
                != Some(query_id)
            {
                continue;
            }
            frames += 1;
            payload_bytes += record.length;
            if let Phase::Granted { reservation } = record.phase {
                record.phase = Phase::Quarantined { reservation };
                tracing::warn!(token = record.grant.token, bytes = record.charged, %query_id, "query cancellation quarantined an unpublished receive grant");
            }
            match record.phase {
                Phase::Quarantined { .. } => quarantined_bytes += record.charged,
                Phase::Copying => copying_bytes += record.charged,
                Phase::Complete(Ok(())) => {
                    completed_frames += 1;
                    completed_payload_bytes += record.length;
                }
                _ => {}
            }
        }
        if frames > 0 {
            tracing::info!(%query_id, frames, requested_payload_bytes = payload_bytes,
                completed_frames, completed_payload_bytes, quarantined_bytes, copying_bytes,
                live_bytes = state.live, peak_bytes = state.peak,
                "owned ingress query retirement accounting");
        }
    }

    pub(crate) fn query_retired(&self, query_id: FragmentInstanceId) -> bool {
        self.state
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .retired_queries
            .contains(&query_id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fragment_executor::{FragmentResult, FragmentRun};
    use crate::proto::starrocks::PUniqueId;
    use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

    #[derive(Debug, Default)]
    struct FakeExecutor {
        next: AtomicU64,
        allocations: AtomicU64,
        leases: Mutex<HashSet<u64>>,
        reservations: Mutex<HashSet<u64>>,
        unavailable: AtomicBool,
    }

    impl FragmentExecutor for FakeExecutor {
        fn run(&self, _: FragmentRun<'_>) -> Result<Option<FragmentResult>, String> {
            Err("no GPU work in lease lifecycle tests".to_string())
        }
        fn staging_info(&self) -> Result<(u64, u64), String> {
            Ok((1 << 20, 64 << 10))
        }
        fn staging_lease(&self, _: u64) -> Result<u64, String> {
            let offset = self.next.fetch_add(256, Ordering::Relaxed);
            self.allocations.fetch_add(1, Ordering::Relaxed);
            assert!(self.leases.lock().unwrap().insert(offset));
            Ok(offset)
        }
        fn staging_release(&self, offset: u64) -> Result<(), String> {
            if self.leases.lock().unwrap().remove(&offset) {
                Ok(())
            } else {
                Err("double release".to_string())
            }
        }
        fn reserve_ingress(&self, _: u64) -> Result<u64, String> {
            if self.unavailable.load(Ordering::Relaxed) {
                return Err("INGRESS_CAPACITY_UNAVAILABLE: pinned pool full".to_string());
            }
            let reservation = self.next.fetch_add(256, Ordering::Relaxed);
            assert!(self.reservations.lock().unwrap().insert(reservation));
            Ok(reservation)
        }
        fn cancel_ingress(&self, reservation: u64) -> Result<(), String> {
            self.reservations.lock().unwrap().remove(&reservation);
            Ok(())
        }
    }

    fn request(id: u64, length: u64) -> PStagingLeaseRequest {
        PStagingLeaseRequest {
            length,
            receiver_epoch: Some(process_epoch()),
            identity: Some(PExchangeLeaseIdentity {
                sender_epoch: 10,
                request_id: id,
                query_id: Some(PUniqueId { hi: 1, lo: 1 }),
                finst_id: Some(PUniqueId { hi: 1, lo: 2 }),
                node_id: Some(7),
                sender_id: Some(0),
                seq: Some(id as i64 - 1),
                canary: Some(false),
            }),
            ..Default::default()
        }
    }

    fn grant(
        ledger: &ReceiveLedger,
        executor: &FakeExecutor,
        request: &PStagingLeaseRequest,
    ) -> Grant {
        match ledger.request(executor, request).unwrap() {
            Admission::Granted(grant) => grant,
            other => panic!("expected grant, got {other:?}"),
        }
    }

    fn publication(request: &PStagingLeaseRequest, grant: &Grant) -> PTransmitPackedParams {
        let identity = request.identity.as_ref().unwrap();
        PTransmitPackedParams {
            identity: Some(identity.clone()),
            receiver_epoch: request.receiver_epoch,
            lease_token: Some(grant.token),
            offset: Some(grant.offset),
            length: Some(request.length),
            finst_id: identity.finst_id,
            node_id: identity.node_id,
            sender_id: identity.sender_id,
            seq: identity.seq,
            canary: identity.canary,
            column_names: vec!["value".into()],
            rows: Some(3),
            eos: Some(false),
            ..Default::default()
        }
    }

    #[test]
    fn lost_grant_reply_replays_one_allocation_and_rejects_conflicts() {
        let ledger = ReceiveLedger::default();
        let executor = FakeExecutor::default();
        let mut request = request(1, 8192);
        let first = grant(&ledger, &executor, &request);
        let replay = grant(&ledger, &executor, &request);
        assert_eq!((first.token, first.address), (replay.token, replay.address));
        assert_eq!(executor.allocations.load(Ordering::Relaxed), 1);
        request.length += 1;
        assert!(
            ledger
                .request(&executor, &request)
                .unwrap_err()
                .contains("immutable")
        );
        assert_eq!(executor.reservations.lock().unwrap().len(), 1);
    }

    #[test]
    fn credits_reserve_evacuation_and_bound_peer_bytes_before_grant() {
        let ledger = ReceiveLedger::default();
        let executor = FakeExecutor::default();
        grant(&ledger, &executor, &request(1, 8192));
        grant(&ledger, &executor, &request(2, 8192));
        assert!(matches!(
            ledger.request(&executor, &request(3, 1)).unwrap(),
            Admission::Unavailable { .. }
        ));
        assert_eq!(executor.allocations.load(Ordering::Relaxed), 2);
        assert_eq!(ledger.state.lock().unwrap().live, 16384);
        assert!(
            ledger
                .request(&executor, &request(4, 16385))
                .unwrap_err()
                .contains("maximum")
        );
    }

    #[test]
    fn evacuation_pressure_returns_unavailable_without_allocating_arena() {
        let ledger = ReceiveLedger::default();
        let executor = FakeExecutor::default();
        executor.unavailable.store(true, Ordering::Relaxed);
        assert!(matches!(
            ledger.request(&executor, &request(1, 1024)).unwrap(),
            Admission::Unavailable { .. }
        ));
        assert_eq!(executor.allocations.load(Ordering::Relaxed), 0);
        assert!(executor.leases.lock().unwrap().is_empty());
    }

    #[test]
    fn publication_ack_waits_for_copy_and_replays_terminal_result() {
        let ledger = ReceiveLedger::default();
        let executor = FakeExecutor::default();
        let request = request(1, 8192);
        let grant = grant(&ledger, &executor, &request);
        let params = publication(&request, &grant);
        let reservation = match ledger.begin_publication(&params, &[1, 2, 3]).unwrap() {
            Publication::Fresh { reservation, .. } => reservation,
            other => panic!("unexpected {other:?}"),
        };
        assert!(matches!(
            ledger.begin_publication(&params, &[1, 2, 3]).unwrap(),
            Publication::Pending
        ));
        let mut abort = request.clone();
        abort.operation = Some(1);
        abort.lease_token = Some(grant.token);
        assert!(matches!(
            ledger.request(&executor, &abort).unwrap(),
            Admission::Unavailable { .. }
        ));
        assert_eq!(executor.leases.lock().unwrap().len(), 1);
        executor.cancel_ingress(reservation).unwrap();
        executor.staging_release(grant.offset).unwrap();
        ledger.complete(grant.token, Ok(()));
        assert!(matches!(
            ledger.begin_publication(&params, &[1, 2, 3]).unwrap(),
            Publication::Complete(Ok(()))
        ));
        assert!(
            ledger
                .begin_publication(&params, &[9])
                .unwrap_err()
                .contains("changed metadata")
        );
        assert_eq!(ledger.state.lock().unwrap().live, 0);
        assert!(
            ledger
                .request(&executor, &request)
                .unwrap_err()
                .contains("completed")
        );
        assert!(matches!(
            ledger.request(&executor, &abort).unwrap(),
            Admission::Released
        ));
        assert!(executor.leases.lock().unwrap().is_empty());
    }

    #[test]
    fn cancelled_unpublished_writes_stay_quarantined_until_quiescent_abort() {
        let ledger = ReceiveLedger::default();
        let executor = FakeExecutor::default();
        let mut first = request(1, 8192);
        let allocation = grant(&ledger, &executor, &first);
        ledger.retire_query(FragmentInstanceId::from_halves(1, 1));
        assert_eq!(ledger.state.lock().unwrap().live, 8192);
        assert_eq!(executor.leases.lock().unwrap().len(), 1);
        assert!(
            ledger
                .request(&executor, &first)
                .unwrap_err()
                .contains("quarantined")
        );
        assert!(
            ledger
                .request(&executor, &request(2, 8192))
                .unwrap_err()
                .contains("retired query")
        );
        first.operation = Some(1);
        first.lease_token = Some(allocation.token);
        assert!(matches!(
            ledger.request(&executor, &first).unwrap(),
            Admission::Released
        ));
        assert!(matches!(
            ledger.request(&executor, &first).unwrap(),
            Admission::Released
        ));
        assert_eq!(ledger.state.lock().unwrap().live, 0);
        assert!(executor.reservations.lock().unwrap().is_empty());
        let mut next = request(3, 8192);
        next.identity.as_mut().unwrap().query_id = Some(PUniqueId { hi: 2, lo: 1 });
        grant(&ledger, &executor, &next);
    }

    #[test]
    fn old_epoch_cannot_publish_or_request_into_restarted_receiver() {
        let ledger = ReceiveLedger::default();
        let executor = FakeExecutor::default();
        let mut request = request(1, 1024);
        request.receiver_epoch = Some(process_epoch().wrapping_add(1));
        assert!(
            ledger
                .request(&executor, &request)
                .unwrap_err()
                .contains("epoch mismatch")
        );
        assert_eq!(executor.allocations.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn grant_reply_loss_can_abort_by_stable_identity_without_a_token() {
        let ledger = ReceiveLedger::default();
        let executor = FakeExecutor::default();
        let mut request = request(1, 1024);
        grant(&ledger, &executor, &request);
        // No WRITE was posted because neither reply containing the address reached the sender.
        request.operation = Some(1);
        assert!(matches!(
            ledger.request(&executor, &request).unwrap(),
            Admission::Released
        ));
        assert!(matches!(
            ledger.request(&executor, &request).unwrap(),
            Admission::Released
        ));
        assert!(executor.leases.lock().unwrap().is_empty());
        assert!(executor.reservations.lock().unwrap().is_empty());
        request.identity.as_mut().unwrap().request_id = 99;
        assert!(matches!(
            ledger.request(&executor, &request).unwrap(),
            Admission::Released
        ));
        request.operation = Some(0);
        assert!(
            ledger
                .request(&executor, &request)
                .unwrap_err()
                .contains("completed")
        );
        assert_eq!(
            executor.allocations.load(Ordering::Relaxed),
            1,
            "delayed grant after abort must not allocate"
        );
    }

    #[test]
    fn tiny_frames_have_a_job_bound_independent_of_bytes() {
        let ledger = ReceiveLedger::default();
        let executor = FakeExecutor::default();
        for id in 1..=32 {
            grant(&ledger, &executor, &request(id, 1));
        }
        assert!(matches!(
            ledger.request(&executor, &request(33, 1)).unwrap(),
            Admission::Unavailable { .. }
        ));
        assert_eq!(executor.allocations.load(Ordering::Relaxed), 32);
    }
}
