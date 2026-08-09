//! Bring-up pre-establishment of every directed nixl peer session.
//!
//! WHY THIS EXISTS — it is not redundant with `TransportState::ensure_session`, and deleting it
//! brings back a 100%-reproducible cold-start hang.
//!
//! Sessions used to be created purely on demand, from *inside* the transport thread, the first
//! time a query wanted to send to a peer. That first contact deadlocks a fresh cluster:
//!
//!   * CN A's transport thread enters `ensure_session(B)` and blocks in the outbound
//!     `exchange_nixl_md` brpc call to B.
//!   * B's `exchange_nixl_md` handler needs *B's* transport thread
//!     (`TransportRequest::ExchangeMd`) — which at that moment is blocked in its own outbound
//!     `exchange_nixl_md` call to A.
//!   * Neither transport thread can answer the other. An all-to-all shuffle first-contacts every
//!     peer at once, so this is the normal case, not a race.
//!
//! The cycle only breaks when `PrpcClient::REPLY_TIMEOUT` (60 s) fires, long after the FE gave
//! up. Reproduced 2026-08-08 on a fresh 4-CN cluster, q14 run once:
//!   * run 1 (COLD): FAILED after 121 s. FE: `exec rpc error, backend [id=10002],
//!     THRIFT_RPC_ERROR, fragmentId=F05`, cause `errorCode=62 method request time out ... 60000
//!     (MILLISECONDS)`, bound channel `R:/127.0.0.1:9102` (cn0's brpc port).
//!   * run 2 (WARM, same cluster, same SQL): 751 ms, correct.
//! The TPC-H harness discards run 0 as a warm-up, so every sweep ever run on this box paid the
//! hang exactly once, invisibly.
//!
//! Two things fix it, and both are here:
//!   1. Every session is established at bring-up, off the query path, so no user query pays
//!      first contact.
//!   2. The `exchange_nixl_md` call is made ON THIS THREAD, and only the finished handshake is
//!      handed to the transport thread (`TransportRequest::WarmSession`). That breaks the cycle:
//!      while this CN waits for a peer's metadata, its own transport thread stays free to answer
//!      that peer's `ExchangeMd`. Warming up any other way would just move the same deadlock
//!      from the first query to bring-up.
//! The remainder of session set-up (`load_remote_md`, the F1 bandwidth canary) still runs on the
//! transport thread because it touches the agent, and that is safe: a peer serves
//! `request_staging_lease` and `transmit_packed` from its blocking pool, never from its
//! transport thread.
//!
//! Peers are NOT known at `NixlTransport::start` — a CN only learns them from the FE, and only
//! after they have registered and heartbeated (their `BrpcPort` is a heartbeat field). So the
//! warmup is a background thread that polls `SHOW PROC '/compute_nodes'` until the cluster has
//! assembled, retrying each peer with backoff because CNs boot at different times. Failures are
//! logged loudly and are never fatal: a peer left cold still works through the lazy
//! `ensure_session` path, just slowly.
//!
//! Environment:
//!   * `SIRIUS_CN_NIXL_WARMUP=0|false|no|off` — kill switch; back to purely lazy sessions.
//!   * `SIRIUS_CN_NIXL_WARMUP_PEERS=host:port,host:port` — explicit peer list, skipping FE
//!     discovery (this CN's own `host:brpc_port` is filtered out either way).
//!   * `SIRIUS_CN_NIXL_WARMUP_TIMEOUT_SECS` — wall-clock budget for the whole loop (default 180).
//!   * `SIRIUS_CN_NIXL_WARMUP_EXPECT_PEERS` — peer count that ends the loop as soon as it is
//!     reached, for operators who know the cluster size (`NUM_CNS - 1`).

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Sender, channel};
use std::time::{Duration, Instant};

use tracing::{info, warn};

use super::agent_tier::rpc_exchange_md;
use super::{SessionWarmup, TransportRequest};
use crate::FeConfig;
use crate::prpc_client::PrpcClient;

/// Gap between FE compute-node polls while the cluster assembles.
const POLL_INTERVAL: Duration = Duration::from_secs(2);
/// Backoff after a peer's first failed attempt; doubles up to [`MAX_BACKOFF`].
const MIN_BACKOFF: Duration = Duration::from_secs(1);
/// Cap on the per-peer retry backoff.
const MAX_BACKOFF: Duration = Duration::from_secs(15);
/// Default wall-clock budget for the whole warmup.
const DEFAULT_BUDGET: Duration = Duration::from_secs(180);
/// How long the discovered peer set must stay unchanged (with every peer established) before the
/// warmup declares the cluster assembled and stops early.
const SETTLE: Duration = Duration::from_secs(20);
/// Bound on one FE compute-node query, so an unresponsive FE cannot park the warmup thread.
const FE_QUERY_TIMEOUT: Duration = Duration::from_secs(10);
/// Stop-flag polling granularity, so shutdown does not wait out a whole [`POLL_INTERVAL`].
const STOP_POLL: Duration = Duration::from_millis(200);

/// Operator-tunable warmup behaviour, resolved once from the environment.
struct Settings {
    /// Explicit peer list, bypassing FE discovery.
    peers: Option<Vec<(String, u16)>>,
    /// Wall-clock budget for the whole loop.
    budget: Duration,
    /// Peer count that ends the loop early once established.
    expect: Option<usize>,
}

impl Settings {
    /// Reads the environment; `None` means the kill switch is set.
    fn from_env() -> Option<Self> {
        if let Some(value) = std::env::var_os("SIRIUS_CN_NIXL_WARMUP") {
            let value = value.to_string_lossy().to_ascii_lowercase();
            if matches!(value.as_str(), "false" | "0" | "no" | "off") {
                return None;
            }
        }
        let peers = std::env::var("SIRIUS_CN_NIXL_WARMUP_PEERS")
            .ok()
            .map(|list| parse_peer_list(&list));
        let budget = std::env::var("SIRIUS_CN_NIXL_WARMUP_TIMEOUT_SECS")
            .ok()
            .and_then(|value| value.parse().ok())
            .map_or(DEFAULT_BUDGET, Duration::from_secs);
        let expect = std::env::var("SIRIUS_CN_NIXL_WARMUP_EXPECT_PEERS")
            .ok()
            .and_then(|value| value.parse().ok());
        Some(Self {
            peers,
            budget,
            expect,
        })
    }
}

/// Parses `host:port,host:port`; malformed entries are dropped with a warning rather than
/// failing bring-up.
fn parse_peer_list(list: &str) -> Vec<(String, u16)> {
    list.split(',')
        .map(str::trim)
        .filter(|entry| !entry.is_empty())
        .filter_map(|entry| match entry.rsplit_once(':') {
            Some((host, port)) => match port.parse::<u16>() {
                Ok(port) if port != 0 && !host.is_empty() => Some((host.to_string(), port)),
                _ => {
                    warn!(
                        entry,
                        "ignoring a malformed SIRIUS_CN_NIXL_WARMUP_PEERS entry"
                    );
                    None
                }
            },
            None => {
                warn!(
                    entry,
                    "ignoring a SIRIUS_CN_NIXL_WARMUP_PEERS entry without a ':port'"
                );
                None
            }
        })
        .collect()
}

/// Starts the warmup thread. `None` when the kill switch disabled it (or the thread could not be
/// spawned) — never an error, because a cold cluster still works through the lazy path.
pub(super) fn spawn(
    agent_name: String,
    local_md: Vec<u8>,
    requests: Sender<TransportRequest>,
    fe: FeConfig,
) -> Option<SessionWarmup> {
    let Some(settings) = Settings::from_env() else {
        warn!(
            "SIRIUS_CN_NIXL_WARMUP is off: peer sessions stay lazy, so the first cross-node \
             query after bring-up can block for up to 60 s per peer on first contact"
        );
        return None;
    };
    let stop = Arc::new(AtomicBool::new(false));
    let thread_stop = Arc::clone(&stop);
    match std::thread::Builder::new()
        .name("nixl-warmup".to_string())
        .spawn(move || {
            run(
                &agent_name,
                &local_md,
                &requests,
                &fe,
                &settings,
                &thread_stop,
            )
        }) {
        Ok(thread) => Some(SessionWarmup { stop, thread }),
        Err(err) => {
            warn!(
                error = %err,
                "failed to spawn the nixl session warmup thread; peer sessions stay lazy"
            );
            None
        }
    }
}

/// Warmup loop: discover peers, pre-establish the ones that are still cold, repeat until the
/// cluster has settled or the budget runs out.
fn run(
    agent_name: &str,
    local_md: &[u8],
    requests: &Sender<TransportRequest>,
    fe: &FeConfig,
    settings: &Settings,
    stop: &AtomicBool,
) {
    let started = Instant::now();
    let budget_ends = started + settings.budget;
    // Only the FE-discovery path needs a runtime (`mysql_async` is async); an explicit peer list
    // is pure configuration.
    let runtime = match settings.peers {
        Some(_) => None,
        None => match tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
        {
            Ok(runtime) => Some(runtime),
            Err(err) => {
                warn!(
                    error = %err,
                    "failed to create the nixl warmup FE-query runtime; peer sessions stay lazy"
                );
                return;
            }
        },
    };
    info!(
        agent = %agent_name,
        budget_secs = settings.budget.as_secs(),
        "pre-establishing nixl peer sessions so no query pays first contact"
    );

    let mut established: BTreeSet<String> = BTreeSet::new();
    let mut seen: BTreeSet<String> = BTreeSet::new();
    // Per-peer (failed attempts, earliest next attempt).
    let mut retry: BTreeMap<String, (u32, Instant)> = BTreeMap::new();
    let mut last_new_peer = Instant::now();

    while !stop.load(Ordering::Relaxed) {
        let peers = match discover(settings, runtime.as_ref(), fe) {
            Ok(peers) => peers,
            Err(err) => {
                warn!(error = %err, "nixl session warmup could not list the FE's compute nodes");
                Vec::new()
            }
        };
        for (host, brpc_port) in peers {
            if stop.load(Ordering::Relaxed) {
                break;
            }
            // The nixl agent name IS `{advertise_host}:{brpc_port}`, so this is the self filter.
            let key = format!("{host}:{brpc_port}");
            if key == agent_name {
                continue;
            }
            if seen.insert(key.clone()) {
                last_new_peer = Instant::now();
            }
            if established.contains(&key) {
                continue;
            }
            if retry
                .get(&key)
                .is_some_and(|(_, next)| Instant::now() < *next)
            {
                continue;
            }
            let attempt_started = Instant::now();
            match warm_one(requests, agent_name, local_md, &host, brpc_port) {
                Ok(()) => {
                    retry.remove(&key);
                    established.insert(key.clone());
                    info!(
                        peer = %key,
                        took_ms = attempt_started.elapsed().as_millis(),
                        established = established.len(),
                        "pre-established a nixl peer session"
                    );
                }
                Err(err) => {
                    let entry = retry.entry(key.clone()).or_insert((0, Instant::now()));
                    entry.0 += 1;
                    let backoff = (MIN_BACKOFF * 2u32.pow(entry.0.min(4) - 1)).min(MAX_BACKOFF);
                    entry.1 = Instant::now() + backoff;
                    warn!(
                        peer = %key,
                        attempt = entry.0,
                        retry_in_secs = backoff.as_secs(),
                        error = %err,
                        "failed to pre-establish a nixl peer session; until it succeeds the \
                         first query sending to this peer pays first contact"
                    );
                }
            }
        }

        // Early exits: the operator-declared peer count, or a peer set that stopped growing with
        // every member established. Otherwise keep polling for late-booting CNs.
        if settings
            .expect
            .is_some_and(|expect| established.len() >= expect)
            || (!seen.is_empty()
                && established.len() == seen.len()
                && last_new_peer.elapsed() >= SETTLE)
            || Instant::now() >= budget_ends
        {
            break;
        }
        sleep_unless_stopped(POLL_INTERVAL, stop);
    }

    let cold: Vec<&str> = seen
        .iter()
        .filter(|peer| !established.contains(*peer))
        .map(String::as_str)
        .collect();
    let elapsed_ms = started.elapsed().as_millis();
    if !cold.is_empty() {
        warn!(
            established = established.len(),
            peers = seen.len(),
            elapsed_ms,
            cold = %cold.join(","),
            "nixl session warmup finished with peers left cold: the first query that sends to \
             one of them pays first contact, which can cost the FE's 60 s rpc timeout"
        );
    } else if seen.is_empty() {
        info!(
            elapsed_ms,
            "nixl session warmup found no peer compute nodes; nothing to pre-establish"
        );
    } else {
        info!(
            established = established.len(),
            elapsed_ms, "nixl session warmup complete: every peer session is established"
        );
    }
}

/// The current peer candidates: the explicit list when configured, otherwise the FE's alive
/// compute nodes.
fn discover(
    settings: &Settings,
    runtime: Option<&tokio::runtime::Runtime>,
    fe: &FeConfig,
) -> Result<Vec<(String, u16)>, String> {
    if let Some(peers) = &settings.peers {
        return Ok(peers.clone());
    }
    let runtime = runtime.ok_or_else(|| "no FE query runtime".to_string())?;
    runtime.block_on(async {
        match tokio::time::timeout(FE_QUERY_TIMEOUT, crate::list_alive_compute_nodes(fe)).await {
            Ok(result) => result.map_err(|err| format!("{err:#}")),
            Err(_) => Err(format!(
                "listing the FE's compute nodes timed out after {FE_QUERY_TIMEOUT:?}"
            )),
        }
    })
}

/// One directed session to `host:brpc_port`: the metadata handshake runs here (see the module
/// comment — this is the call that must stay off the transport thread), the agent-local install
/// and the bandwidth canary run on the transport thread.
fn warm_one(
    requests: &Sender<TransportRequest>,
    agent_name: &str,
    local_md: &[u8],
    host: &str,
    brpc_port: u16,
) -> Result<(), String> {
    let mut client = PrpcClient::new(host, brpc_port);
    let peer = rpc_exchange_md(&mut client, agent_name, local_md)?;
    let (respond_tx, respond_rx) = channel();
    requests
        .send(TransportRequest::WarmSession {
            host: host.to_string(),
            brpc_port,
            client,
            peer,
            respond: respond_tx,
        })
        .map_err(|_| "nixl transport thread is not running".to_string())?;
    respond_rx
        .recv()
        .map_err(|_| "nixl transport thread dropped the response".to_string())?
}

/// Sleeps `total`, waking every [`STOP_POLL`] so a shutdown is noticed promptly.
fn sleep_unless_stopped(total: Duration, stop: &AtomicBool) {
    let ends = Instant::now() + total;
    loop {
        let remaining = ends.saturating_duration_since(Instant::now());
        if remaining.is_zero() || stop.load(Ordering::Relaxed) {
            return;
        }
        std::thread::sleep(remaining.min(STOP_POLL));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The peer list accepts what an operator would write and drops what it cannot use, rather
    /// than failing bring-up over a typo.
    #[test]
    fn peer_list_parses_host_port_pairs_and_drops_junk() {
        assert_eq!(
            parse_peer_list("127.0.0.1:9102, 127.0.0.1:9112 ,,bad,127.0.0.1:0,host:70000"),
            vec![
                ("127.0.0.1".to_string(), 9102),
                ("127.0.0.1".to_string(), 9112),
            ]
        );
    }

    /// An IPv6 literal keeps its brackets: the port is the last `:`-separated field.
    #[test]
    fn peer_list_splits_on_the_last_colon() {
        assert_eq!(
            parse_peer_list("[::1]:9102"),
            vec![("[::1]".to_string(), 9102)]
        );
    }

    /// Backoff grows per attempt and stops at the cap, so a permanently unreachable peer is
    /// retried for the whole budget without being hammered.
    #[test]
    fn retry_backoff_doubles_up_to_the_cap() {
        let backoff = |attempt: u32| (MIN_BACKOFF * 2u32.pow(attempt.min(4) - 1)).min(MAX_BACKOFF);
        assert_eq!(backoff(1), Duration::from_secs(1));
        assert_eq!(backoff(2), Duration::from_secs(2));
        assert_eq!(backoff(4), Duration::from_secs(8));
        assert_eq!(backoff(9), Duration::from_secs(8));
    }
}
