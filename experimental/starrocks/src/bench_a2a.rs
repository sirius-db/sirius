//! All-to-all bandwidth micro-benchmark over the production packed NIXL hop.
//!
//! Shaped after distributed-join's `benchmark/all_to_all.cpp` (and the cascade-tpc-shuttle
//! `bench_a2a` case): each of the two CNs sends `bytes / workers` to its peer as fixed-size
//! chunks, four timed rounds after one warm-up, and reports `per_peer * rounds / time` GB/s.
//!
//! The payload is a raw staging-arena lease (no cudf pack): the sender path is the real
//! `NixlTransport::send_fragment` (Md, Lease, WRITE, Packed announce, EOS), and the receiver
//! releases each lease as its announce arrives instead of parking it for a fragment.

use std::collections::HashMap;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::fragment_executor::{FragmentExecutor, FragmentResult, SenderSlot, StagedBatch};
use crate::result_store::FragmentInstanceId;
use starrocks_plan_translator::TranslatedPlan;

/// Frames with this fragment instance id go to the bench sink, not the exchange rendezvous.
pub(crate) const BENCH_FRAGMENT_INSTANCE: FragmentInstanceId =
    FragmentInstanceId::from_halves(0x5152_4e58_a2a0_0000, 0x0bea_c400);

static RECEIVED_BYTES: AtomicU64 = AtomicU64::new(0);
static RECEIVED_EOS: AtomicU64 = AtomicU64::new(0);

/// Called by the service for every bench frame after its lease was released.
pub(crate) fn record_received(length: u64, eos: bool) {
    RECEIVED_BYTES.fetch_add(length, Ordering::SeqCst);
    if eos {
        RECEIVED_EOS.fetch_add(1, Ordering::SeqCst);
    }
}

/// A [`FragmentExecutor`] whose "parked output" is `chunks` synthetic arena leases per slot.
#[derive(Debug)]
#[cfg_attr(not(feature = "nixl-transport"), allow(dead_code))]
struct BenchExecutor {
    inner: std::sync::Arc<dyn FragmentExecutor>,
    chunk: u64,
    last: u64,
    remaining: Mutex<HashMap<SenderSlot, u64>>,
}

impl FragmentExecutor for BenchExecutor {
    fn execute(&self, _translated: &TranslatedPlan) -> Result<FragmentResult, String> {
        Err("the a2a bench executor runs no plans".to_string())
    }

    fn staging_info(&self) -> Result<(u64, u64), String> {
        self.inner.staging_info()
    }

    fn staging_lease(&self, len: u64) -> Result<u64, String> {
        self.inner.staging_lease(len)
    }

    fn staging_release(&self, offset: u64) -> Result<(), String> {
        self.inner.staging_release(offset)
    }

    fn export_packed_next(&self, slot: SenderSlot) -> Result<Option<StagedBatch>, String> {
        let len = {
            let mut remaining = self.remaining.lock().unwrap_or_else(|p| p.into_inner());
            let left = remaining.entry(slot).or_insert(0);
            if *left == 0 {
                return Ok(None);
            }
            *left -= 1;
            if *left == 0 { self.last } else { self.chunk }
        };
        let offset = self.inner.staging_lease(len)?;
        Ok(Some(StagedBatch {
            metadata: vec![0],
            offset,
            len,
            rows: Some(0),
        }))
    }
}

/// Bench parameters from the CN command line.
#[derive(Clone, Debug)]
pub struct BenchA2a {
    /// Peer brpc address.
    pub peer: std::net::SocketAddr,
    /// Peer nixl agent name (`{host}:{brpc_port}`).
    pub peer_agent_name: String,
    /// Total bytes per GPU, split evenly across the two workers like `all_to_all.cpp`.
    pub bytes: u64,
    /// Bytes per packed frame (one NIXL WRITE each).
    pub chunk: u64,
    /// Timed rounds after one warm-up.
    pub rounds: u32,
}

impl BenchA2a {
    /// Runs the benchmark and returns the result line. Blocks.
    #[cfg(feature = "nixl-transport")]
    pub fn run(
        &self,
        transport: std::sync::Arc<crate::NixlTransport>,
        executor: std::sync::Arc<dyn FragmentExecutor>,
    ) -> Result<String, String> {
        use crate::nixl_transport::RemoteSendSpec;
        use std::time::{Duration, Instant};

        const WORKERS: u64 = 2;
        let per_peer = self.bytes / WORKERS;
        let chunk = self.chunk.max(1);
        let chunks = per_peer.div_ceil(chunk).max(1);
        let last = per_peer - chunk * (chunks - 1);
        let bench = std::sync::Arc::new(BenchExecutor {
            inner: executor,
            chunk,
            last,
            remaining: Mutex::new(HashMap::new()),
        });

        wait_for_peer(self.peer, Duration::from_secs(180))?;

        let round = |index: u32| -> Result<(), String> {
            let slot = SenderSlot {
                fragment_instance_id: BENCH_FRAGMENT_INSTANCE,
                node_id: index as i32,
                sender_id: 0,
            };
            bench
                .remaining
                .lock()
                .unwrap_or_else(|p| p.into_inner())
                .insert(slot, chunks);
            transport.send_fragment(
                RemoteSendSpec {
                    peer: self.peer,
                    peer_agent_name: self.peer_agent_name.clone(),
                    dest_stream: index as i32,
                    sender_id: 0,
                    names: vec!["payload".to_string()],
                    slot,
                },
                bench.clone(),
            )
        };
        let wait_received = |eos: u64| -> Result<(), String> {
            let deadline = Instant::now() + Duration::from_secs(600);
            while RECEIVED_EOS.load(Ordering::SeqCst) < eos {
                if Instant::now() > deadline {
                    return Err(format!(
                        "peer sent {} of {eos} bench rounds",
                        RECEIVED_EOS.load(Ordering::SeqCst)
                    ));
                }
                std::thread::sleep(Duration::from_micros(50));
            }
            Ok(())
        };

        round(0)?;
        wait_received(1)?;
        let started = Instant::now();
        for index in 1..=self.rounds {
            round(index)?;
            wait_received(u64::from(index) + 1)?;
        }
        let secs = started.elapsed().as_secs_f64();
        let gbs = per_peer as f64 * (WORKERS - 1) as f64 * f64::from(self.rounds) / secs / 1e9;
        // Let the peer finish its last round against this CN's brpc server before exit.
        std::thread::sleep(Duration::from_secs(2));
        Ok(format!(
            "bench a2a Size (MB): {:.1}, chunk_bytes={chunk}, chunks_per_round={chunks}, \
             rounds={}, Elapsed time (s): {secs:.6}, Bandwidth per GPU (GB/s): {gbs:.2}",
            self.bytes as f64 / 1e6,
            self.rounds
        ))
    }
}

#[cfg(feature = "nixl-transport")]
fn wait_for_peer(peer: std::net::SocketAddr, timeout: std::time::Duration) -> Result<(), String> {
    let deadline = std::time::Instant::now() + timeout;
    loop {
        if std::net::TcpStream::connect_timeout(&peer, std::time::Duration::from_secs(1)).is_ok() {
            return Ok(());
        }
        if std::time::Instant::now() > deadline {
            return Err(format!("bench peer {peer} never accepted brpc connections"));
        }
        std::thread::sleep(std::time::Duration::from_millis(200));
    }
}
