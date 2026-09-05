//! Pure admission and scheduling policy shared by the real owner loop and CPU-only tests.

#![cfg_attr(not(feature = "nixl-transport"), allow(dead_code))]

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{SyncSender, sync_channel};
use tokio::sync::{Notify, oneshot};

type Job<T> = Box<dyn FnOnce(&mut T) + Send>;

/// One bounded worker per peer, and one for packing. A stalled peer cannot occupy another
/// peer's worker. The Tokio continuation sleeps without consuming a worker between retries.
pub(super) struct Worker<T: Send + 'static> {
    jobs: Option<SyncSender<Job<T>>>,
    join: Option<std::thread::JoinHandle<()>>,
}

impl<T: Send + 'static> Worker<T> {
    pub(super) fn start(name: &str, mut context: T, capacity: usize) -> Result<Self, String> {
        let (jobs, receive) = sync_channel::<Job<T>>(capacity);
        let join = std::thread::Builder::new()
            .name(name.to_string())
            .spawn(move || {
                while let Ok(job) = receive.recv() {
                    job(&mut context);
                }
            })
            .map_err(|err| format!("cannot start {name}: {err}"))?;
        Ok(Self {
            jobs: Some(jobs),
            join: Some(join),
        })
    }

    pub(super) async fn call<R: Send + 'static>(
        &self,
        job: impl FnOnce(&mut T) -> Result<R, String> + Send + 'static,
    ) -> Result<R, String> {
        let (reply, answer) = oneshot::channel();
        self.jobs
            .as_ref()
            .ok_or("control worker has stopped")?
            .try_send(Box::new(move |context| {
                let _ = reply.send(job(context));
            }))
            .map_err(|err| format!("bounded control/export worker admission failed: {err}"))?;
        answer
            .await
            .map_err(|_| "control/export worker dropped its response".to_string())?
    }
}

impl<T: Send + 'static> Drop for Worker<T> {
    fn drop(&mut self) {
        self.jobs.take();
        if let Some(join) = self.join.take() {
            let _ = join.join();
        }
    }
}

#[derive(Default)]
pub(super) struct PublicationOrder {
    next: tokio::sync::Mutex<i64>,
    pub(super) failed: AtomicBool,
    changed: Notify,
}

impl PublicationOrder {
    pub(super) async fn wait(&self, seq: i64) -> Result<(), String> {
        self.wait_before_park(seq, || {}).await
    }

    /// The callback is an observation seam for the precise check-to-sleep boundary. Production
    /// uses a no-op; the deterministic race test injects cancellation at this exact boundary.
    async fn wait_before_park(&self, seq: i64, before_park: impl Fn()) -> Result<(), String> {
        loop {
            let notified = self.changed.notified();
            tokio::pin!(notified);
            notified.as_mut().enable();
            if self.failed.load(Ordering::Acquire) {
                return Err("an earlier frame of this sender failed".to_string());
            }
            if *self.next.lock().await == seq {
                return Ok(());
            }
            before_park();
            notified.await;
        }
    }

    pub(super) async fn advance(&self, seq: i64) {
        let mut next = self.next.lock().await;
        assert_eq!(*next, seq);
        *next += 1;
        self.changed.notify_waiters();
    }

    pub(super) fn fail(&self) {
        self.failed.store(true, Ordering::Release);
        self.changed.notify_waiters();
    }
}

/// A pass visits every admitted peer once, rotating the first peer between passes. A pending
/// operation never consumes the next peer's turn. Frames reserve bytes before packing, so
/// scheduling many tiny frames cannot silently retain unbounded large pack allocations.
pub(super) struct FairPeers<K> {
    peers: VecDeque<K>,
}

impl<K: Clone + PartialEq> FairPeers<K> {
    pub(super) fn new() -> Self {
        Self {
            peers: VecDeque::new(),
        }
    }

    pub(super) fn insert(&mut self, peer: K) {
        if !self.peers.contains(&peer) {
            self.peers.push_back(peer);
        }
    }

    pub(super) fn pass(&mut self) -> Vec<K> {
        let result = self.peers.iter().cloned().collect();
        if let Some(first) = self.peers.pop_front() {
            self.peers.push_back(first);
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stalled_grant_and_publication_do_not_starve_healthy_peer() {
        let mut ring = FairPeers::new();
        for peer in ["grant-stalled", "publish-stalled", "healthy"] {
            ring.insert(peer);
        }
        let mut completions = 0;
        for _ in 0..12 {
            let pass = ring.pass();
            assert_eq!(pass.len(), 3);
            for peer in pass {
                if peer == "healthy" {
                    completions += 1;
                }
            }
        }
        assert_eq!(completions, 12, "healthy peer gets one turn in every pass");
    }

    #[tokio::test]
    async fn windows_one_and_two_bound_packing_and_hold_quarantined_bytes() {
        use std::sync::Arc;
        use tokio::sync::Semaphore;
        for window in [1, 2] {
            let bytes = Arc::new(Semaphore::new(128));
            let frames = Arc::new(Semaphore::new(window));
            let mut reservations = Vec::new();
            let mut slots = Vec::new();
            for _ in 0..window {
                slots.push(Arc::clone(&frames).acquire_owned().await.unwrap());
                reservations.push(Arc::clone(&bytes).acquire_many_owned(64).await.unwrap());
            }
            assert!(frames.try_acquire().is_err());
            assert_eq!(bytes.available_permits(), 128 - 64 * window);
            // Same pessimistic-reservation shrink as production after a small pack completes.
            drop(reservations[0].split(48));
            assert_eq!(bytes.available_permits(), 128 - 64 * window + 48);
            // An uncertain WRITE retains both its actual bytes and the lost-permit accounting.
            reservations.remove(0).forget();
            drop(reservations);
            assert_eq!(bytes.available_permits(), 112);
            assert!(Arc::clone(&bytes).try_acquire_many_owned(113).is_err());
        }
    }

    #[tokio::test]
    async fn independent_control_workers_progress_while_grant_and_publication_stall() {
        use std::sync::{Arc, mpsc};
        use std::time::Duration;
        let (grant_release, grant_wait) = mpsc::channel();
        let (publish_release, publish_wait) = mpsc::channel();
        let grant = Arc::new(Worker::start("test-grant-stall", grant_wait, 2).unwrap());
        let publish = Arc::new(Worker::start("test-publish-stall", publish_wait, 2).unwrap());
        let healthy = Worker::start("test-healthy-peer", (), 2).unwrap();
        let grant_task = tokio::spawn({
            let worker = Arc::clone(&grant);
            async move {
                worker
                    .call(|gate| gate.recv().map_err(|e| e.to_string()))
                    .await
            }
        });
        let publish_task = tokio::spawn({
            let worker = Arc::clone(&publish);
            async move {
                worker
                    .call(|gate| gate.recv().map_err(|e| e.to_string()))
                    .await
            }
        });
        for seq in 0..16 {
            let value =
                tokio::time::timeout(Duration::from_secs(1), healthy.call(move |_| Ok(seq)))
                    .await
                    .unwrap()
                    .unwrap();
            assert_eq!(value, seq);
        }
        grant_release.send(()).unwrap();
        publish_release.send(()).unwrap();
        grant_task.await.unwrap().unwrap();
        publish_task.await.unwrap().unwrap();
    }

    #[tokio::test]
    async fn reversed_write_completions_publish_in_sequence_before_eos() {
        use std::sync::Arc;
        use std::time::Duration;
        let order = Arc::new(PublicationOrder::default());
        // Frame 1 completed its transfer first, but cannot publish or permit EOS.
        assert!(
            tokio::time::timeout(Duration::from_millis(5), order.wait(1))
                .await
                .is_err()
        );
        order.wait(0).await.unwrap();
        order.advance(0).await;
        order.wait(1).await.unwrap();
        order.advance(1).await;
        order.wait(2).await.unwrap();
    }

    #[tokio::test]
    async fn cancellation_wakes_later_publications_without_advancing_sequence() {
        use std::sync::Arc;
        use std::time::Duration;
        let order = Arc::new(PublicationOrder::default());
        let waiter = tokio::spawn({
            let order = Arc::clone(&order);
            async move { order.wait(1).await }
        });
        order.fail();
        let result = tokio::time::timeout(Duration::from_secs(1), waiter)
            .await
            .unwrap()
            .unwrap();
        assert!(result.unwrap_err().contains("earlier frame"));
        assert_eq!(*order.next.lock().await, 0);
    }

    #[tokio::test]
    async fn cancellation_between_sequence_check_and_sleep_is_not_lost() {
        let order = PublicationOrder::default();
        let result = tokio::time::timeout(
            std::time::Duration::from_secs(1),
            order.wait_before_park(1, || order.fail()),
        )
        .await
        .expect("the check-to-sleep notification must wake the waiter");
        assert!(result.unwrap_err().contains("earlier frame"));
    }
}
