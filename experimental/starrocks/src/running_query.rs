//! Which query the engine thread is running, so a purge can stop that run and no other, and the
//! GPU memory a purge has asked to free that is not free yet.
//!
//! The engine's interrupt reaches only a run already inside its gate: one that lands just before
//! the gate opens is dropped. So a purge repeats it for as long as the record, updated under the
//! same lock, says its query's run has not returned. Holding the lock while interrupting means the
//! interrupt can only reach that run, or a closed gate: the engine thread cannot start another
//! query's run before it takes the lock to say so.

use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use std::time::{Duration, Instant};

use crate::recent_queries::{REMEMBER_FOR, RecentQueries};
use crate::result_store::FragmentInstanceId;

/// How many purged queries the engine remembers, so a run of one still queued is refused.
const PURGED_QUERIES: usize = 4096;

#[derive(Debug)]
struct State {
    /// The query of the run in progress, by [`FragmentInstanceId::query_hi`].
    running: Option<u64>,
    /// Queries purged here: a run of one that was still queued never starts.
    purged: RecentQueries<()>,
}

/// The engine thread's current run, and the queries whose runs must stop.
#[derive(Debug)]
pub(crate) struct RunningQuery {
    state: Mutex<State>,
}

impl Default for RunningQuery {
    fn default() -> Self {
        Self {
            state: Mutex::new(State {
                running: None,
                purged: RecentQueries::new(REMEMBER_FOR, PURGED_QUERIES),
            }),
        }
    }
}

impl RunningQuery {
    /// Runs `run` as a run of `query`, so [`interrupt_while_running`](Self::interrupt_while_running)
    /// can stop it. Returns `None`, without running it, when `query` was already purged.
    pub(crate) fn run_as<T>(
        &self,
        query: Option<FragmentInstanceId>,
        run: impl FnOnce() -> T,
    ) -> Option<T> {
        let query = query.map(FragmentInstanceId::query_hi);
        {
            let mut state = self.lock();
            if query.is_some_and(|query| state.purged.get(query).is_some()) {
                return None;
            }
            state.running = query;
        }
        // Cleared however the run ends, a panic included: a stale record would keep a purge
        // interrupting, and its frees pending, for good.
        let _finished = Finished(self);
        Some(run())
    }

    /// Records that `query` was purged: a run of it still queued never starts.
    pub(crate) fn purge(&self, query: FragmentInstanceId) {
        self.lock()
            .purged
            .get_or_insert_with(query.query_hi(), || ());
    }

    /// Whether a run of `query` is in progress.
    pub(crate) fn is_running(&self, query: FragmentInstanceId) -> bool {
        self.lock().running == Some(query.query_hi())
    }

    /// Calls `interrupt` every `every` for as long as a run of `query` is in progress, but no
    /// longer than `limit`, and returns how many times it did. Never while another query's run
    /// is.
    pub(crate) fn interrupt_while_running(
        &self,
        query: FragmentInstanceId,
        every: Duration,
        limit: Duration,
        interrupt: impl Fn(),
    ) -> usize {
        let deadline = Instant::now() + limit;
        let mut sent = 0;
        loop {
            {
                let state = self.lock();
                if state.running != Some(query.query_hi()) || Instant::now() >= deadline {
                    return sent;
                }
                interrupt();
                sent += 1;
            }
            std::thread::sleep(every);
        }
    }

    fn lock(&self) -> MutexGuard<'_, State> {
        self.state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }
}

/// Clears the run record when the run ends.
struct Finished<'a>(&'a RunningQuery);

impl Drop for Finished<'_> {
    fn drop(&mut self) {
        self.0.lock().running = None;
    }
}

/// GPU memory a purge asked to free that is not free yet: parked output whose drop waits for the
/// engine thread, and an interrupted run that has not returned. A receive allocation that finds
/// the pool full waits for these before it gives up.
#[derive(Debug, Default)]
pub struct PendingFrees {
    count: Mutex<Counts>,
    freed: Condvar,
}

#[derive(Debug, Default)]
struct Counts {
    pending: usize,
    /// Frees done so far, so a caller can tell one finished since it last looked.
    done: u64,
}

/// One pending free; dropping it marks it done.
#[derive(Debug)]
pub struct PendingFree(Arc<PendingFrees>);

impl PendingFrees {
    /// Starts one pending free, done when the returned guard drops.
    pub fn begin(self: &Arc<Self>) -> PendingFree {
        self.lock().pending += 1;
        PendingFree(Arc::clone(self))
    }

    /// How many frees are pending.
    pub fn pending(&self) -> usize {
        self.lock().pending
    }

    /// How many frees are done so far.
    pub fn done(&self) -> u64 {
        self.lock().done
    }

    /// Waits up to `timeout` for every pending free to be done. Returns whether they all were.
    pub fn wait_for_none(&self, timeout: Duration) -> bool {
        let deadline = Instant::now() + timeout;
        let mut count = self.lock();
        while count.pending > 0 {
            let left = deadline.saturating_duration_since(Instant::now());
            if left.is_zero() {
                return false;
            }
            count = self
                .freed
                .wait_timeout(count, left)
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .0;
        }
        true
    }

    fn lock(&self) -> MutexGuard<'_, Counts> {
        self.count
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }
}

impl Drop for PendingFree {
    fn drop(&mut self) {
        let mut count = self.0.lock();
        count.pending -= 1;
        count.done += 1;
        if count.pending == 0 {
            self.0.freed.notify_all();
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

    use super::*;

    /// A run that ends only once interrupted, ignoring the first interrupt as one that landed
    /// before its gate opened.
    fn interruptible_run(
        running: &Arc<RunningQuery>,
        query: FragmentInstanceId,
        interrupts: &Arc<AtomicUsize>,
    ) -> std::thread::JoinHandle<Option<()>> {
        let (running, interrupts) = (Arc::clone(running), Arc::clone(interrupts));
        std::thread::spawn(move || {
            running.run_as(Some(query), || {
                while interrupts.load(Ordering::SeqCst) < 2 {
                    std::thread::sleep(Duration::from_millis(1));
                }
            })
        })
    }

    fn wait_until_running(running: &RunningQuery, query: FragmentInstanceId) {
        while !running.is_running(query) {
            std::thread::sleep(Duration::from_millis(1));
        }
    }

    #[test]
    fn a_purge_interrupts_its_querys_run_until_it_returns_and_no_other() {
        let running = Arc::new(RunningQuery::default());
        let (ours, theirs) = (
            FragmentInstanceId::from_halves(1, 3),
            FragmentInstanceId::from_halves(2, 3),
        );
        let interrupts = Arc::new(AtomicUsize::new(0));
        let run = interruptible_run(&running, ours, &interrupts);
        wait_until_running(&running, ours);

        // Another query's purge finds nothing to interrupt.
        let wrong = AtomicBool::new(false);
        assert_eq!(
            running.interrupt_while_running(
                theirs,
                Duration::from_millis(1),
                Duration::from_secs(10),
                || wrong.store(true, Ordering::SeqCst)
            ),
            0
        );
        assert!(!wrong.load(Ordering::SeqCst));

        // Ours repeats past the dropped first interrupt, until the run returns.
        let counted = Arc::clone(&interrupts);
        let sent = running.interrupt_while_running(
            ours,
            Duration::from_millis(1),
            Duration::from_secs(10),
            || {
                counted.fetch_add(1, Ordering::SeqCst);
            },
        );
        assert!(sent >= 2, "{sent}");
        assert_eq!(run.join().unwrap(), Some(()));
        assert!(!running.is_running(ours));
    }

    #[test]
    fn a_run_that_panics_leaves_no_record_and_an_interrupt_gives_up_at_its_limit() {
        let running = Arc::new(RunningQuery::default());
        let query = FragmentInstanceId::from_halves(3, 1);
        let panicked = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            running.run_as(Some(query), || panic!("the engine panicked"))
        }));
        assert!(panicked.is_err());
        assert!(!running.is_running(query), "the record is cleared");

        // A run that ignores interrupts: they stop at the limit.
        let (release, gate) = std::sync::mpsc::channel::<()>();
        let stuck = {
            let running = Arc::clone(&running);
            std::thread::spawn(move || running.run_as(Some(query), || gate.recv().unwrap()))
        };
        wait_until_running(&running, query);
        let started = Instant::now();
        let sent = running.interrupt_while_running(
            query,
            Duration::from_millis(1),
            Duration::from_millis(50),
            || (),
        );
        assert!(sent > 0);
        assert!(started.elapsed() < Duration::from_secs(5));
        release.send(()).unwrap();
        stuck.join().unwrap();
    }

    #[test]
    fn a_queued_run_of_a_purged_query_never_starts() {
        let running = RunningQuery::default();
        running.purge(FragmentInstanceId::from_halves(5, 0));
        assert_eq!(
            running.run_as(Some(FragmentInstanceId::from_halves(5, 7)), || ()),
            None
        );
        assert_eq!(
            running.run_as(Some(FragmentInstanceId::from_halves(6, 7)), || 42),
            Some(42)
        );
    }

    #[test]
    fn pending_frees_are_waited_for_up_to_a_bound() {
        let frees = Arc::new(PendingFrees::default());
        assert!(frees.wait_for_none(Duration::ZERO));
        let first = frees.begin();
        let second = frees.begin();
        assert_eq!(frees.pending(), 2);
        assert!(!frees.wait_for_none(Duration::from_millis(20)));
        drop(first);
        let waiter = {
            let frees = Arc::clone(&frees);
            std::thread::spawn(move || frees.wait_for_none(Duration::from_secs(10)))
        };
        std::thread::sleep(Duration::from_millis(20));
        drop(second);
        assert!(waiter.join().unwrap());
        assert_eq!(frees.pending(), 0);
    }
}
