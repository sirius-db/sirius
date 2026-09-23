//! `SIRIUS_CN_TIMING=1` timeline: one log line per fragment and per packed hop.
//!
//! Wall-clock microseconds since the Unix epoch, so the two CNs on one host can be lined up
//! against each other and against the FE query window by the benchmark runner.

use std::sync::OnceLock;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

/// True when `SIRIUS_CN_TIMING` is set to anything but `0`. Read once.
pub(crate) fn enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("SIRIUS_CN_TIMING")
            .map(|value| !value.is_empty() && value != "0")
            .unwrap_or(false)
    })
}

/// Microseconds since the Unix epoch.
pub(crate) fn unix_us() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_micros() as u64
}

/// Whole microseconds of `elapsed`.
pub(crate) fn us(elapsed: Duration) -> u64 {
    elapsed.as_micros() as u64
}

/// Union length of `[start, end)` intervals in microseconds: time with at least one in flight.
pub(crate) fn busy_us(intervals: &mut [(u64, u64)]) -> u64 {
    intervals.sort_unstable();
    let mut busy = 0;
    let mut current: Option<(u64, u64)> = None;
    for &(start, end) in intervals.iter() {
        match current {
            Some((s, e)) if start <= e => current = Some((s, e.max(end))),
            Some((s, e)) => {
                busy += e - s;
                current = Some((start, end));
            }
            None => current = Some((start, end)),
        }
    }
    if let Some((s, e)) = current {
        busy += e - s;
    }
    busy
}

#[cfg(test)]
mod tests {
    use super::busy_us;

    #[test]
    fn busy_merges_overlapping_intervals() {
        assert_eq!(busy_us(&mut []), 0);
        assert_eq!(busy_us(&mut [(0, 10), (5, 15), (20, 25)]), 20);
        assert_eq!(busy_us(&mut [(20, 25), (0, 10)]), 15);
    }
}
