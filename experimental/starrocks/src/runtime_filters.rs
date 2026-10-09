//! Runtime filters this CN can build itself, and the scans waiting for them.
//!
//! The FE plans a runtime filter at a hash join and has the scans that feed the join's probe side
//! wait for it. When the join broadcasts its build side, every CN running the join receives all
//! the keys through one exchange, so a scan on the same CN can be filtered without the FE's
//! filter transport. A receiver fragment records its broadcast filters when it registers
//! ([`RuntimeFilters::record_builds`]). A scan that probes one is deferred until that exchange is
//! complete on this CN, or until it waits too long and runs unfiltered: filters only drop rows
//! the join would drop anyway.

use std::collections::HashMap;
use std::sync::Mutex;
use std::time::{Duration, Instant};

use starrocks_plan_translator::runtime_filter::BuiltFilter;
use starrocks_thrift::internal_service::TExecPlanFragmentParams;

use crate::fragment_executor::KeyStats;
use crate::local_exchange::ExchangeKey;
use crate::result_store::FragmentInstanceId;

/// Engine stream ids at and above this carry runtime filter keys: `FILTER_STREAM_BASE + filter
/// id`. Exchange node ids are plan node ids, far below.
pub(crate) const FILTER_STREAM_BASE: i32 = 1_000_000;

/// Where this CN receives a filter's keys: one column of one exchange of a registered receiver.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct BuildSite {
    pub(crate) key: ExchangeKey,
    pub(crate) column: usize,
    /// DuckDB type name of the key.
    pub(crate) column_type: String,
}

/// A scan fragment waiting for the filters it probes.
#[derive(Debug)]
pub(crate) struct DeferredScan {
    pub(crate) params: TExecPlanFragmentParams,
    /// The filters it waits for, with where their keys arrive.
    pub(crate) filters: Vec<(i32, BuildSite)>,
    pub(crate) deferred_at: Instant,
}

#[derive(Debug, Default)]
struct State {
    /// By `(query, filter id)`. Kept until the query is purged; a query's handful of entries is
    /// small, and a filter may serve scans that arrive after its receiver ran.
    builds: HashMap<(u64, i32), BuildSite>,
    deferred: HashMap<FragmentInstanceId, DeferredScan>,
}

#[derive(Debug, Default)]
pub(crate) struct RuntimeFilters {
    inner: Mutex<State>,
}

impl RuntimeFilters {
    /// Records the broadcast filters receiver `receiver` builds, keeping the first site of each.
    pub(crate) fn record_builds(&self, receiver: FragmentInstanceId, built: Vec<BuiltFilter>) {
        let mut state = self.lock();
        for filter in built {
            state
                .builds
                .entry((receiver.query_hi(), filter.filter_id))
                .or_insert(BuildSite {
                    key: ExchangeKey {
                        fragment_instance_id: receiver,
                        node_id: filter.exchange_node_id,
                    },
                    column: filter.column,
                    column_type: filter.column_type,
                });
        }
    }

    /// The build sites on this CN of `query`'s filters `filter_ids`.
    pub(crate) fn sites(
        &self,
        query: FragmentInstanceId,
        filter_ids: &[i32],
    ) -> Vec<(i32, BuildSite)> {
        let state = self.lock();
        filter_ids
            .iter()
            .filter_map(|&id| {
                let site = state.builds.get(&(query.query_hi(), id))?;
                Some((id, site.clone()))
            })
            .collect()
    }

    pub(crate) fn defer(&self, instance: FragmentInstanceId, scan: DeferredScan) {
        self.lock().deferred.insert(instance, scan);
    }

    /// Takes every deferred scan whose filters' exchanges are all complete.
    pub(crate) fn take_ready(&self, complete: impl Fn(&BuildSite) -> bool) -> Vec<DeferredScan> {
        let mut state = self.lock();
        let ready: Vec<FragmentInstanceId> = state
            .deferred
            .iter()
            .filter(|(_, scan)| scan.filters.iter().all(|(_, site)| complete(site)))
            .map(|(&id, _)| id)
            .collect();
        ready
            .into_iter()
            .filter_map(|id| state.deferred.remove(&id))
            .collect()
    }

    /// Takes the deferred scan `instance`, if it is still waiting.
    pub(crate) fn take(&self, instance: FragmentInstanceId) -> Option<DeferredScan> {
        self.lock().deferred.remove(&instance)
    }

    /// Forgets everything held for `query`, returning its deferred scans, which never run.
    pub(crate) fn purge_query(&self, query: FragmentInstanceId) -> Vec<DeferredScan> {
        let query_hi = query.query_hi();
        let mut state = self.lock();
        state.builds.retain(|(query, _), _| *query != query_hi);
        let instances: Vec<FragmentInstanceId> = state
            .deferred
            .keys()
            .filter(|instance| instance.query_hi() == query_hi)
            .copied()
            .collect();
        instances
            .iter()
            .filter_map(|instance| state.deferred.remove(instance))
            .collect()
    }

    /// Whether a scan of `query` (any instance id of it) waits for a filter.
    pub(crate) fn holds(&self, query: FragmentInstanceId) -> bool {
        self.lock()
            .deferred
            .keys()
            .any(|instance| instance.query_hi() == query.query_hi())
    }

    /// Scans waiting for a filter. Zero on an idle CN.
    pub(crate) fn deferred(&self) -> usize {
        self.lock().deferred.len()
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, State> {
        self.inner
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }
}

/// Key sets this small are always applied: the join against them costs next to nothing.
pub(crate) const SMALL_KEY_SET: u64 = 1 << 20;

/// Whether a filter with these keys is worth applying: a small key set always is, and a larger
/// one must cover at most `max_density` of its value range. Keys that fill their range (every
/// supplier, say) would keep almost every probe row while costing a join.
pub(crate) fn selective(stats: &KeyStats, max_density: f64) -> bool {
    if stats.rows <= SMALL_KEY_SET {
        return true;
    }
    if stats.max < stats.min {
        return false;
    }
    let range = (stats.max as f64) - (stats.min as f64) + 1.0;
    (stats.rows as f64) / range <= max_density
}

/// `SIRIUS_CN_RUNTIME_FILTERS=0` turns runtime filters off.
pub(crate) fn enabled() -> bool {
    std::env::var("SIRIUS_CN_RUNTIME_FILTERS").as_deref() != Ok("0")
}

/// How long a scan waits for its filters before running unfiltered
/// (`SIRIUS_CN_RUNTIME_FILTER_WAIT_MS`, default 30 s).
pub(crate) fn wait_limit() -> Duration {
    std::env::var("SIRIUS_CN_RUNTIME_FILTER_WAIT_MS")
        .ok()
        .and_then(|ms| ms.parse().ok())
        .map_or(Duration::from_secs(30), Duration::from_millis)
}

/// The largest share of its value range a filter's keys may cover
/// (`SIRIUS_CN_RUNTIME_FILTER_MAX_DENSITY`, default 0.5).
pub(crate) fn max_density() -> f64 {
    std::env::var("SIRIUS_CN_RUNTIME_FILTER_MAX_DENSITY")
        .ok()
        .and_then(|density| density.parse().ok())
        .unwrap_or(0.5)
}

#[cfg(test)]
mod tests {
    use starrocks_thrift::internal_service::InternalServiceVersion;

    use super::*;

    fn params() -> TExecPlanFragmentParams {
        TExecPlanFragmentParams {
            protocol_version: InternalServiceVersion::V1,
            fragment: None,
            desc_tbl: None,
            params: None,
            coord: None,
            backend_num: None,
            query_globals: None,
            query_options: None,
            enable_profile: None,
            resource_info: None,
            import_label: None,
            db_name: None,
            load_job_id: None,
            load_error_hub_info: None,
            is_pipeline: None,
            pipeline_dop: None,
            per_scan_node_dop: None,
            workgroup: None,
            enable_resource_group: None,
            func_version: None,
            enable_shared_scan: None,
            is_stream_pipeline: None,
            adaptive_dop_param: None,
            group_execution_scan_dop: None,
            pred_tree_params: None,
            exec_stats_node_ids: None,
            arrow_flight_sql_version: None,
        }
    }

    fn instance(query: i64, lo: i64) -> FragmentInstanceId {
        FragmentInstanceId::from_halves(query, lo)
    }

    fn built(filter_id: i32, exchange_node_id: i32) -> BuiltFilter {
        BuiltFilter {
            filter_id,
            join_node_id: 14,
            exchange_node_id,
            column: 0,
            column_type: "INTEGER".to_string(),
        }
    }

    fn scan(filters: Vec<(i32, BuildSite)>) -> DeferredScan {
        DeferredScan {
            params: params(),
            filters,
            deferred_at: Instant::now(),
        }
    }

    #[test]
    fn a_scan_waits_until_every_filter_it_probes_is_complete() {
        let filters = RuntimeFilters::default();
        let receiver = instance(9, 1);
        filters.record_builds(receiver, vec![built(0, 13), built(1, 17)]);
        // Another query's filter 0 is a different filter.
        assert!(filters.sites(instance(10, 1), &[0]).is_empty());
        let sites = filters.sites(instance(9, 5), &[0, 1, 2]);
        assert_eq!(sites.len(), 2, "filter 2 is not built on this CN");
        filters.defer(instance(9, 5), scan(sites));
        assert_eq!(filters.deferred(), 1);

        assert!(filters.take_ready(|site| site.key.node_id == 13).is_empty());
        assert_eq!(filters.take_ready(|_| true).len(), 1);
        assert_eq!(filters.deferred(), 0);
        assert!(filters.take(instance(9, 5)).is_none(), "taken once");
    }

    #[test]
    fn purging_a_query_drops_its_builds_and_deferred_scans() {
        let filters = RuntimeFilters::default();
        filters.record_builds(instance(9, 1), vec![built(0, 13)]);
        filters.record_builds(instance(10, 1), vec![built(0, 13)]);
        let sites = filters.sites(instance(9, 5), &[0]);
        filters.defer(instance(9, 5), scan(sites));
        assert_eq!(filters.purge_query(instance(9, 0)).len(), 1);
        assert_eq!(filters.deferred(), 0);
        assert!(filters.sites(instance(9, 5), &[0]).is_empty());
        assert_eq!(filters.sites(instance(10, 5), &[0]).len(), 1);
    }

    #[test]
    fn only_keys_that_leave_most_of_their_range_out_are_selective() {
        let stats = |rows, min, max| KeyStats { rows, min, max };
        // q09's green parts: 5% of part keys.
        assert!(selective(&stats(32_000_000, 1, 600_000_000), 0.5));
        // Every supplier.
        assert!(!selective(&stats(30_000_000, 1, 30_000_000), 0.5));
        // An empty build side keeps nothing, which is the best filter there is.
        assert!(selective(&stats(0, i64::MAX, i64::MIN), 0.5));
        // A small key set is cheap to join against even when it fills its range.
        assert!(selective(&stats(1000, 1, 1000), 0.5));
    }
}
