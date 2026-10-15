//! Runtime filters this CN can build itself, and the scans waiting for them.
//!
//! The FE plans a runtime filter at a hash join and has the scans that feed the join's probe side
//! wait for it. When the join broadcasts its build side, every CN running the join receives all
//! the keys through one exchange, so a scan on the same CN can be filtered without the FE's
//! filter transport. A receiver fragment records its broadcast filters when it registers
//! ([`RuntimeFilters::record_builds`]). A scan that probes one is deferred until that exchange is
//! complete on this CN, or until it waits too long and runs unfiltered: filters only drop rows
//! the join would drop anyway.
//!
//! A partitioned join's instance receives only its share of the keys. Each instance records the
//! filters it holds a share of ([`RuntimeFilters::record_shares`]); once its build exchange is
//! complete it sends its share to every probing scan, which waits for all of them on its own
//! filter exchange (`FILTER_STREAM_BASE + filter id`). The FE sends the probers and the share
//! count of each filter only to the query's root fragment, on the filter's merge node
//! ([`RuntimeFilters::record_topology`]), which answers the other CNs.

use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::Mutex;
use std::time::{Duration, Instant};

use starrocks_plan_translator::runtime_filter::{BuiltFilter, SHARE_BOUNDS, SHARE_KEYS};
use starrocks_thrift::internal_service::TExecPlanFragmentParams;
use starrocks_thrift::runtime_filter::TRuntimeFilterParams;
use starrocks_thrift::types::TNetworkAddress;

use crate::fragment_executor::KeyStats;
use crate::local_exchange::ExchangeKey;
use crate::result_store::FragmentInstanceId;

/// Logs that this CN applies runtime filter `filter_id` to a scan of `instance`.
///
/// Every filter a fragment on this CN builds or probes gets one `runtime filter` line per place
/// it could apply, applied or skipped with a reason, so a run's filters can be tallied from the
/// logs alone (`docs/scripts/rf_logs.py`).
///
/// `exact` says whether the scan keeps the keys themselves, or only the keys within their bounds.
pub(crate) fn log_applied(
    query: Option<FragmentInstanceId>,
    instance: Option<FragmentInstanceId>,
    filter_id: i32,
    exact: bool,
    stats: &KeyStats,
    waited_ms: u64,
) {
    tracing::info!(
        query = %display_id(query),
        instance = %display_id(instance),
        filter_id,
        outcome = "applied",
        keys = if exact { "exact" } else { "range" },
        rows = stats.rows,
        distinct = stats.distinct,
        min = stats.min,
        max = stats.max,
        waited_ms,
        "runtime filter"
    );
}

/// Logs that this CN leaves runtime filter `filter_id` unapplied at `node` (the building join or
/// the probing target, when known), and why. See [`log_applied`].
pub(crate) fn log_skipped(
    query: Option<FragmentInstanceId>,
    instance: Option<FragmentInstanceId>,
    filter_id: i32,
    node: Option<i32>,
    reason: &str,
    detail: &str,
) {
    tracing::info!(
        query = %display_id(query),
        instance = %display_id(instance),
        filter_id,
        node = node.unwrap_or(-1),
        outcome = "skipped",
        reason,
        detail,
        "runtime filter"
    );
}

fn display_id(id: Option<FragmentInstanceId>) -> String {
    id.map_or_else(|| "-".to_string(), |id| id.to_string())
}

/// Engine stream ids at and above this carry runtime filter keys: `FILTER_STREAM_BASE + filter
/// id`. Exchange node ids are plan node ids, far below.
pub(crate) const FILTER_STREAM_BASE: i32 = 1_000_000;

/// Where this CN receives a filter's keys: one column of one exchange.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct BuildSite {
    /// A broadcast join's build exchange, of a registered receiver; or a scan's filter exchange,
    /// where a partitioned join's shares arrive.
    pub(crate) key: ExchangeKey,
    pub(crate) column: usize,
    /// DuckDB type name of the key.
    pub(crate) column_type: String,
    /// For a filter exchange: how many shares make the filter.
    pub(crate) shares: Option<usize>,
}

/// A scan fragment waiting for the filters it probes.
#[derive(Debug)]
pub(crate) struct DeferredScan {
    pub(crate) params: TExecPlanFragmentParams,
    /// The filters it waits for, with where their keys arrive.
    pub(crate) filters: Vec<(i32, BuildSite)>,
    pub(crate) deferred_at: Instant,
}

/// A partitioned join's instance on this CN, holding a share of a filter's keys in its build
/// exchange.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct HeldShare {
    pub(crate) query: FragmentInstanceId,
    pub(crate) filter_id: i32,
    pub(crate) join_node_id: i32,
    /// The build exchange, and the key's column in it.
    pub(crate) site: BuildSite,
    /// The filter's merge node, which knows its probers.
    pub(crate) merge_node: TNetworkAddress,
    /// This instance's sender id among the join's instances, unique per share.
    pub(crate) sender_id: i32,
}

/// Who probes a partitioned join's filter, from the FE's `runtime_filter_params`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct FilterTopology {
    /// How many shares make the filter (`runtime_filter_builder_number`).
    pub(crate) shares: usize,
    /// Every probing fragment instance, with its brpc address (`id_to_prober_params`).
    pub(crate) probers: Vec<(FragmentInstanceId, TNetworkAddress)>,
}

#[derive(Debug, Default)]
struct State {
    /// By `(query, filter id)`. Kept until the query is purged; a query's handful of entries is
    /// small, and a filter may serve scans that arrive after its receiver ran.
    builds: HashMap<(u64, i32), BuildSite>,
    deferred: HashMap<FragmentInstanceId, DeferredScan>,
    /// Shares not sent yet, until their build exchange is complete.
    shares: Vec<HeldShare>,
    /// By `(query, filter id)`, on the filters' merge node.
    topologies: HashMap<(u64, i32), FilterTopology>,
    /// Filter exchanges one of whose holders gave up: their scans run unfiltered.
    abandoned: HashSet<ExchangeKey>,
    /// Queries purged, by [`FragmentInstanceId::query_hi`], oldest first: nothing more is
    /// recorded for them.
    purged: VecDeque<u64>,
}

/// How many purged queries are remembered, as in the exchange.
const PURGED_QUERIES: usize = 1024;

#[derive(Debug, Default)]
pub(crate) struct RuntimeFilters {
    inner: Mutex<State>,
}

impl RuntimeFilters {
    /// Records the broadcast filters receiver `receiver` builds, keeping the first site of each.
    pub(crate) fn record_builds(&self, receiver: FragmentInstanceId, built: Vec<BuiltFilter>) {
        let mut state = self.lock();
        if state.purged.contains(&receiver.query_hi()) {
            return;
        }
        for filter in built {
            state
                .builds
                .entry((receiver.query_hi(), filter.filter_id))
                .or_insert(build_site(receiver, &filter));
        }
    }

    /// Records the partitioned filters `query`'s receiver `receiver`, sender `sender_id` of its
    /// join, holds a share of. Filters without a merge node are left out.
    pub(crate) fn record_shares(
        &self,
        query: FragmentInstanceId,
        receiver: FragmentInstanceId,
        sender_id: i32,
        built: Vec<BuiltFilter>,
    ) {
        let mut state = self.lock();
        if state.purged.contains(&query.query_hi()) {
            return;
        }
        for filter in built {
            let Some(merge_node) = filter.merge_node.clone() else {
                continue;
            };
            state.shares.push(HeldShare {
                query,
                filter_id: filter.filter_id,
                join_node_id: filter.join_node_id,
                site: build_site(receiver, &filter),
                merge_node,
                sender_id,
            });
        }
    }

    /// Takes every share whose build exchange is complete.
    pub(crate) fn take_ready_shares(
        &self,
        complete: impl Fn(&BuildSite) -> bool,
    ) -> Vec<HeldShare> {
        let mut state = self.lock();
        let (ready, waiting) = std::mem::take(&mut state.shares)
            .into_iter()
            .partition(|share| complete(&share.site));
        state.shares = waiting;
        ready
    }

    /// Takes every share receiver `receiver` holds, whatever its build exchange's state.
    pub(crate) fn take_shares_of(&self, receiver: FragmentInstanceId) -> Vec<HeldShare> {
        self.take_ready_shares(|site| site.key.fragment_instance_id == receiver)
    }

    /// Shares held but not sent yet.
    pub(crate) fn held_shares(&self) -> usize {
        self.lock().shares.len()
    }

    /// Records the probers and share count of `query`'s partitioned filters, which the FE sends
    /// with the root fragment. A prober listed twice is kept once: it gets one copy of each share.
    pub(crate) fn record_topology(&self, query: FragmentInstanceId, params: &TRuntimeFilterParams) {
        let counts = params
            .runtime_filter_builder_number
            .iter()
            .flatten()
            .filter_map(|(&filter_id, &count)| Some((filter_id, usize::try_from(count).ok()?)));
        let mut state = self.lock();
        if state.purged.contains(&query.query_hi()) {
            return;
        }
        for (filter_id, shares) in counts {
            let mut probers: Vec<(FragmentInstanceId, TNetworkAddress)> = Vec::new();
            let listed = params
                .id_to_prober_params
                .as_ref()
                .and_then(|probers| probers.get(&filter_id))
                .into_iter()
                .flatten()
                .filter_map(|prober| {
                    Some((
                        FragmentInstanceId::from(prober.fragment_instance_id.as_ref()?),
                        prober.fragment_instance_address.clone()?,
                    ))
                });
            for prober in listed {
                if probers.iter().all(|(instance, _)| *instance != prober.0) {
                    probers.push(prober);
                }
            }
            state.topologies.insert(
                (query.query_hi(), filter_id),
                FilterTopology { shares, probers },
            );
        }
    }

    /// The probers and share count of `query`'s filter `filter_id`, if this CN has them.
    pub(crate) fn topology(
        &self,
        query: FragmentInstanceId,
        filter_id: i32,
    ) -> Option<FilterTopology> {
        self.lock()
            .topologies
            .get(&(query.query_hi(), filter_id))
            .cloned()
    }

    /// Marks filter exchange `key` as missing a share for good: a holder gave up.
    pub(crate) fn abandon(&self, key: ExchangeKey) {
        let mut state = self.lock();
        if !state.purged.contains(&key.fragment_instance_id.query_hi()) {
            state.abandoned.insert(key);
        }
    }

    pub(crate) fn is_abandoned(&self, key: ExchangeKey) -> bool {
        self.lock().abandoned.contains(&key)
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

    /// Takes every deferred scan whose filters' keys are all `ready`. `ready` runs without this
    /// lock held, so it may ask about abandoned shares.
    pub(crate) fn take_ready(&self, ready: impl Fn(&BuildSite) -> bool) -> Vec<DeferredScan> {
        let waiting: Vec<(FragmentInstanceId, Vec<BuildSite>)> = self
            .lock()
            .deferred
            .iter()
            .map(|(&id, scan)| {
                (
                    id,
                    scan.filters.iter().map(|(_, site)| site.clone()).collect(),
                )
            })
            .collect();
        let ready: Vec<FragmentInstanceId> = waiting
            .into_iter()
            .filter(|(_, sites)| sites.iter().all(&ready))
            .map(|(id, _)| id)
            .collect();
        // A scan the timer or another caller took meanwhile is gone: each is taken once.
        let mut state = self.lock();
        ready
            .into_iter()
            .filter_map(|id| state.deferred.remove(&id))
            .collect()
    }

    /// Takes the deferred scan `instance`, if it is still waiting.
    pub(crate) fn take(&self, instance: FragmentInstanceId) -> Option<DeferredScan> {
        self.lock().deferred.remove(&instance)
    }

    /// Forgets everything held for `query`, and returns its deferred scans, which never run. Its
    /// shares are never sent.
    pub(crate) fn purge_query(&self, query: FragmentInstanceId) -> Vec<DeferredScan> {
        let query_hi = query.query_hi();
        let mut state = self.lock();
        state.builds.retain(|(query, _), _| *query != query_hi);
        state
            .shares
            .retain(|share| share.site.key.fragment_instance_id.query_hi() != query_hi);
        state.topologies.retain(|(query, _), _| *query != query_hi);
        state
            .abandoned
            .retain(|key| key.fragment_instance_id.query_hi() != query_hi);
        if !state.purged.contains(&query_hi) {
            state.purged.push_back(query_hi);
            if state.purged.len() > PURGED_QUERIES {
                state.purged.pop_front();
            }
        }
        let purged: Vec<FragmentInstanceId> = state
            .deferred
            .keys()
            .filter(|instance| instance.query_hi() == query_hi)
            .copied()
            .collect();
        purged
            .into_iter()
            .filter_map(|instance| state.deferred.remove(&instance))
            .collect()
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

fn build_site(receiver: FragmentInstanceId, filter: &BuiltFilter) -> BuildSite {
    BuildSite {
        key: ExchangeKey {
            fragment_instance_id: receiver,
            node_id: filter.exchange_node_id,
        },
        column: filter.column,
        column_type: filter.column_type.clone(),
        shares: None,
    }
}

/// The filter exchange of scan `instance` where `filter_id`'s shares arrive.
pub(crate) fn filter_exchange(instance: FragmentInstanceId, filter_id: i32) -> ExchangeKey {
    ExchangeKey {
        fragment_instance_id: instance,
        node_id: FILTER_STREAM_BASE + filter_id,
    }
}

/// What every share of a partitioned join's filter says about itself, in its column names:
/// whether it holds its keys or only their bounds, how many shares make the filter (the count
/// its holder got from the merge node), and the key's type. A prober uses the shares only when
/// every one agrees with the count and key type it expects.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct ShareHeader {
    pub(crate) exact: bool,
    pub(crate) shares: usize,
    pub(crate) key_type: String,
}

impl ShareHeader {
    pub(crate) fn names(&self) -> Vec<String> {
        let kind = if self.exact { SHARE_KEYS } else { SHARE_BOUNDS };
        vec![
            kind.to_string(),
            format!("shares={}", self.shares),
            format!("key_type={}", self.key_type),
        ]
    }

    pub(crate) fn parse(names: &[String]) -> Option<Self> {
        let [kind, shares, key_type] = names else {
            return None;
        };
        let exact = match kind.as_str() {
            SHARE_KEYS => true,
            SHARE_BOUNDS => false,
            _ => return None,
        };
        Some(Self {
            exact,
            shares: shares.strip_prefix("shares=")?.parse().ok()?,
            key_type: key_type.strip_prefix("key_type=")?.to_string(),
        })
    }
}

/// Whether a filter keyed on `key_type` can be sent in shares: `key_stats` and the share plan
/// read signed integers only.
pub(crate) fn shareable_key(key_type: &str) -> bool {
    matches!(key_type, "TINYINT" | "SMALLINT" | "INTEGER" | "BIGINT")
}

/// Key sets with this few distinct keys are always applied: the join against them costs next to
/// nothing.
pub(crate) const SMALL_KEY_SET: u64 = 1 << 20;

/// Whether a filter with these keys is worth applying: a small key set always is, and a larger
/// one must cover at most `max_density` of its value range. Keys that fill their range (every
/// supplier, say) would keep almost every probe row while costing a join.
///
/// Both tests count distinct keys, not rows: a build side that repeats a few keys (q05's five
/// ASIA nations over two million suppliers) is small and selective. `distinct` is summed per
/// batch, so it can only overstate the keys, which skips a filter rather than wrongly applying
/// a dense one.
pub(crate) fn selective(stats: &KeyStats, max_density: f64) -> bool {
    if stats.distinct <= SMALL_KEY_SET {
        return true;
    }
    if stats.max < stats.min {
        return false;
    }
    let range = (stats.max as f64) - (stats.min as f64) + 1.0;
    (stats.distinct as f64) / range <= max_density
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

/// The most distinct keys a partitioned join's instance sends as they are
/// (`SIRIUS_CN_RUNTIME_FILTER_MAX_SHARE_KEYS`, default 2^24). A larger share sends only its
/// smallest and largest key, and its probers keep the keys between the bounds of every share.
///
/// TPC-H keys are spread evenly over their range, so bounds rarely drop anything. At SF1000,
/// q12's `orders` filter is about 7.3M keys per share: sent exactly, it cuts the `orders` rows
/// shipped from 1.5B to 29M; as bounds, by 96 rows.
pub(crate) fn max_share_keys() -> u64 {
    std::env::var("SIRIUS_CN_RUNTIME_FILTER_MAX_SHARE_KEYS")
        .ok()
        .and_then(|keys| keys.parse().ok())
        .unwrap_or(1 << 24)
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
            distribution: starrocks_plan_translator::runtime_filter::BuildDistribution::Broadcast,
            merge_node: None,
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

    fn topology_params(
        probers: &[(i64, i64)],
    ) -> starrocks_thrift::runtime_filter::TRuntimeFilterParams {
        use starrocks_thrift::runtime_filter::TRuntimeFilterProberParams;
        use starrocks_thrift::types::TUniqueId;
        starrocks_thrift::runtime_filter::TRuntimeFilterParams {
            id_to_prober_params: Some(std::collections::BTreeMap::from([(
                0,
                probers
                    .iter()
                    .map(|&(hi, lo)| {
                        TRuntimeFilterProberParams::new(
                            TUniqueId::new(hi, lo),
                            TNetworkAddress::new("127.0.0.1".to_string(), 8060),
                        )
                    })
                    .collect(),
            )])),
            runtime_filter_builder_number: Some(std::collections::BTreeMap::from([(0, 4)])),
            runtime_filter_max_size: None,
            skew_join_runtime_filters: None,
        }
    }

    #[test]
    fn a_prober_listed_twice_gets_one_copy_of_each_share() {
        let filters = RuntimeFilters::default();
        filters.record_topology(instance(9, 0), &topology_params(&[(9, 3), (9, 4), (9, 3)]));
        let topology = filters.topology(instance(9, 0), 0).unwrap();
        assert_eq!(topology.shares, 4);
        assert_eq!(
            topology
                .probers
                .iter()
                .map(|(id, _)| *id)
                .collect::<Vec<_>>(),
            vec![instance(9, 3), instance(9, 4)]
        );
    }

    #[test]
    fn nothing_is_recorded_for_a_purged_query() {
        let filters = RuntimeFilters::default();
        filters.purge_query(instance(9, 0));
        filters.record_topology(instance(9, 0), &topology_params(&[(9, 3)]));
        filters.record_builds(instance(9, 1), vec![built(0, 13)]);
        let mut share = built(1, 17);
        share.distribution =
            starrocks_plan_translator::runtime_filter::BuildDistribution::Partitioned;
        share.merge_node = Some(TNetworkAddress::new("127.0.0.1".to_string(), 8060));
        filters.record_shares(instance(9, 0), instance(9, 1), 0, vec![share.clone()]);
        filters.abandon(filter_exchange(instance(9, 3), 1));
        assert!(filters.topology(instance(9, 0), 0).is_none());
        assert!(filters.sites(instance(9, 5), &[0]).is_empty());
        assert_eq!(filters.held_shares(), 0);
        assert!(!filters.is_abandoned(filter_exchange(instance(9, 3), 1)));
        // Another query is recorded as before.
        filters.record_shares(instance(10, 0), instance(10, 1), 0, vec![share]);
        assert_eq!(filters.held_shares(), 1);
    }

    #[test]
    fn a_share_header_round_trips_through_its_column_names() {
        let header = ShareHeader {
            exact: false,
            shares: 4,
            key_type: "BIGINT".to_string(),
        };
        assert_eq!(ShareHeader::parse(&header.names()), Some(header));
        assert_eq!(ShareHeader::parse(&["rf_key".to_string()]), None);
    }

    #[test]
    fn only_keys_that_leave_most_of_their_range_out_are_selective() {
        let stats = |rows, distinct, min, max| KeyStats {
            rows,
            distinct,
            min,
            max,
        };
        // q09's green parts: 5% of part keys.
        assert!(selective(
            &stats(32_000_000, 32_000_000, 1, 600_000_000),
            0.5
        ));
        // Every supplier.
        assert!(!selective(
            &stats(30_000_000, 30_000_000, 1, 30_000_000),
            0.5
        ));
        // An empty build side keeps nothing, which is the best filter there is.
        assert!(selective(&stats(0, 0, i64::MAX, i64::MIN), 0.5));
        // A small key set is cheap to join against even when it fills its range.
        assert!(selective(&stats(1000, 1000, 1, 1000), 0.5));
        // q05 rf1: two million suppliers' nation keys, five distinct nations in [8, 21].
        assert!(selective(&stats(1_999_620, 5, 8, 21), 0.5));
        // Many distinct keys repeated: judged on the distinct ones.
        assert!(!selective(&stats(8_000_000, 2_000_000, 1, 2_000_000), 0.5));
        assert!(selective(&stats(8_000_000, 2_000_000, 1, 40_000_000), 0.5));
    }
}
