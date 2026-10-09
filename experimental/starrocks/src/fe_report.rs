//! Final `FrontendService.reportExecStatus` reports, one per fragment instance.
//!
//! The FE learns how an instance ended from its `exec_plan_fragment` reply, from `fetch_data` for
//! the result sink, or from a report. Every instance whose dispatch reply was OK sends exactly one
//! final report (`done = true`): OK once it finished, its error if it failed, CANCELLED if the FE
//! cancelled its query. An instance whose dispatch reply carried its error sends none.
//!
//! What the FE does with them (StarRocks `QeProcessorImpl.reportExecStatus`,
//! `DefaultCoordinator.updateFragmentExecStatus`, `FragmentInstanceExecState.updateExecStatus`):
//! - the first non-OK status fails the query, which the FE then cancels everywhere;
//! - a report after the instance's `done = true` is dropped, status and all;
//! - once the result sink returned every row, a CANCELLED status is ignored but any other error
//!   still fails the query. So a failure after that point, or after the FE cancelled the query,
//!   is reported as CANCELLED;
//! - a report for a query the FE already finished comes back NOT_FOUND, which is expected;
//! - with profiling on (`enable_profile`, or a query past `big_query_profile_threshold`), the FE
//!   waits for every cancelled instance's own final report, and an INSERT waits for every
//!   instance's: one that never comes holds the statement until its timeout.
//!
//! Reports leave from a background thread per FE address, so neither the engine thread nor a
//! brpc handler ever waits on the FE, and an FE that does not answer delays no other FE's reports.
//! The address comes from the dispatch request, so like the heartbeat's FE address it must match
//! the configured FE host: a request cannot point the CN's outbound connections elsewhere.

use std::collections::HashMap;
use std::sync::mpsc::{Sender, channel};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use anyhow::{Result, anyhow, bail};
use starrocks_thrift::{
    frontend_service::{
        FrontendServiceVersion, TFrontendServiceSyncClient, TReportExecStatusParams,
    },
    internal_service::TExecPlanFragmentParams,
    status::TStatus,
    status_code::TStatusCode,
    types::{TNetworkAddress, TUniqueId},
};
use tracing::{info, warn};

use crate::Host;
use crate::recent_queries::QueryEnds;
use crate::result_store::FragmentInstanceId;

/// How many times a report is tried when the FE cannot be reached.
const ATTEMPTS: u32 = 3;
/// The wait before trying an unreachable FE again.
const RETRY_WAIT: Duration = Duration::from_secs(1);

/// Sends one `reportExecStatus` to the FE at `coord`, blocking. NOT_FOUND counts as delivered: the
/// FE has already finished the query.
pub(crate) fn report_exec_status(
    coord: &TNetworkAddress,
    query_id: &TUniqueId,
    backend_num: i32,
    fragment_instance_id: &TUniqueId,
    status: TStatus,
    done: bool,
) -> Result<()> {
    // No profile, load or sink counters: the FE reads none of them for a SELECT.
    let params = TReportExecStatusParams::new(
        FrontendServiceVersion::V1,
        query_id.clone(),
        backend_num,
        fragment_instance_id.clone(),
        status,
        done,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    );
    let result = crate::frontend_client(coord)?
        .report_exec_status(params)
        .map_err(|err| {
            anyhow!(
                "reportExecStatus to {}:{}: {err}",
                coord.hostname,
                coord.port
            )
        })?;
    let status = result
        .status
        .ok_or_else(|| anyhow!("the FE's reportExecStatus reply carries no status"))?;
    match status.status_code {
        TStatusCode::OK | TStatusCode::NOT_FOUND => Ok(()),
        code => bail!(
            "the FE refused reportExecStatus ({code:?}): {}",
            status.error_msgs.unwrap_or_default().join("; ")
        ),
    }
}

/// Where and as what one instance reports, from its dispatch params.
#[derive(Clone, Debug, PartialEq, Eq)]
struct ReportTarget {
    coord: TNetworkAddress,
    query_id: TUniqueId,
    backend_num: i32,
    fragment_instance_id: TUniqueId,
}

impl ReportTarget {
    /// `None` when the FE sent no coordinator address or backend number, so no report could name
    /// the instance.
    fn of(params: &TExecPlanFragmentParams) -> Option<Self> {
        let exec = params.params.as_ref()?;
        Some(Self {
            coord: params.coord.clone()?,
            query_id: exec.query_id.clone(),
            backend_num: params.backend_num?,
            fragment_instance_id: exec.fragment_instance_id.clone(),
        })
    }

    fn instance(&self) -> FragmentInstanceId {
        FragmentInstanceId::from(&self.fragment_instance_id)
    }
}

/// One final report waiting for the reporter thread.
#[derive(Debug)]
struct Report {
    target: ReportTarget,
    status: TStatus,
}

/// The instances on this CN that still owe the FE their final report.
#[derive(Debug, Default)]
pub(crate) struct ExecReports {
    /// The configured FE host, which every report's coordinator must match. `None` trusts any.
    frontend_host: Option<Host>,
    /// Queries that ended on this CN. One the FE already ended (it cancelled it, or this CN's
    /// result sink delivered its last row) reports a failure of its instances as CANCELLED.
    ends: Arc<QueryEnds>,
    state: Mutex<ReportState>,
    /// Each FE address's reporter thread queue, started with its first report.
    reporters: Mutex<HashMap<(String, i32), Sender<Report>>>,
}

#[derive(Debug, Default)]
struct ReportState {
    owed: HashMap<FragmentInstanceId, ReportTarget>,
    /// Whether each coordinator host seen so far matches the configured FE host.
    trusted: HashMap<String, bool>,
}

impl ExecReports {
    /// Reports only to coordinators on `frontend_host`, the FE the CN was configured with, and
    /// reads how queries ended from `ends`.
    pub(crate) fn new(frontend_host: Option<Host>, ends: Arc<QueryEnds>) -> Self {
        Self {
            frontend_host,
            ends,
            ..Self::default()
        }
    }

    /// Records a dispatched instance, which now owes one final report. One whose coordinator is
    /// not the configured FE host gets none.
    pub(crate) fn expect(&self, params: &TExecPlanFragmentParams) {
        let Some(target) = ReportTarget::of(params) else {
            return;
        };
        let mut state = self.lock();
        let host = &target.coord.hostname;
        let trusted = match &self.frontend_host {
            None => true,
            Some(expected) => *state
                .trusted
                .entry(host.clone())
                .or_insert_with(|| crate::frontend_host_trusted(expected.as_str(), host)),
        };
        if !trusted {
            warn!(
                instance = %target.instance(),
                coord = %host,
                expected = %self.frontend_host.as_ref().expect("checked above"),
                "not reporting to a coordinator that is not the configured FE host"
            );
            return;
        }
        state.owed.insert(target.instance(), target);
    }

    /// The instance's dispatch reply carries how it ended, so it owes no report.
    pub(crate) fn settled_by_reply(&self, params: &TExecPlanFragmentParams) {
        if let Some(target) = ReportTarget::of(params) {
            self.lock().owed.remove(&target.instance());
        }
    }

    /// Sends the instance's final report, once: OK, or its error. An instance of a query that
    /// ended on the FE's side reports a failure as CANCELLED, which the FE then ignores.
    pub(crate) fn finish(
        &self,
        params: &TExecPlanFragmentParams,
        outcome: std::result::Result<(), &str>,
    ) {
        let Some(instance) = ReportTarget::of(params).map(|target| target.instance()) else {
            return;
        };
        let mut state = self.lock();
        let Some(target) = state.owed.remove(&instance) else {
            return;
        };
        let status = match outcome {
            Ok(()) => status(TStatusCode::OK, None),
            Err(error) if self.ends.ended_by_fe(instance) => {
                status(TStatusCode::CANCELLED, Some(error))
            }
            Err(error) => status(TStatusCode::INTERNAL_ERROR, Some(error)),
        };
        drop(state);
        self.send(Report { target, status });
    }

    /// The FE cancelled `query` (any instance id of it), as recorded in `ends`: every instance of
    /// it that still owes a report sends CANCELLED now, and so does any of them that fails later.
    pub(crate) fn cancel_query(&self, query: FragmentInstanceId, reason: &str) {
        let mut state = self.lock();
        let cancelled: Vec<FragmentInstanceId> = state
            .owed
            .keys()
            .filter(|instance| instance.query_hi() == query.query_hi())
            .copied()
            .collect();
        let targets: Vec<ReportTarget> = cancelled
            .iter()
            .filter_map(|instance| state.owed.remove(instance))
            .collect();
        drop(state);
        for target in targets {
            self.send(Report {
                target,
                status: status(TStatusCode::CANCELLED, Some(reason)),
            });
        }
    }

    /// `query` (any instance id of it) failed on this CN without one instance to blame (a
    /// fragment panicked): every instance of it still owed reports `error`.
    pub(crate) fn fail_query(&self, query: FragmentInstanceId, error: &str) {
        let mut state = self.lock();
        let failed: Vec<FragmentInstanceId> = state
            .owed
            .keys()
            .filter(|instance| instance.query_hi() == query.query_hi())
            .copied()
            .collect();
        let code = if self.ends.ended_by_fe(query) {
            TStatusCode::CANCELLED
        } else {
            TStatusCode::INTERNAL_ERROR
        };
        let targets: Vec<ReportTarget> = failed
            .iter()
            .filter_map(|instance| state.owed.remove(instance))
            .collect();
        drop(state);
        for target in targets {
            self.send(Report {
                target,
                status: status(code, Some(error)),
            });
        }
    }

    /// Whether an instance of `query` (any instance id of it) still owes a report: one still
    /// waiting or running here.
    pub(crate) fn owes(&self, query: FragmentInstanceId) -> bool {
        self.lock()
            .owed
            .keys()
            .any(|instance| instance.query_hi() == query.query_hi())
    }

    /// How many instances still owe a report. Zero on an idle CN.
    pub(crate) fn owed(&self) -> usize {
        self.lock().owed.len()
    }

    /// Queues `report` on its FE address's reporter thread.
    fn send(&self, report: Report) {
        let coord = &report.target.coord;
        let key = (coord.hostname.clone(), coord.port);
        let mut reporters = self
            .reporters
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let reporter = reporters.entry(key).or_insert_with(|| {
            let (reports, queue) = channel::<Report>();
            let spawned = std::thread::Builder::new()
                .name("fe-report".to_string())
                .spawn(move || {
                    for report in queue {
                        deliver(&report);
                    }
                });
            if let Err(err) = spawned {
                warn!(error = %err, "cannot start an FE report thread; its reports are dropped");
            }
            reports
        });
        if reporter.send(report).is_err() {
            warn!("the FE report thread is gone; dropping a report");
        }
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, ReportState> {
        self.state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }
}

/// Sends one final report, trying an unreachable FE again a few times.
fn deliver(report: &Report) {
    let target = &report.target;
    let instance = target.instance();
    let code = report.status.status_code;
    for attempt in 1..=ATTEMPTS {
        match report_exec_status(
            &target.coord,
            &target.query_id,
            target.backend_num,
            &target.fragment_instance_id,
            report.status.clone(),
            true,
        ) {
            Ok(()) => {
                info!(
                    %instance,
                    backend_num = target.backend_num,
                    ?code,
                    "reported an instance's end to the FE"
                );
                return;
            }
            Err(err) if attempt < ATTEMPTS => {
                warn!(%instance, attempt, error = %err, "reportExecStatus failed; trying again");
                std::thread::sleep(RETRY_WAIT);
            }
            Err(err) => warn!(%instance, error = %err, "reportExecStatus failed; giving up"),
        }
    }
}

fn status(code: TStatusCode, message: Option<&str>) -> TStatus {
    TStatus {
        status_code: code,
        error_msgs: message.map(|message| vec![message.to_string()]),
    }
}

/// A stand-in FE that serves `reportExecStatus` over thrift and records what it receives.
#[cfg(test)]
pub(crate) mod fake_frontend {
    use std::net::TcpListener;
    use std::sync::{Arc, Mutex};
    use std::time::{Duration, Instant};

    use starrocks_thrift::{
        frontend_service::{TReportExecStatusParams, TReportExecStatusResult},
        status::TStatus,
        status_code::TStatusCode,
        types::TNetworkAddress,
    };
    use thrift::protocol::{
        TBinaryInputProtocol, TBinaryOutputProtocol, TFieldIdentifier, TInputProtocol,
        TMessageIdentifier, TMessageType, TOutputProtocol, TSerializable, TStructIdentifier, TType,
    };
    use thrift::transport::{
        TBufferedReadTransport, TBufferedWriteTransport, TIoChannel, TTcpChannel,
    };

    pub(crate) struct FakeFrontend {
        pub(crate) address: TNetworkAddress,
        reports: Arc<Mutex<Vec<TReportExecStatusParams>>>,
    }

    impl FakeFrontend {
        /// Starts serving on a free loopback port, answering every report with `reply`.
        pub(crate) fn start(reply: TStatusCode) -> Self {
            let listener = TcpListener::bind("127.0.0.1:0").unwrap();
            let port = listener.local_addr().unwrap().port();
            let reports = Arc::new(Mutex::new(Vec::new()));
            let recorded = Arc::clone(&reports);
            std::thread::spawn(move || {
                for stream in listener.incoming().flatten() {
                    let recorded = Arc::clone(&recorded);
                    std::thread::spawn(move || serve(stream, reply, &recorded));
                }
            });
            Self {
                address: TNetworkAddress::new("127.0.0.1".to_string(), i32::from(port)),
                reports,
            }
        }

        /// The reports received so far, once there are `count`, after a pause long enough for a
        /// report that should not come to arrive.
        pub(crate) fn wait_for(&self, count: usize) -> Vec<TReportExecStatusParams> {
            let deadline = Instant::now() + Duration::from_secs(10);
            while self.reports.lock().unwrap().len() < count && Instant::now() < deadline {
                std::thread::sleep(Duration::from_millis(5));
            }
            std::thread::sleep(Duration::from_millis(100));
            self.reports.lock().unwrap().clone()
        }
    }

    /// Answers each `reportExecStatus` call on one connection.
    fn serve(
        stream: std::net::TcpStream,
        reply: TStatusCode,
        reports: &Mutex<Vec<TReportExecStatusParams>>,
    ) {
        let (read, write) = TTcpChannel::with_stream(stream).split().unwrap();
        let mut input = TBinaryInputProtocol::new(TBufferedReadTransport::new(read), true);
        let mut output = TBinaryOutputProtocol::new(TBufferedWriteTransport::new(write), true);
        while let Ok(call) = input.read_message_begin() {
            assert_eq!(call.name, "reportExecStatus");
            input.read_struct_begin().unwrap();
            loop {
                let field = input.read_field_begin().unwrap();
                match (field.field_type, field.id) {
                    (TType::Stop, _) => break,
                    (_, Some(1)) => reports
                        .lock()
                        .unwrap()
                        .push(TReportExecStatusParams::read_from_in_protocol(&mut input).unwrap()),
                    (other, _) => input.skip(other).unwrap(),
                }
                input.read_field_end().unwrap();
            }
            input.read_struct_end().unwrap();
            input.read_message_end().unwrap();

            let result = TReportExecStatusResult {
                status: Some(TStatus {
                    status_code: reply,
                    error_msgs: None,
                }),
            };
            output
                .write_message_begin(&TMessageIdentifier::new(
                    "reportExecStatus",
                    TMessageType::Reply,
                    call.sequence_number,
                ))
                .unwrap();
            output
                .write_struct_begin(&TStructIdentifier::new("result"))
                .unwrap();
            output
                .write_field_begin(&TFieldIdentifier::new("success", TType::Struct, 0))
                .unwrap();
            result.write_to_out_protocol(&mut output).unwrap();
            output.write_field_end().unwrap();
            output.write_field_stop().unwrap();
            output.write_struct_end().unwrap();
            output.write_message_end().unwrap();
            output.flush().unwrap();
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use starrocks_thrift::internal_service::{InternalServiceVersion, TPlanFragmentExecParams};

    use super::fake_frontend::FakeFrontend;
    use super::*;

    #[test]
    fn a_report_reaches_the_fe_and_not_found_counts_as_delivered() {
        let frontend = FakeFrontend::start(TStatusCode::OK);
        let (query, instance) = (TUniqueId::new(7, 0), TUniqueId::new(7, 3));
        let failed = status(TStatusCode::INTERNAL_ERROR, Some("scan exploded"));
        report_exec_status(
            &frontend.address,
            &query,
            4,
            &instance,
            failed.clone(),
            true,
        )
        .unwrap();
        let reports = frontend.wait_for(1);
        assert_eq!(reports.len(), 1);
        let report = &reports[0];
        assert_eq!(
            (
                report.query_id.as_ref(),
                report.backend_num,
                report.fragment_instance_id.as_ref(),
                report.status.as_ref(),
                report.done
            ),
            (
                Some(&query),
                Some(4),
                Some(&instance),
                Some(&failed),
                Some(true)
            )
        );

        let ok = status(TStatusCode::OK, None);
        let finished = FakeFrontend::start(TStatusCode::NOT_FOUND);
        report_exec_status(&finished.address, &query, 4, &instance, ok.clone(), true).unwrap();
        let refusing = FakeFrontend::start(TStatusCode::INTERNAL_ERROR);
        assert!(report_exec_status(&refusing.address, &query, 4, &instance, ok, true).is_err());
    }

    /// Instance `instance` of query 9, whose backend number is `100 + instance`.
    fn params(frontend: &FakeFrontend, instance: i64) -> TExecPlanFragmentParams {
        TExecPlanFragmentParams {
            protocol_version: InternalServiceVersion::V1,
            fragment: None,
            desc_tbl: None,
            coord: Some(frontend.address.clone()),
            backend_num: Some(100 + instance as i32),
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
            params: Some(TPlanFragmentExecParams::new(
                TUniqueId::new(9, 0),
                TUniqueId::new(9, instance),
                BTreeMap::new(),
                BTreeMap::new(),
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
            )),
        }
    }

    #[test]
    fn a_coordinator_that_is_not_the_configured_fe_gets_no_report() {
        let frontend = FakeFrontend::start(TStatusCode::OK);
        let reports = ExecReports::new(Some(Host::local()), Arc::default());
        let mut elsewhere = params(&frontend, 1);
        elsewhere.coord = Some(TNetworkAddress::new("10.0.0.9".to_string(), 9020));
        reports.expect(&elsewhere);
        assert_eq!(
            reports.owed(),
            0,
            "no connection to 10.0.0.9 is ever opened"
        );
        reports.expect(&params(&frontend, 2));
        assert_eq!(reports.owed(), 1);
    }

    #[test]
    fn an_fe_that_does_not_answer_delays_no_other_fe() {
        // Accepts connections and never replies, so each try waits out the read timeout.
        let silent = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let silent_port = silent.local_addr().unwrap().port();
        let frontend = FakeFrontend::start(TStatusCode::OK);
        let reports = ExecReports::default();
        let mut stuck = params(&frontend, 1);
        stuck.coord = Some(TNetworkAddress::new(
            "127.0.0.1".to_string(),
            i32::from(silent_port),
        ));
        reports.expect(&stuck);
        reports.expect(&params(&frontend, 2));
        reports.finish(&stuck, Ok(()));
        let started = std::time::Instant::now();
        reports.finish(&params(&frontend, 2), Ok(()));
        assert_eq!(ends(&frontend.wait_for(1)), [(102, TStatusCode::OK)]);
        assert!(
            started.elapsed() < Duration::from_secs(3),
            "{:?}",
            started.elapsed()
        );
        drop(silent);
    }

    fn ends(reports: &[TReportExecStatusParams]) -> Vec<(i32, TStatusCode)> {
        let mut ends: Vec<_> = reports
            .iter()
            .map(|report| {
                assert_eq!(report.done, Some(true));
                (
                    report.backend_num.unwrap(),
                    report.status.as_ref().unwrap().status_code,
                )
            })
            .collect();
        ends.sort_unstable_by_key(|(backend_num, _)| *backend_num);
        ends
    }

    #[test]
    fn each_instance_reports_once_and_after_its_query_ended_only_cancelled() {
        let frontend = FakeFrontend::start(TStatusCode::OK);
        let query_ends = Arc::new(QueryEnds::default());
        let reports = ExecReports::new(None, Arc::clone(&query_ends));
        for instance in 1..=4 {
            reports.expect(&params(&frontend, instance));
        }
        reports.finish(&params(&frontend, 1), Ok(()));
        reports.finish(&params(&frontend, 1), Err("after done"));
        reports.finish(&params(&frontend, 2), Err("scan exploded"));
        reports.settled_by_reply(&params(&frontend, 3));
        reports.finish(&params(&frontend, 3), Err("in the dispatch reply"));
        query_ends.cancel(FragmentInstanceId::from_halves(9, 0), "user cancelled");
        reports.cancel_query(FragmentInstanceId::from_halves(9, 0), "user cancelled");
        reports.finish(&params(&frontend, 4), Err("after the cancel"));
        assert_eq!(reports.owed(), 0);
        assert_eq!(
            ends(&frontend.wait_for(3)),
            [
                (101, TStatusCode::OK),
                (102, TStatusCode::INTERNAL_ERROR),
                (104, TStatusCode::CANCELLED),
            ]
        );

        // An instance of a query whose result this CN delivered in full fails as CANCELLED.
        let delivered = FakeFrontend::start(TStatusCode::OK);
        let query_ends = Arc::new(QueryEnds::default());
        let reports = ExecReports::new(None, Arc::clone(&query_ends));
        reports.expect(&params(&delivered, 5));
        query_ends.delivered(FragmentInstanceId::from_halves(9, 1));
        reports.finish(&params(&delivered, 5), Err("teardown failed"));
        assert_eq!(
            ends(&delivered.wait_for(1)),
            [(105, TStatusCode::CANCELLED)]
        );
    }
}
