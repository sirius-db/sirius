//! End-to-end FE↔CN integration tests.
//!
//! Each test boots a real StarRocks frontend (built on demand) and runs the compute node
//! in-process via [`starrocks_test_harness::TestCluster`]. Run them with a pixi env that has
//! both Java and Rust active, e.g. `pixi run -e default cargo test`. The first run builds the
//! FE (minutes); later runs reuse `starrocks/output/fe` and are fast. FE/CN logs for a failed
//! run land under the cluster's temp `STARROCKS_HOME` (path is printed in the failure message).
//!
//! FE startup is the dominant cost, so related assertions are grouped to minimise FE boots:
//! one cluster covers registration + liveness + graceful shutdown, a second covers FE-restart
//! resilience.

use std::time::Duration;

use starrocks_test_harness::TestCluster;

/// Liveness can take a few heartbeat cycles after registration; keep this generous for CI.
const CN_ALIVE_TIMEOUT: Duration = Duration::from_secs(60);

/// The CN registers with the FE, becomes visible and alive, learns FE identity over heartbeats,
/// and then shuts down cleanly.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn registration_liveness_and_shutdown() {
    let mut cluster = TestCluster::start()
        .await
        .expect("failed to start FE + CN cluster");

    // Registration: ComputeNode::start only returns after the FE confirms the node, so it must
    // already be visible in SHOW COMPUTE NODES.
    assert!(
        cluster
            .cn_registered()
            .await
            .expect("failed to query compute node registration"),
        "compute node should be registered with the FE",
    );

    // Liveness: the FE heartbeats the CN until it reports alive.
    cluster
        .wait_for_cn_alive(CN_ALIVE_TIMEOUT)
        .await
        .expect("compute node did not become alive");

    // Heartbeat identity: a live CN must have learned the FE's cluster id, epoch, and the
    // compute-node id the FE assigned it.
    let snapshot = cluster.cn_state().snapshot();
    assert!(
        snapshot.cluster_id.is_some(),
        "CN should learn FE cluster id"
    );
    assert!(snapshot.epoch.is_some(), "CN should learn FE epoch");
    assert!(
        snapshot.compute_node_id.is_some(),
        "CN should learn its FE-assigned compute node id",
    );

    // Graceful shutdown: stopping the CN tears down the servers and maintenance loops cleanly.
    cluster
        .stop_cn()
        .await
        .expect("compute node shutdown failed");

    cluster.stop().await.expect("cluster teardown failed");
}

/// After the FE process is bounced (reusing its meta dir), the CN re-establishes liveness.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cn_recovers_after_fe_restart() {
    let mut cluster = TestCluster::start()
        .await
        .expect("failed to start FE + CN cluster");

    cluster
        .wait_for_cn_alive(CN_ALIVE_TIMEOUT)
        .await
        .expect("compute node did not become alive before restart");

    cluster.restart_fe().await.expect("failed to restart FE");

    // The CN's registration-refresh and heartbeat handling should bring it back to alive.
    cluster
        .wait_for_cn_alive(CN_ALIVE_TIMEOUT)
        .await
        .expect("compute node did not recover after FE restart");

    cluster.stop().await.expect("cluster teardown failed");
}
