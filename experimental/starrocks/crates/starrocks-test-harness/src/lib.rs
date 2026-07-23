//! Integration-test harness for the Sirius StarRocks compute node (CN).
//!
//! [`TestCluster`] boots a real StarRocks frontend (FE, Java) and runs the CN in-process so a
//! plain `cargo test` (inside the pixi env that provides both Java and Rust) can exercise the
//! full FE↔CN handshake: SQL registration, FE→CN heartbeats, and CN→FE inventory reports.
//!
//! The FE binary is built on demand: if `starrocks/output/fe` is missing the harness shells out
//! to `pixi run fe-build` (the single source of truth for the FE build). Each cluster runs in an
//! isolated, throwaway `STARROCKS_HOME` (own meta dir, log dir, config, and ports) so runs are
//! reproducible and several clusters can coexist. Everything is torn down on
//! [`TestCluster::stop`] or when the value is dropped.

use std::{
    fs,
    net::{Ipv4Addr, SocketAddr, TcpListener},
    os::unix::fs::symlink,
    path::{Path, PathBuf},
    process::{Child, Command, Stdio},
    sync::Mutex,
    time::{Duration, Instant},
};

use anyhow::{Context, Result, anyhow, bail};
use mysql_async::{Conn, OptsBuilder, Row, prelude::Queryable};
use sirius_starrocks_cn::{ComputeNode, ComputeNodeConfig, FeConfig, Host, RegistrationConfig};
use tempfile::TempDir;
use tracing::{info, warn};

/// How long to wait for the FE to accept MySQL connections and report itself alive.
const FE_READY_TIMEOUT: Duration = Duration::from_secs(180);
/// Polling interval while waiting on FE readiness or compute-node state transitions.
const POLL_INTERVAL: Duration = Duration::from_millis(500);
/// Grace period for the FE JVM to exit on SIGTERM before it is force-killed.
const FE_SHUTDOWN_GRACE: Duration = Duration::from_secs(20);
/// Heap cap for the test FE: the production default (`-Xmx8192m`) is wasteful for tests and can
/// fail to start on smaller CI runners.
const TEST_FE_JAVA_OPTS: &str = "-Dlog4j2.formatMsgNoLookups=true -Xmx2g -XX:+UseG1GC \
     -Djava.security.policy=${STARROCKS_HOME}/conf/udf_security.policy";

/// A running FE + in-process CN, with the FE's throwaway `STARROCKS_HOME` and process handle.
pub struct TestCluster {
    fe: FrontendProcess,
    cn: Option<ComputeNode>,
    fe_config: FeConfig,
    cn_config: ComputeNodeConfig,
}

impl TestCluster {
    /// Boots the FE (building it first if needed) and starts the CN in-process against it.
    pub async fn start() -> Result<Self> {
        let project = Project::locate()?;
        project.ensure_fe_built()?;

        let mut fe = FrontendProcess::spawn(&project)?;
        // On any startup failure, preserve the FE home (logs + conf) so the failure can be
        // diagnosed — error messages point at it. The happy path falls through and the home is
        // cleaned up on drop/`stop`.
        match Self::bring_up(&mut fe).await {
            Ok((fe_config, cn_config, cn)) => Ok(Self {
                fe,
                cn: Some(cn),
                fe_config,
                cn_config,
            }),
            Err(err) => {
                fe.keep_home_for_debug();
                Err(err)
            }
        }
    }

    /// Waits for the FE to be ready and starts the CN against it, returning the resolved configs.
    async fn bring_up(
        fe: &mut FrontendProcess,
    ) -> Result<(FeConfig, ComputeNodeConfig, ComputeNode)> {
        let mut conn = fe.wait_until_ready().await?;

        // Use the FE's self-advertised IP for both the MySQL control connection and the CN's
        // FE-host trust check, so CN→FE reports target an address the CN already trusts.
        let fe_host = frontend_advertised_host(&mut conn).await?;
        drop(conn);

        let fe_config = FeConfig {
            host: Host::new(fe_host).context("FE advertised an empty host")?,
            query_port: fe.ports.query_port,
            ..FeConfig::default()
        };
        let cn_config = test_compute_node_config()?;

        let cn = ComputeNode::start(
            fe_config.clone(),
            cn_config.clone(),
            RegistrationConfig::default(),
        )
        .await
        .context("failed to start in-process compute node")?;

        Ok((fe_config, cn_config, cn))
    }

    /// Opens a fresh MySQL connection to the FE query port (caller drops it when done).
    pub async fn fe_conn(&self) -> Result<Conn> {
        connect_fe(&self.fe_config).await
    }

    /// Shared heartbeat state of the running CN, for asserting learned FE identity.
    pub fn cn_state(&self) -> sirius_starrocks_cn::SharedHeartbeatState {
        self.cn.as_ref().expect("compute node is running").state()
    }

    /// The CN's advertised host and heartbeat port, as registered with the FE.
    pub fn cn_advertise(&self) -> (String, u16) {
        (
            self.cn_config.advertise_host.to_string(),
            self.cn_config.heartbeat_port,
        )
    }

    /// Stops the in-process CN (idempotent; a no-op if already stopped).
    pub async fn stop_cn(&mut self) -> Result<()> {
        if let Some(cn) = self.cn.take() {
            cn.shutdown()
                .await
                .context("compute node shutdown failed")?;
        }
        Ok(())
    }

    /// Restarts the CN in-process against the same FE (the CN must be stopped first or this
    /// stops it). Used by resilience tests after toggling CN or FE availability.
    pub async fn restart_cn(&mut self) -> Result<()> {
        self.stop_cn().await?;
        let cn = ComputeNode::start(
            self.fe_config.clone(),
            self.cn_config.clone(),
            RegistrationConfig::default(),
        )
        .await
        .context("failed to restart in-process compute node")?;
        self.cn = Some(cn);
        Ok(())
    }

    /// Restarts the FE process, reusing the same meta dir and ports so cluster identity and the
    /// CN registration survive. Used to test that the CN recovers after an FE bounce.
    pub async fn restart_fe(&mut self) -> Result<()> {
        self.fe.terminate();
        // Transfer the home temp dir to the respawned process before the old one is dropped, so
        // dropping the old process does not delete the STARROCKS_HOME the new one reuses.
        let ports = self.fe.ports.clone();
        let home_path = self.fe.home();
        let home_dir = self.fe.take_home();
        self.fe = FrontendProcess::respawn(ports, home_path, home_dir)?;
        self.fe.wait_until_ready().await?;
        Ok(())
    }

    /// Returns true if the CN appears in `SHOW COMPUTE NODES` (registered, regardless of liveness).
    pub async fn cn_registered(&self) -> Result<bool> {
        let (host, port) = self.cn_advertise();
        let mut conn = self.fe_conn().await?;
        let present = compute_node_present(&mut conn, &host, port).await?;
        drop(conn);
        Ok(present)
    }

    /// Polls `SHOW COMPUTE NODES` until the CN row reports `Alive = true`, or times out.
    pub async fn wait_for_cn_alive(&self, timeout: Duration) -> Result<()> {
        let (host, port) = self.cn_advertise();
        let deadline = Instant::now() + timeout;
        loop {
            let mut conn = self.fe_conn().await?;
            let alive = compute_node_alive(&mut conn, &host, port).await?;
            drop(conn);
            if alive {
                return Ok(());
            }
            if Instant::now() >= deadline {
                bail!("compute node {host}:{port} did not become alive within {timeout:?}");
            }
            tokio::time::sleep(POLL_INTERVAL).await;
        }
    }

    /// Stops the CN and the FE and removes the throwaway `STARROCKS_HOME`.
    pub async fn stop(mut self) -> Result<()> {
        self.stop_cn().await?;
        self.fe.terminate();
        Ok(())
    }
}

/// Resolved layout of the `experimental/starrocks` pixi project.
struct Project {
    root: PathBuf,
}

impl Project {
    /// Resolves the project root (`experimental/starrocks`) relative to this crate.
    fn locate() -> Result<Self> {
        // CARGO_MANIFEST_DIR is .../experimental/starrocks/crates/starrocks-test-harness.
        let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let root = manifest_dir
            .ancestors()
            .nth(2)
            .ok_or_else(|| anyhow!("failed to resolve project root from {manifest_dir:?}"))?
            .to_path_buf();
        if !root.join("pixi.toml").is_file() {
            bail!("expected pixi.toml at project root {root:?}");
        }
        Ok(Self { root })
    }

    /// Directory containing the packaged FE (`bin/`, `conf/`, `lib/`, ...).
    fn fe_output(&self) -> PathBuf {
        self.root.join("starrocks/output/fe")
    }

    /// The Sirius FE config template applied (with port/heap overrides) to each cluster.
    fn fe_conf_template(&self) -> PathBuf {
        self.root.join("conf/fe.conf")
    }

    /// Builds the FE via `pixi run fe-build` if the packaged output is missing.
    ///
    /// `cargo test` runs the integration tests concurrently, so several clusters can call this at
    /// once on a fresh checkout. The lock serializes them: the first builds while the rest wait,
    /// then they observe the finished `start_fe.sh` and return. This guards against concurrent
    /// Maven builds into the same tree and against reading a half-written launcher within this
    /// test process (the only process that triggers the build).
    fn ensure_fe_built(&self) -> Result<()> {
        static FE_BUILD_LOCK: Mutex<()> = Mutex::new(());
        let _guard = FE_BUILD_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());

        if self.fe_output().join("bin/start_fe.sh").is_file() {
            return Ok(());
        }
        info!("StarRocks FE output missing; building via `pixi run fe-build` (this is slow)");
        let status = Command::new("pixi")
            .args(["run", "fe-build"])
            .current_dir(&self.root)
            .status()
            .context(
                "failed to launch `pixi run fe-build`; is pixi on PATH? \
                 build the FE manually with `pixi run fe-build`",
            )?;
        if !status.success() {
            bail!("`pixi run fe-build` failed with status {status}");
        }
        if !self.fe_output().join("bin/start_fe.sh").is_file() {
            bail!(
                "`pixi run fe-build` completed but {:?} is missing",
                self.fe_output().join("bin/start_fe.sh")
            );
        }
        Ok(())
    }
}

/// The four FE ports a cluster listens on; allocated free so clusters don't collide.
#[derive(Clone, Debug)]
struct FrontendPorts {
    http_port: u16,
    rpc_port: u16,
    query_port: u16,
    edit_log_port: u16,
}

impl FrontendPorts {
    fn allocate() -> Result<Self> {
        Ok(Self {
            http_port: free_port()?,
            rpc_port: free_port()?,
            query_port: free_port()?,
            edit_log_port: free_port()?,
        })
    }
}

/// A spawned FE JVM plus its isolated `STARROCKS_HOME` temp dir.
struct FrontendProcess {
    child: Child,
    ports: FrontendPorts,
    // Owns the throwaway STARROCKS_HOME: dropping it removes the directory. On an FE restart the
    // `TempDir` is transferred to the respawned process (see `take_home`) so the home survives the
    // bounce; whichever `FrontendProcess` finally drops with it cleans the directory up.
    home_dir: Option<TempDir>,
    home_path: PathBuf,
}

impl FrontendProcess {
    /// Provisions a fresh isolated `STARROCKS_HOME` and launches the FE.
    fn spawn(project: &Project) -> Result<Self> {
        let ports = FrontendPorts::allocate()?;
        let home = provision_fe_home(project, &ports)?;
        let home_path = home.path().to_path_buf();
        let child = launch_fe(&home_path)?;
        Ok(Self {
            child,
            ports,
            home_dir: Some(home),
            home_path,
        })
    }

    /// Relaunches the FE in an existing `STARROCKS_HOME` (reusing meta + ports), taking over
    /// ownership of the home temp dir so it is not deleted while still in use.
    fn respawn(
        ports: FrontendPorts,
        home_path: PathBuf,
        home_dir: Option<TempDir>,
    ) -> Result<Self> {
        let child = launch_fe(&home_path)?;
        Ok(Self {
            child,
            ports,
            home_dir,
            home_path,
        })
    }

    fn home(&self) -> PathBuf {
        self.home_path.clone()
    }

    /// Takes ownership of the home temp dir, so a respawned process can keep it alive.
    fn take_home(&mut self) -> Option<TempDir> {
        self.home_dir.take()
    }

    /// Disarms the home temp dir's auto-cleanup so the FE logs/config survive for diagnosis, and
    /// prints where they are. Used on startup failure and (via `Drop`) on test panics.
    fn keep_home_for_debug(&mut self) {
        if let Some(home) = self.home_dir.take() {
            let path = home.keep();
            eprintln!(
                "starrocks-test-harness: preserved FE home for debugging at {} \
                 (logs under {}/log, launcher output in {}/fe.bootstrap.log)",
                path.display(),
                path.display(),
                path.display()
            );
        }
    }

    /// Waits until the FE accepts MySQL connections and reports a live frontend, returning an
    /// open control connection. Fails fast if the FE process exits during startup.
    async fn wait_until_ready(&mut self) -> Result<Conn> {
        let fe = FeConfig {
            host: Host::local(),
            query_port: self.ports.query_port,
            ..FeConfig::default()
        };
        let deadline = Instant::now() + FE_READY_TIMEOUT;
        loop {
            if let Some(status) = self.exited() {
                bail!(
                    "FE process exited during startup ({status}); see logs under {:?}",
                    self.home_path.join("log")
                );
            }
            match connect_fe(&fe).await {
                Ok(mut conn) => {
                    if frontend_is_alive(&mut conn).await.unwrap_or(false) {
                        return Ok(conn);
                    }
                    drop(conn);
                }
                Err(err) => {
                    if Instant::now() >= deadline {
                        return Err(err).with_context(|| {
                            format!(
                                "FE did not accept connections within {FE_READY_TIMEOUT:?}; \
                                 see logs under {:?}",
                                self.home_path.join("log")
                            )
                        });
                    }
                }
            }
            if Instant::now() >= deadline {
                bail!(
                    "FE did not report a live frontend within {FE_READY_TIMEOUT:?}; \
                     see logs under {:?}",
                    self.home_path.join("log")
                );
            }
            tokio::time::sleep(POLL_INTERVAL).await;
        }
    }

    /// Returns the exit status if the FE process has already terminated.
    fn exited(&mut self) -> Option<std::process::ExitStatus> {
        self.child.try_wait().ok().flatten()
    }

    /// Sends SIGTERM to the FE, waits a grace period, then force-kills if still alive.
    fn terminate(&mut self) {
        let pid = self.child.id() as i32;
        unsafe {
            libc::kill(pid, libc::SIGTERM);
        }
        let deadline = Instant::now() + FE_SHUTDOWN_GRACE;
        loop {
            match self.child.try_wait() {
                Ok(Some(_)) => return,
                Ok(None) => {}
                Err(err) => {
                    warn!(error = %err, "failed to poll FE process; force-killing");
                    break;
                }
            }
            if Instant::now() >= deadline {
                warn!("FE did not exit on SIGTERM within grace period; force-killing");
                break;
            }
            std::thread::sleep(POLL_INTERVAL);
        }
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

impl Drop for FrontendProcess {
    fn drop(&mut self) {
        // Best-effort: never leak the FE JVM if the cluster was dropped without `stop`.
        if self.child.try_wait().ok().flatten().is_none() {
            self.terminate();
        }
        // If the cluster is being dropped while a test is panicking (a failed assertion or an
        // `expect` on a later step), preserve the FE home so the logs survive for diagnosis.
        if std::thread::panicking() {
            self.keep_home_for_debug();
        }
    }
}

/// Builds an isolated `STARROCKS_HOME` for one FE: real `bin/`+`conf/` (so `fe.pid` and the
/// overridden `fe.conf` stay per-cluster) and symlinked heavy artifacts (`lib/`, `webroot/`, ...).
/// The home lives under `target/fe-it/` (not `/tmp`) so a preserved-on-failure home is easy to
/// find locally and to collect as a CI artifact.
fn provision_fe_home(project: &Project, ports: &FrontendPorts) -> Result<TempDir> {
    let output = project.fe_output();
    let base = project.root.join("target/fe-it");
    fs::create_dir_all(&base)
        .with_context(|| format!("failed to create FE home base dir {base:?}"))?;
    let home = tempfile::Builder::new()
        .prefix("fe-")
        .tempdir_in(&base)
        .context("failed to create FE home temp dir")?;
    let home_path = home.path();

    copy_dir(&output.join("bin"), &home_path.join("bin"))
        .context("failed to copy FE bin directory")?;
    copy_dir(&output.join("conf"), &home_path.join("conf"))
        .context("failed to copy FE conf directory")?;
    for artifact in ["lib", "spark-dpp", "hive-udf", "webroot"] {
        let src = output.join(artifact);
        if src.exists() {
            symlink(&src, home_path.join(artifact))
                .with_context(|| format!("failed to symlink FE artifact {artifact}"))?;
        }
    }
    fs::create_dir_all(home_path.join("log")).context("failed to create FE log dir")?;
    fs::create_dir_all(home_path.join("meta")).context("failed to create FE meta dir")?;

    write_fe_conf(
        &project.fe_conf_template(),
        &home_path.join("conf/fe.conf"),
        ports,
    )?;
    Ok(home)
}

/// Renders the per-cluster `fe.conf` from the Sirius template, appending overrides (duplicate
/// keys: last value wins) for the allocated ports and a test-sized JVM heap. The FE's advertised
/// IP is read back at runtime and used as the CN's FE host, so no `priority_networks` pin is
/// needed (and a loopback one is rejected by some StarRocks versions).
fn write_fe_conf(template: &Path, dest: &Path, ports: &FrontendPorts) -> Result<()> {
    let mut conf = fs::read_to_string(template)
        .with_context(|| format!("failed to read FE conf template {template:?}"))?;
    conf.push_str(&format!(
        "\n# ---- Sirius integration-test overrides (appended; last value wins) ----\n\
         http_port = {}\n\
         rpc_port = {}\n\
         query_port = {}\n\
         edit_log_port = {}\n\
         JAVA_OPTS=\"{TEST_FE_JAVA_OPTS}\"\n",
        ports.http_port, ports.rpc_port, ports.query_port, ports.edit_log_port,
    ));
    fs::write(dest, conf).with_context(|| format!("failed to write FE conf {dest:?}"))?;
    Ok(())
}

/// Spawns `bin/start_fe.sh` in the given `STARROCKS_HOME`. Without `--logconsole` the FE redirects
/// its own output to `log/fe.out`; the pre-exec script output is captured to `fe.bootstrap.log`.
fn launch_fe(home_path: &Path) -> Result<Child> {
    let bootstrap_log = fs::File::create(home_path.join("fe.bootstrap.log"))
        .context("failed to create FE bootstrap log")?;
    let stderr_log = bootstrap_log
        .try_clone()
        .context("failed to clone FE bootstrap log handle")?;
    let child = Command::new(home_path.join("bin/start_fe.sh"))
        .current_dir(home_path)
        .stdin(Stdio::null())
        .stdout(Stdio::from(bootstrap_log))
        .stderr(Stdio::from(stderr_log))
        .spawn()
        .with_context(|| {
            format!(
                "failed to spawn FE via {:?}",
                home_path.join("bin/start_fe.sh")
            )
        })?;
    Ok(child)
}

/// Default CN config for tests: loopback bind/advertise on freshly allocated ports.
fn test_compute_node_config() -> Result<ComputeNodeConfig> {
    Ok(ComputeNodeConfig {
        bind_host: Host::local(),
        advertise_host: Host::local(),
        heartbeat_port: free_port()?,
        thrift_port: free_port()?,
        http_port: free_port()?,
        brpc_port: free_port()?,
        ..ComputeNodeConfig::default()
    })
}

/// Opens a MySQL connection to the FE using its configured host/port and root credentials.
async fn connect_fe(fe: &FeConfig) -> Result<Conn> {
    let opts = OptsBuilder::default()
        .ip_or_hostname(fe.host.to_string())
        .tcp_port(fe.query_port)
        .prefer_socket(false)
        .user(Some(fe.user.clone()))
        .pass(Some(fe.password.expose_secret().to_string()));
    Conn::new(opts)
        .await
        .with_context(|| format!("failed to connect to FE at {}:{}", fe.host, fe.query_port))
}

/// Returns true once at least one frontend reports `Alive = true` in `SHOW FRONTENDS`.
async fn frontend_is_alive(conn: &mut Conn) -> Result<bool> {
    let rows: Vec<Row> = conn
        .query("SHOW FRONTENDS")
        .await
        .context("failed to query SHOW FRONTENDS")?;
    Ok(rows.iter().any(|row| row_bool(row, "Alive")))
}

/// Reads the FE's self-advertised IP from `SHOW FRONTENDS` (the `IP` column).
async fn frontend_advertised_host(conn: &mut Conn) -> Result<String> {
    let rows: Vec<Row> = conn
        .query("SHOW FRONTENDS")
        .await
        .context("failed to query SHOW FRONTENDS")?;
    rows.iter()
        .find_map(|row| row.get::<String, _>("IP").filter(|ip| !ip.is_empty()))
        .ok_or_else(|| anyhow!("SHOW FRONTENDS returned no usable IP"))
}

/// Returns true if a compute node with the given host + heartbeat port reports `Alive = true`.
async fn compute_node_alive(conn: &mut Conn, host: &str, heartbeat_port: u16) -> Result<bool> {
    matching_compute_node(conn, host, heartbeat_port)
        .await
        .map(|row| row.is_some_and(|row| row_bool(&row, "Alive")))
}

/// Returns true if a compute node with the given host + heartbeat port is registered with the FE.
async fn compute_node_present(conn: &mut Conn, host: &str, heartbeat_port: u16) -> Result<bool> {
    matching_compute_node(conn, host, heartbeat_port)
        .await
        .map(|row| row.is_some())
}

/// Finds the `SHOW COMPUTE NODES` row for the given host + heartbeat port, if registered.
async fn matching_compute_node(
    conn: &mut Conn,
    host: &str,
    heartbeat_port: u16,
) -> Result<Option<Row>> {
    let rows: Vec<Row> = conn
        .query("SHOW COMPUTE NODES")
        .await
        .context("failed to query SHOW COMPUTE NODES")?;
    let port = heartbeat_port.to_string();
    Ok(rows.into_iter().find(|row| {
        row.get::<String, _>("IP").as_deref() == Some(host)
            && row.get::<String, _>("HeartbeatPort").as_deref() == Some(port.as_str())
    }))
}

/// Reads a StarRocks boolean-ish column (`true`/`false`) by name, defaulting to false.
fn row_bool(row: &Row, column: &str) -> bool {
    row.get::<String, _>(column)
        .map(|value| value.eq_ignore_ascii_case("true"))
        .unwrap_or(false)
}

/// Recursively copies a directory tree of real files (resolving any symlinks to file contents).
fn copy_dir(src: &Path, dest: &Path) -> Result<()> {
    fs::create_dir_all(dest).with_context(|| format!("failed to create {dest:?}"))?;
    for entry in fs::read_dir(src).with_context(|| format!("failed to read {src:?}"))? {
        let entry = entry?;
        let file_type = entry.file_type()?;
        let target = dest.join(entry.file_name());
        if file_type.is_dir() {
            copy_dir(&entry.path(), &target)?;
        } else {
            fs::copy(entry.path(), &target)
                .with_context(|| format!("failed to copy {:?}", entry.path()))?;
        }
    }
    Ok(())
}

/// Picks a currently-free TCP port on loopback (subject to a benign allocate-then-bind race).
fn free_port() -> Result<u16> {
    let listener = TcpListener::bind(SocketAddr::from((Ipv4Addr::LOCALHOST, 0)))
        .context("failed to bind a free port")?;
    Ok(listener.local_addr()?.port())
}
