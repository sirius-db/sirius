//! One registry for the CN's data-transport and dispatch tunables.
//!
//! Every knob the exchange path has is declared here with its environment name, its default,
//! and its valid range (or accepted set, for [`FusionMode`]), and the whole set is resolved ONCE
//! at bring-up ([`Tunables::resolve`]) so a bad value fails the CN before it accepts a query
//! rather than mid-sweep.
//!
//! Three rules, each of which the previous ad-hoc parsing broke somewhere:
//!
//! 1. **Out-of-range and unparsable values are REJECTED, never clamped and never ignored.**
//!    An earlier ad-hoc parser did `.and_then(|v| v.parse().ok())`, which turns
//!    `SIRIUS_CN_NIXL_WARMUP_TIMEOUT_SECS=6O` (letter O) into the 180 s default with no
//!    diagnostic — the operator sees the knob having no effect and cannot tell why.
//! 2. **The resolved values are logged at bring-up.** The launcher's echo is what you asked
//!    for; this line is what the CN got.
//! 3. **Unset means the documented default**, so an operator who sets nothing gets exactly the
//!    behaviour these constants had when they were hardcoded.
//!
//! The defaults are the values that were compiled in before this module existed, so resolving
//! an empty environment is a no-op change.

use std::sync::OnceLock;
use std::time::Duration;

/// Filled by [`Tunables::resolve`] at bring-up. Reads that happen before that (unit tests) see
/// [`Tunables::DEFAULTS`].
static RESOLVED: OnceLock<Tunables> = OnceLock::new();

/// Bound on a PRPC connect and on waiting for a reply.
///
/// Generous because a peer's `request_staging_lease` queues behind whatever fragment its engine
/// thread is currently running; a peer that exceeds this is treated as wedged and the query
/// fails loudly. Worth raising at large scale factors: a CN whose engine thread is inside a
/// multi-minute stage cannot answer a lease request, and the sender then fails a query that
/// would have completed — an SF100 q08 refusal at 60758 ms (observed 2026-08-08) was exactly
/// this timeout firing.
const RPC_TIMEOUT_SECS: Knob<u64> = Knob {
    name: "SIRIUS_CN_RPC_TIMEOUT_SECS",
    default: 60,
    min: 1,
    max: 3600,
};

/// Bound on waiting for one posted nixl WRITE to reach DONE.
///
/// This covers the RDMA/NVLink transfer alone, not any queueing behind a peer's engine thread,
/// so it is much tighter than [`RPC_TIMEOUT_SECS`] and should stay that way: it is the one
/// signal that distinguishes a wedged fabric from a busy peer.
const XFER_TIMEOUT_SECS: Knob<u64> = Knob {
    name: "SIRIUS_CN_NIXL_XFER_TIMEOUT_SECS",
    default: 30,
    min: 1,
    max: 3600,
};

/// Bytes of the mandatory first-contact bandwidth canary.
///
/// Pool memory over `cuda_ipc` silently degrades ~220x while still transferring correct bytes,
/// so a slow link must be refused, not tolerated. Large enough that the measurement is
/// bandwidth and not latency; the lower bound keeps it that way.
const CANARY_BYTES: Knob<u64> = Knob {
    name: "SIRIUS_CN_NIXL_CANARY_BYTES",
    default: 16 << 20,
    min: 1 << 20,
    max: 1 << 30,
};

/// Floor under which the link is declared degraded and the transport tier is refused.
///
/// Measured reference points: same-host `cuda_ipc` ~85-90 GB/s on A100 and 322-399 GB/s on the
/// GB200 NV18 mesh; the degraded staged-copy path ~0.4 GB/s; a cross-host `cudaMalloc` IPC
/// handle bounced through the host at 0.32-0.43 GB/s. The default sits an order of magnitude
/// above the degraded paths and two below the healthy ones, so it separates them cleanly on
/// every fabric measured so far.
///
/// `0` disables the check. That is an escape hatch for bringing up a fabric whose healthy
/// bandwidth is genuinely below the floor, not a way to make a failing cluster start — it is
/// logged as a warning at bring-up, because with it set the ~220x silent-degradation trap this
/// canary exists to catch is once again silent.
const CANARY_FLOOR_GBPS: Knob<f64> = Knob {
    name: "SIRIUS_CN_NIXL_CANARY_FLOOR_GBPS",
    default: 2.0,
    min: 0.0,
    max: 10_000.0,
};

/// Wall-clock budget for the whole bring-up session-warmup loop.
///
/// The warmup is best-effort — exhausting the budget is loud but never fails bring-up, because
/// a cold peer still works (slowly) through the lazy `ensure_session` path.
const WARMUP_TIMEOUT_SECS: Knob<u64> = Knob {
    name: "SIRIUS_CN_NIXL_WARMUP_TIMEOUT_SECS",
    default: 180,
    min: 1,
    max: 3600,
};

/// Peer count that ends the warmup loop early once established. Unset means "keep trying every
/// peer the FE reports until the budget runs out".
const WARMUP_EXPECT_PEERS: Knob<u64> = Knob {
    name: "SIRIUS_CN_NIXL_WARMUP_EXPECT_PEERS",
    default: 0, // sentinel: no early exit
    min: 0,
    max: 4096,
};

/// Kill switch for the bring-up session warmup. `off` returns to purely lazy sessions: the first
/// cross-node query after bring-up pays first contact, and on a cold cluster that is the
/// deadlock the warmup exists to prevent (see `nixl_transport/warmup.rs`).
const WARMUP: Switch = Switch {
    name: "SIRIUS_CN_NIXL_WARMUP",
    default: true,
};

/// Explicit warmup peer list, `host:port,host:port`, bypassing FE discovery. Unset means "ask
/// the FE for its alive compute nodes"; this CN's own `host:brpc_port` is filtered out either
/// way. Rule 1 applies per entry: a malformed one rejects the whole list instead of being
/// dropped with a warning, which is how a typo'd peer used to stay cold unnoticed.
const WARMUP_PEERS: PeerList = PeerList {
    name: "SIRIUS_CN_NIXL_WARMUP_PEERS",
};

/// Frames admitted per peer by the optimized transport. Byte reservations are enforced in
/// addition to this count; a handle slot never grants permission to overfill the TX arena.
const TRANSFER_WINDOW: Knob<u64> = Knob {
    name: "SIRIUS_CN_NIXL_TRANSFER_WINDOW",
    default: 1,
    min: 1,
    max: 8,
};

/// Homogeneous protocol/ownership mode shared by the Rust wrapper and C++ engine. Unlike the
/// older on/off switches this deliberately accepts only 0 and 1, matching the C++ gate.
const OPTIMIZED_EXCHANGE: Knob<u64> = Knob {
    name: "SIRIUS_EXCHANGE_OPTIMIZED",
    default: 0,
    min: 0,
    max: 1,
};

/// Bound on independently serviced control peers. One unresponsive peer can occupy only its
/// own worker, leaving every other admitted peer's RPCs and the NIXL owner free to progress.
const TRANSFER_PEERS: Knob<u64> = Knob {
    name: "SIRIUS_CN_NIXL_TRANSFER_PEERS",
    default: 32,
    min: 1,
    max: 128,
};

/// One environment-backed knob: where it is read from, what it is when unset, and the range
/// outside which a value is an error rather than a clamp.
struct Knob<T> {
    name: &'static str,
    default: T,
    min: T,
    max: T,
}

impl Knob<u64> {
    /// The configured value, or the default when unset.
    ///
    /// # Errors
    /// When the value does not parse as an unsigned integer, or falls outside `[min, max]`.
    fn read(&self) -> Result<u64, String> {
        let Some(raw) = env_value(self.name) else {
            return Ok(self.default);
        };
        let value: u64 = raw
            .trim()
            .parse()
            .map_err(|_| self.rejected(&raw, "expected a non-negative integer"))?;
        self.in_range(value, &raw)
    }

    /// Rejects an out-of-range value, naming the range and the default.
    fn in_range(&self, value: u64, raw: &str) -> Result<u64, String> {
        if value < self.min || value > self.max {
            return Err(self.rejected(
                raw,
                &format!("must be between {} and {}", self.min, self.max),
            ));
        }
        Ok(value)
    }

    fn rejected(&self, raw: &str, why: &str) -> String {
        format!(
            "{}: {why}, got \"{raw}\" (unset means the default, {})",
            self.name, self.default
        )
    }
}

impl Knob<f64> {
    /// The configured value, or the default when unset.
    ///
    /// # Errors
    /// When the value does not parse as a finite float, or falls outside `[min, max]`.
    fn read(&self) -> Result<f64, String> {
        let Some(raw) = env_value(self.name) else {
            return Ok(self.default);
        };
        let value: f64 = raw
            .trim()
            .parse()
            .map_err(|_| self.rejected(&raw, "expected a number"))?;
        // NaN fails every comparison, so the range check below would ADMIT it; reject first.
        if !value.is_finite() {
            return Err(self.rejected(&raw, "expected a finite number"));
        }
        if value < self.min || value > self.max {
            return Err(self.rejected(
                &raw,
                &format!("must be between {} and {}", self.min, self.max),
            ));
        }
        Ok(value)
    }

    fn rejected(&self, raw: &str, why: &str) -> String {
        format!(
            "{}: {why}, got \"{raw}\" (unset means the default, {})",
            self.name, self.default
        )
    }
}

/// An on/off knob. Accepts the usual spellings of a boolean and rejects anything else.
struct Switch {
    name: &'static str,
    default: bool,
}

impl Switch {
    /// The configured value, or the default when unset.
    ///
    /// # Errors
    /// When the value is none of `1`/`0`, `true`/`false`, `yes`/`no`, `on`/`off` (any case).
    fn read(&self) -> Result<bool, String> {
        let Some(raw) = env_value(self.name) else {
            return Ok(self.default);
        };
        match raw.trim().to_ascii_lowercase().as_str() {
            "1" | "true" | "yes" | "on" => Ok(true),
            "0" | "false" | "no" | "off" => Ok(false),
            _ => Err(format!(
                "{}: expected one of 1/0, true/false, yes/no, on/off, got \"{raw}\" (unset means \
                 the default, {})",
                self.name, self.default
            )),
        }
    }
}

/// A `host:port,host:port` list knob. Unset means `None`, not an empty list.
struct PeerList {
    name: &'static str,
}

impl PeerList {
    /// The configured pairs, or `None` when unset.
    ///
    /// # Errors
    /// When any entry is not a non-empty host and a port in `1..=65535`.
    fn read(&self) -> Result<Option<Vec<(String, u16)>>, String> {
        env_value(self.name)
            .map(|raw| {
                parse_peer_list(&raw).map_err(|why| format!("{}: {why}, got \"{raw}\"", self.name))
            })
            .transpose()
    }
}

/// Parses `host:port,host:port`. Blank entries are skipped; every other entry must be a
/// non-empty host and a port in `1..=65535`. The port is the last `:`-separated field, so an
/// IPv6 literal keeps its brackets.
fn parse_peer_list(list: &str) -> Result<Vec<(String, u16)>, String> {
    list.split(',')
        .map(str::trim)
        .filter(|entry| !entry.is_empty())
        .map(|entry| {
            let Some((host, port)) = entry.rsplit_once(':') else {
                return Err(format!("entry \"{entry}\" has no ':port'"));
            };
            match port.parse::<u16>() {
                Ok(port) if port != 0 && !host.is_empty() => Ok((host.to_string(), port)),
                _ => Err(format!(
                    "entry \"{entry}\" is not host:port with a port between 1 and 65535"
                )),
            }
        })
        .collect()
}

/// The variable's value, or `None` when it is unset OR set to the empty string.
///
/// Empty counts as unset on purpose: `export FOO=${FOO:-}` and an unset variable are the same
/// operator intent, and rejecting `""` would fail bring-up on a launcher typo that means
/// "leave it alone".
fn env_value(name: &str) -> Option<String> {
    match std::env::var(name) {
        Ok(value) if !value.trim().is_empty() => Some(value),
        _ => None,
    }
}

/// Which same-node senders are fused into their receiver's plan instead of running and parking
/// their rows (`compute_node_service.rs`, `try_defer_sender`).
///
/// Not a [`Knob`]: the value is a word, not a number in a range, so it has its own reader with
/// the same three rules (reject, log, unset means the default).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub(crate) enum FusionMode {
    /// Every sender runs and parks (the behaviour before fusion existed).
    Off = 0,
    /// Leaf senders (no exchange input) with a `HASH_PARTITIONED` single local destination whose
    /// receiving exchange is plain and not under an aggregation. The shipped default: the
    /// shuffle shape that parks a fact table whole at 1 CN, and nothing else.
    Leaf = 1,
    /// Every single-local-destination leaf, any partition type. Removes the broadcast parking of
    /// dimension tables at the price of estimated instead of exact cardinalities for them.
    LeafAny = 2,
}

/// Environment name of the fusion mode.
const FUSION_MODE_NAME: &str = "SIRIUS_CN_FRAGMENT_FUSION";

impl FusionMode {
    /// The mode when the variable is unset.
    pub(crate) const DEFAULT: Self = Self::Leaf;

    /// The accepted spellings, for the rejection message.
    const ACCEPTED: &'static str = "off, leaf, leaf-any";

    /// The configured mode, or the default when unset. Trimmed and case-insensitive; `off`, `0`
    /// and `false` all turn fusion off.
    ///
    /// # Errors
    /// Any other value, naming the variable, the value and the accepted set.
    fn read() -> Result<Self, String> {
        let Some(raw) = env_value(FUSION_MODE_NAME) else {
            return Ok(Self::DEFAULT);
        };
        match raw.trim().to_ascii_lowercase().as_str() {
            "off" | "0" | "false" => Ok(Self::Off),
            "leaf" => Ok(Self::Leaf),
            "leaf-any" => Ok(Self::LeafAny),
            _ => Err(format!(
                "{FUSION_MODE_NAME}: expected one of {}, got \"{raw}\" (unset means the default, \
                 leaf)",
                Self::ACCEPTED
            )),
        }
    }

    /// The mode as the byte an `AtomicU8` holds; [`from_code`](Self::from_code) inverts it.
    pub(crate) const fn code(self) -> u8 {
        self as u8
    }

    /// Inverse of [`code`](Self::code); `None` for a byte no mode produces.
    pub(crate) const fn from_code(code: u8) -> Option<Self> {
        match code {
            0 => Some(Self::Off),
            1 => Some(Self::Leaf),
            2 => Some(Self::LeafAny),
            _ => None,
        }
    }
}

/// The resolved transport tunables. Clone so call sites can hold one cheaply: the peer list is
/// the one heap field, and it is `None` in every configuration but an explicit
/// `SIRIUS_CN_NIXL_WARMUP_PEERS`.
///
/// `main` calls [`resolve`](Self::resolve) at bring-up; the transport, the PRPC client and the
/// session warmup read the fields through [`get`](Self::get). Those consumers are part of the
/// library crate's public API, so the fields are `pub` rather than `pub(crate)`.
#[derive(Clone, Debug, PartialEq)]
pub struct Tunables {
    /// Whether the owned, early-ingress exchange protocol is enabled for this process.
    pub optimized_exchange: bool,
    /// Maximum optimized in-flight frame count per peer.
    pub transfer_window: usize,
    /// Maximum optimized control peers (one bounded worker per peer).
    pub transfer_peers: usize,
    /// See [`RPC_TIMEOUT_SECS`].
    pub rpc_timeout: Duration,
    /// See [`XFER_TIMEOUT_SECS`].
    pub xfer_timeout: Duration,
    /// See [`CANARY_BYTES`].
    pub canary_bytes: u64,
    /// See [`CANARY_FLOOR_GBPS`]; `0.0` means the check is disabled.
    pub canary_floor_gbps: f64,
    /// See [`WARMUP_TIMEOUT_SECS`].
    pub warmup_timeout: Duration,
    /// See [`WARMUP_EXPECT_PEERS`]; `None` when the sentinel `0` leaves early exit off.
    pub warmup_expect_peers: Option<usize>,
    /// See [`WARMUP`]; `false` leaves every peer session lazy.
    pub warmup: bool,
    /// See [`WARMUP_PEERS`]; `None` means discover the peers from the FE.
    pub warmup_peers: Option<Vec<(String, u16)>>,
    /// See [`FusionMode`].
    pub(crate) fusion_mode: FusionMode,
}

impl Tunables {
    /// Exactly the values these knobs had as hardcoded constants (and, for the fusion mode, the
    /// shipped default).
    const DEFAULTS: Self = Self {
        optimized_exchange: false,
        transfer_window: TRANSFER_WINDOW.default as usize,
        transfer_peers: TRANSFER_PEERS.default as usize,
        rpc_timeout: Duration::from_secs(RPC_TIMEOUT_SECS.default),
        xfer_timeout: Duration::from_secs(XFER_TIMEOUT_SECS.default),
        canary_bytes: CANARY_BYTES.default,
        canary_floor_gbps: CANARY_FLOOR_GBPS.default,
        warmup_timeout: Duration::from_secs(WARMUP_TIMEOUT_SECS.default),
        warmup_expect_peers: None,
        warmup: WARMUP.default,
        warmup_peers: None,
        fusion_mode: FusionMode::DEFAULT,
    };

    /// Reads and validates every knob without touching the global.
    ///
    /// # Errors
    /// The first knob that rejects its value, naming the variable, the value, and the range.
    /// One knob per error rather than a collected list: the message is a startup failure an
    /// operator fixes one line at a time.
    fn from_env() -> Result<Self, String> {
        Ok(Self {
            optimized_exchange: OPTIMIZED_EXCHANGE.read()? == 1,
            transfer_window: TRANSFER_WINDOW.read()? as usize,
            transfer_peers: TRANSFER_PEERS.read()? as usize,
            rpc_timeout: Duration::from_secs(RPC_TIMEOUT_SECS.read()?),
            xfer_timeout: Duration::from_secs(XFER_TIMEOUT_SECS.read()?),
            canary_bytes: CANARY_BYTES.read()?,
            canary_floor_gbps: CANARY_FLOOR_GBPS.read()?,
            warmup_timeout: Duration::from_secs(WARMUP_TIMEOUT_SECS.read()?),
            // The sentinel keeps the "engine decides" path and the explicit override in one
            // knob, so both are reachable from the same variable.
            warmup_expect_peers: match WARMUP_EXPECT_PEERS.read()? {
                0 => None,
                peers => Some(peers as usize),
            },
            warmup: WARMUP.read()?,
            warmup_peers: WARMUP_PEERS.read()?,
            fusion_mode: FusionMode::read()?,
        })
    }

    /// Resolves every knob, publishes the result for [`get`](Self::get), and logs what the CN
    /// actually got. Call once, at bring-up, before the transport starts.
    ///
    /// A second call is a no-op that returns the already-published values — the first caller
    /// wins, so a test that resolves early cannot be overwritten by a later one.
    ///
    /// # Errors
    /// Propagates the first rejected knob, so a typo'd tunable fails CN startup instead of
    /// silently reverting to a default mid-sweep.
    pub fn resolve() -> Result<Self, String> {
        if let Some(already) = RESOLVED.get() {
            return Ok(already.clone());
        }
        let tunables = Self::from_env()?;
        // A racing resolve may have won; either way the published value is the one in force.
        let published = RESOLVED.get_or_init(|| tunables).clone();
        if published.canary_floor_gbps == 0.0 {
            tracing::warn!(
                knob = CANARY_FLOOR_GBPS.name,
                "the nixl bandwidth canary floor is disabled: a link that has silently fallen \
                 back to staged host copies (~220x slower, still correct bytes) will now be \
                 ACCEPTED instead of refused"
            );
        }
        tracing::info!(
            optimized_exchange = published.optimized_exchange,
            transfer_window = published.transfer_window,
            transfer_peers = published.transfer_peers,
            rpc_timeout_secs = published.rpc_timeout.as_secs(),
            xfer_timeout_secs = published.xfer_timeout.as_secs(),
            canary_bytes = published.canary_bytes,
            canary_floor_gbps = published.canary_floor_gbps,
            warmup_timeout_secs = published.warmup_timeout.as_secs(),
            warmup_expect_peers = published.warmup_expect_peers,
            warmup = published.warmup,
            warmup_peers = ?published.warmup_peers,
            fusion_mode = ?published.fusion_mode,
            "resolved CN transport tunables"
        );
        Ok(published)
    }

    /// The tunables in force. Before [`resolve`](Self::resolve) runs this is
    /// [`DEFAULTS`](Self::DEFAULTS) — which is correct for unit tests, and unreachable in
    /// production because `main` resolves before it binds a port or starts the engine.
    pub fn get() -> Self {
        RESOLVED.get().cloned().unwrap_or(Self::DEFAULTS)
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// Serializes the tests that mutate the process environment. `cargo test` runs a module's
    /// tests on multiple threads, and `set_var` is process-wide -- so this is the test binary's
    /// only such lock, and every module that writes the environment goes through [`with_env`].
    static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    /// Sets `name` to `value`, or removes it for `None`.
    ///
    /// # Safety
    /// The caller must hold [`ENV_LOCK`]: `set_var`/`remove_var` are process-wide.
    unsafe fn assign(name: &str, value: Option<&str>) {
        unsafe {
            match value {
                Some(value) => std::env::set_var(name, value),
                None => std::env::remove_var(name),
            }
        }
    }

    /// Runs `body` with every `(name, value)` applied — `None` meaning "unset" — and restores
    /// the previous environment after.
    ///
    /// Takes the whole set at once rather than one variable per call: [`ENV_LOCK`] is a plain
    /// non-reentrant `Mutex`, so a nested `with_env` would deadlock against itself.
    pub(crate) fn with_env<T>(vars: &[(&str, Option<&str>)], body: impl FnOnce() -> T) -> T {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|err| err.into_inner());
        let restore: Vec<_> = vars
            .iter()
            .map(|(name, value)| {
                let previous = std::env::var(name).ok();
                // SAFETY: ENV_LOCK is held, so no other test thread touches the environment.
                unsafe { assign(name, *value) };
                (*name, previous)
            })
            .collect();
        let outcome = body();
        for (name, previous) in restore {
            // SAFETY: still under ENV_LOCK.
            unsafe { assign(name, previous.as_deref()) };
        }
        outcome
    }

    #[test]
    fn an_empty_environment_reproduces_the_previously_hardcoded_constants() {
        let resolved = with_env(
            &[
                (OPTIMIZED_EXCHANGE.name, None),
                (TRANSFER_WINDOW.name, None),
                (TRANSFER_PEERS.name, None),
                (RPC_TIMEOUT_SECS.name, None),
                (XFER_TIMEOUT_SECS.name, None),
                (CANARY_BYTES.name, None),
                (CANARY_FLOOR_GBPS.name, None),
                (WARMUP_TIMEOUT_SECS.name, None),
                (WARMUP_EXPECT_PEERS.name, None),
                (WARMUP.name, None),
                (WARMUP_PEERS.name, None),
                (FUSION_MODE_NAME, None),
            ],
            Tunables::from_env,
        )
        .expect("an empty environment resolves");
        assert_eq!(resolved, Tunables::DEFAULTS);
        // The constants these replaced, spelled out so a default change has to touch this test.
        assert_eq!(resolved.rpc_timeout, Duration::from_secs(60));
        assert_eq!(resolved.xfer_timeout, Duration::from_secs(30));
        assert_eq!(resolved.canary_bytes, 16 << 20);
        assert_eq!(resolved.canary_floor_gbps, 2.0);
        assert_eq!(resolved.warmup_timeout, Duration::from_secs(180));
        assert_eq!(resolved.warmup_expect_peers, None);
        assert!(resolved.warmup);
        assert_eq!(resolved.warmup_peers, None);
        assert_eq!(resolved.fusion_mode, FusionMode::Leaf);
    }

    /// The fusion knob: unset is `leaf`; the off spellings, case and whitespace are tolerated;
    /// anything else fails the whole resolution naming the variable, the value and the accepted
    /// set (so `all`, a future value for middle fragments, is a bring-up error today, not a silent `leaf`).
    #[test]
    fn fusion_mode_parses_off_leaf_leaf_any_and_rejects_others() {
        assert_eq!(
            with_env(&[(FUSION_MODE_NAME, None)], FusionMode::read),
            Ok(FusionMode::Leaf)
        );
        for off in ["off", "0", "false", " OFF "] {
            assert_eq!(
                with_env(&[(FUSION_MODE_NAME, Some(off))], FusionMode::read),
                Ok(FusionMode::Off),
                "{off:?}"
            );
        }
        assert_eq!(
            with_env(&[(FUSION_MODE_NAME, Some("LEAF"))], FusionMode::read),
            Ok(FusionMode::Leaf)
        );
        assert_eq!(
            with_env(&[(FUSION_MODE_NAME, Some("leaf-any"))], FusionMode::read),
            Ok(FusionMode::LeafAny)
        );
        for bad in ["all", "on"] {
            let error = with_env(&[(FUSION_MODE_NAME, Some(bad))], FusionMode::read)
                .expect_err("not an accepted mode");
            assert!(
                error.contains(FUSION_MODE_NAME)
                    && error.contains(bad)
                    && error.contains("off, leaf, leaf-any"),
                "{error}"
            );
        }

        let resolved = with_env(&[(FUSION_MODE_NAME, Some("leaf-any"))], Tunables::from_env)
            .expect("leaf-any resolves");
        assert_eq!(resolved.fusion_mode, FusionMode::LeafAny);
        let error = with_env(&[(FUSION_MODE_NAME, Some("all"))], Tunables::from_env)
            .expect_err("a bad mode fails the whole resolution");
        assert!(error.contains(FUSION_MODE_NAME), "{error}");
    }

    /// The byte an `AtomicU8` holds round-trips; a byte no mode produces decodes to nothing.
    #[test]
    fn fusion_mode_codes_round_trip() {
        for mode in [FusionMode::Off, FusionMode::Leaf, FusionMode::LeafAny] {
            assert_eq!(FusionMode::from_code(mode.code()), Some(mode));
        }
        assert_eq!(FusionMode::from_code(3), None);
    }

    #[test]
    fn a_configured_value_is_taken() {
        let value = with_env(&[(XFER_TIMEOUT_SECS.name, Some("90"))], || {
            XFER_TIMEOUT_SECS.read()
        });
        assert_eq!(value, Ok(90));
    }

    #[test]
    fn surrounding_whitespace_is_tolerated() {
        let value = with_env(&[(XFER_TIMEOUT_SECS.name, Some(" 90 "))], || {
            XFER_TIMEOUT_SECS.read()
        });
        assert_eq!(value, Ok(90));
    }

    #[test]
    fn an_empty_value_reads_as_unset() {
        let value = with_env(&[(XFER_TIMEOUT_SECS.name, Some(""))], || {
            XFER_TIMEOUT_SECS.read()
        });
        assert_eq!(value, Ok(XFER_TIMEOUT_SECS.default));
    }

    /// The defect this module exists to fix: the old `.parse().ok()` turned a typo into the
    /// default with no diagnostic, so the knob appeared to have no effect.
    #[test]
    fn an_unparsable_value_is_rejected_rather_than_silently_defaulted() {
        let error = with_env(&[(WARMUP_TIMEOUT_SECS.name, Some("6O"))], || {
            WARMUP_TIMEOUT_SECS.read()
        })
        .expect_err("a letter O is not a digit");
        assert!(error.contains(WARMUP_TIMEOUT_SECS.name), "{error}");
        assert!(error.contains("6O"), "{error}");
    }

    #[test]
    fn an_out_of_range_value_is_rejected_rather_than_clamped() {
        let error = with_env(&[(XFER_TIMEOUT_SECS.name, Some("0"))], || {
            XFER_TIMEOUT_SECS.read()
        })
        .expect_err("zero is below the minimum");
        assert!(error.contains("between 1 and 3600"), "{error}");

        let error = with_env(&[(CANARY_BYTES.name, Some("1024"))], || CANARY_BYTES.read())
            .expect_err("1 KiB is below the 1 MiB minimum");
        assert!(error.contains(CANARY_BYTES.name), "{error}");
    }

    #[test]
    fn a_float_knob_takes_a_value_and_rejects_a_non_finite_one() {
        let value = with_env(&[(CANARY_FLOOR_GBPS.name, Some("50.5"))], || {
            CANARY_FLOOR_GBPS.read()
        });
        assert_eq!(value, Ok(50.5));

        // NaN compares false against both bounds, so a naive range check would admit it.
        let error = with_env(&[(CANARY_FLOOR_GBPS.name, Some("NaN"))], || {
            CANARY_FLOOR_GBPS.read()
        })
        .expect_err("NaN is not a usable floor");
        assert!(error.contains("finite"), "{error}");

        let error = with_env(&[(CANARY_FLOOR_GBPS.name, Some("-1"))], || {
            CANARY_FLOOR_GBPS.read()
        })
        .expect_err("a negative floor is meaningless");
        assert!(error.contains("between 0 and 10000"), "{error}");
    }

    /// `0` is the documented escape hatch, not an out-of-range value.
    #[test]
    fn a_zero_canary_floor_is_accepted_as_the_disable_sentinel() {
        let value = with_env(&[(CANARY_FLOOR_GBPS.name, Some("0"))], || {
            CANARY_FLOOR_GBPS.read()
        });
        assert_eq!(value, Ok(0.0));
    }

    /// The expect-peers sentinel: `0` means "no early exit", not "exit after zero peers".
    #[test]
    fn the_expect_peers_sentinel_maps_to_none() {
        let zero = with_env(&[(WARMUP_EXPECT_PEERS.name, Some("0"))], Tunables::from_env)
            .expect("zero resolves");
        assert_eq!(zero.warmup_expect_peers, None);

        let three = with_env(&[(WARMUP_EXPECT_PEERS.name, Some("3"))], Tunables::from_env)
            .expect("3 resolves");
        assert_eq!(three.warmup_expect_peers, Some(3));
    }

    /// The warmup kill switch takes the spellings an operator would type and, per rule 1,
    /// rejects anything else rather than reading it as "on".
    #[test]
    fn the_warmup_switch_accepts_the_usual_spellings_and_rejects_the_rest() {
        for (raw, expected) in [
            ("0", false),
            ("false", false),
            ("No", false),
            ("OFF", false),
            ("1", true),
            ("true", true),
            ("yes", true),
            (" On ", true),
        ] {
            let value = with_env(&[(WARMUP.name, Some(raw))], || WARMUP.read());
            assert_eq!(value, Ok(expected), "{raw:?}");
        }
        let error = with_env(&[(WARMUP.name, Some("maybe"))], || WARMUP.read())
            .expect_err("'maybe' is not a boolean");
        assert!(error.contains(WARMUP.name), "{error}");
        assert!(error.contains("maybe"), "{error}");
    }

    /// The peer list accepts what an operator would write; junk fails bring-up (rule 1) instead
    /// of being dropped, which used to leave a typo'd peer cold with only a warning to show.
    #[test]
    fn peer_list_parses_host_port_pairs_and_rejects_junk() {
        assert_eq!(
            parse_peer_list("127.0.0.1:9102, 127.0.0.1:9112 ,,"),
            Ok(vec![
                ("127.0.0.1".to_string(), 9102),
                ("127.0.0.1".to_string(), 9112),
            ])
        );
        for junk in [
            "bad",
            "127.0.0.1:0",
            "host:70000",
            ":9102",
            "127.0.0.1:9102,bad",
        ] {
            assert!(parse_peer_list(junk).is_err(), "{junk:?} must be rejected");
        }

        let error = with_env(&[(WARMUP_PEERS.name, Some("127.0.0.1:9102,bad"))], || {
            WARMUP_PEERS.read()
        })
        .expect_err("one junk entry rejects the whole list");
        assert!(error.contains(WARMUP_PEERS.name), "{error}");
        assert!(error.contains("bad"), "{error}");

        let unset = with_env(&[(WARMUP_PEERS.name, None)], || WARMUP_PEERS.read());
        assert_eq!(unset, Ok(None));
    }

    /// An IPv6 literal keeps its brackets: the port is the last `:`-separated field.
    #[test]
    fn peer_list_splits_on_the_last_colon() {
        assert_eq!(
            parse_peer_list("[::1]:9102"),
            Ok(vec![("[::1]".to_string(), 9102)])
        );
    }

    /// A bad knob has to fail the whole resolution, not just its own field.
    #[test]
    fn one_rejected_knob_fails_the_whole_resolution() {
        let error = with_env(&[(RPC_TIMEOUT_SECS.name, Some("-5"))], Tunables::from_env)
            .expect_err("a negative timeout is not an unsigned integer");
        assert!(error.contains(RPC_TIMEOUT_SECS.name), "{error}");
    }
}
