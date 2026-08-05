"""Discrete-event engine: resource-constrained replay of one query's task graph.

Resources modeled per GPU device (mirroring the Super Sirius executor, see
docs/super-sirius/pipeline-execution.md):

- executor thread slots (bounded_thread_pool, count from the trace),
- an admission queue with *head-of-line* blocking on memory reservation
  (the manager loop pops one task and cannot pop the next until the head's
  reservation is granted). Queue discipline (``queue_order``):
  * ``"traced"`` (default): among released tasks, dispatch in traced
    queue-entry order. The trace order encodes task-creator/scheduler
    decisions (hint-chain recursion, byte-threshold batching) that v0 does
    not model; using sim arrival order instead lets microsecond-level drift
    flip the order of a tiny task vs. a large scan burst and cascade into
    hundreds of ms of spurious reordering (observed on tpch_q09_iter3:
    +24% error with arrival order, +0.1% with traced order). Admission
    *timing* stays fully emergent — only relative order is anchored.
  * ``"arrival"``: strict FIFO by simulated enqueue time (used by unit
    tests and available for experiments).
- a GPU memory pool: occupancy = active reservations + published-but-unconsumed
  GPU-resident batches; a task is admitted only when
  occupancy + min(reservation, capacity) <= capacity,
- fluid transfer channels serving Preparing materializations: each transfer
  demands its traced achieved rate x c2c knob; when aggregate demand exceeds
  channel capacity (traced peak aggregate x c2c knob) all active transfers are
  throttled proportionally.
- a fluid GPU device-compute resource (gap G4b, docs/simulator-design.md):
  when ``device_capacity`` is configured and tasks carry ``dev_work_ns``
  (kernel work, set by the physics retime layer), the compute phase of a task
  becomes a fluid job -- work ``W`` served at natural demand rate ``W / D``
  (``D`` = the phase's natural, knob-scaled duration) against a shared
  per-device capacity in kernel-ns per wall-ns. When concurrent tasks'
  aggregate kernel demand exceeds capacity all phases stretch proportionally,
  so queue-wait on a saturated device EMERGES from demand vs capacity instead
  of being replayed from the trace (the measured L-lane failure mode:
  span-level "host" time that is really device wait).

Nothing is hand-coded per knob beyond event durations/rates: back-pressure,
queueing and saturation emerge from admission and capacity.

Spill / downgrade layer (gap G5, see docs/spill-model.md). Three modes,
resolved from ``spill_mode="auto"``:

- ``"off"`` (unpressured trace, gpu_mem_capacity == 1): v0 semantics,
  byte-identical — a memory-blocked head waits; a complete stall force-admits.
- ``"replay"`` (the trace itself contains Downgrading / OOM-rescheduled
  tasks): pure bookkeeping. When the head is memory-blocked, idle resident
  batches are evicted at ZERO time cost — every real cost is already inside
  the traced spans (the downgrade wait sits in grant_ns, the re-upgrade H2D
  sits in the retry tasks' Preparing spans, and the recompute waste IS the
  traced failed task attempts, replayed like any other task).
- ``"model"`` (capacity knob moved on an unpressured trace): the engine
  mirrors the real admission/downgrade policy: it demotes idle resident
  batches to the HOST pool (capacity-bounded) at a calibrated rate while the
  manager loop stalls; consumers of demoted batches pay the re-upgrade
  transfer in their prep; and a head whose reservation still cannot be
  granted does what the real engine does — it is dispatched anyway, OOMs,
  and is rescheduled: modeled as a "spin" that burns a thread slot for one
  calibrated OOM-reschedule cycle and re-queues at the tail (capped at the
  engine's retry budget, then force-admitted).

Progress guarantee: if the simulation stalls completely (no pending event can
free memory) a memory-blocked queue head is force-admitted — occurrences are
counted and reported.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Set, Tuple

from .knobs import Knobs
from .model import BatchSpec, QueryGraph, TaskSpec

ChannelKey = Tuple[str, str, int]  # (origin_tier, target_tier, device)

_BYTES_EPS = 0.5
_TIME_EPS = 1e-6
_WORK_EPS = 0.5  # ns of device kernel work (float-accumulation slack)

SPILL_MODES = ("auto", "off", "replay", "model")


@dataclass
class SpillParams:
    """Costs of the downgrade / OOM-reschedule mechanism (docs/spill-model.md).

    Rates are bytes/ns (== GB/s). ``downgrade_rate``/``upgrade_rate`` and
    ``oom_cycle_ns`` are MEASURED from the E2 pressure captures (D2H convert
    ~30 GB/s from data_batch InTransit; re-materialization ~29 GB/s effective
    from retry Preparing spans; median failed attempt = admit ->
    partial-progress compute -> OOM unwind -> requeue ~= 47-50 ms on both
    E2-mid q21 and E2-lo q9). ``min_progress`` — the floor fraction of a
    task's work that one OOM-rescheduled attempt banks via its
    reschedule_intermediate outputs when almost no memory is grantable — is
    the CALIBRATED parameter (fit on one pressured point, validated on the
    other held out). Each attempt banks
    ``f = clamp(available / (need + rematerialize), min_progress, 1)``,
    so attempts-per-task and hence the thrash cost scale with pressure depth
    emergently.
    """

    downgrade_rate: float = 30.0  # GB/s: demoting idle batches GPU -> HOST
    upgrade_rate: float = 29.0  # GB/s: re-materializing demoted inputs
    downgrade_base_ns: float = 250_000.0  # fixed manager cost per downgrade
    oom_cycle_ns: float = 50_000_000.0  # one admit->OOM->reschedule cycle
    # Off-thread delay between an OOM unwind and the retry re-entering the
    # queue (the executor's reschedule backoff, ~50 ms/retry) — paces retries
    # without burning a thread slot.
    retry_backoff_ns: float = 50_000_000.0
    min_progress: float = 0.02  # progress floor per attempt (measured: ~50
    # attempts per thrashed logical task on both E2 pressure captures)
    # Fraction of a rescheduled attempt's intermediate output that
    # materializes on the HOST tier (born downgraded) instead of occupying
    # the GPU pool. CALIBRATED — this is the dial that sets how fast the
    # pool drains under thrash and hence the depth of the spill cliff.
    # Default 0.47 = joint fit of the two E2 pressure points (q21@0.25x
    # +38.8%, q9@0.15x -20.4%); see docs/spill-model.md for the held-out
    # protocol results and the bistability caveat.
    spin_output_host_fraction: float = 0.47
    # A downgrade sweep frees down to this fraction of the pool (the engine's
    # downgrade_stop_fraction, memory-management.md), not just enough for the
    # blocked head — this hysteresis opens admission windows.
    downgrade_stop_fraction: float = 0.7
    max_oom_retries: int = 100  # engine MAX_RETRIES, then force-admit


@dataclass
class TaskTimes:
    release: float = -1.0
    enqueue: float = -1.0
    admit: float = -1.0
    prep_start: float = -1.0
    prep_end: float = -1.0
    finish: float = -1.0
    mem_wait_ns: float = 0.0  # time this task spent head-of-queue blocked on memory
    forced_admission: bool = False
    oom_spins: int = 0  # modeled OOM-reschedule cycles before admission
    spin_ns: float = 0.0  # thread time burnt in those cycles

    @property
    def queue_wait_ns(self) -> float:
        return max(0.0, self.admit - self.enqueue)


@dataclass
class ChannelStats:
    capacity: Optional[float]  # bytes/ns after knob, None = unlimited
    moved_bytes: float = 0.0
    busy_ns: float = 0.0
    throttled_ns: float = 0.0
    peak_active: int = 0

    def utilization_rate(self) -> float:
        return self.moved_bytes / self.busy_ns if self.busy_ns > 0 else 0.0


@dataclass
class SimResult:
    wall_ns: float
    task_times: Dict[int, TaskTimes]
    block_totals: Dict[int, Dict[str, float]]  # device -> {"threads": ns, "memory": ns}
    channel_stats: Dict[ChannelKey, ChannelStats]
    thread_timeline: Dict[int, List[Tuple[float, int]]]  # device -> (t, busy)
    pool_timeline: Dict[
        int, List[Tuple[float, float, float]]
    ]  # (t, reserved, resident)
    peak_pool: Dict[int, float]
    pool_capacity: Dict[int, float]
    n_threads: Dict[int, int]
    thread_busy_ns: Dict[int, float]
    forced_admissions: int = 0
    dep_cycle_breaks: int = 0
    orphan_gpu_batches: int = 0
    # fluid GPU device-compute resource (gap G4b); key = device id.
    # ChannelStats reused with work-ns in place of bytes: capacity is
    # kernel-ns per wall-ns, moved_bytes is served kernel work (ns).
    device_stats: Dict[int, ChannelStats] = field(default_factory=dict)
    # spill / downgrade layer (docs/spill-model.md)
    spill_mode: str = "off"
    downgrade_events: int = 0
    downgraded_bytes: float = 0.0
    reupgraded_bytes: float = 0.0
    oom_retries: int = 0
    spin_ns: float = 0.0
    retry_cap_forced: int = 0

    def binding_constraint(self, device: int = 0) -> str:
        bt = self.block_totals.get(device, {})
        threads = bt.get("threads", 0.0)
        memory = bt.get("memory", 0.0) + bt.get("spill", 0.0)
        chan = sum(c.throttled_ns for c in self.channel_stats.values())
        dev = sum(d.throttled_ns for d in self.device_stats.values())
        best = max(threads, memory, chan, dev)
        if best <= 0:
            return "dependencies"
        if best == dev:
            return "gpu_device"
        if best == memory:
            return "gpu_memory"
        if best == chan:
            return "transfer_channel"
        return "executor_threads"


class _FluidChannel:
    def __init__(self, key: ChannelKey, capacity: Optional[float]) -> None:
        self.key = key
        self.capacity = capacity  # bytes/ns, None = unlimited
        self.active: Dict[int, List[float]] = {}  # tid -> [remaining, demand]
        self.last_t = 0.0
        self.token = 0
        self.stats = ChannelStats(capacity=capacity)

    def _factor(self) -> float:
        if self.capacity is None or not self.active:
            return 1.0
        total = sum(v[1] for v in self.active.values())
        if total <= self.capacity:
            return 1.0
        return self.capacity / total

    def _advance(self, now: float) -> None:
        dt = now - self.last_t
        if dt > 0 and self.active:
            f = self._factor()
            for v in self.active.values():
                v[0] -= v[1] * f * dt
                self.stats.moved_bytes += v[1] * f * dt
            self.stats.busy_ns += dt
            if f < 1.0 - 1e-9:
                self.stats.throttled_ns += dt
        self.last_t = max(self.last_t, now)

    def _next_finish(self, now: float) -> Optional[float]:
        if not self.active:
            return None
        f = self._factor()
        return min(now + max(0.0, v[0]) / (v[1] * f) for v in self.active.values())


class _FluidDevice(_FluidChannel):
    """Shared GPU device-compute resource (gap G4b).

    Same fluid mechanics as a transfer channel, different units: a job is a
    task compute phase with ``work`` = knob-scaled kernel-ns and ``demand`` =
    work / natural-phase-duration (<= 1 by construction: the kernel share of
    a span cannot exceed the span). Capacity is kernel-ns servable per
    wall-ns (~1.0 for one GPU whose kernels serialize; > 1 only when the
    baseline capture shows multi-stream overlap). When Sum(demand) exceeds
    capacity every active phase stretches proportionally -- emergent
    queue-wait; when it does not, phases run at their natural durations,
    which collapses to the section-7 physics behavior on uncontended lanes.
    """

    def __init__(self, device: int, capacity: float) -> None:
        super().__init__(("GPU-DEVICE", "compute", device), capacity)
        self.device = device


class Engine:
    def __init__(
        self,
        graph: QueryGraph,
        knobs: Knobs,
        n_threads: Dict[int, int],
        pool_capacity: Dict[int, int],
        channel_capacity: Dict[ChannelKey, float],
        queue_order: str = "traced",
        spill_mode: str = "auto",
        spill: Optional[SpillParams] = None,
        host_capacity: Optional[float] = None,
        device_capacity: Optional[Dict[int, float]] = None,
    ) -> None:
        self.graph = graph
        self.knobs = knobs
        if queue_order not in ("traced", "arrival"):
            raise ValueError(
                f"queue_order must be 'traced' or 'arrival', got {queue_order!r}"
            )
        self.queue_order = queue_order
        self.tasks: Dict[int, TaskSpec] = graph.tasks
        self.batches: Dict[int, BatchSpec] = graph.batches

        if spill_mode not in SPILL_MODES:
            raise ValueError(
                f"spill_mode must be one of {SPILL_MODES}, got {spill_mode!r}"
            )
        if spill_mode == "auto":
            if any(t.t_downgrading >= 0 or not t.success for t in self.tasks.values()):
                spill_mode = "replay"
            elif knobs.gpu_mem_capacity != 1.0:
                spill_mode = "model"
            else:
                spill_mode = "off"
        self.spill_mode = spill_mode
        self.spill = spill if spill is not None else SpillParams()

        self._heap: List[Tuple[float, int, Callable, tuple]] = []
        self._seq = 0
        self.now = 0.0

        devices = {t.device for t in self.tasks.values()} or {0}
        self.n_threads = {
            d: max(1, n_threads.get(d, n_threads.get(0, 4))) for d in devices
        }
        self.pool_capacity = {
            d: float(pool_capacity.get(d, pool_capacity.get(0, 1 << 62)))
            * knobs.gpu_mem_capacity
            for d in devices
        }

        # per-device mutable state
        self._fifo: Dict[int, List[Tuple[float, int]]] = {d: [] for d in devices}
        self._slots_free: Dict[int, int] = dict(self.n_threads)
        self._reserved: Dict[int, float] = {d: 0.0 for d in devices}
        self._resident: Dict[int, float] = {d: 0.0 for d in devices}
        self._n_res: Dict[int, int] = {d: 0 for d in devices}
        self._granted: Dict[int, float] = {}  # tid -> granted reservation bytes
        self._block_kind: Dict[int, Optional[str]] = {d: None for d in devices}
        self._block_since: Dict[int, float] = {d: 0.0 for d in devices}
        self._block_head: Dict[int, Optional[int]] = {d: None for d in devices}

        self.block_totals: Dict[int, Dict[str, float]] = {
            d: {"threads": 0.0, "memory": 0.0} for d in devices
        }
        self.thread_timeline: Dict[int, List[Tuple[float, int]]] = {
            d: [(0.0, 0)] for d in devices
        }
        self.pool_timeline: Dict[int, List[Tuple[float, float, float]]] = {
            d: [(0.0, 0.0, 0.0)] for d in devices
        }
        self.peak_pool: Dict[int, float] = {d: 0.0 for d in devices}
        self.thread_busy_ns: Dict[int, float] = {d: 0.0 for d in devices}
        self._busy_since: Dict[int, Tuple[float, int]] = {d: (0.0, 0) for d in devices}

        self.channels: Dict[ChannelKey, _FluidChannel] = {}
        self._channel_capacity_cfg = channel_capacity

        # fluid GPU device-compute resource (gap G4b): only devices with a
        # configured capacity get one; tasks without dev_work_ns bypass it.
        self.devices: Dict[int, _FluidDevice] = {}
        if device_capacity:
            for d, cap in device_capacity.items():
                if d in devices and cap and cap > 0:
                    self.devices[d] = _FluidDevice(d, float(cap))

        self.rec: Dict[int, TaskTimes] = {tid: TaskTimes() for tid in self.tasks}
        self.forced_admissions = 0
        self.dep_cycle_breaks = 0
        self.orphan_gpu_batches = 0

        # spill / downgrade state
        for d in devices:
            self.block_totals[d]["spill"] = 0.0
        self._host_cap = (
            float(host_capacity) * knobs.cpu_mem_capacity
            if host_capacity
            else float("inf")
        )
        self._host_used = 0.0
        # dev -> {bid: bytes} in publish order (LRU eviction candidates)
        self._lru: Dict[int, Dict[int, float]] = {d: {} for d in devices}
        self._evicted: Dict[int, float] = {}  # bid -> bytes now on HOST
        self._pins: Dict[int, int] = {}  # bid -> running consumers
        self._mgr_busy_until: Dict[int, float] = {d: 0.0 for d in devices}
        self._max_prio: Dict[int, float] = {d: 0.0 for d in devices}
        self._banked: Dict[int, float] = {}  # tid -> work fraction banked
        self._prepub: Dict[int, float] = {}  # bid -> bytes published early
        self.downgrade_events = 0
        self.downgraded_bytes = 0.0
        self.reupgraded_bytes = 0.0
        self.oom_retries = 0
        self.spin_ns_total = 0.0
        self.retry_cap_forced = 0

        # dependency bookkeeping
        self._pending_deps: Dict[int, int] = {}
        self._rdeps: Dict[int, Set[int]] = {tid: set() for tid in self.tasks}
        for tid, t in self.tasks.items():
            deps = {d for d in t.deps if d in self.tasks}
            self._pending_deps[tid] = len(deps)
            for d in deps:
                self._rdeps[d].add(tid)

        self._batch_pending: Dict[int, int] = {}
        for bid, b in self.batches.items():
            consumers = {c for c in b.consumer_tids if c in self.tasks}
            self._batch_pending[bid] = len(consumers)
            if b.gpu_resident and (
                b.producer_tid is None or b.producer_tid not in self.tasks
            ):
                # No producing task in the replay -> resident from t=0.
                self.orphan_gpu_batches += 1
                dev = b.device if b.device in devices else next(iter(devices))
                self._resident[dev] += b.nbytes
                self._lru[dev][bid] = float(b.nbytes)

        self._unfinished = set(self.tasks)
        self._released: Set[int] = set()

    # ------------------------------------------------------------------ utils

    def schedule(self, t: float, fn: Callable, *args) -> None:
        self._seq += 1
        heapq.heappush(self._heap, (t, self._seq, fn, args))

    def _channel_for(self, task: TaskSpec) -> _FluidChannel:
        key = (task.prep_origin, task.prep_target, task.device)
        ch = self.channels.get(key)
        if ch is None:
            cap = self._channel_capacity_cfg.get(key)
            cap = cap * self.knobs.c2c_bandwidth if cap is not None else None
            ch = self.channels[key] = _FluidChannel(key, cap)
            ch.last_t = self.now
        return ch

    def _sample_threads(self, dev: int) -> None:
        busy = self.n_threads[dev] - self._slots_free[dev]
        since, prev_busy = self._busy_since[dev]
        self.thread_busy_ns[dev] += prev_busy * (self.now - since)
        self._busy_since[dev] = (self.now, busy)
        tl = self.thread_timeline[dev]
        if tl and tl[-1][0] == self.now:
            tl[-1] = (self.now, busy)
        else:
            tl.append((self.now, busy))

    def _sample_pool(self, dev: int) -> None:
        occ = self._reserved[dev] + self._resident[dev]
        self.peak_pool[dev] = max(self.peak_pool[dev], occ)
        tl = self.pool_timeline[dev]
        entry = (self.now, self._reserved[dev], self._resident[dev])
        if tl and tl[-1][0] == self.now:
            tl[-1] = entry
        else:
            tl.append(entry)

    def _set_block(self, dev: int, kind: Optional[str], head: Optional[int]) -> None:
        prev = self._block_kind[dev]
        if prev == kind and self._block_head[dev] == head:
            return
        if prev is not None:
            dt = self.now - self._block_since[dev]
            self.block_totals[dev][prev] += dt
            if prev == "memory" and self._block_head[dev] is not None:
                self.rec[self._block_head[dev]].mem_wait_ns += dt
        self._block_kind[dev] = kind
        self._block_head[dev] = head
        self._block_since[dev] = self.now

    # ------------------------------------------------------------- lifecycle

    def _release(self, tid: int) -> None:
        if tid in self._released:
            return
        self._released.add(tid)
        rec = self.rec[tid]
        rec.release = self.now
        self.schedule(self.now + self.tasks[tid].pre_queue_ns, self._enqueue, tid)

    def _enqueue(self, tid: int) -> None:
        task = self.tasks[tid]
        self.rec[tid].enqueue = self.now
        if self.queue_order == "traced":
            prio = float(task.t_queued if task.t_queued >= 0 else task.t_created)
        else:
            prio = self.now
        heapq.heappush(self._fifo[task.device], (prio, task.tid))
        if prio > self._max_prio[task.device]:
            self._max_prio[task.device] = prio
        self._pump(task.device)

    def _pump(self, dev: int) -> None:
        fifo = self._fifo[dev]
        while True:
            if not fifo:
                self._set_block(dev, None, None)
                return
            if (
                self.spill_mode == "model"
                and self.now < self._mgr_busy_until[dev] - _TIME_EPS
            ):
                # Manager loop is blocked inside request_downgrade(): no
                # admissions until the downgrade completes.
                self._set_block(dev, "memory", fifo[0][1])
                return
            if self._slots_free[dev] == 0:
                self._set_block(dev, "threads", fifo[0][1])
                return
            tid = fifo[0][1]
            task = self.tasks[tid]
            cap = self.pool_capacity[dev]
            need = min(float(task.reservation_bytes), cap)
            extra = 0.0
            if self.spill_mode == "model":
                # Demoted inputs must be re-materialized on admission; the
                # engine clamps the total ask to what the space can grant.
                extra = min(self._evicted_input_bytes(task), max(0.0, cap - need))
            if (
                self._reserved[dev] + self._resident[dev] + need + extra
                > cap + _BYTES_EPS
            ):
                if self.spill_mode != "off":
                    action = self._spill_head(dev, tid, task, need, extra)
                    if action == "continue":
                        continue
                    if action == "wait":
                        return
                self._set_block(dev, "memory", tid)
                return
            self._admit(dev)

    def _admit(self, dev: int, forced: bool = False) -> None:
        """Pop the queue head and grant it a (possibly clamped) reservation."""
        _, tid = heapq.heappop(self._fifo[dev])
        task = self.tasks[tid]
        need = min(float(task.reservation_bytes), self.pool_capacity[dev])
        if forced:
            self.forced_admissions += 1
            self.rec[tid].forced_admission = True
        self._set_block(dev, None, None)
        self._slots_free[dev] -= 1
        self._n_res[dev] += 1
        self._reserved[dev] += need
        self._granted[tid] = need
        self.rec[tid].admit = self.now
        extra_prep_ns = 0.0
        if self.spill_mode != "off":
            for bid in task.input_batches:
                self._pins[bid] = self._pins.get(bid, 0) + 1
            if self.spill_mode == "model":
                extra_prep_ns = self._reupgrade_inputs(dev, task)
        self._sample_threads(dev)
        self._sample_pool(dev)
        self.schedule(self.now + task.grant_ns + extra_prep_ns, self._prep_start, tid)

    # ------------------------------------------------------- spill mechanics

    def _evicted_input_bytes(self, task: TaskSpec) -> float:
        if not self._evicted:
            return 0.0
        return sum(self._evicted.get(bid, 0.0) for bid in set(task.input_batches))

    def _reupgrade_inputs(self, dev: int, task: TaskSpec) -> float:
        """Bring demoted input batches back to the GPU: bytes rejoin the pool,
        HOST frees, and the task pays the transfer in its prep. Returns the
        extra prep ns."""
        moved = 0.0
        for bid in set(task.input_batches):
            nb = self._evicted.pop(bid, 0.0)
            if nb <= 0.0:
                continue
            self._host_used -= nb
            self._resident[dev] += nb
            self._lru[dev][bid] = self._lru[dev].get(bid, 0.0) + nb
            moved += nb
        if moved <= 0.0:
            return 0.0
        self.reupgraded_bytes += moved
        return moved / self.spill.upgrade_rate

    def _spill_head(
        self, dev: int, tid: int, task: TaskSpec, need: float, extra: float
    ) -> str:
        """Memory-blocked head under an active spill mode. Returns:
        "continue" (head admitted or spinning; keep pumping),
        "wait" (manager stalled on a downgrade; a pump is scheduled),
        "block" (waiting suffices / nothing to evict; v0 blocking)."""
        cap = self.pool_capacity[dev]
        # Deficit that waiting for active reservations can NEVER free: the
        # resident-data overshoot. When it is <= 0, the real make_reservation
        # simply blocks until running tasks release their reservations — the
        # traces confirm no Downgrading is emitted in that regime (e.g. the
        # marginal q9 point: pool 99.6% peaked, 0 Downgrading events).
        deficit = self._resident[dev] + need + extra - cap
        if deficit <= _BYTES_EPS:
            return "block"
        # A real downgrade sweep frees down to downgrade_stop_fraction of the
        # pool (hysteresis), not just enough for the head.
        evict_target = deficit
        if self.spill_mode == "model":
            evict_target = max(
                deficit,
                self._resident[dev]
                + need
                + extra
                - self.spill.downgrade_stop_fraction * cap,
            )
        own_inputs = set(task.input_batches)
        lru = self._lru[dev]
        evicted = 0.0
        for bid in list(lru):
            if evicted >= evict_target - _BYTES_EPS:
                break
            if self._pins.get(bid):
                continue
            if bid in own_inputs:
                continue
            nb = lru[bid]
            if (
                self.spill_mode == "model"
                and self._host_used + nb > self._host_cap + _BYTES_EPS
            ):
                continue  # HOST pool cannot take this batch
            self._host_used += nb
            del lru[bid]
            self._resident[dev] -= nb
            self._evicted[bid] = nb
            evicted += nb

        if evicted > 0.0:
            self.downgrade_events += 1
            self.downgraded_bytes += evicted
            self._sample_pool(dev)
            if self.spill_mode == "replay":
                # Zero-cost bookkeeping: the traced spans already carry every
                # real downgrade cost. Admit if the head now fits.
                if self._reserved[dev] + self._resident[dev] + need <= cap + _BYTES_EPS:
                    self._admit(dev)
                    return "continue"
                return "block"
            stall = self.spill.downgrade_base_ns + evicted / self.spill.downgrade_rate
            self._mgr_busy_until[dev] = max(self._mgr_busy_until[dev], self.now + stall)
            self._set_block(dev, "memory", tid)
            self.schedule(self._mgr_busy_until[dev], self._pump, dev)
            return "wait"

        if self.spill_mode == "replay":
            return "block"

        # Model mode, nothing (more) evictable: the real engine dispatches
        # anyway with whatever reservation it got; the task OOMs at the point
        # its allocations outgrow the grant and is RESCHEDULED — its
        # reschedule_intermediate outputs preserve partial progress, so each
        # attempt banks a fraction of the work (resume-at-operator).
        rec = self.rec[tid]
        if rec.oom_spins >= self.spill.max_oom_retries:
            self.retry_cap_forced += 1
            self._admit(dev, forced=True)
            return "continue"
        heapq.heappop(self._fifo[dev])
        avail = max(0.0, cap - self._reserved[dev] - self._resident[dev])
        denom = max(need + extra, 1.0)
        f = min(1.0, max(self.spill.min_progress, avail / denom))
        banked = self._banked.get(tid, 0.0)
        f = min(f, 1.0 - banked)
        self._banked[tid] = banked + f
        # Partial progress consumes inputs incrementally: the attempt's
        # processed input batches are freed (the real engine's rescheduled
        # tasks keep reschedule_intermediate outputs and release consumed
        # inputs) — this drains residency and re-opens admission windows.
        if f > 0.0:
            touched = False
            for bid in set(task.input_batches):
                nb = lru.get(bid)
                if not nb:
                    continue
                delta = min(nb, f * float(self.batches[bid].nbytes))
                lru[bid] = nb - delta
                self._resident[dev] -= delta
                touched = True
            # ... and publishes its intermediate (reschedule_intermediate)
            # outputs incrementally. Under pressure a calibrated fraction of
            # them is BORN DOWNGRADED — materialized on the HOST tier
            # directly (the E2-mid capture shows 1153 GB of H2D
            # re-materialization against only 23 GB of D2H conversions), so
            # that share does not occupy the GPU pool; consumers pay the
            # re-upgrade transfer instead. The rest stays GPU-resident.
            beta = self.spill.spin_output_host_fraction
            for bid in set(task.output_batches):
                b = self.batches[bid]
                if not b.gpu_resident:
                    continue
                delta = f * float(b.nbytes)
                self._prepub[bid] = self._prepub.get(bid, 0.0) + delta
                if beta > 0.0:
                    self._evicted[bid] = self._evicted.get(bid, 0.0) + beta * delta
                    self._host_used += beta * delta
                if beta < 1.0:
                    lru[bid] = lru.get(bid, 0.0) + (1.0 - beta) * delta
                    self._resident[dev] += (1.0 - beta) * delta
                touched = True
            if touched:
                self._sample_pool(dev)
        compute = sum(
            base / self.knobs.op_scale(name) for (name, _oid, base, _b) in task.ops
        )
        cycle = self.spill.oom_cycle_ns + f * compute
        rec.oom_spins += 1
        rec.spin_ns += cycle
        self.oom_retries += 1
        self.spin_ns_total += cycle
        self.block_totals[dev]["spill"] += cycle
        self._set_block(dev, None, None)
        self._slots_free[dev] -= 1
        self._sample_threads(dev)
        self.schedule(self.now + cycle, self._spin_done, tid)
        return "continue"

    def _spin_done(self, tid: int) -> None:
        task = self.tasks[tid]
        dev = task.device
        if self._banked.get(tid, 0.0) >= 1.0 - 1e-9:
            # Final attempt completed the work: keep the thread slot and
            # transition straight to prep/finish (ops already banked). Grant
            # whatever the pool can give — the big allocations already
            # happened inside the banked attempts.
            cap = self.pool_capacity[dev]
            need = min(float(task.reservation_bytes), cap)
            grant = min(need, max(0.0, cap - self._reserved[dev] - self._resident[dev]))
            self._n_res[dev] += 1
            self._reserved[dev] += grant
            self._granted[tid] = grant
            self.rec[tid].admit = self.now
            extra_prep_ns = 0.0
            for bid in task.input_batches:
                self._pins[bid] = self._pins.get(bid, 0) + 1
            extra_prep_ns = self._reupgrade_inputs(dev, task)
            self._sample_pool(dev)
            self.schedule(
                self.now + task.grant_ns + extra_prep_ns, self._prep_start, tid
            )
            return
        self._slots_free[dev] += 1
        self._sample_threads(dev)
        if self.spill.retry_backoff_ns > 0:
            self.schedule(self.now + self.spill.retry_backoff_ns, self._requeue, tid)
        else:
            self._requeue(tid)
        self._pump(dev)

    def _requeue(self, tid: int) -> None:
        dev = self.tasks[tid].device
        prio = max(self.now, self._max_prio[dev] + 1.0)
        self._max_prio[dev] = prio
        heapq.heappush(self._fifo[dev], (prio, tid))
        self._pump(dev)

    def _prep_start(self, tid: int) -> None:
        task = self.tasks[tid]
        self.rec[tid].prep_start = self.now
        if task.is_transfer_prep and task.prep_ns > 1000 and task.prep_bytes > 0:
            ch = self._channel_for(task)
            ch._advance(self.now)
            demand = task.prep_bytes / task.prep_ns * self.knobs.c2c_bandwidth
            ch.active[tid] = [float(task.prep_bytes), demand]
            ch.stats.peak_active = max(ch.stats.peak_active, len(ch.active))
            self._reschedule_channel(ch)
        else:
            prep_ns = task.prep_ns
            if prep_ns > 0 and not task.is_transfer_prep:
                # Same-tier Preparing (pinned-cache decompress etc.) is GPU
                # work in the v0 conflated sense -- scale it with gpu_speed
                # like Computing spans (roadmap item, validation-results.md
                # section 8.6: v0 missed x3.7 Preparing inflation on the
                # pinned late-mat lane). Identity at knobs=1 is unchanged.
                # The physics path pre-splits Preparing and neutralizes the
                # engine GPU knobs, so nothing is scaled twice there.
                prep_ns = prep_ns / self.knobs.gpu_speed
            self.schedule(self.now + prep_ns, self._prep_done, tid)

    def _reschedule_channel(self, ch: _FluidChannel) -> None:
        ch.token += 1
        nxt = ch._next_finish(self.now)
        if nxt is not None:
            self.schedule(nxt, self._channel_event, ch.key, ch.token)

    def _channel_event(self, key: ChannelKey, token: int) -> None:
        ch = self.channels[key]
        if token != ch.token:
            return
        ch._advance(self.now)
        done = [tid for tid, v in ch.active.items() if v[0] <= _BYTES_EPS]
        for tid in done:
            del ch.active[tid]
        self._reschedule_channel(ch)
        for tid in done:
            self._prep_done(tid)

    def _prep_done(self, tid: int) -> None:
        task = self.tasks[tid]
        self.rec[tid].prep_end = self.now
        remaining = 1.0
        if self._banked:
            # Work already banked by OOM-rescheduled attempts (model mode)
            # is not re-done: only the remaining fraction runs here.
            remaining = max(0.0, 1.0 - self._banked.get(tid, 0.0))
        comp = remaining * sum(
            base / self.knobs.op_scale(name) for (name, _oid, base, _b) in task.ops
        )
        dev = self.devices.get(task.device)
        work = remaining * float(getattr(task, "dev_work_ns", 0.0) or 0.0)
        if dev is not None and work > _TIME_EPS and comp > _TIME_EPS:
            # G4b: the compute phase is a fluid job on the shared device.
            # Natural duration = comp (already knob-scaled); demand = work /
            # comp; unthrottled it completes in exactly comp, and it stretches
            # proportionally when aggregate demand exceeds device capacity.
            dev._advance(self.now)
            dev.active[tid] = [work, work / comp]
            dev.stats.peak_active = max(dev.stats.peak_active, len(dev.active))
            self._reschedule_device(dev)
        else:
            self.schedule(self.now + comp + task.tail_ns, self._finish, tid)

    def _reschedule_device(self, dev: _FluidDevice) -> None:
        dev.token += 1
        nxt = dev._next_finish(self.now)
        if nxt is not None:
            self.schedule(nxt, self._device_event, dev.device, dev.token)

    def _device_event(self, device: int, token: int) -> None:
        dev = self.devices[device]
        if token != dev.token:
            return
        dev._advance(self.now)
        done = [tid for tid, v in dev.active.items() if v[0] <= _WORK_EPS]
        for tid in done:
            del dev.active[tid]
        self._reschedule_device(dev)
        for tid in done:
            self.schedule(self.now + self.tasks[tid].tail_ns, self._finish, tid)

    def _finish(self, tid: int) -> None:
        task = self.tasks[tid]
        dev = task.device
        self.rec[tid].finish = self.now
        self._unfinished.discard(tid)
        self._slots_free[dev] += 1
        self._n_res[dev] -= 1
        self._reserved[dev] -= self._granted.pop(tid, 0.0)
        devs_to_pump = {dev}
        spill_on = self.spill_mode != "off"
        # publish outputs
        for bid in task.output_batches:
            b = self.batches[bid]
            if b.gpu_resident:
                d = b.device if b.device in self._resident else dev
                if spill_on:
                    rest = max(0.0, float(b.nbytes) - self._prepub.pop(bid, 0.0))
                    self._resident[d] += rest
                    self._lru[d][bid] = self._lru[d].get(bid, 0.0) + rest
                else:
                    self._resident[d] += b.nbytes
        # unpin inputs
        if spill_on:
            for bid in task.input_batches:
                n = self._pins.get(bid, 0) - 1
                if n > 0:
                    self._pins[bid] = n
                else:
                    self._pins.pop(bid, None)
        # consume inputs
        for bid in task.input_batches:
            self._batch_pending[bid] -= 1
            if self._batch_pending[bid] == 0:
                b = self.batches[bid]
                host_part = self._evicted.pop(bid, None)
                if host_part is not None:
                    # demoted / born-downgraded part consumed from HOST
                    # (replay: its re-read cost sits in the traced spans)
                    self._host_used -= host_part
                if b.gpu_resident:
                    d = b.device if b.device in self._resident else dev
                    if spill_on:
                        # remaining accounted GPU bytes (spins may have
                        # consumed or demoted part of this batch already)
                        self._resident[d] -= self._lru[d].pop(bid, 0.0)
                    else:
                        self._resident[d] -= b.nbytes
                    devs_to_pump.add(d)
        self._sample_threads(dev)
        self._sample_pool(dev)
        # notify dependents
        for dtid in self._rdeps.get(tid, ()):  # pragma: no branch
            self._pending_deps[dtid] -= 1
            if self._pending_deps[dtid] == 0 and dtid not in self._released:
                self.schedule(
                    self.now + self.tasks[dtid].creation_lag_ns, self._release, dtid
                )
        for d in devs_to_pump:
            self._pump(d)

    # ------------------------------------------------------------------- run

    def run(self) -> SimResult:
        for tid, task in self.tasks.items():
            if self._pending_deps[tid] == 0:
                off = (
                    task.release_offset_ns if task.release_offset_ns is not None else 0
                )
                self.schedule(float(off), self._release, tid)

        while self._unfinished:
            if not self._heap:
                # No event can make progress. Two legitimate stall causes:
                # 1) memory-blocked queue head with nothing left running that
                #    could free the pool -> force-admit (the real engine would
                #    spill/downgrade here; counted + reported);
                # 2) a dependency cycle from batch->producer mis-attribution
                #    -> break it deterministically (counted + reported).
                blocked = sorted(
                    d
                    for d, k in self._block_kind.items()
                    if k == "memory" and self._fifo[d] and self._slots_free[d] > 0
                )
                if blocked:
                    self._admit(blocked[0], forced=True)
                    continue
                pend = [tid for tid in self._unfinished if tid not in self._released]
                if not pend:
                    raise RuntimeError(
                        "hwsim engine stalled with released-but-unfinished "
                        f"tasks: {sorted(self._unfinished)[:10]}"
                    )
                victim = min(pend, key=lambda t: self.tasks[t].t_created)
                self._pending_deps[victim] = 0
                self.dep_cycle_breaks += 1
                self.schedule(self.now, self._release, victim)
            t, _seq, fn, args = heapq.heappop(self._heap)
            self.now = max(self.now, t)
            fn(*args)

        for dev in self._slots_free:
            self._set_block(dev, None, None)
            self._sample_threads(dev)
        for ch in self.channels.values():
            ch._advance(self.now)
        for fd in self.devices.values():
            fd._advance(self.now)

        wall = (
            max((r.finish for r in self.rec.values()), default=0.0)
            + self.graph.finish_tail_ns
        )
        return SimResult(
            wall_ns=wall,
            task_times=self.rec,
            block_totals=self.block_totals,
            channel_stats={k: ch.stats for k, ch in self.channels.items()},
            device_stats={d: fd.stats for d, fd in self.devices.items()},
            thread_timeline=self.thread_timeline,
            pool_timeline=self.pool_timeline,
            peak_pool=self.peak_pool,
            pool_capacity=self.pool_capacity,
            n_threads=self.n_threads,
            thread_busy_ns=self.thread_busy_ns,
            forced_admissions=self.forced_admissions,
            dep_cycle_breaks=self.dep_cycle_breaks,
            orphan_gpu_batches=self.orphan_gpu_batches,
            spill_mode=self.spill_mode,
            downgrade_events=self.downgrade_events,
            downgraded_bytes=self.downgraded_bytes,
            reupgraded_bytes=self.reupgraded_bytes,
            oom_retries=self.oom_retries,
            spin_ns=self.spin_ns_total,
            retry_cap_forced=self.retry_cap_forced,
        )


def simulate_query(
    model,
    graph: QueryGraph,
    knobs: Knobs,
    spill_mode: str = "auto",
    spill: Optional[SpillParams] = None,
) -> SimResult:
    """Convenience wrapper wiring session-level resource facts into the engine."""
    pool = {}
    for ms in model.memory_spaces.values():
        if ms.tier == "GPU":
            pool[ms.device_id] = ms.capacity_bytes
    host = getattr(model, "host_pool_capacity", 0)
    return Engine(
        graph,
        knobs,
        n_threads=model.n_executor_threads,
        pool_capacity=pool,
        channel_capacity=dict(model.channel_peak_rate),
        spill_mode=spill_mode,
        spill=spill,
        host_capacity=host if host else None,
    ).run()
