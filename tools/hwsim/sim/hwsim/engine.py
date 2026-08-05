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

Nothing is hand-coded per knob beyond event durations/rates: back-pressure,
queueing and saturation emerge from admission and capacity.

Progress guarantee: if the simulation stalls completely (no pending event can
free memory) a memory-blocked queue head is force-admitted — the real engine
would spill / downgrade at that point, whose cost v0 cannot price (the sample
trace has zero Downgrading events); occurrences are counted and reported.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Set, Tuple

from .knobs import Knobs
from .model import BatchSpec, QueryGraph, TaskSpec

ChannelKey = Tuple[str, str, int]  # (origin_tier, target_tier, device)

_BYTES_EPS = 0.5


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

    def binding_constraint(self, device: int = 0) -> str:
        bt = self.block_totals.get(device, {})
        threads = bt.get("threads", 0.0)
        memory = bt.get("memory", 0.0)
        chan = sum(c.throttled_ns for c in self.channel_stats.values())
        best = max(threads, memory, chan)
        if best <= 0:
            return "dependencies"
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


class Engine:
    def __init__(
        self,
        graph: QueryGraph,
        knobs: Knobs,
        n_threads: Dict[int, int],
        pool_capacity: Dict[int, int],
        channel_capacity: Dict[ChannelKey, float],
        queue_order: str = "traced",
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

        self.rec: Dict[int, TaskTimes] = {tid: TaskTimes() for tid in self.tasks}
        self.forced_admissions = 0
        self.dep_cycle_breaks = 0
        self.orphan_gpu_batches = 0

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
                self._resident[
                    b.device if b.device in devices else next(iter(devices))
                ] += b.nbytes

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
        self._pump(task.device)

    def _pump(self, dev: int) -> None:
        fifo = self._fifo[dev]
        while True:
            if not fifo:
                self._set_block(dev, None, None)
                return
            if self._slots_free[dev] == 0:
                self._set_block(dev, "threads", fifo[0][1])
                return
            tid = fifo[0][1]
            task = self.tasks[tid]
            cap = self.pool_capacity[dev]
            need = min(float(task.reservation_bytes), cap)
            if self._reserved[dev] + self._resident[dev] + need > cap + _BYTES_EPS:
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
        self._sample_threads(dev)
        self._sample_pool(dev)
        self.schedule(self.now + task.grant_ns, self._prep_start, tid)

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
            self.schedule(self.now + task.prep_ns, self._prep_done, tid)

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
        dur = task.tail_ns
        for name, _oid, base, _b in task.ops:
            dur += base / self.knobs.op_scale(name)
        self.schedule(self.now + dur, self._finish, tid)

    def _finish(self, tid: int) -> None:
        task = self.tasks[tid]
        dev = task.device
        self.rec[tid].finish = self.now
        self._unfinished.discard(tid)
        self._slots_free[dev] += 1
        self._n_res[dev] -= 1
        self._reserved[dev] -= self._granted.pop(tid, 0.0)
        devs_to_pump = {dev}
        # publish outputs
        for bid in task.output_batches:
            b = self.batches[bid]
            if b.gpu_resident:
                self._resident[
                    b.device if b.device in self._resident else dev
                ] += b.nbytes
        # consume inputs
        for bid in task.input_batches:
            self._batch_pending[bid] -= 1
            if self._batch_pending[bid] == 0:
                b = self.batches[bid]
                if b.gpu_resident:
                    d = b.device if b.device in self._resident else dev
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

        wall = (
            max((r.finish for r in self.rec.values()), default=0.0)
            + self.graph.finish_tail_ns
        )
        return SimResult(
            wall_ns=wall,
            task_times=self.rec,
            block_totals=self.block_totals,
            channel_stats={k: ch.stats for k, ch in self.channels.items()},
            thread_timeline=self.thread_timeline,
            pool_timeline=self.pool_timeline,
            peak_pool=self.peak_pool,
            pool_capacity=self.pool_capacity,
            n_threads=self.n_threads,
            thread_busy_ns=self.thread_busy_ns,
            forced_admissions=self.forced_admissions,
            dep_cycle_breaks=self.dep_cycle_breaks,
            orphan_gpu_batches=self.orphan_gpu_batches,
        )


def simulate_query(model, graph: QueryGraph, knobs: Knobs) -> SimResult:
    """Convenience wrapper wiring session-level resource facts into the engine."""
    pool = {}
    for ms in model.memory_spaces.values():
        if ms.tier == "GPU":
            pool[ms.device_id] = ms.capacity_bytes
    return Engine(
        graph,
        knobs,
        n_threads=model.n_executor_threads,
        pool_capacity=pool,
        channel_capacity=dict(model.channel_peak_rate),
    ).run()
