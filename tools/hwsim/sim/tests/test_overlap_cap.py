"""Per-span kernel-overlap cap (WS20, section-7 split path).

The RTX PRO 6000 external validation's two pessimistic outliers (q17/q20,
real x ~ 1.0 at MPS-50 but charged +15.6/+19.7%) motivated capping the
charged kernel share on spans whose kernel time is hidden under concurrent
host work: `f_kernel_overlap` = share of a span's kernel-busy union NOT
covered by a same-thread sync wait (attribute.py), consumed by
`_capped_kernel_share` in integrate.py. Constraints pinned here:

- knobs=1 identity, and exact old behavior at overlap=0 / on old profiles;
- the cap charges max(host, hidden kernels) instead of their sum, with the
  span's host residue as hiding capacity;
- the G4b fluid-device gate takes precedence: when it engages (serialized
  kernels) the cap stands down.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

from hwsim.knobs import Knobs  # noqa: E402
from hwsim.physics.attribute import (  # noqa: E402
    _HostSpanIndex,
    _kernel_overlap_frac,
)
from hwsim.physics.integrate import (  # noqa: E402
    _capped_kernel_share,
    _op_inv_factor,
    retime_graph,
)
from hwsim.physics.join import join_graph  # noqa: E402
from hwsim.physics.reader import HostSpanRow  # noqa: E402
from hwsim.physics.schema import OpPhysics, PrepPhysics  # noqa: E402

from test_physics_join_integrate import (  # noqa: E402
    make_graph,
    make_task,
    op_phys,
    profile_for,
    task_phys,
)

MS = 1_000_000


def _sync(gtid, start, end):
    return HostSpanRow(
        start=start, end=end, global_tid=gtid, api="cudaStreamSynchronize",
        kind="sync",
    )


def _op_with_overlap(overlap, **fracs):
    # op_phys() derives f_host from the remaining kwargs, so the overlap
    # fraction is set after construction (it is not a span-share class).
    op = op_phys(**fracs)
    op.f_kernel_overlap = overlap
    return op


class TestCappedKernelShareMath(unittest.TestCase):
    def test_identity_at_knobs1(self):
        # kern_scaled == kern -> charged == kern for any overlap/host
        for ov in (0.0, 0.3, 1.0):
            for fh in (0.0, 0.5):
                self.assertAlmostEqual(
                    _capped_kernel_share(0.2, 0.2, fh, ov), 0.2
                )

    def test_zero_overlap_is_passthrough(self):
        self.assertAlmostEqual(_capped_kernel_share(0.2, 0.4, 0.7, 0.0), 0.4)

    def test_fully_hidden_within_capacity_charges_nothing_extra(self):
        # kern 0.1 doubled to 0.2, ov=1, host residue 0.7 absorbs the 0.1
        self.assertAlmostEqual(_capped_kernel_share(0.1, 0.2, 0.7, 1.0), 0.1)

    def test_hidden_stretch_beyond_capacity_pays_the_excess(self):
        # kern 0.2 scaled 4x to 0.8, ov=1, host 0.3: charged =
        # max(0.2, 0.8 - 0.3) = 0.5 (excess over hidden+host headroom)
        self.assertAlmostEqual(_capped_kernel_share(0.2, 0.8, 0.3, 1.0), 0.5)

    def test_partial_overlap_above_gate_splits_serial_and_hidden(self):
        # kern 0.2 doubled to 0.4, ov=0.9, host 0.6: serial tenth charges
        # 0.04; hidden max(0.18, 0.36-0.6)=0.18 -> 0.22
        self.assertAlmostEqual(_capped_kernel_share(0.2, 0.4, 0.6, 0.9), 0.22)

    def test_below_gate_is_full_charge(self):
        # the hypothesis is about ENTIRELY-hidden kernel time: at low overlap
        # the uncovered sliver is launch-side latency and the stretch lands
        # in the sync wait -- full charge (keeps the validated GB300 section
        # 7.2 medians within 0.5 pp; ungated linear discounting moved them
        # 1-3.5 pp more optimistic)
        self.assertAlmostEqual(_capped_kernel_share(0.2, 0.4, 0.6, 0.5), 0.4)
        self.assertAlmostEqual(_capped_kernel_share(0.2, 0.4, 0.6, 0.89), 0.4)

    def test_shrinking_hidden_kernels_floors_at_base(self):
        # gpu_compute > 1: hidden kernel time cannot shrink the span below
        # the co-running host work
        self.assertAlmostEqual(_capped_kernel_share(0.2, 0.1, 0.5, 1.0), 0.2)
        # ...but the serial share still speeds up
        self.assertAlmostEqual(
            _capped_kernel_share(0.2, 0.1, 0.5, 0.95), 0.005 + 0.19
        )


class TestOverlapFracFromSyncSpans(unittest.TestCase):
    def test_no_kernels_is_zero(self):
        self.assertEqual(
            _kernel_overlap_frac([], _HostSpanIndex([]), 1, 0, 100), 0.0
        )

    def test_uncovered_kernel_time_is_hidden(self):
        k_union = [(10.0, 30.0), (50.0, 60.0)]  # 30 ns busy
        hidx = _HostSpanIndex([_sync(1, 20, 40)])  # covers 10 of the 30
        self.assertAlmostEqual(
            _kernel_overlap_frac(k_union, hidx, 1, 0, 100), 20.0 / 30.0
        )

    def test_other_thread_sync_does_not_count(self):
        k_union = [(10.0, 30.0)]
        hidx = _HostSpanIndex([_sync(2, 0, 100)])  # different thread
        self.assertAlmostEqual(
            _kernel_overlap_frac(k_union, hidx, 1, 0, 100), 1.0
        )

    def test_fully_sync_covered_is_zero_overlap(self):
        k_union = [(10.0, 30.0)]
        hidx = _HostSpanIndex([_sync(1, 5, 35)])
        self.assertAlmostEqual(
            _kernel_overlap_frac(k_union, hidx, 1, 0, 100), 0.0
        )


class TestRetimeWithOverlapCap(unittest.TestCase):
    def _retimed_dur(self, op, knobs, serial_frac=None):
        t = make_task(0, ops=[("OP(5)", 5, 10 * MS)])
        g = make_graph([t])
        prof = profile_for({3: [task_phys(ops=[op])]})
        ann, _ = join_graph(prof, g)
        g2, stats = retime_graph(g, ann, knobs, prof.curves, serial_frac=serial_frac)
        return g2.tasks[0].ops[-1][2], stats

    def test_hidden_kernel_does_not_stretch_span(self):
        # 20% kernel fully hidden under 80% host: gpu_compute=0.5 must not
        # charge the stretch (q17/q20 shape)
        op = _op_with_overlap(1.0, f_comp=0.2)
        dur, stats = self._retimed_dur(op, Knobs(gpu_compute=0.5), serial_frac=0.6)
        self.assertEqual(dur, 10 * MS)
        self.assertTrue(stats.overlap_cap_active)
        self.assertAlmostEqual(stats.overlap_hidden_kernel_ns, 0.2 * 10 * MS)

    def test_unhidden_kernel_still_charged(self):
        op = _op_with_overlap(0.0, f_comp=0.2)
        dur, _ = self._retimed_dur(op, Knobs(gpu_compute=0.5), serial_frac=0.6)
        self.assertEqual(dur, 12 * MS)  # +0.2*10ms

    def test_knobs1_identity_with_overlap(self):
        op = _op_with_overlap(1.0, f_comp=0.2)
        dur, _ = self._retimed_dur(op, Knobs(), serial_frac=0.6)
        self.assertEqual(dur, 10 * MS)

    def test_g4b_gate_takes_precedence(self):
        # serialized kernels (serial_frac >= 0.9): fluid device engages and
        # the cap must stand down -- span keeps the uncapped section-7 charge
        op = _op_with_overlap(1.0, f_comp=0.2)
        dur, stats = self._retimed_dur(op, Knobs(gpu_compute=0.5), serial_frac=0.95)
        self.assertEqual(dur, 12 * MS)
        self.assertTrue(stats.device_model_active)
        self.assertFalse(stats.overlap_cap_active)

    def test_deep_throttle_pays_excess_over_host_headroom(self):
        # 40% kernel fully hidden, 60% host; at gpu_compute=0.25 the hidden
        # share stretches to 1.6 -> charged max(0.4, 1.6-0.6)=1.0; span =
        # (0.6 + 1.0) * 10ms = 16ms (uncapped would be 0.6+1.6=22ms)
        op = _op_with_overlap(1.0, f_comp=0.4)
        dur, _ = self._retimed_dur(op, Knobs(gpu_compute=0.25), serial_frac=0.6)
        self.assertEqual(dur, 16 * MS)

    def test_old_profiles_bit_identical(self):
        # an OpPhysics without the field (pre-WS20 profile shape) must take
        # exactly the uncapped path
        op = OpPhysics(op_id=5, op_name="OP", span_ns=10 * MS,
                       f_comp=0.2, f_host=0.8)
        self.assertFalse(hasattr(op, "f_kernel_overlap") and op.f_kernel_overlap)
        k = Knobs(gpu_compute=0.5)
        self.assertAlmostEqual(
            _op_inv_factor(op, k, overlap_cap=True),
            _op_inv_factor(op, k, overlap_cap=False),
        )

    def test_prep_kernel_share_capped_too(self):
        # same-tier prep, 30% kernel (decompress) fully hidden under host
        t = make_task(0, prep_ns=10 * MS)
        g = make_graph([t])
        prep = PrepPhysics(span_ns=10 * MS, f_xfer=0.0, f_comp=0.3,
                           f_host=0.7, f_kernel_overlap=1.0)
        prof = profile_for({3: [task_phys(prep=prep)]})
        ann, _ = join_graph(prof, g)
        g2, _ = retime_graph(g, ann, Knobs(gpu_compute=0.5), prof.curves,
                             serial_frac=0.6)
        # non-transfer prep becomes PHYS::PREP; hidden decompress absorbed
        self.assertEqual(g2.tasks[0].ops[0][0], "PHYS::PREP")
        self.assertEqual(g2.tasks[0].ops[0][2], 10 * MS)


if __name__ == "__main__":
    unittest.main()
