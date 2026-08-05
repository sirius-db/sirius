"""Spec-sheet target mode (WS19): derive the full knob vector for hardware we
do NOT have, from advertised spec-sheet values.

    python -m hwsim simulate <trace> --physics p.json --target rtx.yaml

The trace comes from the SOURCE machine; ``--target`` names a descriptor of
the machine to predict. Each knob is a ratio ``target_achievable /
source_achievable`` where "achievable" resolves by provenance priority:

    measured-in-trace (WS9 G6 engine attrs / physics-profile wire curves)
      > descriptor ``measured:`` block
        > advertised x class derate (DERATING TABLE below)

Two vectors are derived: **derated-nominal** (best-available on both sides)
and **advertised-optimistic** (the target hits its spec sheet; the source
stays at its achieved values — so the optimistic bound does NOT collapse to
1.0 at target==source: your own box does not achieve its spec sheet either).
Full semantics, anchor sources and limits: docs/spec-sheet-mode.md.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .descriptor import HwDescriptor
from .knobs import Knobs

# ---------------------------------------------------------------------------
# DERATING TABLE — advertised -> achievable factors per resource class.
#
# Anchored on the TWO measured platforms (zero free parameters; do not tune):
#   GB300 (pmgb300ws-0163, Grace+Blackwell-Ultra, aarch64/C2C):
#     membw-throttle.md / compute-throttle.md / io-throttle.md
#   RTX PRO 6000 Blackwell Workstation (x86/PCIe):
#     docs/external-validation-rtx-pro-6000.md "Environment"
# Where the two anchors disagree for a class, unanchored classes get the
# RANGE [lo, hi] and the geometric-ish midpoint as nominal.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Derate:
    nominal: float
    lo: float
    hi: float
    source: str


# GPU memory bandwidth: CE D2D copy traffic (2x bytes = read+write, i.e. true
# HBM/GDDR traffic) / advertised peak.
MEMBW_DERATE: Dict[str, Derate] = {
    "hbm3e": Derate(
        0.702,
        0.702,
        0.702,
        "GB300: 5619 GB/s CE-measured / 8000 advertised (membw-throttle.md; "
        "NVIDIA Blackwell Ultra 8 TB/s per GPU)",
    ),
    "gddr7": Derate(
        0.821,
        0.821,
        0.821,
        "RTX PRO 6000: 1471 GB/s CE-measured / 1792 advertised "
        "(external-validation-rtx-pro-6000.md; NVIDIA datasheet 1792 GB/s)",
    ),
}
MEMBW_DERATE_DEFAULT = Derate(
    0.76,
    0.702,
    0.821,
    "no anchor for this memory class — range spans the two measured anchors "
    "(hbm3e 0.70, gddr7 0.82)",
)

# Host<->GPU link: pinned-copy per-direction payload rate / advertised
# per-direction peak.
LINK_DERATE: Dict[str, Derate] = {
    "pcie": Derate(
        0.902,
        0.895,
        0.902,
        "RTX box: 57.7 (H2D) / 64.0 GB/s PCIe gen5 x16 per dir "
        "(external-validation report; D2H 57.3 -> 0.895)",
    ),
    "c2c": Derate(
        0.851,
        0.829,
        0.851,
        "GB300: 383 (H2D) / 450 GB/s NVLink-C2C per dir "
        "(membw-throttle.md; NVIDIA C2C 900 GB/s bidirectional; "
        "D2H 373 -> 0.829)",
    ),
    "nvlink": Derate(
        0.851,
        0.829,
        0.902,
        "NO ANCHOR for device NVLink as a host link — borrowed from the "
        "C2C/PCIe anchors; order-of-magnitude only",
    ),
}

# FP32 compute: throttle-kit register-only FMA TFLOP/s / advertised (or
# theoretical SMs x 128 x 2 x clock) dense-FP32 peak. Two anchors nearly
# agree across arch (0.648 vs 0.692) — that ~7% spread is the honest
# cross-arch IPC error floor of the SMs x clock law.
FMA_DERATE = Derate(
    0.67,
    0.648,
    0.692,
    "GB300 fma 52.16 / 80.6 theoretical = 0.648 (compute-throttle.md); "
    "RTX PRO 6000 fma 87.2 / 126 advertised = 0.692 (external report)",
)

# Host DRAM: 8-thread memcpy traffic (2x bytes) / advertised peak. ONE anchor
# (Grace); the measurement is thread- and NUMA-bound, not a STREAM ceiling —
# order-of-magnitude class.
DRAM_DERATE = Derate(
    0.38,
    0.30,
    0.55,
    "Grace: 196 GB/s 8-thr memcpy / 512 GB/s advertised LPDDR5X "
    "(membw-throttle.md; NVIDIA Grace datasheet). Single anchor; the RTX "
    "box's DIMM config is unknown so x86 cannot anchor. WIDE band.",
)

# NVMe sequential read: 8-stream O_DIRECT achievable / advertised.
STORAGE_DERATE = Derate(
    0.91,
    0.85,
    1.0,
    "GB300: 6.525 GB/s 8-stream / 7.167 QD32 device ceiling "
    "(io-throttle.md; OEM drive has no public spec). NVMe seq read "
    "typically reaches 0.85-1.0 of spec.",
)

# Trace-derived channel capacities above this are coherent-C2C artifacts,
# never wire rates (physics/sanity.py uses the same ceiling).
PHYSICAL_LINK_MAX_GBPS = 1000.0

CROSS_CHECK_TOLERANCE = 0.15  # SMs x clock vs fp32-tflops disagreement gate


# ---------------------------------------------------------------------------
# Source characterization
# ---------------------------------------------------------------------------


def read_trace_engine_attrs(session_dir: str) -> Dict[str, object]:
    """Flatten the WS9 G6 engine Init custom_attributes of a session.

    Read directly from ``<session>/engine/*.ndjson`` (tiny files) so the
    pickled SessionModel cache format does not change. Returns {} on old
    traces / missing dir — callers fall back to ``--source``.
    """
    out: Dict[str, object] = {}
    d = os.path.join(session_dir, "engine")
    if not os.path.isdir(d):
        return out
    for fname in sorted(os.listdir(d)):
        if not fname.endswith(".ndjson"):
            continue
        with open(os.path.join(d, fname)) as f:
            for line in f:
                if '"Init"' not in line:
                    continue
                try:
                    ev = json.loads(line)
                except ValueError:
                    continue
                init = ev.get("data", {}).get("Init")
                if not isinstance(init, dict):
                    continue
                impl = init.get("implementation") or {}
                for attr in impl.get("custom_attributes") or []:
                    key = attr.get("key")
                    val = attr.get("value")
                    if key is None or not isinstance(val, dict) or not val:
                        continue
                    out[key] = next(iter(val.values()))
    return out


@dataclass
class Resolved:
    """One resolved quantity with its provenance."""

    value: float
    provenance: str  # "measured" | "trace" | "advertised" | "derated" | ...

    def __str__(self) -> str:
        return f"{self.value:g} ({self.provenance})"


@dataclass
class Side:
    """One machine's resolved characteristics (source or target)."""

    name: str = "?"
    engine_smclk: Optional[Resolved] = None  # SMs x MHz
    fp32_adv: Optional[Resolved] = None  # TFLOP/s, advertised/theoretical
    fp32_meas: Optional[Resolved] = None  # TFLOP/s, throttle-kit fma
    mem_class: str = ""
    membw_adv: Optional[Resolved] = None  # GB/s advertised
    membw_meas: Optional[Resolved] = None  # GB/s CE-measured (2x bytes)
    vram_adv: Optional[Resolved] = None  # GB advertised
    vram_meas: Optional[Resolved] = None  # GB usable (driver-reported)
    link_type: str = ""
    link_adv: Optional[Resolved] = None  # GB/s per dir advertised
    link_meas: Optional[Resolved] = None  # GB/s per dir measured (H2D)
    dram_adv: Optional[Resolved] = None
    dram_meas: Optional[Resolved] = None
    storage_adv: Optional[Resolved] = None
    storage_meas: Optional[Resolved] = None
    cpu_engine: Optional[Resolved] = None  # cores x GHz
    cpu_cores: Optional[Resolved] = None
    grace_colimit: Optional[bool] = None


def side_from_descriptor(desc: HwDescriptor) -> Side:
    s = Side(name=desc.name)
    g, m = desc.gpu, desc.measured
    if g.sm_count and g.boost_clock_mhz:
        s.engine_smclk = Resolved(
            g.sm_count * g.boost_clock_mhz,
            f"descriptor ({g.sm_count} SMs x {g.boost_clock_mhz:g} MHz)",
        )
    if g.fp32_tflops_peak:
        s.fp32_adv = Resolved(float(g.fp32_tflops_peak), "advertised")
    elif g.theoretical_fp32_tflops():
        s.fp32_adv = Resolved(
            g.theoretical_fp32_tflops(), "theoretical (SMs x 128 x 2 x clk)"
        )
    if m.fp32_tflops:
        s.fp32_meas = Resolved(float(m.fp32_tflops), "measured fma")
    s.mem_class = (g.mem_class or "").lower()
    if g.mem_bandwidth_gbs_peak:
        s.membw_adv = Resolved(float(g.mem_bandwidth_gbs_peak), "advertised")
    if m.gpu_mem_gbs:
        s.membw_meas = Resolved(float(m.gpu_mem_gbs), "measured CE")
    if g.vram_gb:
        s.vram_adv = Resolved(float(g.vram_gb), "advertised")
    if m.vram_usable_gb:
        s.vram_meas = Resolved(float(m.vram_usable_gb), "measured usable")
    s.link_type = (desc.link.type or "").lower()
    if desc.link.peak_gbs():
        detail = (
            f"gen{desc.link.gen} x{desc.link.lanes}"
            if desc.link.type == "pcie" and desc.link.gen
            else desc.link.type
        )
        s.link_adv = Resolved(desc.link.peak_gbs(), f"advertised ({detail})")
    if m.link_h2d_gbs:
        s.link_meas = Resolved(float(m.link_h2d_gbs), "measured H2D")
    if desc.cpu.dram_peak_gbs():
        s.dram_adv = Resolved(desc.cpu.dram_peak_gbs(), "advertised")
    if m.cpu_dram_gbs:
        s.dram_meas = Resolved(float(m.cpu_dram_gbs), "measured 8-thr memcpy")
    if desc.storage.seq_read_gbs:
        s.storage_adv = Resolved(float(desc.storage.seq_read_gbs), "advertised")
    if m.storage_seq_read_gbs:
        s.storage_meas = Resolved(
            float(m.storage_seq_read_gbs), "measured 8-stream O_DIRECT"
        )
    if desc.cpu.engine():
        clk = desc.cpu.boost_clock_ghz or desc.cpu.base_clock_ghz
        s.cpu_engine = Resolved(
            desc.cpu.engine(),
            f"descriptor ({desc.cpu.cores} cores x {clk:g} GHz)",
        )
    if desc.cpu.cores:
        s.cpu_cores = Resolved(float(desc.cpu.cores), "descriptor cores")
    s.grace_colimit = desc.link.grace_colimit
    return s


def _link_meas_from_profile(physics_path: Optional[str]) -> Optional[Resolved]:
    """Wire-side H2D peak aggregate from a physics profile (measured transfer
    curves beat any spec). Gated at the physical ceiling so coherent-C2C
    artifact rates are never used."""
    if not physics_path or not os.path.exists(physics_path):
        return None
    try:
        with open(physics_path) as f:
            prof = json.load(f)
    except (OSError, ValueError):
        return None
    peaks = (prof.get("diagnostics") or {}).get("channel_peak_gbps") or {}
    h2d = [
        v
        for k, v in peaks.items()
        if isinstance(v, (int, float))
        and 0 < v <= PHYSICAL_LINK_MAX_GBPS
        and ("Host-to-Device" in k or "HtoD" in k)
    ]
    if not h2d:
        return None
    return Resolved(max(h2d), "trace physics profile (nsys wire peak)")


def resolve_source(
    session_dir: str,
    source_desc: Optional[HwDescriptor] = None,
    physics_path: Optional[str] = None,
) -> Tuple[Side, List[str]]:
    """Characterize the SOURCE machine: trace G6 attrs > ``--source``
    descriptor measured > descriptor advertised. Returns (side, warnings)."""
    warnings: List[str] = []
    side = side_from_descriptor(source_desc) if source_desc else Side()
    attrs = read_trace_engine_attrs(session_dir)
    gname = attrs.get("gpu.0.name")
    if gname:
        side.name = str(gname) if not source_desc else side.name
    if side.name == "?":
        side.name = "source"
    sm = attrs.get("gpu.0.sm_count")
    clk_khz = attrs.get("gpu.0.sm_clock_khz")
    if sm and clk_khz:
        eng = float(sm) * float(clk_khz) / 1000.0
        if (
            side.engine_smclk is not None
            and abs(eng / side.engine_smclk.value - 1.0) > 0.02
        ):
            warnings.append(
                f"source: trace G6 says {sm} SMs x {float(clk_khz)/1000:g} MHz "
                f"but --source descriptor says {side.engine_smclk.provenance} "
                "— trace wins (is the descriptor for the right box?)"
            )
        side.engine_smclk = Resolved(
            eng, f"trace G6 ({sm} SMs x {float(clk_khz)/1000:g} MHz)"
        )
        if side.fp32_adv is None:
            side.fp32_adv = Resolved(
                float(sm) * 128 * 2 * float(clk_khz) * 1e3 / 1e12,
                "theoretical from trace G6 (SMs x 128 x 2 x clk)",
            )
    mem_clk = attrs.get("gpu.0.mem_clock_khz")
    bus = attrs.get("gpu.0.mem_bus_width_bits")
    if mem_clk and bus and side.membw_adv is None:
        # CUDA-derived peak (2 x mem clock x bus width); can sit below the
        # marketing number (GB300: 7.16 vs 8.0 TB/s).
        side.membw_adv = Resolved(
            2.0 * float(mem_clk) * 1e3 * float(bus) / 8.0 / 1e9,
            "trace G6 CUDA-derived (2 x mem_clk x bus_width)",
        )
    if side.cpu_cores is None and attrs.get("hw.host_cores"):
        side.cpu_cores = Resolved(
            float(attrs["hw.host_cores"]), "trace G6 hw.host_cores"
        )
    prof_link = _link_meas_from_profile(physics_path)
    if prof_link is not None:
        side.link_meas = prof_link
    if not attrs and source_desc is None:
        warnings.append(
            "source: trace has no WS9 G6 engine attributes and no --source "
            "descriptor was given — source characteristics are unknown"
        )
    return side, warnings


# ---------------------------------------------------------------------------
# Knob derivation
# ---------------------------------------------------------------------------


@dataclass
class DerivedKnob:
    name: str
    nominal: float
    optimistic: float
    provenance: str  # "<target basis> / <source basis>"
    tier: str
    note: str = ""


@dataclass
class TargetDerivation:
    source_name: str
    target_name: str
    knobs: List[DerivedKnob] = field(default_factory=list)
    grace_colimit: bool = True
    warnings: List[str] = field(default_factory=list)

    def _vector(self, optimistic: bool) -> Knobs:
        k = Knobs()
        vals = {d.name: (d.optimistic if optimistic else d.nominal) for d in self.knobs}
        for name, v in vals.items():
            if name == "gpu_mem_bandwidth":
                # A derived HBM ratio of exactly 1.0 keeps the default None
                # (v0 "tracks gpu_compute" semantics; physics resolves None
                # to traced HBM = 1.0). Required for the target==source and
                # sm-halved consistency guarantees (byte-identical output vs
                # the equivalent --knob invocation). Consequence on the v0
                # path only: a same-membw target with gpu_compute > 1 scales
                # whole spans by gpu_compute (the usual v0 conflation,
                # warned) instead of pinning them at 1.0 — use --physics for
                # the honest split.
                if v == 1.0:
                    continue
                k.gpu_mem_bandwidth = v
            else:
                setattr(k, name, v)
        # platform law (physics/laws.py law 2), descriptor-driven
        k.grace_colimit = self.grace_colimit  # dynamic attr; see laws.py
        return k

    def nominal_knobs(self) -> Knobs:
        return self._vector(optimistic=False)

    def optimistic_knobs(self) -> Knobs:
        return self._vector(optimistic=True)

    def header_text(self) -> str:
        from .report import _table

        rows = []
        for d in self.knobs:
            rows.append(
                [
                    d.name,
                    f"{d.nominal:.4g}",
                    f"{d.optimistic:.4g}",
                    d.provenance,
                    d.tier,
                ]
            )
        lines = [
            f"=== TARGET MODE: {self.source_name} -> {self.target_name} ===",
            _table(
                [
                    "knob",
                    "nominal",
                    "optimistic",
                    "provenance (target / source)",
                    "confidence tier",
                ],
                rows,
            ),
            "platform law: Grace C2C<->DRAM co-limit "
            + (
                "ON (c2c-class target link)"
                if self.grace_colimit
                else "OFF (non-c2c target link: link multiplier = "
                "c2c_bandwidth alone)"
            ),
            "band: 'nominal' = best-available (measured > advertised x "
            "derate); 'optimistic' = target hits its spec sheet. "
            "See docs/spec-sheet-mode.md.",
        ]
        for n in [d.note for d in self.knobs if d.note]:
            lines.append(f"note: {n}")
        return "\n".join(lines)


def _ach(meas: Optional[Resolved], adv: Optional[Resolved], derate: Derate):
    """(achievable Resolved, used_derate: bool) by provenance priority."""
    if meas is not None:
        return meas, False
    if adv is not None:
        return (
            Resolved(
                adv.value * derate.nominal,
                f"{adv.provenance} x{derate.nominal:g} derate",
            ),
            True,
        )
    return None, False


def _ratio_knob(
    name: str,
    tgt_meas: Optional[Resolved],
    tgt_adv: Optional[Resolved],
    src_meas: Optional[Resolved],
    src_adv: Optional[Resolved],
    tgt_derate: Derate,
    src_derate: Derate,
    tier: str,
    warnings: List[str],
    optimistic_derate: float = 1.0,
    note: str = "",
) -> Optional[DerivedKnob]:
    src, _ = _ach(src_meas, src_adv, src_derate)
    tgt, tgt_derated = _ach(tgt_meas, tgt_adv, tgt_derate)
    if src is None or tgt is None:
        warnings.append(
            f"{name}: unresolved ({'source' if src is None else 'target'} "
            "side has neither measured nor advertised values) — knob left "
            "at 1.0"
        )
        return None
    nominal = tgt.value / src.value
    if tgt_adv is not None:
        optimistic = tgt_adv.value * optimistic_derate / src.value
    else:
        optimistic = nominal
    if tgt_derated and tgt_derate.lo != tgt_derate.hi:
        note = (note + " " if note else "") + (
            f"{name}: target class unanchored — derate range "
            f"[{tgt_derate.lo:g}, {tgt_derate.hi:g}] "
            f"=> knob range [{tgt_adv.value * tgt_derate.lo / src.value:.4g}, "
            f"{tgt_adv.value * tgt_derate.hi / src.value:.4g}]"
        )
    return DerivedKnob(
        name=name,
        nominal=nominal,
        optimistic=optimistic,
        provenance=f"{tgt.provenance} / {src.provenance}",
        tier=tier,
        note=note,
    )


def derive(source: Side, target: Side) -> TargetDerivation:
    d = TargetDerivation(source_name=source.name, target_name=target.name)
    w = d.warnings

    # -- gpu_compute -------------------------------------------------------
    # nominal  = target_achievable_fp32 / source_achievable_fp32
    #            (measured fma > advertised x FMA_DERATE.nominal)
    # optimistic = target advertised FP32 x best measured anchor fraction
    #            (advertised x 1.0 would be theoretical peak no silicon
    #             sustains — both anchors sit at 0.648/0.692)
    # cross-check: SMs x clock ratio vs FP32-TFLOPS ratio (>15% => arch IPC
    # difference, e.g. 64 vs 128 FP32 lanes/SM; warn loudly).
    eng = tfl = None
    if source.engine_smclk and target.engine_smclk:
        eng = target.engine_smclk.value / source.engine_smclk.value
    if source.fp32_adv and target.fp32_adv:
        tfl = target.fp32_adv.value / source.fp32_adv.value
    if eng is not None and tfl is not None and eng > 0:
        if abs(tfl / eng - 1.0) > CROSS_CHECK_TOLERANCE:
            w.append(
                f"gpu_compute: SMs x clock ratio ({eng:.3f}) and FP32-TFLOPS "
                f"ratio ({tfl:.3f}) disagree by "
                f"{abs(tfl / eng - 1.0) * 100:.0f}% — arch-generation IPC "
                "difference (e.g. FP32 lanes per SM). The FP32 ratio wins; "
                "kernels that are not FP32-throughput-bound may follow the "
                "SM ratio instead."
            )
    src_c, _ = _ach(source.fp32_meas, source.fp32_adv, FMA_DERATE)
    tgt_c, _ = _ach(target.fp32_meas, target.fp32_adv, FMA_DERATE)
    if src_c is not None and tgt_c is not None:
        nominal = tgt_c.value / src_c.value
        if target.fp32_adv is not None:
            optimistic = target.fp32_adv.value * FMA_DERATE.hi / src_c.value
        else:
            optimistic = nominal
        note = ""
        if eng is not None and nominal and abs(eng / nominal - 1.0) > 0.05:
            note = (
                f"gpu_compute: SMs x clock ratio is {eng:.3f} vs nominal "
                f"{nominal:.3f} — the two anchors' achieved-fraction spread "
                "(0.648 vs 0.692) is the cross-arch error floor (~7%)."
            )
        tier = (
            "validated <=1 (use --physics; suite median +6.8%); "
            "gpu_compute>1 is order-of-magnitude (speedup band)"
            if nominal <= 1.0
            else "order-of-magnitude (speedup: emergent waits cannot shrink "
            "below traced demand; report [G4b, v0] band)"
        )
        d.knobs.append(
            DerivedKnob(
                "gpu_compute",
                nominal,
                optimistic,
                f"{tgt_c.provenance} / {src_c.provenance}",
                tier,
                note,
            )
        )
    else:
        w.append(
            "gpu_compute: unresolved (need SM count + clock or FP32 numbers "
            "on both sides) — knob left at 1.0"
        )

    # -- gpu_mem_bandwidth ---------------------------------------------------
    kb = _ratio_knob(
        "gpu_mem_bandwidth",
        target.membw_meas,
        target.membw_adv,
        source.membw_meas,
        source.membw_adv,
        MEMBW_DERATE.get(target.mem_class, MEMBW_DERATE_DEFAULT),
        MEMBW_DERATE.get(source.mem_class, MEMBW_DERATE_DEFAULT),
        "validated with --physics on host-dominated lanes (E4 rescore "
        "median -2.8/-5.3%); v0 = pessimistic roofline",
        w,
    )
    if kb:
        d.knobs.append(kb)

    # -- gpu_mem_capacity ----------------------------------------------------
    kb = _ratio_knob(
        "gpu_mem_capacity",
        target.vram_meas,
        target.vram_adv,
        source.vram_meas,
        source.vram_adv,
        Derate(1.0, 1.0, 1.0, "capacity is not derated"),
        Derate(1.0, 1.0, 1.0, "capacity is not derated"),
        "validated above the spill knee (<=0.5%); below it "
        "order-of-magnitude (~+/-40%, G5)",
        w,
        note=(
            "gpu_mem_capacity assumes the SAME pool convention on both "
            "boxes (usage_limit fraction x VRAM). If the target config uses "
            "an absolute usage_limit_bytes, override with --knob "
            "gpu_mem_capacity=<target_pool/source_pool>."
        ),
    )
    if kb:
        d.knobs.append(kb)

    # -- c2c_bandwidth (host<->GPU link) -------------------------------------
    src_lt = source.link_type or "c2c"
    tgt_lt = target.link_type or src_lt
    kb = _ratio_knob(
        "c2c_bandwidth",
        target.link_meas,
        target.link_adv,
        source.link_meas,
        source.link_adv,
        LINK_DERATE.get(tgt_lt, LINK_DERATE["nvlink"]),
        LINK_DERATE.get(src_lt, LINK_DERATE["nvlink"]),
        "validated on link-bound lanes with --physics + wire cap "
        "(-9.1/-1.5%); INERT on coherent-C2C traces (warned at simulate)",
        w,
    )
    if kb:
        if src_lt != tgt_lt:
            kb.note = (
                f"cross-link-class ({src_lt} source -> {tgt_lt} target): "
                "ratio semantics validated per side but never end-to-end "
                "across classes; latency/small-copy behavior differs beyond "
                "bandwidth."
            )
        d.knobs.append(kb)
    # platform law: Grace co-limit only when the TARGET link is C2C-class
    if target.grace_colimit is not None:
        d.grace_colimit = bool(target.grace_colimit)
    else:
        d.grace_colimit = tgt_lt == "c2c"

    # -- io_bandwidth --------------------------------------------------------
    kb = _ratio_knob(
        "io_bandwidth",
        target.storage_meas,
        target.storage_adv,
        source.storage_meas,
        source.storage_adv,
        STORAGE_DERATE,
        STORAGE_DERATE,
        "validated +/-5% (<=1) on scan-bound cold lanes; decode-heavy "
        "queries <=+20% pessimistic; INERT on GPU-pinned lanes; >1 is an "
        "optimistic bound (G1 scales decode too)",
        w,
    )
    if kb:
        d.knobs.append(kb)

    # -- cpu_mem_bandwidth ---------------------------------------------------
    kb = _ratio_knob(
        "cpu_mem_bandwidth",
        target.dram_meas,
        target.dram_adv,
        source.dram_meas,
        source.dram_adv,
        DRAM_DERATE,
        DRAM_DERATE,
        "co-limit input only (link law); UNVALIDATED standalone — no host "
        "bandwidth events; v0: no effect",
        w,
    )
    if kb:
        d.knobs.append(kb)

    # -- cpu_compute ---------------------------------------------------------
    if source.cpu_engine and target.cpu_engine:
        r = target.cpu_engine.value / source.cpu_engine.value
        prov = f"{target.cpu_engine.provenance} / {source.cpu_engine.provenance}"
    elif source.cpu_cores and target.cpu_cores:
        r = target.cpu_cores.value / source.cpu_cores.value
        prov = (
            f"cores-only ratio ({target.cpu_cores.value:g}/"
            f"{source.cpu_cores.value:g}) — clocks unknown, assumed equal"
        )
    else:
        r = None
        prov = ""
    if r is not None:
        d.knobs.append(
            DerivedKnob(
                "cpu_compute",
                r,
                r,
                prov,
                "UNVALIDATED — no physical validation on ANY platform; "
                "cross-CPU-arch host-time scaling is the model's weakest "
                "knob; v0: no effect, physics: scales host glue only",
            )
        )
        w.append(
            "cpu_compute derived as a cores x clock ratio is UNVALIDATED: "
            "no throttle-kit or cross-machine measurement backs it anywhere. "
            f"Treat host-time scaling ({r:.3f}) as a guess; consider "
            "--knob cpu_compute=1 to freeze host time instead."
        )
    else:
        w.append("cpu_compute: unresolved (missing cpu cores) — left at 1.0")

    return d
