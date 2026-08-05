"""Per-kernel membw-bound vs compute-bound classification.

Strategy per nsys-extraction.md section 5.1, precedence order:

1. explicit override table (kernel name -> class), fed by offline ncu
   spot-checks (``ncu --set roofline`` per kernel family);
2. Tier B gpu-metrics integration (device-wide DRAM throughput integrated over
   the kernel's interval) — only for kernels that own the device exclusively;
3. name-based priors for the cuDF/CUB kernel families;
4. otherwise "unknown" — which the retiming scales with the v0 conflated rule,
   never dropped.

Classes:
- "membw"   : scales with gpu_mem_bandwidth (coupled per laws.py),
- "compute" : scales with gpu_compute,
- "mixed"   : split 50/50 between the two,
- "unknown" : v0 conflated scaling.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Tuple

CLASS_MEMBW = "membw"
CLASS_COMPUTE = "compute"
CLASS_MIXED = "mixed"
CLASS_UNKNOWN = "unknown"

VALID_CLASSES = (CLASS_MEMBW, CLASS_COMPUTE, CLASS_MIXED, CLASS_UNKNOWN)

# Name priors (matched as lowercase substrings, first hit wins). Sources:
# nsys-extraction.md section 5.1 + the "decompress is SM-bound" measurement.
NAME_PRIORS: List[Tuple[str, str]] = [
    # decompression: measured SM-bound on this box (do NOT scale with links)
    ("decompress", CLASS_COMPUTE),
    ("snappy", CLASS_COMPUTE),
    ("zstd", CLASS_COMPUTE),
    ("inflate", CLASS_COMPUTE),
    ("unsnap", CLASS_COMPUTE),
    ("gpuinflate", CLASS_COMPUTE),
    # data-movement families: membw-bound
    ("gather", CLASS_MEMBW),
    ("scatter", CLASS_MEMBW),
    ("copy_if", CLASS_MEMBW),
    ("copy_range", CLASS_MEMBW),
    ("concatenate", CLASS_MEMBW),
    ("materiali", CLASS_MEMBW),  # materialize / materialization
    ("radixsort", CLASS_MEMBW),
    ("radix_sort", CLASS_MEMBW),
    ("devicescan", CLASS_MEMBW),
    ("device_scan", CLASS_MEMBW),
    ("devicereduce", CLASS_MEMBW),
    ("devicememcpy", CLASS_MEMBW),
    ("memset", CLASS_MEMBW),
    ("partition", CLASS_MEMBW),
    ("merge", CLASS_MEMBW),
    # hash join/groupby probe+build: latency/membw mixed
    ("hash", CLASS_MIXED),
    ("join", CLASS_MIXED),
    ("groupby", CLASS_MIXED),
    ("aggregate", CLASS_MIXED),
]

# gpu-metrics (Tier B) thresholds, interpreted as percent-of-peak DRAM
# throughput averaged over the kernel interval. VERIFY on the first Tier B
# capture (metric value semantics were not checkable without a run).
METRICS_MEMBW_PCT = 40.0
METRICS_COMPUTE_PCT = 15.0


class Classifier:
    def __init__(
        self,
        overrides: Optional[Dict[str, str]] = None,
        priors: Optional[List[Tuple[str, str]]] = None,
    ) -> None:
        self.overrides: Dict[str, str] = {}
        for name, cls in (overrides or {}).items():
            if cls not in VALID_CLASSES:
                raise ValueError(
                    f"override for {name!r}: invalid class {cls!r}; "
                    f"valid: {', '.join(VALID_CLASSES)}"
                )
            self.overrides[name] = cls
        self._overrides_lower = {k.lower(): v for k, v in self.overrides.items()}
        self.priors = priors if priors is not None else NAME_PRIORS
        # kernel name -> mean pct-of-peak DRAM throughput (exclusive intervals)
        self.metrics_pct: Dict[str, float] = {}

    # -- Tier B calibration -------------------------------------------------

    def calibrate_from_metrics(
        self,
        kernel_rows: Iterable,  # objects with .name, .start, .end
        dram_samples: List[Tuple[float, float]],  # (timestamp, pct_of_peak)
    ) -> int:
        """Classify by integrating device-wide DRAM throughput over kernel
        intervals. Restricted to kernels that own the device exclusively
        (nsys-extraction.md section 5.1 caveat); returns #names calibrated."""
        rows = sorted(kernel_rows, key=lambda k: k.start)
        if not rows or not dram_samples:
            return 0
        samples = sorted(dram_samples)
        acc: Dict[str, List[float]] = {}
        for i, k in enumerate(rows):
            exclusive = True
            if i > 0 and rows[i - 1].end > k.start:
                exclusive = False
            if i + 1 < len(rows) and rows[i + 1].start < k.end:
                exclusive = False
            if not exclusive:
                continue
            vals = [v for (t, v) in samples if k.start <= t < k.end]
            if vals:
                acc.setdefault(k.name, []).append(sum(vals) / len(vals))
        for name, means in acc.items():
            self.metrics_pct[name] = sum(means) / len(means)
        return len(acc)

    # -- classification -----------------------------------------------------

    def classify(self, kernel_name: str) -> str:
        low = kernel_name.lower()
        # 1. overrides (exact, then substring)
        cls = self._overrides_lower.get(low)
        if cls:
            return cls
        for pat, cls in self._overrides_lower.items():
            if pat in low:
                return cls
        # 2. gpu-metrics calibration
        pct = self.metrics_pct.get(kernel_name)
        if pct is not None:
            if pct >= METRICS_MEMBW_PCT:
                return CLASS_MEMBW
            if pct <= METRICS_COMPUTE_PCT:
                return CLASS_COMPUTE
            return CLASS_MIXED
        # 3. name priors
        for pat, cls in self.priors:
            if pat in low:
                return cls
        return CLASS_UNKNOWN
