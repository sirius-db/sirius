"""hwsim.physics — nsys physics join (v1, gap G4).

Ingests a paired nsys sqlite export (schema per tools/hwsim/docs/nsys-extraction.md,
nsys 2025.6.3) and produces per-task physics annotations for a Quent trace:
which share of each traced span was compute-bound kernel time, membw-bound
kernel time, explicit transfer time, or host-side glue. This enables honest
split knob semantics:

- ``gpu_compute`` scales only kernel-busy compute-bound portions,
- ``gpu_mem_bandwidth`` scales membw-bound portions (coupled to gpu_compute
  below the SM-issue line, see ``laws.py``),
- transfers are re-timed through the measured per-size bandwidth curve, with
  the Grace C2C/host-DRAM co-limit applied.

Unmatched GPU time degrades gracefully to the v0 conflated behavior (spans /
min(gpu_compute, gpu_mem_bandwidth)) with a warning — never silently dropped.

Design doc: tools/hwsim/docs/nsys-join.md.
"""

from .schema import PhysicsProfile  # noqa: F401
from .ingest import ingest_nsys  # noqa: F401
from .join import join_graph  # noqa: F401
from .integrate import simulate_with_physics  # noqa: F401
