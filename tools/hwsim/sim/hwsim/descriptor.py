"""Hardware descriptor files — spec-sheet target mode (WS19).

A descriptor is a small YAML file of ADVERTISED spec-sheet values for one
machine, plus an optional ``measured:`` block of achieved values for boxes we
have benchmarked with the throttle kit. Library lives in
``tools/hwsim/hw-descriptors/``; schema reference and every anchor number's
source: ``tools/hwsim/docs/spec-sheet-mode.md``.

hwsim is pure-stdlib, so descriptors are parsed with a strict YAML *subset*
parser (:func:`parse_simple_yaml`): nested maps by indentation, scalar values
(int / float / bool / quoted or bare strings), ``#`` comments. No lists, no
anchors, no multi-line strings — loading a file outside the subset raises
``DescriptorError`` with a line number.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


class DescriptorError(ValueError):
    pass


# ---------------------------------------------------------------------------
# Strict YAML-subset parser (stdlib-only)
# ---------------------------------------------------------------------------


def _parse_scalar(raw: str):
    s = raw.strip()
    if s.startswith('"') and s.endswith('"') and len(s) >= 2:
        return s[1:-1]
    if s.startswith("'") and s.endswith("'") and len(s) >= 2:
        return s[1:-1]
    low = s.lower()
    if low in ("true", "yes", "on"):
        return True
    if low in ("false", "no", "off"):
        return False
    if low in ("null", "none", "~", ""):
        return None
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        pass
    return s


def _strip_comment(line: str) -> str:
    """Remove a trailing ``#`` comment (respecting simple quoting)."""
    out = []
    quote = None
    for ch in line:
        if quote:
            out.append(ch)
            if ch == quote:
                quote = None
            continue
        if ch in ("'", '"'):
            quote = ch
            out.append(ch)
            continue
        if ch == "#":
            break
        out.append(ch)
    return "".join(out).rstrip()


def parse_simple_yaml(text: str, origin: str = "<string>") -> Dict[str, Any]:
    """Parse the YAML subset used by hw descriptors into nested dicts."""
    root: Dict[str, Any] = {}
    # stack of (indent, dict)
    stack = [(-1, root)]
    pending: Optional[tuple] = None  # (indent, key, dict-to-attach-into)
    for lineno, rawline in enumerate(text.splitlines(), start=1):
        line = _strip_comment(rawline)
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        if "\t" in line[:indent] or line.lstrip(" ").startswith("- "):
            raise DescriptorError(
                f"{origin}:{lineno}: outside the supported YAML subset "
                "(tabs / lists are not supported)"
            )
        body = line.strip()
        if ":" not in body:
            raise DescriptorError(f"{origin}:{lineno}: expected 'key: value'")
        key, _, val = body.partition(":")
        key = key.strip()
        # resolve the containing dict for this indent
        if pending is not None:
            p_indent, p_key, p_parent = pending
            if indent > p_indent:
                child: Dict[str, Any] = {}
                p_parent[p_key] = child
                stack.append((p_indent, child))
                pending = None
            else:
                p_parent[p_key] = None  # empty section
                pending = None
        while stack and indent <= stack[-1][0]:
            stack.pop()
        if not stack:
            raise DescriptorError(f"{origin}:{lineno}: bad indentation")
        parent = stack[-1][1]
        if val.strip() == "":
            pending = (indent, key, parent)
        else:
            parent[key] = _parse_scalar(val)
    if pending is not None:
        pending[2][pending[1]] = None
    return root


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

# PCIe payload-rate table, GB/s per lane per direction (128b/130b encoded
# line rate; gen5 x16 = 64 GB/s, matching the RTX PRO 6000 anchor basis).
PCIE_LANE_GBS = {3: 1.0, 4: 2.0, 5: 4.0, 6: 8.0}

LINK_TYPES = ("pcie", "c2c", "nvlink")


@dataclass
class GpuDesc:
    name: str = ""
    sm_count: Optional[int] = None
    boost_clock_mhz: Optional[float] = None
    fp32_tflops_peak: Optional[float] = None  # advertised dense FP32
    mem_class: str = ""  # hbm3e | gddr7 | hbm3 | hbm2e | gddr6 | ...
    mem_bandwidth_gbs_peak: Optional[float] = None
    vram_gb: Optional[float] = None
    l2_mb: Optional[float] = None

    def theoretical_fp32_tflops(self) -> Optional[float]:
        """128 FP32 lanes/SM x 2 FLOP/FMA x clock — the Volta+ datasheet
        formula (A100 is 64 lanes/SM: its fp32_tflops_peak field catches
        that; the SMs x clock cross-check warns)."""
        if self.sm_count and self.boost_clock_mhz:
            return self.sm_count * 128 * 2 * self.boost_clock_mhz * 1e6 / 1e12
        return None


@dataclass
class LinkDesc:
    type: str = ""  # pcie | c2c | nvlink
    gen: Optional[int] = None
    lanes: Optional[int] = None
    gbs_peak_per_dir: Optional[float] = None
    # Optional explicit platform-law override; None => derived from type
    # (c2c => Grace DRAM co-limit ON, else OFF). See physics/laws.py law 2.
    grace_colimit: Optional[bool] = None

    def peak_gbs(self) -> Optional[float]:
        if self.gbs_peak_per_dir is not None:
            return float(self.gbs_peak_per_dir)
        if self.type == "pcie" and self.gen in PCIE_LANE_GBS and self.lanes:
            return PCIE_LANE_GBS[self.gen] * self.lanes
        return None


@dataclass
class CpuDesc:
    arch: str = ""
    cores: Optional[int] = None
    base_clock_ghz: Optional[float] = None
    boost_clock_ghz: Optional[float] = None
    dram_channels: Optional[int] = None
    dram_speed_mts: Optional[float] = None
    dram_gbs_peak: Optional[float] = None

    def dram_peak_gbs(self) -> Optional[float]:
        if self.dram_gbs_peak is not None:
            return float(self.dram_gbs_peak)
        if self.dram_channels and self.dram_speed_mts:
            return self.dram_channels * self.dram_speed_mts * 8.0 / 1000.0
        return None

    def engine(self) -> Optional[float]:
        """cores x clock (GHz); clock falls back boost -> base -> None."""
        clk = self.boost_clock_ghz or self.base_clock_ghz
        if self.cores and clk:
            return self.cores * clk
        return None


@dataclass
class StorageDesc:
    seq_read_gbs: Optional[float] = None  # advertised sequential read


@dataclass
class MeasuredDesc:
    """Achieved values measured with the hwsim throttle kit (same conventions
    on every box: fma victim TFLOP/s; CE D2D copy at 2x-bytes traffic
    accounting; pinned H2D loop per direction; 8-thread host memcpy at
    2x-bytes; 8-stream O_DIRECT sequential read)."""

    fp32_tflops: Optional[float] = None
    gpu_mem_gbs: Optional[float] = None
    link_h2d_gbs: Optional[float] = None
    link_d2h_gbs: Optional[float] = None
    cpu_dram_gbs: Optional[float] = None
    storage_seq_read_gbs: Optional[float] = None
    vram_usable_gb: Optional[float] = None


@dataclass
class HwDescriptor:
    name: str = ""
    gpu: GpuDesc = field(default_factory=GpuDesc)
    link: LinkDesc = field(default_factory=LinkDesc)
    cpu: CpuDesc = field(default_factory=CpuDesc)
    storage: StorageDesc = field(default_factory=StorageDesc)
    measured: MeasuredDesc = field(default_factory=MeasuredDesc)
    path: str = ""


def _fill(obj, d: Optional[Dict[str, Any]], origin: str, section: str):
    if d is None:
        return obj
    if not isinstance(d, dict):
        raise DescriptorError(f"{origin}: section {section!r} must be a map")
    known = set(obj.__dataclass_fields__)
    for k, v in d.items():
        if k not in known:
            raise DescriptorError(
                f"{origin}: unknown key {section}.{k} "
                f"(known: {', '.join(sorted(known))})"
            )
        setattr(obj, k, v)
    return obj


def load_descriptor(path: str) -> HwDescriptor:
    with open(path) as f:
        raw = parse_simple_yaml(f.read(), origin=path)
    known_sections = {"name", "gpu", "link", "cpu", "storage", "measured"}
    unknown = set(raw) - known_sections
    if unknown:
        raise DescriptorError(
            f"{path}: unknown top-level section(s): {', '.join(sorted(unknown))}"
        )
    desc = HwDescriptor(path=path)
    desc.name = str(raw.get("name") or os.path.splitext(os.path.basename(path))[0])
    _fill(desc.gpu, raw.get("gpu"), path, "gpu")
    _fill(desc.link, raw.get("link"), path, "link")
    _fill(desc.cpu, raw.get("cpu"), path, "cpu")
    _fill(desc.storage, raw.get("storage"), path, "storage")
    _fill(desc.measured, raw.get("measured"), path, "measured")
    if desc.link.type and desc.link.type not in LINK_TYPES:
        raise DescriptorError(
            f"{path}: link.type must be one of {LINK_TYPES}, " f"got {desc.link.type!r}"
        )
    return desc
