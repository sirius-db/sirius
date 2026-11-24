# Copyright 2026, Sirius Contributors. SPDX-License-Identifier: Apache-2.0
"""Verify native S3 connections use kernel TLS for every benchmark worker."""
from datetime import datetime, timezone
from pathlib import Path
import json, platform


def snapshot():
    assert Path("/sys/module/tls").is_dir(), "Kernel TLS module is not loaded"
    values = {}
    for line in Path("/proc/net/tls_stat").read_text().splitlines():
        key, value = line.split()
        values[key] = int(value)
    return {
        "module_loaded": True,
        "kernel": platform.release(),
        "utc": datetime.now(timezone.utc).isoformat(),
        "counters": values,
    }


def validate(worker, before, after):
    b, a = before["counters"], after["counters"]
    delta = {k: a[k] - b.get(k, 0) for k in a if not k.startswith("TlsCurr")}
    tx = delta.get("TlsTxSw", 0) + delta.get("TlsTxDevice", 0)
    rx = delta.get("TlsRxSw", 0) + delta.get("TlsRxDevice", 0)
    logs = list(worker.glob("benchmark/*/log_dir/*.log"))
    fallback = []
    for path in logs:
        for line in path.read_text(errors="replace").splitlines():
            if "kTLS not engaged" in line or "kTLS disabled by policy" in line:
                fallback.append(line)
    proof = {
        "correct": bool(
            before["module_loaded"]
            and after["module_loaded"]
            and tx > 0
            and rx > 0
            and logs
            and not fallback
        ),
        "new_kernel_tls_tx_connections": tx,
        "new_kernel_tls_rx_connections": rx,
        "counter_deltas": delta,
        "native_logs_checked": len(logs),
        "user_space_fallback_messages": fallback,
        "before": before,
        "after": after,
    }
    (worker / "ktls-verification.json").write_text(json.dumps(proof, indent=2) + "\n")
    return proof
