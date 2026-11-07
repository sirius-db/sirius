#!/usr/bin/env python3
"""Prints the distributed: section of sirius.yaml as shell defaults, for the launch scripts to eval.

Looks for the file where Sirius does: $SIRIUS_CONFIG_FILE, ./sirius.yaml, ~/.sirius/sirius.yaml.
Each line is `export VAR=${VAR:-value}`, so a variable the user exported wins over the file. The
value is shell-quoted, so it cannot run code when eval'd. On any error it prints nothing to stdout
and exits non-zero: the scripts stop instead of running with half the file applied.
"""

import os
import shlex
import sys
from pathlib import Path

# Key path in sirius.distributed -> the variable the scripts read.
KEYS = {
    ("cluster",): "SIRIUS_CLUSTER",
    ("gateway", "port"): "SIRIUS_GATEWAY_PORT",
    ("gateway", "host"): "SIRIUS_GATEWAY_HOST",
    ("gateway", "networks"): "SIRIUS_GATEWAY_PRIORITY_NETWORKS",
    ("gateway", "min_ranks"): "SIRIUS_GATEWAY_MIN_RANKS",
    ("gateway", "join_wait"): "SIRIUS_GATEWAY_JOIN_WAIT",
    ("node", "gpus"): "SIRIUS_NODE_GPUS",
    ("node", "networks"): "SIRIUS_NODE_NETWORKS",
    ("node", "quack_port"): "SIRIUS_NODE_QUACK_PORT",
    ("node", "exchange_port"): "SIRIUS_NODE_EXCHANGE_PORT",
    ("exchange", "mnnvl"): "SIRIUS_GATEWAY_EXCHANGE_MNNVL",
    ("exchange", "frame_bytes"): "SIRIUS_GATEWAY_EXCHANGE_FRAME_BYTES",
    ("exchange", "send_frames"): "SIRIUS_GATEWAY_EXCHANGE_SEND_FRAMES",
    (
        "exchange",
        "recv_frames_per_peer",
    ): "SIRIUS_GATEWAY_EXCHANGE_RECV_FRAMES_PER_PEER",
    ("ucx", "tls"): "UCX_TLS",
    ("ucx", "net_devices"): "UCX_NET_DEVICES",
}


def config_path():
    for candidate in (
        os.environ.get("SIRIUS_CONFIG_FILE"),
        "sirius.yaml",
        "~/.sirius/sirius.yaml",
    ):
        if candidate and Path(candidate).expanduser().is_file():
            return Path(candidate).expanduser()
    return None


def flatten(node, prefix=()):
    for key, value in node.items():
        if isinstance(value, dict):
            yield from flatten(value, prefix + (key,))
        else:
            yield prefix + (key,), value


def scalar(value):
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (str, int, float)):
        return str(value)
    return None


def shell_value(value):
    if isinstance(value, list):  # gpus: [0, 1]
        items = [scalar(v) for v in value]
        return None if None in items else ",".join(items)
    return scalar(value)


def main():
    path = config_path()
    if path is None:
        return 0
    try:
        import yaml
    except ImportError:
        sys.exit(f"reading {path} needs PyYAML: run the script through pixi run")
    try:
        doc = yaml.safe_load(path.read_text()) or {}
    except yaml.YAMLError as error:
        sys.exit(f"{path}: {error}")
    sirius = doc.get("sirius") if isinstance(doc, dict) else None
    section = (sirius.get("distributed") if isinstance(sirius, dict) else None) or {}
    if not isinstance(section, dict):
        sys.exit(f"{path}: sirius.distributed must be a mapping")
    # Check every key before printing any line.
    lines = []
    for keys, value in flatten(section):
        name = "sirius.distributed." + ".".join(str(k) for k in keys)
        var = KEYS.get(keys)
        if var is None:
            known = ", ".join(".".join(k) for k in KEYS)
            sys.exit(f"{path}: unknown key {name} (known: {known})")
        text = shell_value(value)
        if text is None:
            sys.exit(f"{path}: {name} must be a string, a number or a list of them")
        lines.append(f"export {var}=${{{var}:-{shlex.quote(text)}}}")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
