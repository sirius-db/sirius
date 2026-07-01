#!/usr/bin/env python3
"""Patch a YAML config file with key=value overrides.

Backs up the original file before modifying it. Supports adding new keys
into existing groups, creating intermediate groups as needed.

Usage:
    python patch_config.py sirius.yaml \\
        --opt sirius.executor.pipeline.num_threads=4 \\
        --opt sirius.operator_params.scan_task_batch_size=536870912
"""

import argparse
import shutil
from datetime import datetime
from pathlib import Path

import yaml


def parse_value(value: str):
    """Parse a string value into the appropriate Python type."""
    v = value.strip()
    if v in ("true", "True"):
        return True
    if v in ("false", "False"):
        return False
    try:
        return int(v)
    except ValueError:
        pass
    try:
        return float(v)
    except ValueError:
        pass
    # Strip surrounding quotes if present
    if len(v) >= 2 and v[0] == '"' and v[-1] == '"':
        return v[1:-1]
    return v


def set_nested(data: dict, dot_path: str, value):
    """Set a value in a nested dict using a dot-separated path."""
    parts = dot_path.split(".")
    current = data
    for i, part in enumerate(parts[:-1]):
        if part in current and not isinstance(current[part], dict):
            ancestor = ".".join(parts[: i + 1])
            raise ValueError(
                f"Cannot set '{dot_path}': ancestor '{ancestor}' is a "
                f"{type(current[part]).__name__}, not a mapping"
            )
        if part not in current:
            current[part] = {}
        current = current[part]
    current[parts[-1]] = value


def main():
    parser = argparse.ArgumentParser(
        description="Patch a YAML config file with --opt KEY=VALUE overrides.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("config", type=Path, help="Path to the YAML config file")
    parser.add_argument(
        "--opt",
        action="append",
        metavar="KEY=VALUE",
        default=[],
        help="Override a config key (dot-separated path). Can be repeated.",
    )
    args = parser.parse_args()

    config_path: Path = args.config.resolve()
    if not config_path.exists():
        parser.error(f"Config file not found: {config_path}")

    if not args.opt:
        print("No --opt arguments provided. Nothing to do.")
        return

    # Parse opts
    opts: list[tuple[str, str]] = []
    for opt in args.opt:
        if "=" not in opt:
            parser.error(f"Invalid --opt format (expected KEY=VALUE): {opt!r}")
        key, _, val = opt.partition("=")
        opts.append((key.strip(), val.strip()))

    # Backup
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = config_path.with_suffix(f".{timestamp}.bak")
    shutil.copy2(config_path, backup_path)
    print(f"Backup: {backup_path}")

    # Load, apply, save
    with open(config_path) as f:
        data = yaml.safe_load(f) or {}

    for dot_path, value in opts:
        set_nested(data, dot_path, parse_value(value))

    with open(config_path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    print(f"Patched {config_path} with {len(opts)} change(s):")
    for dot_path, value in opts:
        print(f"  {dot_path} = {value}")


if __name__ == "__main__":
    main()
