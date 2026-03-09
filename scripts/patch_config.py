#!/usr/bin/env python3
"""Patch a libconfig file with key=value overrides.

Backs up the original file before modifying it. Supports adding new keys
into existing groups, creating intermediate groups as needed.

Usage:
    python patch_config.py sirius.cfg \\
        --opt sirius.executor.pipeline.num_threads=4 \\
        --opt sirius.executor.duckdb_scan.cache=true \\
        --opt sirius.operator_params.scan_task_batch_size=536870912
"""

import argparse
import re
import shutil
from datetime import datetime
from pathlib import Path


def format_value(value: str) -> str:
    """Return the value as it should appear in the config (no quotes for booleans/numbers)."""
    v = value.strip()
    if v in ("true", "false"):
        return v
    try:
        int(v)
        return v
    except ValueError:
        pass
    try:
        float(v)
        return v
    except ValueError:
        pass
    # String: add quotes if missing
    if not (v.startswith('"') and v.endswith('"')):
        return f'"{v}"'
    return v


def strip_comment(line: str) -> str:
    """Return the line with inline // comment removed, stripped."""
    return re.sub(r"//[^\n]*", "", line).strip()


def scan(lines: list[str]) -> tuple[dict, dict, dict]:
    """
    Scan lines and return:
      path_map:    dot-path -> line index of the scalar assignment
      group_open:  dot-path -> line index of the 'name = {' line
      group_close: dot-path -> line index of the '};' or '}' closing line
    """
    path_map: dict[str, int] = {}
    group_open: dict[str, int] = {}
    group_close: dict[str, int] = {}
    stack: list[tuple[str, int]] = []  # (name, line_idx)

    for i, raw in enumerate(lines):
        code = strip_comment(raw)
        if not code:
            continue

        # Group opening: word = {
        m = re.match(r"^(\w+)\s*=\s*\{$", code)
        if m:
            stack.append((m.group(1), i))
            path = ".".join(n for n, _ in stack)
            group_open[path] = i
            continue

        # Group closing: }; or }
        if code in ("};", "}"):
            if stack:
                path = ".".join(n for n, _ in stack)
                group_close[path] = i
                stack.pop()
            continue

        # Scalar: word = value;
        m = re.match(r"^(\w+)\s*=\s*(.+);$", code)
        if m and not m.group(2).strip().startswith("{"):
            path = ".".join([n for n, _ in stack] + [m.group(1)])
            path_map[path] = i

    return path_map, group_open, group_close


def group_content_indent(lines: list[str], open_idx: int, close_idx: int) -> str:
    """Determine the indentation string used for content inside a group."""
    for i in range(open_idx + 1, close_idx):
        stripped = lines[i].lstrip()
        if stripped and not stripped.startswith("//"):
            return lines[i][: len(lines[i]) - len(stripped)]
    # Fallback: opening line indent + 4 spaces
    open_line = lines[open_idx]
    base = open_line[: len(open_line) - len(open_line.lstrip())]
    return base + "    "


def apply_one(lines: list[str], dot_path: str, value: str) -> list[str]:
    """Apply a single dot-path=value change to lines, returning updated lines."""
    path_map, group_open, group_close = scan(lines)
    fval = format_value(value)
    key_name = dot_path.split(".")[-1]

    if dot_path in path_map:
        # Modify existing scalar in-place, preserving inline comment
        idx = path_map[dot_path]
        line = lines[idx]
        # Replace value between '=' and ';', keeping everything else
        new_line = re.sub(
            r"(\b" + re.escape(key_name) + r"\s*=\s*)([^;]+)(;)",
            lambda m: m.group(1) + fval + m.group(3),
            line,
        )
        lines[idx] = new_line
        return lines

    # Need to insert. Find deepest existing ancestor group.
    parts = dot_path.split(".")
    parent_parts = parts[:-1]

    existing_depth = 0
    for depth in range(len(parent_parts), 0, -1):
        candidate = ".".join(parts[:depth])
        if candidate in group_open and candidate in group_close:
            existing_depth = depth
            break

    if existing_depth == 0:
        raise ValueError(
            f"Cannot find any existing ancestor group for path: {dot_path}\n"
            f"Known groups: {sorted(group_open)}"
        )

    parent_path = ".".join(parts[:existing_depth])
    insert_at = group_close[parent_path]  # insert before the closing };

    indent = group_content_indent(
        lines, group_open[parent_path], group_close[parent_path]
    )

    # Build lines to insert
    new_groups = parts[existing_depth:-1]  # intermediate groups to create
    new_lines: list[str] = []

    for depth, gname in enumerate(new_groups):
        new_lines.append(f"{indent}{'    ' * depth}{gname} = {{\n")

    key_indent = indent + "    " * len(new_groups)
    new_lines.append(f"{key_indent}{key_name} = {fval};\n")

    for depth in range(len(new_groups) - 1, -1, -1):
        new_lines.append(f"{indent}{'    ' * depth}}};\n")

    lines[insert_at:insert_at] = new_lines
    return lines


def main():
    parser = argparse.ArgumentParser(
        description="Patch a libconfig file with --opt KEY=VALUE overrides.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("config", type=Path, help="Path to the libconfig .cfg file")
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

    # Apply changes
    lines = config_path.read_text().splitlines(keepends=True)
    if lines and not lines[-1].endswith("\n"):
        lines[-1] += "\n"

    for dot_path, value in opts:
        lines = apply_one(lines, dot_path, value)

    config_path.write_text("".join(lines))

    print(f"Patched {config_path} with {len(opts)} change(s):")
    for dot_path, value in opts:
        print(f"  {dot_path} = {format_value(value)}")


if __name__ == "__main__":
    main()
