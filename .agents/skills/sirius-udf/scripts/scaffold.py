#!/usr/bin/env python3
# Copyright 2026, Sirius Contributors. SPDX-License-Identifier: Apache-2.0
"""Create an external, synthetic UDF project without copying user source."""

import argparse
import json
from pathlib import Path


TARGETS = ("sql", "cudf", "cuda")
SKILL_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = (
    SKILL_ROOT.parent.parent.parent
    if SKILL_ROOT.parent.name == "skills" and SKILL_ROOT.parent.parent.name == ".agents"
    else None
)


def external_output(path, repository_root=PROJECT_ROOT):
    """Resolve symlinks and reject existing or Sirius-local destinations."""
    path = Path(path)
    if not path.is_absolute():
        raise ValueError("--output must be an absolute path")
    if path.exists() or path.is_symlink():
        raise ValueError("output already exists; choose a new directory")
    resolved = path.resolve()
    if not resolved.parent.is_dir():
        raise ValueError("output parent must already exist")
    if repository_root is not None:
        repository_root = Path(repository_root).resolve()
        if resolved == repository_root or repository_root in resolved.parents:
            raise ValueError("output must be outside the Sirius repository")
    return resolved


def scaffold(target, output, assets=None, repository_root=PROJECT_ROOT):
    if target not in TARGETS:
        raise ValueError(f"unknown target: {target}")
    destination = external_output(output, repository_root)
    assets = Path(assets) if assets is not None else SKILL_ROOT / "assets"
    trees = [assets / "common", assets / target]
    if target != "sql":
        trees.insert(1, assets / "native")
    # Read and validate everything before creating the destination.
    files = {}
    for tree in trees:
        if not tree.is_dir():
            raise ValueError(f"missing template directory: {tree}")
        for source in sorted(tree.rglob("*")):
            if source.is_file():
                relative = source.relative_to(tree)
                if "__pycache__" in relative.parts or source.suffix in (".pyc", ".pyo"):
                    continue
                if relative in files:
                    raise ValueError(f"duplicate template output: {relative}")
                files[relative] = source.read_text(encoding="utf-8")
    destination.mkdir(
        mode=0o700
    )  # Exclusive creation protects against concurrent runs.
    for relative, content in files.items():
        result = destination / relative
        result.parent.mkdir(parents=True, exist_ok=True)
        result.write_text(content, encoding="utf-8")
    (destination / "project.json").write_text(
        json.dumps(
            {"target": target, "synthetic": True, "reference": "example contract"},
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return destination


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", choices=TARGETS, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        destination = scaffold(args.target, args.output)
    except (ValueError, OSError) as error:
        parser.exit(1, f"scaffold: {error}\n")
    print(f"Created synthetic {args.target} project: {destination}")
    print(f"Build and validation instructions: {destination / 'USAGE.md'}")


if __name__ == "__main__":
    main()
