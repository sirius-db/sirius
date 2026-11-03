#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Build a test-only probe against an explicitly selected existing Sirius build."""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess


def digest(p):
    h = hashlib.sha256()
    with open(p, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--engine-root", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    a = p.parse_args()
    root = a.engine_root.resolve(strict=True)
    build = root / "build/release"
    out = a.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    source = Path(__file__).with_name("probe.cpp").resolve(strict=True)

    def git(*args):
        return subprocess.check_output(
            ["git", "-C", str(root), *args], text=True
        ).strip()

    if git("status", "--porcelain", "--", "src", "pixi.lock", "pixi.toml"):
        raise RuntimeError(
            "engine inputs are dirty; commit and bind candidate before building"
        )
    entries = json.loads((build / "compile_commands.json").read_text())
    entry = next(
        x for x in entries if x["file"].endswith("/test/io/s3_throughput_test.cpp")
    )
    original = shlex.split(entry["command"])
    args = []
    i = 0
    while i < len(original):
        word = original[i]
        if word in ("-o", "-MF", "-MT", "-MQ"):
            i += 2
            continue
        if word in ("-MD", "-MMD"):
            i += 1
            continue
        if word.endswith("/sccache") or word == "sccache":
            i += 1
            continue
        args.append(str(source) if word == entry["file"] else word)
        i += 1
    obj = out / "probe.o"
    args += ["-o", str(obj)]
    ninja = root / ".pixi/envs/default/bin/ninja"
    commands = subprocess.check_output(
        [str(ninja), "-C", str(build), "-t", "commands", "s3_throughput_test"],
        text=True,
    ).splitlines()
    line = commands[-1]
    if not line.startswith(": && ") or not line.endswith(" && :"):
        raise RuntimeError("unexpected CMake linker command wrapper")
    tokens = shlex.split(line[5:-5])
    link = []
    i = 0
    while i < len(tokens):
        word = tokens[i]
        if word == "-o":
            link += ["-o", str(out / "sirius_rest_probe")]
            i += 2
            continue
        if word.endswith(".o") or word.startswith("-Wl,--dependency-file="):
            i += 1
            continue
        link.append(word)
        i += 1
    link.insert(1, str(obj))
    libraries = []
    for word in link:
        candidate = Path(word) if word.startswith("/") else build / word
        if (
            not word.startswith("-")
            and candidate.is_file()
            and (".so" in candidate.name or candidate.suffix == ".a")
        ):
            libraries.append(
                dict(path=str(candidate.resolve()), sha256=digest(candidate))
            )
    record = dict(
        engine_commit=git("rev-parse", "HEAD"),
        engine_tree=git("rev-parse", "HEAD^{tree}"),
        engine_root=str(root),
        lock_sha256=digest(root / "pixi.lock"),
        probe_source_sha256=digest(source),
        compile=args,
        compile_cwd=entry["directory"],
        link=link,
        link_cwd=str(build),
        libraries=libraries,
        provenance="Reuses the explicitly selected existing engine build; does not rebuild it. Validate its prior build receipt separately.",
    )
    for name, cmd, cwd in [
        ("compile", args, entry["directory"]),
        ("link", link, build),
    ]:
        with (out / (name + ".stdout")).open("w") as stdout, (
            out / (name + ".stderr")
        ).open("w") as stderr:
            result = subprocess.run(
                cmd, cwd=cwd, stdout=stdout, stderr=stderr, timeout=180
            )
        record[name + "_exit"] = result.returncode
        (out / "build-manifest.json").write_text(json.dumps(record, indent=2) + "\n")
        result.check_returncode()
    record["binary_sha256"] = digest(out / "sirius_rest_probe")
    (out / "build-manifest.json").write_text(json.dumps(record, indent=2) + "\n")
    print(record["binary_sha256"])


if __name__ == "__main__":
    main()
