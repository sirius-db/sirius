"""Exercise the production embedders with content changes and incremental builds."""

import argparse
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import unittest

SOURCE = Path(__file__).resolve().parents[2]
CMAKE = "cmake"
XXHSUM = "xxhsum"
IDENTITY_TOOL = None
ROOTS = (
    "cub/block/block_reduce.cuh",
    "cub/block/block_scan.cuh",
    "cub/block/block_exchange.cuh",
    "cuda/std/cstdint",
    "cuda/std/cstddef",
    "cuda/std/climits",
    "cuda/std/type_traits",
)
HEADERS = (
    *ROOTS,
    "cub/version.cuh",
    "thrust/version.h",
    "cub/detail/unreferenced.cuh",
    "thrust/detail/unreferenced.h",
    "cuda/std/__detail/unreferenced",
    "nv/target",
)


def run(*args):
    result = subprocess.run(args, text=True, capture_output=True)
    if result.returncode:
        raise AssertionError(result.stdout + result.stderr)
    return result.stdout


def write(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def fingerprint(path, symbol):
    match = re.search(rf'{symbol}\[\] = "([0-9a-f]+)"', path.read_text())
    if match is None:
        raise AssertionError(f"missing generated identity: {symbol}")
    return match.group(1)


def bundle_bytes(headers):
    records = b"simpatico-headers-v2\n"
    for name, content in sorted(headers.items()):
        normalized = content.replace("\r\n", "\n").replace("\r", "\n").encode()
        records += (
            f"{len(name.encode())}:{name}{len(normalized)}:".encode() + normalized
        )
    return records


def checksum(data):
    result = subprocess.run(
        [XXHSUM, "-H128"], input=data, capture_output=True, check=True
    )
    return result.stdout.decode().split()[0]


def manifest(headers):
    return checksum(bundle_bytes(headers))


class HeaderManifests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="simpatico-headers-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.headers = self.root / "headers"
        for name in HEADERS:
            write(self.headers / name, "// fixture 1\n")
        self.include_list = self.root / "includes.txt"
        self.output = self.root / "generated" / "headers.cpp"
        self.output.parent.mkdir()

    def cccl(self, roots=None):
        roots = roots or [self.headers]
        write(self.include_list, "".join(f"{root}\n" for root in roots))
        run(
            CMAKE,
            f"-DINCLUDE_DIRS_FILE={self.include_list}",
            f"-DOUT={self.output}",
            f"-DXXHSUM_EXECUTABLE={XXHSUM}",
            "-P",
            str(SOURCE / "cmake/embed_cccl_headers.cmake"),
        )
        return fingerprint(self.output, "kCcclEmbeddedHeadersIdentity")

    def test_cccl_content_and_prefix(self):
        original = self.cccl()
        self.assertEqual(
            original, manifest({name: "// fixture 1\n" for name in HEADERS})
        )
        self.assertEqual(len(original), 32)
        copy = self.root / "different-prefix"
        shutil.copytree(self.headers, copy)
        self.assertEqual(original, self.cccl([copy]))
        write(copy / ROOTS[0], "// fixture 2\n")
        self.assertNotEqual(original, self.cccl([copy]))
        self.assertEqual(original, self.cccl([self.headers, copy]))
        self.assertNotEqual(original, self.cccl([copy, self.headers]))

    def test_disjoint_root_order_and_full_inventory(self):
        other = self.root / "other"
        other.mkdir()
        shutil.move(self.headers / "cuda", other / "cuda")
        original = self.cccl([self.headers, other])
        self.assertEqual(original, self.cccl([other, self.headers]))
        # An unreferenced header still belongs to the bundle.
        write(self.headers / "cub/detail/new.cuh", "// new 1\n")
        added = self.cccl([self.headers, other])
        self.assertNotEqual(original, added)
        write(self.headers / "cub/detail/new.cuh", "// new 2\n")
        self.assertNotEqual(added, self.cccl([self.headers, other]))

    def test_names_additions_removals_and_allow_list(self):
        original = self.cccl()
        write(self.headers / "unapproved/random.h", "// not a CCCL header\n")
        self.assertEqual(original, self.cccl())
        added_header = self.headers / "thrust/detail/another.h"
        write(added_header, "// unused\n")
        added = self.cccl()
        self.assertNotEqual(original, added)
        renamed = added_header.with_name("renamed.h")
        added_header.rename(renamed)
        self.assertNotEqual(added, self.cccl())
        renamed.unlink()
        self.assertEqual(original, self.cccl())

    def test_normalized_embedded_bytes(self):
        header = self.headers / "cub/detail/unreferenced.cuh"
        content = "// unicode π: ; \\\nint value;\n"
        header.write_bytes(content.replace("\n", "\r\n").encode())
        crlf = self.cccl()
        self.assertIn(content, self.output.read_text())
        self.assertNotIn(b"\r", self.output.read_bytes())
        header.write_bytes(content.replace("\n", "\r").encode())
        self.assertEqual(crlf, self.cccl())
        header.write_bytes(content.encode())
        self.assertEqual(crlf, self.cccl())
        headers = {name: "// fixture 1\n" for name in HEADERS}
        headers["cub/detail/unreferenced.cuh"] = content
        self.assertEqual(crlf, manifest(headers))

    def test_runtime_cli_equivalence(self):
        path = self.root / "hash input.bin"
        for content in (
            b"",
            b"abc",
            bytes(range(256)) * 1027,
            bundle_bytes({name: "// fixture 1\n" for name in HEADERS}),
        ):
            path.write_bytes(content)
            self.assertEqual(
                checksum(content), run(IDENTITY_TOOL, "--hash-file", str(path)).strip()
            )
        for missing in (self.root / "absent", self.root):
            result = subprocess.run(
                [IDENTITY_TOOL, "--hash-file", str(missing)], capture_output=True
            )
            self.assertNotEqual(result.returncode, 0)

    def test_bundle_field_boundaries(self):
        self.assertNotEqual(manifest({"a": "b12:c"}), manifest({"ab": "12:c"}))
        self.assertNotEqual(manifest({"a": "b", "c": "d"}), manifest({"a": "b1:c1:d"}))

    def test_missing_required_library(self):
        (self.headers / "cub/version.cuh").unlink()
        with self.assertRaisesRegex(AssertionError, "cub/version.cuh"):
            self.cccl()

    def test_project_headers(self):
        stdint = self.root / "stdint.hpp"
        rle = self.root / "rle.cuh"
        write(stdint, "// stdint 1\n")
        write(rle, "// rle 1\n")

        def generate():
            run(
                CMAKE,
                f"-DIN_STDINT={stdint}",
                f"-DIN_RLE={rle}",
                f"-DOUT={self.output}",
                f"-DXXHSUM_EXECUTABLE={XXHSUM}",
                "-P",
                str(SOURCE / "cmake/embed_jit_headers.cmake"),
            )
            return fingerprint(self.output, "kEmbeddedJitHeadersIdentity")

        original = generate()
        self.assertEqual(
            original,
            manifest(
                {
                    "codegen/stdint_shim.hpp": "// stdint 1\n",
                    "codegen/decode/rle_block.cuh": "// rle 1\n",
                }
            ),
        )
        write(stdint, "// stdint 2\n")
        self.assertNotEqual(original, generate())
        write(stdint, "// stdint 1\n")
        write(rle, "// rle 2\n")
        self.assertNotEqual(original, generate())

    def test_incremental_content_and_shadowing(self):
        earlier = self.root / "earlier"
        earlier.mkdir()
        write(self.include_list, f"{earlier}\n{self.headers}\n")
        source = self.root / "cmake-project"
        build = self.root / "build"
        script = SOURCE / "cmake/embed_cccl_headers.cmake"
        write(
            source / "CMakeLists.txt",
            f"""cmake_minimum_required(VERSION 3.24)
project(embed_fixture LANGUAGES NONE)
add_custom_command(OUTPUT "{self.output}"
  COMMAND "${{CMAKE_COMMAND}}" "-DINCLUDE_DIRS_FILE={self.include_list}"
    "-DOUT={self.output}" "-DDEPFILE={self.output}.d"
    "-DXXHSUM_EXECUTABLE={XXHSUM}" -P "{script}"
  DEPENDS "{self.include_list}" "{script}"
    "{SOURCE / 'cmake/jit_header_manifest.cmake'}" "{XXHSUM}"
  DEPFILE "{self.output}.d" VERBATIM)
add_custom_target(embed ALL DEPENDS "{self.output}")
""",
        )
        run(CMAKE, "-S", str(source), "-B", str(build), "-G", "Ninja")

        def rebuild():
            run(CMAKE, "--build", str(build))
            return fingerprint(self.output, "kCcclEmbeddedHeadersIdentity")

        def mark_changed(path):
            # Fast edits can share a filesystem timestamp with the previous
            # output. Order only the changed dependency after that output;
            # touching other inputs could hide a missing dependency edge.
            timestamp = self.output.stat().st_mtime_ns + 1
            os.utime(path, ns=(path.stat().st_atime_ns, timestamp))
            self.assertGreater(path.stat().st_mtime_ns, timestamp - 1)

        def assert_no_rebuild(expected):
            timestamp = self.output.stat().st_mtime_ns
            self.assertEqual(expected, rebuild())
            self.assertEqual(timestamp, self.output.stat().st_mtime_ns)

        original = rebuild()
        assert_no_rebuild(original)
        write(self.headers / ROOTS[0], "// fixture 2\n")
        mark_changed(self.headers / ROOTS[0])
        changed = rebuild()
        self.assertNotEqual(original, changed)
        assert_no_rebuild(changed)
        write(earlier / ROOTS[0], "// fixture 3\n")
        mark_changed(earlier)
        shadowed = rebuild()
        self.assertNotEqual(changed, shadowed)
        assert_no_rebuild(shadowed)
        (earlier / ROOTS[0]).unlink()
        mark_changed((earlier / ROOTS[0]).parent)
        self.assertEqual(changed, rebuild())
        assert_no_rebuild(changed)

        # Entirely new, unreferenced subtrees must also trigger regeneration.
        new_header = self.headers / "cuda/new/detail/header"
        write(new_header, "// new unreferenced subtree\n")
        mark_changed(self.headers / "cuda")
        added = rebuild()
        self.assertNotEqual(changed, added)
        assert_no_rebuild(added)
        write(new_header, "// edited unreferenced header\n")
        mark_changed(new_header)
        edited = rebuild()
        self.assertNotEqual(added, edited)
        assert_no_rebuild(edited)
        renamed = new_header.with_name("renamed")
        new_header.rename(renamed)
        mark_changed(new_header.parent)
        self.assertNotEqual(edited, rebuild())
        renamed.unlink()
        mark_changed(renamed.parent)
        self.assertEqual(changed, rebuild())
        assert_no_rebuild(changed)

    def test_static_compiler_artifacts(self):
        libraries = {}
        for role in ("NVRTC", "BUILTINS", "PTX"):
            libraries[role] = self.root / f"{role}.a"
            libraries[role].write_bytes(b"archive 1")

        def generate():
            run(
                CMAKE,
                *(f"-D{role}={path}" for role, path in libraries.items()),
                f"-DOUT={self.output}",
                f"-DXXHSUM_EXECUTABLE={XXHSUM}",
                "-P",
                str(SOURCE / "cmake/compiler_identity.cmake"),
            )
            return self.output.read_text()

        original = generate()
        digest = checksum(b"archive 1")
        self.assertIn(
            f"static-v2:xxh3-128:nvrtc:{digest}:builtins:{digest}:ptx:{digest}",
            original,
        )
        for path in libraries.values():
            path.write_bytes(b"archive 2")
            self.assertNotEqual(original, generate())
            path.write_bytes(b"archive 1")
        relocated = self.root / "different-library-prefix"
        relocated.mkdir()
        for role, path in list(libraries.items()):
            shutil.copyfile(path, relocated / path.name)
            libraries[role] = relocated / path.name
        self.assertEqual(original, generate())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cmake", default="cmake")
    parser.add_argument("--xxhsum", default="xxhsum")
    parser.add_argument("--identity-tool", required=True)
    arguments, remaining = parser.parse_known_args()
    CMAKE = arguments.cmake
    XXHSUM = arguments.xxhsum
    IDENTITY_TOOL = arguments.identity_tool
    unittest.main(argv=[__file__, *remaining])
