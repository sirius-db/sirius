"""Exercise the production embedders with content changes and incremental builds."""

import argparse
import hashlib
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import unittest

SOURCE = Path(__file__).resolve().parents[2]
CMAKE = "cmake"
ROOTS = (
    "cub/block/block_reduce.cuh",
    "cub/block/block_scan.cuh",
    "cub/block/block_exchange.cuh",
    "cuda/std/cstdint",
    "cuda/std/cstddef",
    "cuda/std/climits",
    "cuda/std/type_traits",
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


def manifest(headers):
    records = "simpatico-headers-v1\n"
    for name, content in sorted(headers.items()):
        normalized = content.replace("\r\n", "\n").replace("\r", "\n").encode()
        records += (
            f"{len(name.encode())}:{name}{hashlib.sha256(normalized).hexdigest()}\n"
        )
    return hashlib.sha256(records.encode()).hexdigest()


class HeaderManifests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="simpatico-headers-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.headers = self.root / "headers"
        for name in (*ROOTS, "cub/version.cuh", "thrust/version.h"):
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
            "-P",
            str(SOURCE / "cmake/embed_cccl_headers.cmake"),
        )
        return fingerprint(self.output, "kCcclEmbeddedHeadersIdentity")

    def test_cccl_content_and_prefix(self):
        original = self.cccl()
        self.assertEqual(original, manifest({name: "// fixture 1\n" for name in ROOTS}))
        copy = self.root / "different-prefix"
        shutil.copytree(self.headers, copy)
        self.assertEqual(original, self.cccl([copy]))
        write(copy / ROOTS[0], "// fixture 2\n")
        self.assertNotEqual(original, self.cccl([copy]))
        self.assertEqual(original, self.cccl([self.headers, copy]))
        self.assertNotEqual(original, self.cccl([copy, self.headers]))

    def test_disjoint_root_order_and_closure(self):
        other = self.root / "other"
        other.mkdir()
        shutil.move(self.headers / "cuda", other / "cuda")
        original = self.cccl([self.headers, other])
        self.assertEqual(original, self.cccl([other, self.headers]))
        write(self.headers / ROOTS[0], "#include <cub/detail/new.cuh>\n")
        write(self.headers / "cub/detail/new.cuh", "// new 1\n")
        added = self.cccl([self.headers, other])
        self.assertNotEqual(original, added)
        write(self.headers / "cub/detail/new.cuh", "// new 2\n")
        self.assertNotEqual(added, self.cccl([self.headers, other]))

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
    "-DOUT={self.output}" "-DDEPFILE={self.output}.d" -P "{script}"
  DEPENDS "{self.include_list}" "{script}"
  DEPFILE "{self.output}.d" VERBATIM)
add_custom_target(embed ALL DEPENDS "{self.output}")
""",
        )
        run(CMAKE, "-S", str(source), "-B", str(build), "-G", "Ninja")

        def rebuild():
            run(CMAKE, "--build", str(build))
            return fingerprint(self.output, "kCcclEmbeddedHeadersIdentity")

        original = rebuild()
        timestamp = self.output.stat().st_mtime_ns
        self.assertEqual(original, rebuild())
        self.assertEqual(timestamp, self.output.stat().st_mtime_ns)
        write(self.headers / ROOTS[0], "// fixture 2\n")
        changed = rebuild()
        self.assertNotEqual(original, changed)
        write(earlier / ROOTS[0], "// fixture 3\n")
        shadowed = rebuild()
        self.assertNotEqual(changed, shadowed)
        (earlier / ROOTS[0]).unlink()
        self.assertEqual(changed, rebuild())

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
                "-P",
                str(SOURCE / "cmake/compiler_identity.cmake"),
            )
            return self.output.read_text()

        original = generate()
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
    arguments, remaining = parser.parse_known_args()
    CMAKE = arguments.cmake
    unittest.main(argv=[__file__, *remaining])
