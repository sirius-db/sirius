#!/usr/bin/env python3
# Copyright 2026, Sirius Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Extract the THIRD_PARTY_NOTICES document embedded in a Sirius extension binary.

The document lives in the .sirius_notices ELF section (see
sirius-duckdb/cmake/third-party-notices.cmake), so this works without loading the
extension or having a GPU. Accepts gzipped binaries.
"""

import argparse
import gzip
import struct
import sys

SECTION = b".sirius_notices"


def read_binary(path: str) -> bytes:
    with open(path, "rb") as f:
        data = f.read()
    if data[:2] == b"\x1f\x8b":
        data = gzip.decompress(data)
    return data


def section(data: bytes, name: bytes) -> bytes:
    if data[:4] != b"\x7fELF" or data[4] != 2 or data[5] != 1:
        raise ValueError("not a little-endian ELF64 file")
    (shoff,) = struct.unpack_from("<Q", data, 0x28)
    shentsize, shnum, shstrndx = struct.unpack_from("<HHH", data, 0x3A)

    def header(index: int):
        # sh_name, sh_type, sh_flags, sh_addr, sh_offset, sh_size
        return struct.unpack_from("<IIQQQQ", data, shoff + index * shentsize)

    names_offset = header(shstrndx)[4]
    for index in range(shnum):
        name_offset, _, _, _, offset, size = header(index)
        start = names_offset + name_offset
        if data[start : data.index(b"\0", start)] == name:
            return data[offset : offset + size]
    raise ValueError(
        f"no {name.decode()} section; the binary predates embedded notices"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("extension", help=".duckdb_extension file, optionally gzipped")
    parser.add_argument("-o", "--output", help="output file (default: stdout)")
    parser.add_argument(
        "--require",
        action="append",
        default=[],
        metavar="COMPONENT",
        help="fail unless the document lists this component (repeatable)",
    )
    args = parser.parse_args()
    try:
        # The section ends with a NUL terminator after the document.
        document = section(read_binary(args.extension), SECTION).rstrip(b"\0")
    except (OSError, ValueError, struct.error) as error:
        print(f"{args.extension}: {error}", file=sys.stderr)
        return 1
    # The component list precedes the first notice's rule: one unindented line per
    # component, its name then its version.
    index = document.decode("utf-8", "replace").split("\n" + "=" * 80, 1)[0]
    listed = index.splitlines()
    missing = [
        name
        for name in args.require
        if not any(line == name or line.startswith(name + " ") for line in listed)
    ]
    if missing:
        print(
            f"{args.extension}: notices do not list {', '.join(missing)}",
            file=sys.stderr,
        )
        return 1
    if args.output:
        with open(args.output, "wb") as f:
            f.write(document)
    else:
        sys.stdout.buffer.write(document)
    return 0


if __name__ == "__main__":
    sys.exit(main())
