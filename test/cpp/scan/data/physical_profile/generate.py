"""Regenerate T5's pinned PyArrow corpus and read back the actual encodings.

Run through pixi. The test KMS deliberately wraps test keys in base64; it is not
an example for production key management. Regenerate on a cuDF/PyArrow bump.
"""

import argparse
import base64
import json
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.parquet.encryption as pe

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, required=True)
ROOT = parser.parse_args().output.resolve()
ROOT.mkdir(parents=True, exist_ok=True)
MANIFEST = {"pyarrow": pa.__version__, "libcudf": "26.08.01", "files": []}
values = {
    "int": (
        pa.array([None if i % 7 == 0 else i for i in range(128)], pa.int32()),
        ["PLAIN", "DELTA_BINARY_PACKED", "DICTIONARY"],
    ),
    "double": (
        pa.array([None if i % 7 == 0 else i / 4 for i in range(128)], pa.float64()),
        ["PLAIN", "BYTE_STREAM_SPLIT", "DICTIONARY"],
    ),
    "string": (
        pa.array([None if i % 7 == 0 else f"value-{i % 31:04}" for i in range(128)]),
        ["PLAIN", "DELTA_LENGTH_BYTE_ARRAY", "DELTA_BYTE_ARRAY", "DICTIONARY"],
    ),
}
for codec in ["NONE", "SNAPPY", "GZIP", "ZSTD", "LZ4", "BROTLI"]:
    for kind, (array, encodings) in values.items():
        for encoding in encodings:
            name = f"{kind}-{codec}-{encoding}.parquet"
            options = {"use_dictionary": encoding == "DICTIONARY"}
            if encoding != "DICTIONARY":
                options["column_encoding"] = encoding
            pq.write_table(
                pa.table({"x": array}),
                ROOT / name,
                compression=codec,
                row_group_size=64,
                **options,
            )
            metadata = pq.read_metadata(ROOT / name)
            chunks = [
                metadata.row_group(i).column(0) for i in range(metadata.num_row_groups)
            ]
            MANIFEST["files"].append(
                {
                    "file": name,
                    "requested_codec": codec,
                    "requested_encoding": encoding,
                    "type": str(array.type),
                    "actual": [
                        {
                            "codec": c.compression,
                            "encodings": c.encodings,
                            "physical_type": c.physical_type,
                        }
                        for c in chunks
                    ],
                }
            )


test_keys = {}


class TestKms(pe.KmsClient):
    def wrap_key(self, key_bytes, master_key_identifier):
        test_keys[master_key_identifier] = base64.b64encode(key_bytes).decode()
        return base64.b64encode(key_bytes)

    def unwrap_key(self, wrapped_key, master_key_identifier):
        return base64.b64decode(wrapped_key)


factory = pe.CryptoFactory(lambda config: TestKms())
kms = pe.KmsConnectionConfig(
    custom_kms_conf={"footer": "0123456789012345", "column": "0123456789012345"}
)
for plaintext in [False, True]:
    name = "encrypted-columns.parquet" if plaintext else "encrypted-footer.parquet"
    config = pe.EncryptionConfiguration(
        footer_key="footer",
        column_keys={"column": ["x"]} if plaintext else None,
        uniform_encryption=not plaintext,
        plaintext_footer=plaintext,
        double_wrapping=False,
    )
    props = factory.file_encryption_properties(kms, config)
    if not plaintext:
        (ROOT / "encrypted-footer.key").write_text(test_keys["footer"] + "\n")
    pq.write_table(
        pa.table({"x": pa.array([1, 2, 3], pa.int32())}),
        ROOT / name,
        encryption_properties=props,
    )
    actual = pq.read_table(
        ROOT / name, decryption_properties=factory.file_decryption_properties(kms)
    )
    assert actual.column(0).to_pylist() == [1, 2, 3]
    MANIFEST["files"].append(
        {
            "file": name,
            "encryption": "columns" if plaintext else "footer",
            "magic": (ROOT / name).read_bytes()[-4:].decode(),
            "readback": [1, 2, 3],
        }
    )
# Change only the footer encoding union; the data pages stay byte-identical.
from footer_encoding import rewrite_encoding_lists
import struct

raw = (ROOT / "int-SNAPPY-PLAIN.parquet").read_bytes()
size = struct.unpack("<I", raw[-8:-4])[0]
for name, replacement in [("levels", bytes([0x25, 0, 8])), ("empty", bytes([0x05]))]:
    footer = rewrite_encoding_lists(raw[-8 - size : -8], replacement)
    path = ROOT / f"int-SNAPPY-{name}.parquet"
    path.write_bytes(
        raw[: -8 - size] + footer + struct.pack("<I", len(footer)) + b"PAR1"
    )
    metadata = pq.read_metadata(path)
    assert pq.read_table(path).equals(pq.read_table(ROOT / "int-SNAPPY-PLAIN.parquet"))
    MANIFEST["files"].append(
        {
            "file": path.name,
            "footer_variant": name,
            "actual": [
                {
                    "codec": metadata.row_group(i).column(0).compression,
                    "encodings": metadata.row_group(i).column(0).encodings,
                }
                for i in range(metadata.num_row_groups)
            ],
        }
    )

(ROOT / "manifest.json").write_text(json.dumps(MANIFEST, indent=2) + "\n")

expected = json.loads((Path(__file__).resolve().parent / "manifest.json").read_text())
assert (
    json.loads(json.dumps(MANIFEST)) == expected
), "actual codec/encoding manifest changed; review the pinned corpus"
