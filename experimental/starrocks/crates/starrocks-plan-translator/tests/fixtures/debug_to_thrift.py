"""Re-encodes a recorded `TExecPlanFragmentParams` dump as thrift binary-protocol bytes.

The CN's `SIRIUS_CN_DUMP_FRAGMENTS` writes each fragment it receives in Rust `{:#?}` Debug
format. The translator tests read fixtures with the thrift binary protocol instead, the way
the FE sends them, so this script parses the Debug text and writes the same values back out
using the field ids and types from the StarRocks IDL.

Usage, from `experimental/starrocks`:

    pixi run -e cn python3 crates/starrocks-plan-translator/tests/fixtures/debug_to_thrift.py \\
        fragment-0004.txt crates/starrocks-plan-translator/tests/fixtures/<dir>/<name>.bin
"""

import json
import os
import re
import shutil
import struct
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
WORKSPACE = os.path.normpath(os.path.join(HERE, "..", "..", "..", ".."))
IDL = os.path.join(WORKSPACE, "starrocks", "gensrc", "thrift")

# ---------------------------------------------------------------------------------------------
# Rust Debug parser. Structs become {"__t__": name, field: value}; tuple structs and enums
# become {"__t__": name, "__v__": [values]}; Some(x) is x and None is None; maps are dicts
# and sets are lists.
# ---------------------------------------------------------------------------------------------

TOKEN = re.compile(
    r'\s+|("(?:[^"\\]|\\.)*")|([A-Za-z_][A-Za-z0-9_]*)|(-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)'
    r"|([{}()\[\],:])"
)


def tokenize(text):
    tokens, pos = [], 0
    for match in TOKEN.finditer(text):
        if match.start() != pos:
            raise ValueError(f"unexpected text at {pos}: {text[pos:pos + 40]!r}")
        pos = match.end()
        string, ident, number, punct = match.groups()
        if string is not None:
            tokens.append(("s", json.loads(string)))
        elif ident is not None:
            tokens.append(("i", ident))
        elif number is not None:
            is_float = any(c in number for c in ".eE")
            tokens.append(("n", float(number) if is_float else int(number)))
        elif punct is not None:
            tokens.append(("p", punct))
    if pos != len(text):
        raise ValueError("trailing text")
    return tokens


class Parser:
    def __init__(self, tokens):
        self.tokens, self.pos = tokens, 0

    def peek(self):
        return self.tokens[self.pos][1] if self.pos < len(self.tokens) else None

    def next(self):
        token = self.tokens[self.pos]
        self.pos += 1
        return token

    def skip_comma(self):
        if self.peek() == ",":
            self.pos += 1

    def items(self, close):
        items = []
        while self.peek() != close:
            items.append(self.value())
            self.skip_comma()
        self.pos += 1
        return items

    def value(self):
        kind, token = self.next()
        if kind in ("s", "n"):
            return token
        if kind == "i":
            if token in ("None", "true", "false"):
                return {"None": None, "true": True, "false": False}[token]
            if self.peek() == "{":
                self.pos += 1
                fields = {"__t__": token}
                while self.peek() != "}":
                    _, name = self.next()
                    assert self.next()[1] == ":"
                    fields[name] = self.value()
                    self.skip_comma()
                self.pos += 1
                return fields
            if self.peek() == "(":
                self.pos += 1
                values = self.items(")")
                return values[0] if token == "Some" else {"__t__": token, "__v__": values}
            return {"__t__": token}
        if token == "[":
            return self.items("]")
        if token == "{":
            mapping, elements = {}, []
            while self.peek() != "}":
                key = self.value()
                if self.peek() == ":":
                    self.pos += 1
                    mapping[key] = self.value()
                else:
                    elements.append(key)
                self.skip_comma()
            self.pos += 1
            return elements if elements else mapping
        raise ValueError(f"unexpected token {token!r}")


# ---------------------------------------------------------------------------------------------
# Thrift binary protocol encoder driven by the IDL (`thrift --gen json:merge`).
# ---------------------------------------------------------------------------------------------

TYPE_CODE = {
    "bool": 2, "i8": 3, "byte": 3, "double": 4, "i16": 6, "i32": 8, "enum": 8, "i64": 10,
    "string": 11, "binary": 11, "struct": 12, "map": 13, "set": 14, "list": 15,
}


def load_idl(root_file):
    out = tempfile.mkdtemp()
    try:
        thrift = shutil.which("thrift") or os.path.join(WORKSPACE, ".pixi/envs/cn/bin/thrift")
        subprocess.run(
            [thrift, "--gen", "json:merge", "-I", IDL, "-out", out, root_file],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        with open(os.path.join(out, os.path.basename(root_file)[:-len(".thrift")] + ".json")) as f:
            idl = json.load(f)
    finally:
        shutil.rmtree(out)
    return {s["name"]: s["fields"] for s in idl["structs"]}


def normalize(name):
    """Matches IDL names to the Rust generator's snake_case field names (`slotType`/`slot_type`,
    `fn`/`fn_`)."""
    return name.replace("_", "").lower()


def type_of(field, prefix=""):
    """Returns {typeId, class?, elemType?, keyType?, valueType?} for a field or container part."""
    if prefix:
        ty = field.get(prefix + "Type")
        return ty if ty is not None else {"typeId": field[prefix + "TypeId"]}
    return field.get("type") or {"typeId": field["typeId"]}


class Encoder:
    def __init__(self, structs):
        self.structs = structs
        self.out = bytearray()

    def write_struct(self, class_name, value):
        fields = self.structs[class_name]
        by_name = {normalize(f["name"]): f for f in fields}
        for key, item in value.items():
            if key == "__t__" or item is None:
                continue
            field = by_name.get(normalize(key))
            if field is None:
                raise KeyError(f"{class_name} has no field {key!r}")
            ty = type_of(field)
            self.out += struct.pack(">bh", TYPE_CODE[ty["typeId"]], field["key"])
            self.write(ty, item)
        self.out += b"\x00"

    def write(self, ty, value):
        type_id = ty["typeId"]
        # Newtype wrappers in the Debug text: enums (`TPlanNodeType(6)`) and `OrderedFloat(x)`.
        if isinstance(value, dict) and "__v__" in value and type_id != "struct":
            (value,) = value["__v__"]
        if type_id == "bool":
            self.out += struct.pack(">?", value)
        elif type_id in ("i8", "byte"):
            self.out += struct.pack(">b", value)
        elif type_id == "i16":
            self.out += struct.pack(">h", value)
        elif type_id in ("i32", "enum"):
            self.out += struct.pack(">i", value)
        elif type_id == "i64":
            self.out += struct.pack(">q", value)
        elif type_id == "double":
            self.out += struct.pack(">d", value)
        elif type_id in ("string", "binary"):
            data = bytes(value) if isinstance(value, list) else value.encode()
            self.out += struct.pack(">i", len(data)) + data
        elif type_id == "struct":
            self.write_struct(ty["class"], value)
        elif type_id in ("list", "set"):
            elem = type_of(ty, "elem")
            items = list(value.values()) if isinstance(value, dict) else value
            assert not isinstance(value, dict) or not value, "non-empty map where a list belongs"
            self.out += struct.pack(">bi", TYPE_CODE[elem["typeId"]], len(items))
            for item in items:
                self.write(elem, item)
        elif type_id == "map":
            key_ty, value_ty = type_of(ty, "key"), type_of(ty, "value")
            entries = value if isinstance(value, dict) else {}
            assert isinstance(value, dict) or not value, "non-empty list where a map belongs"
            self.out += struct.pack(
                ">bbi", TYPE_CODE[key_ty["typeId"]], TYPE_CODE[value_ty["typeId"]], len(entries)
            )
            for key, item in entries.items():
                self.write(key_ty, key)
                self.write(value_ty, item)
        else:
            raise TypeError(f"unsupported thrift type {type_id}")


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    source, target = sys.argv[1:]
    with open(source) as f:
        params = Parser(tokenize(f.read())).value()
    encoder = Encoder(load_idl(os.path.join(IDL, "InternalService.thrift")))
    encoder.write_struct(params["__t__"], params)
    with open(target, "wb") as f:
        f.write(encoder.out)


if __name__ == "__main__":
    main()
