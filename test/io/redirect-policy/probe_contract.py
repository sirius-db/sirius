# SPDX-License-Identifier: Apache-2.0
"""Invocation contract shared by successful and expected-failure validators."""
import re
from urllib.parse import quote

DEFAULTS = dict(
    target="",
    reuse="off",
    retention="expanded",
    connections=1,
    workers=1,
    reads=4,
    footer_first=False,
    request_log=True,
)
REQUIRED = {"endpoint", "bucket", "key", "fixture", "out"}


def invocation(command):
    if not isinstance(command, list) or len(command) < 2 or len(command) % 2 != 1:
        raise ValueError("expected executable and unique option/value pairs")
    values = dict(DEFAULTS)
    seen = set()
    for option, value in zip(command[1::2], command[2::2]):
        if (
            not isinstance(option, str)
            or not option.startswith("--")
            or not isinstance(value, str)
        ):
            raise ValueError("invalid option/value")
        key = option[2:].replace("-", "_")
        if (
            option != "--" + key.replace("_", "-")
            or key not in REQUIRED | DEFAULTS.keys()
            or key in seen
        ):
            raise ValueError("duplicate/unknown option")
        seen.add(key)
        if key in ("connections", "workers", "reads"):
            if not re.fullmatch(r"[0-9]+", value):
                raise ValueError("invalid unsigned option")
            value = int(value)
        elif key in ("footer_first", "request_log"):
            if value not in ("true", "false"):
                raise ValueError("invalid boolean")
            value = value == "true"
        values[key] = value
    if not REQUIRED <= seen:
        raise ValueError("missing required options")
    if not (
        1 <= values["connections"] <= 64
        and values["workers"] == 1
        and 1 <= values["reads"] <= 256
    ):
        raise ValueError("outside serial workload limits")
    if values["reuse"] not in ("off", "same_open") or values["retention"] not in (
        "legacy",
        "expanded",
    ):
        raise ValueError("invalid policy mode")
    return values


def bind_invocation(receipt, config, size):
    values = invocation(receipt["command"])
    for key, value in values.items():
        if type(config.get(key)) is not type(value) or config[key] != value:
            raise ValueError("invocation mismatch: " + key)
    if config.get("fixture_size") != size:
        raise ValueError("fixture size mismatch")
    expected_path = (
        "/" + quote(values["bucket"], safe="") + "/" + quote(values["key"], safe="/")
    )
    origin = [r for r in receipt["trace"] if r["role"] == "origin"]
    if not origin or any(r["path"] != expected_path for r in origin):
        raise ValueError("origin object differs from invocation")
    return values
