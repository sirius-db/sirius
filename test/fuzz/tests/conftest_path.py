"""Put test/fuzz on sys.path so `import siriusfuzz` works under `python -m unittest`,
and locate a DuckDB shell for the tests that execute SQL."""

import pathlib
import sys

FUZZ_DIR = pathlib.Path(__file__).resolve().parent.parent
if str(FUZZ_DIR) not in sys.path:
    sys.path.insert(0, str(FUZZ_DIR))


def available_shell():
    """The DuckDB shell to test against, or None (tests needing one are skipped)."""
    from siriusfuzz.session import SessionError, find_shell

    try:
        return find_shell(None, "build/release/duckdb")
    except SessionError:
        return None
