"""Put test/fuzz on sys.path so `import siriusfuzz` works under `python -m unittest`."""

import pathlib
import sys

FUZZ_DIR = pathlib.Path(__file__).resolve().parent.parent
if str(FUZZ_DIR) not in sys.path:
    sys.path.insert(0, str(FUZZ_DIR))
