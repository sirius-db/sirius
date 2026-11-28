# Stock Paimon planning admission experiment

This CPU-only diagnostic exercises caller-side admission for a private,
immutable copy of the checked-in `append_basic` fixture at snapshot 2. It checks
the actual manifest list references and bounded Zstandard frames before
optionally running a reader command. The command receives the fixture path in
`PAIMON_PROBE_TABLE`. Exit statuses 2 and 3 indicate refusal and errors in
the diagnostic. A reader that runs supplies its own exit status.

Run from this directory with Python 3.11+ and libzstd installed. The tests
generate private fixture variants with `fastavro` 1.12.1 and `zstandard` 0.25.0.

```sh
python3 -m pip install fastavro==1.12.1 zstandard==0.25.0
python3 -m unittest -v test_check.py
python3 check.py ../warehouse/reference.db/append_basic
```

If library discovery fails, set `PAIMON_PROBE_ZSTD` to the absolute path of
libzstd. The library must export `ZSTD_getFrameHeader`.

To chain a stock SDK probe after admission, add `-- /path/to/reader` after
the fixture path. Configure the reader to select snapshot 2 explicitly. The
diagnostic passes the table path and the reader owns snapshot selection. The
regression tests use a sentinel to check dispatch.
The actual native CPU run belongs to the SDK and reader pair. This
diagnostic assumes these specific fixture schemas and a sealed view of the
files between validation and reading. Codec workspace, physical backing and
production SQL integration remain outside the diagnostic's evidence.
