# Compression plans for `.hpln` generation

The same Pareto-picked plans as `src/compression/simpatico_codegen/plans/tpch_sf1000`, with the
four `*_disabled.txt` ones enabled.

They are disabled for **pins** on purpose: §3.14 measured enabling them at −0.27% of suite time,
i.e. nothing, so the pin path leaves those columns raw and keeps the pin cheap. A **file** is a
different calculus — its bytes are what sits on disk and crosses the network, and a table stored
`input -> identity` is roughly 3x its parquet size. Comparing an uncompressed `.hpln` against a
snappy parquet would measure which tables happen to have a plan, not the format.

`nation` and `region` have no plan at either setting; they are a few KB.
