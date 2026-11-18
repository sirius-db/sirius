# Regression corpus

Each `.sql` file has setup statements and one final query. The header names the
issue or saved signature, the switch that hides it from default generation, and
the engine revision where it last failed. Run with `pixi run fuzz corpus`.

Three failures need their original data layout or query history and have no
file here: the validity row offset (#1667), the illegal CUDA address (#2597) and
the Parquet timestamp drift (#2595). See those issues.
The validity fault may depend on earlier queries in the same shell; one fresh
replay alone is not proof that it cleared.
