# SQL target

Use a DuckDB scalar macro when expansion into existing expressions preserves the
reference behavior. The macro is bound by DuckDB before Sirius translates the
logical plan. A legal DuckDB expression can still be unsupported by Sirius:
inspect `src/expression/from_duckdb.cpp` and the current evaluator, then verify on
the GPU rather than relying on a list copied from Spark.

Keep explicit casts where the source distinguishes integer, decimal, or floating
arithmetic. Check NULL handling, overflow, Unicode, locale, calendar, and error
semantics for the function's actual domain. Do not use textual SQL substitution
to replace function calls.

Generate externally:

```bash
python .agents/skills/sirius-udf/scripts/scaffold.py --target sql --output /tmp/example-udf-sql
```

The synthetic macro, independent CPU expression, setup, validation runner, and
loading commands are in the generated project. The example is an infrastructure
smoke test, not the CPU oracle for a customer's original UDF. Replace the
reference with the original executable behavior or an explicitly approved
contract before claiming a conversion.

Use relational queries for routines that read tables or traverse graphs. Record
snapshot and refresh assumptions in the external deliverable. Do not add table
access or mutable state to a scalar callback.
