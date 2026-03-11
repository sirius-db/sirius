# Testing this extension
This directory contains all the tests for this extension. The `sql` directory holds tests that are written as [SQLLogicTests](https://duckdb.org/dev/sqllogictest/intro.html). DuckDB aims to have most its tests in this format as SQL statements, so for the quack extension, this should probably be the goal too.

To run the SQLLogicTests:
```bash
pixi run sql-test
```
or for the debug preset:
```bash
pixi run sql-test debug
```
