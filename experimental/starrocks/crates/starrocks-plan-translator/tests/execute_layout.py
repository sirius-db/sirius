"""Execute the translated mapping fixture with distinct same-type aggregate values."""

import sys

import duckdb

connection = duckdb.connect()
connection.execute("LOAD substrait")
connection.execute("CREATE SCHEMA tpch")
connection.execute("CREATE TABLE tpch.users(value BIGINT, other BIGINT)")
connection.execute("INSERT INTO tpch.users VALUES (10, 100), (20, 200)")
result = connection.execute("CALL from_substrait(?)", [sys.stdin.buffer.read()])
assert [column[0] for column in result.description] == [
    "count", "total", "count_1", "expr_3"
]
rows = result.fetchall()
assert rows == [(2, 30, 2, 32)], rows
