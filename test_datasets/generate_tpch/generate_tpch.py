#!/usr/bin/env python3
"""
TPC-H Data Generator for DuckDB

This script generates TPC-H benchmark data at a specified scale factor
and saves it to either Parquet files or a DuckDB database.
"""

import duckdb
import argparse
import os
from pathlib import Path


def generate_tpch_data(
    scale_factor: float,
    output_dir: str = "tpch_data",
    format: str = "parquet",
    db_file: str = None,
):
    """
    Generate TPC-H data using DuckDB's built-in TPC-H extension.

    Args:
        scale_factor: TPC-H scale factor (e.g., 0.01, 0.1, 1, 10)
        output_dir: Directory to save output files (for parquet format)
        format: Output format ('parquet' or 'database')
        db_file: Path to DuckDB database file (for database format)
    """

    # TPC-H table names
    tables = [
        "customer",
        "lineitem",
        "nation",
        "orders",
        "part",
        "partsupp",
        "region",
        "supplier",
    ]

    if format == "parquet":
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        print(f"Generating TPC-H data at scale factor {scale_factor}...")
        print(f"Output directory: {output_dir}")

        # Create in-memory DuckDB connection
        conn = duckdb.connect()

        # Install and load TPC-H extension
        conn.execute("INSTALL tpch")
        conn.execute("LOAD tpch")

        # Generate TPC-H data at specified scale factor
        conn.execute(f"CALL dbgen(sf={scale_factor})")

        # Export each table to parquet
        for table in tables:
            print(f"Generating table: {table}...")
            output_file = os.path.join(output_dir, f"{table}.parquet")

            # Export table to parquet
            conn.execute(
                f"""
                COPY {table}
                TO '{output_file}'
                (FORMAT PARQUET, COMPRESSION 'SNAPPY')
            """
            )

            # Get row count
            result = conn.execute(
                f"SELECT COUNT(*) FROM read_parquet('{output_file}')"
            ).fetchone()
            print(f"  → {result[0]:,} rows written to {output_file}")

        conn.close()
        print(f"\nTPC-H data generation complete!")
        print(f"Files saved in: {output_dir}")

    elif format == "database":
        db_path = db_file or f"tpch_sf{scale_factor}.duckdb"
        print(f"Generating TPC-H data at scale factor {scale_factor}...")
        print(f"Database file: {db_path}")

        # Create DuckDB database
        conn = duckdb.connect(db_path)

        # Install and load TPC-H extension
        conn.execute("INSTALL tpch")
        conn.execute("LOAD tpch")

        # Generate TPC-H data at specified scale factor
        print("Generating TPC-H data...")
        conn.execute(f"CALL dbgen(sf={scale_factor})")

        # Display row counts for each table
        for table in tables:
            print(f"Table: {table}...")

            # Get row count
            result = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
            print(f"  → {result[0]:,} rows")

        conn.close()
        print(f"\nTPC-H data generation complete!")
        print(f"Database saved as: {db_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate TPC-H benchmark data for DuckDB",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate SF=1 data as Parquet files
  python generate_tpch.py --scale-factor 1

  # Generate SF=0.1 data into a DuckDB database
  python generate_tpch.py --scale-factor 0.1 --format database

  # Generate SF=10 with custom output location
  python generate_tpch.py --scale-factor 10 --output-dir ./data/tpch_sf10

  # Generate into a specific database file
  python generate_tpch.py --scale-factor 1 --format database --db-file my_tpch.duckdb

Common scale factors:
  0.01  - ~10MB (useful for testing)
  0.1   - ~100MB
  1     - ~1GB
  10    - ~10GB
  100   - ~100GB
        """,
    )

    parser.add_argument(
        "--scale-factor",
        "-sf",
        type=float,
        required=True,
        help="TPC-H scale factor (e.g., 0.01, 0.1, 1, 10, 100)",
    )

    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default="tpch_data",
        help="Output directory for Parquet files (default: tpch_data)",
    )

    parser.add_argument(
        "--format",
        "-f",
        type=str,
        choices=["parquet", "database"],
        default="parquet",
        help="Output format: 'parquet' for separate files or 'database' for DuckDB database (default: parquet)",
    )

    parser.add_argument(
        "--db-file",
        type=str,
        help="Database file path (only for database format, default: tpch_sf{scale_factor}.duckdb)",
    )

    args = parser.parse_args()

    generate_tpch_data(
        scale_factor=args.scale_factor,
        output_dir=args.output_dir,
        format=args.format,
        db_file=args.db_file,
    )


if __name__ == "__main__":
    main()
