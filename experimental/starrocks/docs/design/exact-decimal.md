# Plan: exact DECIMAL

## Why

- **Audited runs:** an official TPC-H run is certified by its sponsor, and the spec requires DECIMAL arithmetic. FP64 isn't enough.
- **On the POC branch:** 17/22 queries matched with native DECIMAL, and 20/22 with decimals lowered to FP64. The extra differences were floating-point rounding.

## Today

- **The translator lowers decimal arithmetic, SUM and AVG to FP64** (`expr_translator.rs:405,735`). Its comment gives the reason: the Sirius GPU expression and aggregate path can't consume decimal arithmetic.
- **Results aren't cast back.** A slot the FE declared DECIMAL arrives as a double (#1687).
- **The DuckDB check** casts decimals to doubles and compares with a 5e-3 relative tolerance.
- **Sirius does handle DECIMAL in places:** its compression and scan code carries `DECIMAL64` / `DECIMAL128` columns. Whether its expressions and aggregates still can't is unverified.

## Steps

### Step 1: find which layer can't handle DECIMAL

Run, in Sirius directly (DuckDB SQL, no StarRocks), the decimal shapes TPC-H needs:
- `l_extendedprice * (1 - l_discount)`;
- `sum(...)` and `avg(...)` over DECIMAL(15,2);
- a comparison against a decimal literal.

For each, record whether it runs on the GPU, falls back to the CPU, or fails, and at which layer: the Sirius planner, the expression translator, or cuDF.

**Output:** a short table added to this doc, and a decision on whether Step 2 needs engine work.

### Step 2: keep DECIMAL through the translator

- Remove the FP64 lowering, emitting decimal types and casts in Substrait.
- Keep the FE's result scale: SUM gives DECIMAL(38,s), and multiplication adds the scales.
- Update `partial_state.rs` (two-phase aggregation) so a decimal SUM is sent between steps as DECIMAL128.

### Step 3: exact checks

- Make the DuckDB check compare decimals exactly, and keep tolerance only for real FP64 columns.
- Re-run all 22 queries: the pass count shouldn't drop.

## Open questions

- Timing for an audited run, which sets when Step 2 has to land.

## Out of scope

- DECIMAL256.
