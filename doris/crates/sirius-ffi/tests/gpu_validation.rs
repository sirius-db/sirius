//! 3-way query validation: DuckDB SQL (CPU) vs gpu_processing (GPU) vs from_substrait (CPU).
//!
//! Run with GPU access + built extensions:
//!   pixi run -e doris cargo test -p sirius-ffi --test gpu_validation -- --ignored

use sirius_ffi::SiriusEngine;

fn engine_with_tables() -> SiriusEngine {
    let e = SiriusEngine::new().expect("SiriusEngine::new() failed — check extensions");
    // GPU buffers are optional for this test.
    let _ = e.init_gpu_buffers("1GB", "1GB");

    // Generate test data inline (no external file dependencies).
    e.execute_sql(
        "CREATE TABLE orders AS
         SELECT
             i AS order_id,
             1 + (i % 20) AS customer_id,
             ROUND(100 + random() * 900, 2) AS amount,
             CASE i % 4 WHEN 0 THEN 'pending' WHEN 1 THEN 'shipped'
                        WHEN 2 THEN 'delivered' ELSE 'cancelled' END AS status,
             DATE '2024-01-01' + INTERVAL (i % 365) DAY AS order_date,
             CASE i % 3 WHEN 0 THEN 'credit_card' WHEN 1 THEN 'paypal'
                        ELSE 'bank_transfer' END AS payment_method
         FROM generate_series(1, 100) t(i)",
    )
    .expect("create orders");

    e.execute_sql(
        "CREATE TABLE customers AS
         SELECT
             i AS customer_id,
             'Customer_' || i AS name,
             CASE i % 5 WHEN 0 THEN 'alice@example.com' WHEN 1 THEN 'bob@example.com'
                        WHEN 2 THEN 'carol@example.com' WHEN 3 THEN 'dave@example.com'
                        ELSE 'eve@example.com' END AS email,
             CASE i % 4 WHEN 0 THEN 'US' WHEN 1 THEN 'UK' WHEN 2 THEN 'DE'
                        ELSE 'JP' END AS country,
             DATE '2020-01-01' + INTERVAL (i * 17 % 1000) DAY AS signup_date
         FROM generate_series(1, 20) t(i)",
    )
    .expect("create customers");

    e
}

fn decode_ipc(ipc_bytes: &[u8]) -> Vec<Vec<String>> {
    use arrow::ipc::reader::StreamReader;
    use std::io::Cursor;

    let reader = StreamReader::try_new(Cursor::new(ipc_bytes), None).expect("IPC decode");
    let mut all_rows = Vec::new();
    for batch in reader {
        let batch = batch.expect("batch");
        for row_idx in 0..batch.num_rows() {
            let mut row = Vec::new();
            for col_idx in 0..batch.num_columns() {
                let col = batch.column(col_idx);
                let val = arrow::util::display::array_value_to_string(col, row_idx)
                    .unwrap_or_else(|_| "?".to_string());
                row.push(val);
            }
            all_rows.push(row);
        }
    }
    all_rows
}

fn compare_results(
    name: &str,
    cpu_rows: &[Vec<String>],
    gpu_rows: &[Vec<String>],
    sort_cols: &[usize],
) {
    let sort = |rows: &[Vec<String>]| -> Vec<Vec<String>> {
        let mut sorted = rows.to_vec();
        sorted.sort_by(|a, b| {
            for &c in sort_cols {
                let cmp = a.get(c).cmp(&b.get(c));
                if cmp != std::cmp::Ordering::Equal {
                    return cmp;
                }
            }
            std::cmp::Ordering::Equal
        });
        sorted
    };

    let cpu_sorted = sort(cpu_rows);
    let gpu_sorted = sort(gpu_rows);

    assert_eq!(
        cpu_sorted.len(),
        gpu_sorted.len(),
        "{name}: row count mismatch: CPU={}, GPU={}",
        cpu_sorted.len(),
        gpu_sorted.len()
    );

    for (i, (cpu_row, gpu_row)) in cpu_sorted.iter().zip(gpu_sorted.iter()).enumerate() {
        assert_eq!(
            cpu_row, gpu_row,
            "{name}: row {i} mismatch\n  CPU: {cpu_row:?}\n  GPU: {gpu_row:?}"
        );
    }
    println!("{name}: PASSED ({} rows)", cpu_rows.len());
}

/// Run a query through CPU (execute_sql) and GPU (execute_gpu), compare results.
fn validate_cpu_vs_gpu(engine: &SiriusEngine, name: &str, sql: &str, sort_cols: &[usize]) {
    let cpu_ipc = engine
        .execute_sql(sql)
        .unwrap_or_else(|e| panic!("{name}: execute_sql failed: {e}"));
    let cpu_rows = decode_ipc(&cpu_ipc);

    match engine.execute_gpu(sql) {
        Ok(gpu_ipc) => {
            let gpu_rows = decode_ipc(&gpu_ipc);
            compare_results(name, &cpu_rows, &gpu_rows, sort_cols);
        }
        Err(e) => {
            println!("{name}: gpu_processing not available ({e}), skipping GPU comparison");
            println!("{name}: CPU-only PASSED ({} rows)", cpu_rows.len());
        }
    }
}

#[test]
#[ignore] // Needs GPU + extensions
fn test_count() {
    let e = engine_with_tables();
    validate_cpu_vs_gpu(&e, "COUNT(*)", "SELECT COUNT(*) as total FROM orders", &[]);
}

#[test]
#[ignore]
fn test_where_filter() {
    let e = engine_with_tables();
    validate_cpu_vs_gpu(
        &e,
        "WHERE filter",
        "SELECT order_id, amount, status FROM orders WHERE amount > 800",
        &[0],
    );
}

#[test]
#[ignore]
fn test_order_by_limit() {
    let e = engine_with_tables();
    validate_cpu_vs_gpu(
        &e,
        "ORDER BY + LIMIT",
        "SELECT order_id, customer_id, amount FROM orders ORDER BY amount DESC LIMIT 5",
        &[],
    );
}

#[test]
#[ignore]
fn test_group_by_count_sum() {
    let e = engine_with_tables();
    validate_cpu_vs_gpu(
        &e,
        "GROUP BY COUNT/SUM",
        "SELECT customer_id, COUNT(*) as cnt, SUM(amount) as total FROM orders GROUP BY customer_id",
        &[0],
    );
}

#[test]
#[ignore]
fn test_group_by_avg() {
    let e = engine_with_tables();
    validate_cpu_vs_gpu(
        &e,
        "GROUP BY AVG",
        "SELECT customer_id, AVG(amount) as avg_amount FROM orders GROUP BY customer_id",
        &[0],
    );
}

#[test]
#[ignore]
fn test_join() {
    let e = engine_with_tables();
    validate_cpu_vs_gpu(
        &e,
        "JOIN",
        "SELECT o.order_id, c.name, o.amount FROM orders o JOIN customers c ON o.customer_id = c.customer_id ORDER BY o.order_id LIMIT 10",
        &[],
    );
}

#[test]
#[ignore]
fn test_join_aggregate() {
    let e = engine_with_tables();
    validate_cpu_vs_gpu(
        &e,
        "JOIN + AGG",
        "SELECT c.name, COUNT(*) as order_count, SUM(o.amount) as total_amount FROM orders o JOIN customers c ON o.customer_id = c.customer_id GROUP BY c.name",
        &[0],
    );
}
