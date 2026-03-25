//! Reorder Arrow IPC columns to match FE expected output order.
//!
//! When DuckDB executes a Substrait plan, the output column order follows
//! the plan's internal order (GROUP BY keys first, then measures for AGG;
//! left+right for JOINs). The Doris FE expects columns in the SELECT list
//! order specified by `output_exprs`. This module reorders columns to match.

use std::io::Cursor;
use std::sync::Arc;

use arrow::array::{Array, RecordBatch, new_null_array};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::ipc::reader::StreamReader;
use arrow::ipc::writer::StreamWriter;
use tracing::info;

/// Reorder IPC columns to match FE output order, padding with NULLs for missing columns.
///
/// `fe_output_names` is in the FE's expected order. Names like `"expr_N"` represent
/// aggregate/computed expressions that don't exist in the IPC schema by name —
/// these are matched by elimination (unmatched IPC columns fill unmatched FE slots).
///
/// Returns the original IPC unchanged only if the column order already matches
/// AND all names match exactly (no `expr_N` wildcards).
///
/// `ipc_name_aliases` maps IPC column names (e.g. "col_0") to their real names
/// from the descriptor table. This handles packed GPU exchange tables where the
/// C++ registration uses generic names.
pub fn reorder_and_pad_ipc(
    ipc_bytes: &[u8],
    fe_output_names: &[String],
    ipc_name_aliases: Option<&[String]>,
) -> Result<Vec<u8>, String> {
    let reader = StreamReader::try_new(Cursor::new(ipc_bytes), None)
        .map_err(|e| format!("parse IPC for reorder+pad: {e}"))?;
    let schema = reader.schema();
    // Use aliases if provided (maps col_0→real_name), otherwise use IPC field names.
    let ipc_names: Vec<&str> = if let Some(aliases) = ipc_name_aliases {
        // Aliases are positional — alias[i] is the real name for IPC column i.
        aliases.iter().map(|s| s.as_str()).collect()
    } else {
        schema.fields().iter().map(|f| f.name().as_str()).collect()
    };

    // Check if columns are already in the right order with EXACT name matching.
    // expr_N names are NOT considered exact matches — they always trigger reorder.
    if schema.fields().len() == fe_output_names.len() {
        let exact_match = ipc_names
            .iter()
            .zip(fe_output_names.iter())
            .all(|(a, b)| *a == b.as_str());
        if exact_match {
            return Ok(ipc_bytes.to_vec());
        }
    }

    // Build name → IPC column index map (uses aliases if provided).
    let ipc_name_to_idx: std::collections::HashMap<&str, usize> = ipc_names
        .iter()
        .enumerate()
        .map(|(i, name)| (*name, i))
        .collect();

    // For each FE output position, find the source IPC column index.
    // When aliases are provided, all names are treated as exact matches
    // (aliases already contain "expr_N" placeholders for aggregates).
    // Without aliases, "expr_N" names are matched by elimination.
    let use_exact = ipc_name_aliases.is_some();
    let mut used_ipc: std::collections::HashSet<usize> = std::collections::HashSet::new();
    let mut mapping: Vec<Option<usize>> = Vec::with_capacity(fe_output_names.len());
    for name in fe_output_names {
        if use_exact || !name.starts_with("expr_") {
            if let Some(&idx) = ipc_name_to_idx.get(name.as_str()) {
                mapping.push(Some(idx));
                used_ipc.insert(idx);
                continue;
            }
        }
        mapping.push(None); // Will be resolved below
    }
    // Assign unmatched IPC columns to unmatched FE positions by order.
    let mut unmatched_ipc: Vec<usize> = (0..schema.fields().len())
        .filter(|i| !used_ipc.contains(i))
        .collect();
    for slot in &mut mapping {
        if slot.is_none() && !unmatched_ipc.is_empty() {
            *slot = Some(unmatched_ipc.remove(0));
        }
    }

    let matched = mapping.iter().filter(|m| m.is_some()).count();
    let null_count = mapping.iter().filter(|m| m.is_none()).count();
    info!(
        ipc_cols = schema.fields().len(),
        fe_cols = fe_output_names.len(),
        matched,
        null_count,
        fe_names = ?fe_output_names,
        ipc_names = ?ipc_names,
        mapping = ?mapping,
        "reordering IPC columns to match FE output order"
    );

    // Build reordered schema.
    let mut fields: Vec<Arc<Field>> = Vec::with_capacity(fe_output_names.len());
    for (i, maybe_idx) in mapping.iter().enumerate() {
        match maybe_idx {
            Some(idx) => fields.push(Arc::new(schema.field(*idx).clone())),
            None => fields.push(Arc::new(Field::new(
                &fe_output_names[i],
                DataType::Utf8,
                true,
            ))),
        }
    }
    let reordered_schema = Arc::new(Schema::new(fields));

    let mut buf = Vec::new();
    {
        let mut writer = StreamWriter::try_new(&mut buf, &reordered_schema)
            .map_err(|e| format!("IPC writer: {e}"))?;
        for batch_result in reader {
            let batch = batch_result.map_err(|e| format!("read IPC batch: {e}"))?;
            let num_rows = batch.num_rows();
            let mut columns: Vec<Arc<dyn Array>> = Vec::with_capacity(fe_output_names.len());
            for maybe_idx in &mapping {
                match maybe_idx {
                    Some(idx) => columns.push(batch.column(*idx).clone()),
                    None => columns.push(new_null_array(&DataType::Utf8, num_rows)),
                }
            }
            let reordered = RecordBatch::try_new(reordered_schema.clone(), columns)
                .map_err(|e| format!("build reordered batch: {e}"))?;
            writer.write(&reordered).map_err(|e| format!("write batch: {e}"))?;
        }
        writer.finish().map_err(|e| format!("finish IPC: {e}"))?;
    }

    Ok(buf)
}

/// Helper: create Arrow IPC bytes from column names, types, and data.
#[cfg(test)]
fn make_ipc(names: &[&str], arrays: Vec<Arc<dyn Array>>) -> Vec<u8> {
    use arrow::ipc::writer::StreamWriter;

    let fields: Vec<Arc<Field>> = names
        .iter()
        .zip(arrays.iter())
        .map(|(name, arr)| Arc::new(Field::new(*name, arr.data_type().clone(), true)))
        .collect();
    let schema = Arc::new(Schema::new(fields));
    let batch = RecordBatch::try_new(schema.clone(), arrays).unwrap();

    let mut buf = Vec::new();
    let mut writer = StreamWriter::try_new(&mut buf, &schema).unwrap();
    writer.write(&batch).unwrap();
    writer.finish().unwrap();
    buf
}

/// Helper: read column names from Arrow IPC bytes.
#[cfg(test)]
fn read_ipc_column_names(ipc_bytes: &[u8]) -> Vec<String> {
    let reader = StreamReader::try_new(Cursor::new(ipc_bytes), None).unwrap();
    reader
        .schema()
        .fields()
        .iter()
        .map(|f| f.name().clone())
        .collect()
}

/// Helper: read first row values as strings from Arrow IPC.
#[cfg(test)]
fn read_ipc_first_row(ipc_bytes: &[u8]) -> Vec<String> {
    let reader = StreamReader::try_new(Cursor::new(ipc_bytes), None).unwrap();
    let mut values = Vec::new();
    for batch in reader {
        let batch = batch.unwrap();
        if batch.num_rows() > 0 {
            for col_idx in 0..batch.num_columns() {
                let val = arrow::util::display::array_value_to_string(batch.column(col_idx), 0)
                    .unwrap_or_else(|_| "?".to_string());
                values.push(val);
            }
            break;
        }
    }
    values
}

#[cfg(test)]
mod ipc_reorder_tests {
    use super::*;
    use arrow::array::{Float64Array, Int64Array, StringArray};

    #[test]
    fn test_exact_match_no_reorder() {
        let ipc = make_ipc(
            &["a", "b", "c"],
            vec![
                Arc::new(Int64Array::from(vec![1])),
                Arc::new(Int64Array::from(vec![2])),
                Arc::new(Int64Array::from(vec![3])),
            ],
        );
        let result = reorder_and_pad_ipc(
            &ipc,
            &["a".into(), "b".into(), "c".into()],
            None,
        )
        .unwrap();
        assert_eq!(read_ipc_column_names(&result), vec!["a", "b", "c"]);
    }

    #[test]
    fn test_simple_reorder() {
        let ipc = make_ipc(
            &["a", "b", "c"],
            vec![
                Arc::new(Int64Array::from(vec![1])),
                Arc::new(Int64Array::from(vec![2])),
                Arc::new(Int64Array::from(vec![3])),
            ],
        );
        let result = reorder_and_pad_ipc(
            &ipc,
            &["c".into(), "a".into(), "b".into()],
            None,
        )
        .unwrap();
        assert_eq!(read_ipc_column_names(&result), vec!["c", "a", "b"]);
        assert_eq!(read_ipc_first_row(&result), vec!["3", "1", "2"]);
    }

    #[test]
    fn test_q3_column_reorder() {
        // Q3: DuckDB returns [l_orderkey, o_orderdate, o_shippriority, sum(...)]
        // FE expects [l_orderkey, revenue(=sum), o_orderdate, o_shippriority]
        let ipc = make_ipc(
            &["l_orderkey", "o_orderdate", "o_shippriority", "sum(revenue)"],
            vec![
                Arc::new(Int64Array::from(vec![100])),
                Arc::new(Int64Array::from(vec![19050])),       // date as int
                Arc::new(Int64Array::from(vec![0])),           // shippriority
                Arc::new(Float64Array::from(vec![406181.01])), // revenue
            ],
        );
        // FE names: l_orderkey is known, expr_1 = revenue, o_orderdate, o_shippriority
        let result = reorder_and_pad_ipc(
            &ipc,
            &[
                "l_orderkey".into(),
                "expr_1".into(),       // aggregate → matched by elimination
                "o_orderdate".into(),
                "o_shippriority".into(),
            ],
            None,
        )
        .unwrap();
        let names = read_ipc_column_names(&result);
        let values = read_ipc_first_row(&result);
        // The aggregate column should now be at position 1
        assert_eq!(names, vec!["l_orderkey", "sum(revenue)", "o_orderdate", "o_shippriority"]);
        assert_eq!(values, vec!["100", "406181.01", "19050", "0"]);
    }

    #[test]
    fn test_expr_placeholder_matches_unmatched() {
        // IPC has: [a, b, computed_col]
        // FE expects: [a, expr_0, b] — expr_0 should get computed_col
        let ipc = make_ipc(
            &["a", "b", "computed_col"],
            vec![
                Arc::new(Int64Array::from(vec![1])),
                Arc::new(Int64Array::from(vec![2])),
                Arc::new(Int64Array::from(vec![99])),
            ],
        );
        let result = reorder_and_pad_ipc(
            &ipc,
            &["a".into(), "expr_0".into(), "b".into()],
            None,
        )
        .unwrap();
        let names = read_ipc_column_names(&result);
        let values = read_ipc_first_row(&result);
        assert_eq!(names, vec!["a", "computed_col", "b"]);
        assert_eq!(values, vec!["1", "99", "2"]);
    }

    #[test]
    fn test_fewer_ipc_cols_pads_with_null() {
        let ipc = make_ipc(
            &["a", "b"],
            vec![
                Arc::new(Int64Array::from(vec![1])),
                Arc::new(Int64Array::from(vec![2])),
            ],
        );
        let result = reorder_and_pad_ipc(
            &ipc,
            &["a".into(), "b".into(), "missing".into()],
            None,
        )
        .unwrap();
        let names = read_ipc_column_names(&result);
        assert_eq!(names, vec!["a", "b", "missing"]);
        let values = read_ipc_first_row(&result);
        assert_eq!(values[2], ""); // NULL → empty string
    }

    #[test]
    fn test_more_ipc_cols_keeps_matched_only() {
        let ipc = make_ipc(
            &["a", "b", "c", "extra"],
            vec![
                Arc::new(Int64Array::from(vec![1])),
                Arc::new(Int64Array::from(vec![2])),
                Arc::new(Int64Array::from(vec![3])),
                Arc::new(Int64Array::from(vec![99])),
            ],
        );
        let result = reorder_and_pad_ipc(
            &ipc,
            &["c".into(), "a".into()],
            None,
        )
        .unwrap();
        let names = read_ipc_column_names(&result);
        let values = read_ipc_first_row(&result);
        assert_eq!(names, vec!["c", "a"]);
        assert_eq!(values, vec!["3", "1"]);
    }

    #[test]
    fn test_multiple_expr_placeholders() {
        // Two aggregate expressions
        let ipc = make_ipc(
            &["group_key", "sum_a", "sum_b"],
            vec![
                Arc::new(StringArray::from(vec!["key1"])),
                Arc::new(Float64Array::from(vec![100.0])),
                Arc::new(Float64Array::from(vec![200.0])),
            ],
        );
        let result = reorder_and_pad_ipc(
            &ipc,
            &["expr_0".into(), "group_key".into(), "expr_2".into()],
            None,
        )
        .unwrap();
        let names = read_ipc_column_names(&result);
        let values = read_ipc_first_row(&result);
        // expr_0 gets sum_a (first unmatched), expr_2 gets sum_b (second unmatched)
        assert_eq!(names, vec!["sum_a", "group_key", "sum_b"]);
        assert_eq!(values, vec!["100.0", "key1", "200.0"]);
    }

    #[test]
    fn test_alias_based_reorder() {
        // IPC has col_0, col_1, col_2, col_3 (generic from packed GPU).
        // Aliases: col_0→l_orderkey, col_1→o_orderdate, col_2→o_shippriority, col_3→revenue
        // FE expects: l_orderkey, expr_1(=revenue), o_orderdate, o_shippriority
        let ipc = make_ipc(
            &["col_0", "col_1", "col_2", "col_3"],
            vec![
                Arc::new(Int64Array::from(vec![100])),
                Arc::new(Int64Array::from(vec![19050])),
                Arc::new(Int64Array::from(vec![0])),
                Arc::new(Float64Array::from(vec![406181.01])),
            ],
        );
        let aliases = vec![
            "l_orderkey".to_string(),
            "o_orderdate".to_string(),
            "o_shippriority".to_string(),
            "expr_1".to_string(), // revenue aggregate
        ];
        let result = reorder_and_pad_ipc(
            &ipc,
            &[
                "l_orderkey".into(),
                "expr_1".into(),
                "o_orderdate".into(),
                "o_shippriority".into(),
            ],
            Some(&aliases),
        )
        .unwrap();
        let values = read_ipc_first_row(&result);
        // Revenue (406181.01) should be at position 1, dates at 2
        assert_eq!(values, vec!["100", "406181.01", "19050", "0"]);
    }

    #[test]
    fn test_empty_ipc_returns_error() {
        let result = reorder_and_pad_ipc(&[], &["a".into()], None);
        assert!(result.is_err());
    }
}
