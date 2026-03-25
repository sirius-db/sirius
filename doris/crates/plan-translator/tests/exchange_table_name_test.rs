//! Tests for exchange table name resolution.
//!
//! Validates that `resolve_exchange_table_name` correctly maps Doris
//! EXCHANGE_NODE node_ids to the query-scoped DuckDB table names
//! (`__EXCH_{hex8}_{node_id}`) used by the exchange receiver.

use std::collections::HashMap;

use plan_translator::node_translator::resolve_exchange_table_name;

fn schemas_with(names: &[&str]) -> HashMap<String, Vec<String>> {
    names
        .iter()
        .map(|&n| (n.to_string(), vec!["col1".to_string()]))
        .collect()
}

#[test]
fn test_exact_match_single_table() {
    let schemas = schemas_with(&["__EXCH_12345678_7"]);
    assert_eq!(
        resolve_exchange_table_name(7, &schemas),
        "__EXCH_12345678_7"
    );
}

#[test]
fn test_no_match_returns_legacy_fallback() {
    let schemas = schemas_with(&["some_table"]);
    assert_eq!(
        resolve_exchange_table_name(7, &schemas),
        "__EXCHANGE_TABLE_7"
    );
}

#[test]
fn test_empty_schemas_returns_legacy_fallback() {
    let schemas: HashMap<String, Vec<String>> = HashMap::new();
    assert_eq!(
        resolve_exchange_table_name(3, &schemas),
        "__EXCHANGE_TABLE_3"
    );
}

#[test]
fn test_no_ambiguity_node_3_vs_13() {
    // node_id=3 must NOT match __EXCH_xxx_13 (suffix "_13" ends with "3")
    let schemas = schemas_with(&["__EXCH_abcdef01_13"]);
    assert_eq!(
        resolve_exchange_table_name(3, &schemas),
        "__EXCHANGE_TABLE_3",
        "node_id=3 should not match table ending in _13"
    );
}

#[test]
fn test_no_ambiguity_node_13_matches_13() {
    let schemas = schemas_with(&["__EXCH_abcdef01_13"]);
    assert_eq!(
        resolve_exchange_table_name(13, &schemas),
        "__EXCH_abcdef01_13"
    );
}

#[test]
fn test_multiple_exchange_tables() {
    let schemas = schemas_with(&[
        "__EXCH_abcdef01_3",
        "__EXCH_abcdef01_7",
        "__EXCH_abcdef01_13",
    ]);
    assert_eq!(
        resolve_exchange_table_name(3, &schemas),
        "__EXCH_abcdef01_3"
    );
    assert_eq!(
        resolve_exchange_table_name(7, &schemas),
        "__EXCH_abcdef01_7"
    );
    assert_eq!(
        resolve_exchange_table_name(13, &schemas),
        "__EXCH_abcdef01_13"
    );
}

#[test]
fn test_legacy_format_matched() {
    let schemas = schemas_with(&["__EXCHANGE_TABLE_5"]);
    assert_eq!(
        resolve_exchange_table_name(5, &schemas),
        "__EXCHANGE_TABLE_5"
    );
}

#[test]
fn test_prefers_exch_over_legacy() {
    let schemas = schemas_with(&[
        "__EXCH_12345678_5",
        "__EXCHANGE_TABLE_5",
    ]);
    assert_eq!(
        resolve_exchange_table_name(5, &schemas),
        "__EXCH_12345678_5",
        "should prefer __EXCH_ format over legacy __EXCHANGE_TABLE_"
    );
}

#[test]
fn test_mixed_tables_ignores_non_exchange() {
    let schemas = schemas_with(&[
        "lineitem_1",
        "orders_2",
        "__EXCH_abcdef01_3",
        "nation_4",
    ]);
    assert_eq!(
        resolve_exchange_table_name(3, &schemas),
        "__EXCH_abcdef01_3"
    );
    // node_id=1 should not match "lineitem_1"
    assert_eq!(
        resolve_exchange_table_name(1, &schemas),
        "__EXCHANGE_TABLE_1"
    );
}

#[test]
fn test_node_id_zero() {
    let schemas = schemas_with(&["__EXCH_00000000_0"]);
    assert_eq!(
        resolve_exchange_table_name(0, &schemas),
        "__EXCH_00000000_0"
    );
}

#[test]
fn test_large_node_id() {
    let schemas = schemas_with(&["__EXCH_ffffffff_999"]);
    assert_eq!(
        resolve_exchange_table_name(999, &schemas),
        "__EXCH_ffffffff_999"
    );
}
