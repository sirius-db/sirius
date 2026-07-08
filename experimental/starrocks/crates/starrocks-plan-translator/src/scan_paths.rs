//! Per-scan-node parquet file paths collected from a fragment's broker scan ranges.

use std::collections::HashMap;

use starrocks_thrift::exprs::TExprNodeType;
use starrocks_thrift::internal_service::{TExecPlanFragmentParams, TScanRangeParams};
use starrocks_thrift::plan_nodes::{
    TBrokerRangeDesc, TBrokerScanRangeParams, TFileFormatType, TFileScanType,
};
use starrocks_thrift::types::TFileType;

use crate::descriptor_table::DescriptorTable;
use crate::error::{Result, TranslateError};

/// Parquet file paths for each scan node in a fragment, keyed by plan node id.
///
/// Built from the fragment's broker scan ranges so `FILE_SCAN` nodes emit
/// Substrait `local_files` reads DuckDB resolves as `parquet_scan(<paths>)`.
/// Nodes without broker ranges have no entry and fall back to a named-table read.
///
/// Collection fails closed: only the slice that maps faithfully onto a
/// `parquet_scan` over local files is accepted — a `FILES()` query (not a load),
/// reading whole local parquet files whose columns are direct passthroughs to the
/// scan tuple. Anything else (loads, byte-range splits, remote schemes, casts or
/// other column transforms, path-derived or flexibly-mapped columns) is rejected
/// with [`TranslateError::UnsupportedScanRange`] rather than silently producing
/// wrong results.
#[derive(Debug, Default)]
pub(crate) struct ScanFilePaths {
    by_node: HashMap<i32, Vec<String>>,
}

impl ScanFilePaths {
    /// Collects whole-file parquet paths from `params`' broker scan ranges.
    ///
    /// Ranges arrive either per node (`per_node_scan_ranges`) or, for pipeline
    /// fragments, per driver sequence (`node_to_per_driver_seq_scan_ranges`);
    /// both are collected.
    pub(crate) fn from_fragment(
        params: &TExecPlanFragmentParams,
        desc: &DescriptorTable,
    ) -> Result<Self> {
        let mut paths = Self::default();
        let Some(exec_params) = params.params.as_ref() else {
            return Ok(paths);
        };
        for (node_id, ranges) in &exec_params.per_node_scan_ranges {
            paths.add_ranges(*node_id, ranges, desc)?;
        }
        if let Some(per_driver) = exec_params.node_to_per_driver_seq_scan_ranges.as_ref() {
            for (node_id, per_seq) in per_driver {
                // A node must appear in exactly one of the two scan-range maps.
                // StarRocks keeps them disjoint (LocalFragmentAssignmentStrategy),
                // but that contract is invisible here, so enforce it locally: a node
                // in both would have its whole-file paths collected — and read —
                // twice, silently duplicating rows.
                if exec_params.per_node_scan_ranges.contains_key(node_id) {
                    return Err(Self::unsupported(
                        *node_id,
                        "scan node appears in both per-node and per-driver scan-range maps",
                    ));
                }
                for ranges in per_seq.values() {
                    paths.add_ranges(*node_id, ranges, desc)?;
                }
            }
        }
        Ok(paths)
    }

    /// Returns the parquet file paths collected for `node_id`, empty when the node
    /// has no broker scan ranges (it then falls back to a named-table read).
    pub(crate) fn for_node(&self, node_id: i32) -> &[String] {
        self.by_node
            .get(&node_id)
            .map(Vec::as_slice)
            .unwrap_or_default()
    }

    /// Validates and appends the whole-file parquet paths in `ranges` to `node_id`.
    fn add_ranges(
        &mut self,
        node_id: i32,
        ranges: &[TScanRangeParams],
        desc: &DescriptorTable,
    ) -> Result<()> {
        for range in ranges {
            let Some(broker) = range.scan_range.broker_scan_range.as_ref() else {
                continue;
            };
            Self::check_params(node_id, &broker.params, desc)?;
            for range_desc in &broker.ranges {
                Self::check_range(node_id, range_desc)?;
                self.by_node
                    .entry(node_id)
                    .or_default()
                    .push(range_desc.path.clone());
            }
        }
        Ok(())
    }

    /// Rejects scan-range params outside the supported `FILES()`-query slice.
    fn check_params(
        node_id: i32,
        params: &TBrokerScanRangeParams,
        desc: &DescriptorTable,
    ) -> Result<()> {
        // Only FILES() *query* scans are plain reads; LOAD and FILES_INSERT carry
        // load semantics (strict mode, transforms) this translation does not model.
        if params.file_scan_type != Some(TFileScanType::FILES_QUERY) {
            return Err(Self::unsupported(
                node_id,
                "only FILES() query scans are supported",
            ));
        }
        // A FILES() query sets `use_broker` explicitly: `Some(false)` is direct
        // access (what Sirius's reader does), `Some(true)` routes through a broker
        // process, and an absent value also selects the broker filesystem. Require
        // explicit direct access.
        if params.use_broker != Some(false) {
            return Err(Self::unsupported(
                node_id,
                "only direct (non-broker) file access is supported",
            ));
        }
        // Flexible column mapping is name-based with null filling for missing
        // columns — semantics a positional/by-name `parquet_scan` cannot reproduce.
        if params.flexible_column_mapping == Some(true) {
            return Err(Self::unsupported(
                node_id,
                "flexible column mapping is not supported",
            ));
        }
        // Each destination slot must be a direct passthrough of the file column of
        // the *same name*. The emitted read carries no per-column expression, and
        // the reader binds parquet columns to the destination names, so:
        //   - a cast, default, or other transform (anything but a bare slot
        //     reference) would be silently dropped, and
        //   - a bare reference to a differently-named source column would read the
        //     wrong column under that name.
        // Reject both rather than produce wrong values.
        if let Some(exprs) = params.expr_of_dest_slot.as_ref() {
            for (dest_slot_id, expr) in exprs {
                let slot_ref = match expr.nodes.as_slice() {
                    [node] if node.node_type == TExprNodeType::SLOT_REF => node.slot_ref.as_ref(),
                    _ => None,
                };
                let Some(slot_ref) = slot_ref else {
                    return Err(Self::unsupported(
                        node_id,
                        "scan column with a cast or transform is not supported",
                    ));
                };
                let dest_name = desc
                    .slot(params.dest_tuple_id, *dest_slot_id)?
                    .output_name();
                let src_name = desc
                    .slot(slot_ref.tuple_id, slot_ref.slot_id)?
                    .output_name();
                if dest_name != src_name {
                    return Err(Self::unsupported(
                        node_id,
                        "renamed or reordered column mapping is not supported",
                    ));
                }
            }
        }
        Ok(())
    }

    /// Rejects a single broker range outside the supported whole-file local-parquet slice.
    fn check_range(node_id: i32, desc: &TBrokerRangeDesc) -> Result<()> {
        if desc.format_type != TFileFormatType::FORMAT_PARQUET {
            return Err(Self::unsupported(
                node_id,
                "only parquet broker scan ranges are supported",
            ));
        }
        // A FILES() query carries a broker file descriptor (even for local paths);
        // a stream (or other) descriptor is a load shape, not a readable file scan.
        if desc.file_type != TFileType::FILE_BROKER {
            return Err(Self::unsupported(
                node_id,
                "only broker-descriptor file scan ranges are supported",
            ));
        }
        // Whole-file only: DuckDB's `local_files` reader ignores per-file byte
        // offsets, so accept a range only when it provably covers the whole,
        // non-empty file. A FILES() query stats each file, so require a known
        // positive `file_size` reached by reading to EOF (`size == -1`) or by a
        // bounded size at least that large. Unknown size, an empty file, or a
        // bounded size short of the file are rejected rather than silently reading
        // the wrong bytes.
        let whole_file = desc.start_offset == 0
            && match desc.file_size {
                Some(file_size) => file_size > 0 && (desc.size == -1 || desc.size >= file_size),
                None => false,
            };
        if !whole_file {
            return Err(Self::unsupported(
                node_id,
                "byte-range split scan ranges are not supported",
            ));
        }
        // StarRocks appends path-derived columns after the physical parquet
        // columns; `parquet_scan` over the path alone would not produce them, so
        // the descriptor slot order would no longer match.
        if desc
            .columns_from_path
            .as_ref()
            .is_some_and(|columns| !columns.is_empty())
        {
            return Err(Self::unsupported(
                node_id,
                "path-derived columns (columns_from_path) are not supported",
            ));
        }
        Self::check_local_path(node_id, &desc.path)
    }

    /// Rejects non-local URI schemes and glob metacharacters in a scan path.
    fn check_local_path(node_id: i32, path: &str) -> Result<()> {
        // `parquet_scan` treats `*`, `?` and `[` as globs; an already-expanded
        // literal path containing them would re-expand and read the wrong files.
        if path.contains(['*', '?', '[']) {
            return Err(Self::unsupported(
                node_id,
                "glob metacharacters in scan paths are not supported",
            ));
        }
        // Only local files reach Sirius's reader without access configuration. Allow
        // a bare path and the local `file:` forms; reject every other scheme
        // (s3://, hdfs://, hdfs:/, oss://, ...) and any non-local `file://`
        // authority, whose credentials/endpoints are not propagated.
        //
        // A scheme is `<scheme>:` immediately followed by `/` (so a bare path or a
        // filename containing `:` is not mistaken for one).
        if let Some(colon) = path.find(':') {
            let scheme = &path[..colon];
            let after = &path[colon + 1..];
            let is_scheme = after.starts_with('/')
                && scheme
                    .bytes()
                    .next()
                    .is_some_and(|b| b.is_ascii_alphabetic())
                && scheme
                    .bytes()
                    .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'+' | b'-' | b'.'));
            if is_scheme {
                if !scheme.eq_ignore_ascii_case("file") {
                    return Err(Self::unsupported(
                        node_id,
                        "only local file paths are supported",
                    ));
                }
                // `file://<authority>/...`: only an empty authority or `localhost`
                // is local. `file:/...` (no `//`) has no authority and is local.
                if let Some(after_slashes) = after.strip_prefix("//") {
                    let authority = after_slashes.split('/').next().unwrap_or_default();
                    if !(authority.is_empty() || authority.eq_ignore_ascii_case("localhost")) {
                        return Err(Self::unsupported(
                            node_id,
                            "only local file paths are supported",
                        ));
                    }
                }
            }
        }
        Ok(())
    }

    /// Builds an [`TranslateError::UnsupportedScanRange`] for `node_id`.
    fn unsupported(node_id: i32, reason: &'static str) -> TranslateError {
        TranslateError::UnsupportedScanRange { node_id, reason }
    }
}
