//! The parquet files a fragment's `FILE_SCAN_NODE`s read, collected from the dispatch's scan
//! ranges and validated fail-closed.
//!
//! The FE assigns each scan node its ranges per fragment instance
//! (`TPipelineFragmentParams.local_params[i].per_node_scan_ranges[node_id]`), each range a
//! `TFileRangeDesc` with a path and a byte window, and the per-node scan parameters
//! (`file_scan_params[node_id]`: format, destination tuple, required slots) in the first
//! fragment of the batch. A `local()` TVF over parquet produces one whole-file range per file.
//!
//! What is accepted, and why the rest is refused rather than approximated:
//! - exactly one fragment instance: the ranges of several instances would have to be merged,
//!   and a single Substrait plan runs the scan once (`parallel_pipeline_task_num=1`);
//! - parquet only, and every range covers its whole file (`start_offset == 0`,
//!   `size == file_size`): DuckDB's `parquet_scan` reads whole files, so a byte-range split
//!   would either duplicate or drop rows;
//! - no `columns_from_path` (hive partition columns synthesized from the path) and only file
//!   slots in `required_slots`: the read's schema is the destination tuple's columns by name.

use std::collections::BTreeMap;

use doris_thrift::palo_internal_service::TPipelineFragmentParams;
use doris_thrift::plan_nodes::{TFileFormatType, TFileScanRangeParams};

use crate::error::{Result, TranslateError};

/// Parquet paths per scan node, in range order.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ScanRanges {
    /// File paths keyed by scan node id.
    paths: BTreeMap<i32, Vec<String>>,
}

impl ScanRanges {
    /// Collects and validates the scan ranges of every scan node in the fragment.
    pub fn from_params(params: &TPipelineFragmentParams) -> Result<Self> {
        let instances = params.local_params.as_deref().unwrap_or_default();
        let Some(instance) = instances.first() else {
            return Ok(Self::default());
        };
        if instances.len() != 1 {
            return Err(TranslateError::malformed(format!(
                "fragment has {} instances; one Substrait plan can only run one",
                instances.len()
            )));
        }
        let mut paths = BTreeMap::new();
        for (node_id, ranges) in &instance.per_node_scan_ranges {
            let shared_params = params
                .file_scan_params
                .as_ref()
                .and_then(|by_node| by_node.get(node_id));
            if let Some(scan_params) = shared_params {
                validate_scan_params(*node_id, scan_params)?;
            }
            let mut node_paths = Vec::new();
            for range in ranges {
                let file_range = range
                    .scan_range
                    .ext_scan_range
                    .as_ref()
                    .and_then(|ext| ext.file_scan_range.as_ref())
                    .ok_or(TranslateError::UnsupportedScanRange {
                        node_id: *node_id,
                        reason: "scan range is not a file scan range",
                    })?;
                // Older FEs ship the parameters inside every range instead of once per node.
                let range_params = file_range.params.as_ref().or(shared_params);
                if let Some(scan_params) = file_range.params.as_ref() {
                    validate_scan_params(*node_id, scan_params)?;
                }
                for desc in file_range.ranges.iter().flatten() {
                    let format = desc
                        .format_type
                        .or(range_params.and_then(|p| p.format_type));
                    if format != Some(TFileFormatType::FORMAT_PARQUET) {
                        return Err(TranslateError::UnsupportedScanRange {
                            node_id: *node_id,
                            reason: "only parquet files are supported",
                        });
                    }
                    if desc
                        .columns_from_path
                        .as_ref()
                        .is_some_and(|columns| !columns.is_empty())
                    {
                        return Err(TranslateError::UnsupportedScanRange {
                            node_id: *node_id,
                            reason: "columns synthesized from the file path are not supported",
                        });
                    }
                    let whole_file = desc.start_offset.unwrap_or(0) == 0
                        && desc.file_size.is_some()
                        && desc.size == desc.file_size;
                    if !whole_file {
                        return Err(TranslateError::UnsupportedScanRange {
                            node_id: *node_id,
                            reason: "byte-range splits are not supported; each range must cover its whole file",
                        });
                    }
                    let path = desc.path.as_deref().filter(|path| !path.is_empty()).ok_or(
                        TranslateError::MissingField {
                            context: "TFileRangeDesc",
                            field: "path",
                        },
                    )?;
                    node_paths.push(path.to_string());
                }
            }
            paths.insert(*node_id, node_paths);
        }
        Ok(Self { paths })
    }

    /// Builds ranges directly from paths (fixtures and tests).
    pub fn from_paths(paths: impl IntoIterator<Item = (i32, Vec<String>)>) -> Self {
        Self {
            paths: paths.into_iter().collect(),
        }
    }

    /// The parquet paths of a scan node; empty when the node has no ranges on this instance.
    pub fn for_node(&self, node_id: i32) -> &[String] {
        self.paths
            .get(&node_id)
            .map(Vec::as_slice)
            .unwrap_or_default()
    }

    /// Scan node ids that have ranges.
    pub fn node_ids(&self) -> impl Iterator<Item = i32> + '_ {
        self.paths.keys().copied()
    }
}

/// Refuses scan parameters whose column mapping is not "destination slots by file column name".
fn validate_scan_params(node_id: i32, scan_params: &TFileScanRangeParams) -> Result<()> {
    if scan_params
        .format_type
        .is_some_and(|format| format != TFileFormatType::FORMAT_PARQUET)
    {
        return Err(TranslateError::UnsupportedScanRange {
            node_id,
            reason: "only parquet files are supported",
        });
    }
    for slot in scan_params.required_slots.iter().flatten() {
        if slot.is_file_slot == Some(false) {
            return Err(TranslateError::UnsupportedScanRange {
                node_id,
                reason: "required slot is not a file column (partition or virtual column)",
            });
        }
    }
    if scan_params.pre_filter_exprs.is_some()
        || scan_params
            .pre_filter_exprs_list
            .as_ref()
            .is_some_and(|exprs| !exprs.is_empty())
    {
        return Err(TranslateError::UnsupportedScanRange {
            node_id,
            reason: "pre-filter expressions on the source tuple are not supported",
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use doris_thrift::palo_internal_service::{TPipelineInstanceParams, TScanRangeParams};
    use doris_thrift::plan_nodes::{
        TExternalScanRange, TFileRangeDesc, TFileScanRange, TFileScanSlotInfo, TScanRange,
    };
    use doris_thrift::types::TUniqueId;

    use super::*;

    fn range(path: &str, start: i64, size: i64, file_size: i64) -> TFileRangeDesc {
        TFileRangeDesc {
            path: Some(path.to_string()),
            start_offset: Some(start),
            size: Some(size),
            file_size: Some(file_size),
            format_type: Some(TFileFormatType::FORMAT_PARQUET),
            ..Default::default()
        }
    }

    fn instance(per_node: Vec<(i32, Vec<TFileRangeDesc>)>) -> TPipelineInstanceParams {
        TPipelineInstanceParams {
            fragment_instance_id: TUniqueId::new(1, 1),
            per_node_scan_ranges: per_node
                .into_iter()
                .map(|(node_id, ranges)| {
                    (
                        node_id,
                        vec![TScanRangeParams {
                            scan_range: TScanRange {
                                ext_scan_range: Some(TExternalScanRange {
                                    file_scan_range: Some(TFileScanRange {
                                        ranges: Some(ranges),
                                        ..Default::default()
                                    }),
                                }),
                                ..Default::default()
                            },
                            ..Default::default()
                        }],
                    )
                })
                .collect(),
            ..Default::default()
        }
    }

    fn params(instances: Vec<TPipelineInstanceParams>) -> TPipelineFragmentParams {
        TPipelineFragmentParams {
            query_id: TUniqueId::new(1, 2),
            local_params: Some(instances),
            ..Default::default()
        }
    }

    #[test]
    fn collects_whole_file_parquet_ranges_per_node() {
        let params = params(vec![instance(vec![
            (
                0,
                vec![
                    range("/data/lineitem/part.0.parquet", 0, 100, 100),
                    range("/data/lineitem/part.1.parquet", 0, 7, 7),
                ],
            ),
            (2, vec![range("/data/orders.parquet", 0, 5, 5)]),
        ])]);
        let ranges = ScanRanges::from_params(&params).unwrap();
        assert_eq!(
            ranges.for_node(0),
            [
                "/data/lineitem/part.0.parquet",
                "/data/lineitem/part.1.parquet"
            ]
        );
        assert_eq!(ranges.for_node(2), ["/data/orders.parquet"]);
        assert!(ranges.for_node(9).is_empty());
        assert_eq!(ranges.node_ids().collect::<Vec<_>>(), vec![0, 2]);
    }

    #[test]
    fn no_instances_means_no_ranges() {
        let params = params(vec![]);
        assert_eq!(
            ScanRanges::from_params(&params).unwrap(),
            ScanRanges::default()
        );
    }

    #[test]
    fn several_instances_are_rejected() {
        let params = params(vec![instance(vec![]), instance(vec![])]);
        assert!(matches!(
            ScanRanges::from_params(&params).unwrap_err(),
            TranslateError::MalformedPlan(_)
        ));
    }

    #[test]
    fn partial_ranges_non_parquet_and_path_columns_are_rejected() {
        let split = params(vec![instance(vec![(
            0,
            vec![range("/f.parquet", 0, 50, 100)],
        )])]);
        assert!(matches!(
            ScanRanges::from_params(&split).unwrap_err(),
            TranslateError::UnsupportedScanRange { node_id: 0, reason } if reason.contains("whole file")
        ));
        let mut csv = range("/f.csv", 0, 10, 10);
        csv.format_type = Some(TFileFormatType::FORMAT_CSV_PLAIN);
        let csv = params(vec![instance(vec![(0, vec![csv])])]);
        assert!(matches!(
            ScanRanges::from_params(&csv).unwrap_err(),
            TranslateError::UnsupportedScanRange { reason, .. } if reason.contains("parquet")
        ));
        let mut partitioned = range("/dt=2024/f.parquet", 0, 10, 10);
        partitioned.columns_from_path = Some(vec!["2024".to_string()]);
        let partitioned = params(vec![instance(vec![(0, vec![partitioned])])]);
        assert!(matches!(
            ScanRanges::from_params(&partitioned).unwrap_err(),
            TranslateError::UnsupportedScanRange { reason, .. } if reason.contains("path")
        ));
    }

    #[test]
    fn scan_params_must_map_file_columns_only() {
        let mut params = params(vec![instance(vec![(
            0,
            vec![range("/f.parquet", 0, 10, 10)],
        )])]);
        params.file_scan_params = Some(
            [(
                0,
                TFileScanRangeParams {
                    format_type: Some(TFileFormatType::FORMAT_PARQUET),
                    required_slots: Some(vec![TFileScanSlotInfo {
                        slot_id: Some(4),
                        is_file_slot: Some(false),
                        ..Default::default()
                    }]),
                    ..Default::default()
                },
            )]
            .into_iter()
            .collect(),
        );
        assert!(matches!(
            ScanRanges::from_params(&params).unwrap_err(),
            TranslateError::UnsupportedScanRange { reason, .. } if reason.contains("file column")
        ));
    }
}
