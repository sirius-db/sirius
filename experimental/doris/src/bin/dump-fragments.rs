//! Pretty-prints a captured `exec_plan_fragment` payload (`batch-NN-request.tcompact` from
//! `SIRIUS_BE_DUMP_FRAGMENTS` / `tests/fixtures/tpch/qNN/`).
//!
//!   dump-fragments <file.tcompact>            every fragment in Debug form, root first
//!   dump-fragments --summary <file.tcompact>  one shape line per fragment
//!
//! The file holds a TCompact `TPipelineFragmentParamsList`; it is decoded through the same
//! path the backend uses, so what prints here is what the translator sees.

use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::Parser;
use doris_proto::{PExecPlanFragmentRequest, PFragmentRequestVersion};
use sirius_doris_be::decode_fragment_params_list;

#[derive(Debug, Parser)]
struct Args {
    /// Captured dispatch payload (TCompact `TPipelineFragmentParamsList`).
    file: PathBuf,
    /// Print one shape line per fragment instead of the full Debug form.
    #[arg(long)]
    summary: bool,
    /// The payload is thrift binary rather than compact (`use_compact_thrift_rpc=false`).
    #[arg(long)]
    binary: bool,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let bytes = std::fs::read(&args.file)
        .with_context(|| format!("failed to read {}", args.file.display()))?;
    let request = PExecPlanFragmentRequest {
        request: Some(bytes),
        compact: Some(!args.binary),
        version: Some(PFragmentRequestVersion::Version3 as i32),
    };
    let batch = decode_fragment_params_list(&request).map_err(anyhow::Error::msg)?;
    println!("query_id={:x}-{:x}", batch.query_id.hi, batch.query_id.lo);
    println!("fragments={}", batch.fragments.len());
    for fragment in &batch.fragments {
        println!("[{}] {}", fragment.index, fragment.shape());
        if !args.summary {
            println!("{:#?}", fragment.params);
        }
    }
    Ok(())
}
