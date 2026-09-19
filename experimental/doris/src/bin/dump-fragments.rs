//! Pretty-prints a captured `exec_plan_fragment` payload (`batch-NN-request.tcompact` from
//! `SIRIUS_BE_DUMP_FRAGMENTS` / `tests/fixtures/tpch/qNN/`).
//!
//!   dump-fragments <file.tcompact>              every fragment in Debug form, root first
//!   dump-fragments --summary <file.tcompact>    one shape line per fragment
//!   dump-fragments --translate <file.tcompact>  each fragment's Substrait plan (explain text)
//!                                               or the translation error
//!   dump-fragments --stitch <file.tcompact>     the whole dispatch stitched into one plan
//!   dump-fragments --stitch --write-plan q.substrait [--rewrite-path OLD=NEW] <file.tcompact>
//!                                               also write the stitched plan's protobuf bytes
//!                                               (what the engine, or DuckDB's `from_substrait`
//!                                               in the CPU differential, consumes), with the
//!                                               captured scan paths re-rooted
//!
//! The file holds a TCompact `TPipelineFragmentParamsList`; it is decoded through the same
//! path the backend uses, so what prints here is what the translator sees.

use std::path::PathBuf;

use anyhow::{Context, Result, bail};
use clap::Parser;
use doris_proto::{PExecPlanFragmentRequest, PFragmentRequestVersion};
use doris_thrift::palo_internal_service::TPipelineFragmentParams;
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
    /// Translate each fragment on its own and print the Substrait explain text (or the error).
    #[arg(long)]
    translate: bool,
    /// Stitch the dispatch into one plan (MVP-A0) and print its Substrait explain text.
    #[arg(long)]
    stitch: bool,
    /// With --stitch: write the stitched plan as Substrait protobuf bytes to this file.
    #[arg(long, value_name = "FILE", requires = "stitch")]
    write_plan: Option<PathBuf>,
    /// Replace the prefix OLD of every scan range path with NEW before translating (the
    /// corpus was captured against /tmp/tpch-sf1; the plan must point at a local copy).
    #[arg(long, value_name = "OLD=NEW")]
    rewrite_path: Option<String>,
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
    let mut batch = decode_fragment_params_list(&request).map_err(anyhow::Error::msg)?;
    if let Some(rule) = &args.rewrite_path {
        let Some((old, new)) = rule.split_once('=') else {
            bail!("--rewrite-path expects OLD=NEW, got {rule:?}");
        };
        let mut rewritten = 0;
        for fragment in &mut batch.fragments {
            rewritten += rewrite_scan_paths(&mut fragment.params, old, new);
        }
        eprintln!("rewrote {rewritten} scan range path(s): {old} -> {new}");
    }
    println!("query_id={:x}-{:x}", batch.query_id.hi, batch.query_id.lo);
    println!("fragments={}", batch.fragments.len());
    let translator = doris_plan_translator::PlanTranslator::new();
    if args.stitch {
        let fragments: Vec<_> = batch.fragments.iter().map(|f| &f.params).collect();
        match translator.translate_batch(&fragments) {
            Ok(plan) => {
                println!("{}", plan.explain());
                if let Some(path) = &args.write_plan {
                    std::fs::write(path, plan.to_substrait_bytes())
                        .with_context(|| format!("failed to write {}", path.display()))?;
                    eprintln!("wrote {} (output {:?})", path.display(), plan.output_names);
                }
            }
            Err(err) => {
                println!("stitch error: {err}");
                if args.write_plan.is_some() {
                    bail!("no plan written: {err}");
                }
            }
        }
        return Ok(());
    }
    for fragment in &batch.fragments {
        println!("[{}] {}", fragment.index, fragment.shape());
        if args.translate {
            match translator.translate_fragment(&fragment.params) {
                Ok(plan) => println!("{}", plan.explain()),
                Err(err) => println!("translation error: {err}"),
            }
        } else if !args.summary {
            println!("{:#?}", fragment.params);
        }
    }
    Ok(())
}

/// Re-roots every `TFileRangeDesc.path` under `old` to `new`; returns how many changed.
fn rewrite_scan_paths(params: &mut TPipelineFragmentParams, old: &str, new: &str) -> usize {
    let mut count = 0;
    for instance in params.local_params.iter_mut().flatten() {
        for ranges in instance.per_node_scan_ranges.values_mut() {
            for range in ranges {
                let Some(file_range) = range
                    .scan_range
                    .ext_scan_range
                    .as_mut()
                    .and_then(|ext| ext.file_scan_range.as_mut())
                else {
                    continue;
                };
                for desc in file_range.ranges.iter_mut().flatten() {
                    if let Some(path) = desc.path.as_mut()
                        && let Some(rest) = path.strip_prefix(old)
                    {
                        *path = format!("{new}{rest}");
                        count += 1;
                    }
                }
            }
        }
    }
    count
}
