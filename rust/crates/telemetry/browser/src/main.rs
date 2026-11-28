// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Convert Sirius telemetry output into a `.quent-browser` file.

use std::path::PathBuf;
use std::process::ExitCode;

use clap::Parser;
use sirius_telemetry_browser::{MODEL_NAME, generate::generate_bundle};

/// Convert a Sirius telemetry directory or Quent ZIP into a browser bundle.
///
/// The input must contain exactly one Sirius engine. Upload the result with a
/// `quent-browser` role; the printed model name, schema digest, and format version
/// select the analyzer that opens it.
#[derive(Parser)]
#[command(version)]
struct Args {
    /// Telemetry output directory (containing one directory per context) or Quent ZIP.
    input: PathBuf,
    /// Browser bundle to write.
    #[arg(short, long)]
    output: PathBuf,
}

fn main() -> ExitCode {
    let args = Args::parse();
    match generate_bundle(&args.input, &args.output) {
        Ok(summary) => {
            println!("wrote {}", args.output.display());
            println!("model_name: {MODEL_NAME}");
            println!("schema_digest: {}", summary.schema_digest);
            println!("format_version: {}", summary.format_version);
            println!("engine_id: {}", summary.engine_id);
            println!("contexts: {}", summary.context_count);
            println!("events: {}", summary.event_count);
            ExitCode::SUCCESS
        }
        Err(error) => {
            eprintln!("error: {error}");
            ExitCode::FAILURE
        }
    }
}
