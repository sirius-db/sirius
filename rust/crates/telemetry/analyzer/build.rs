// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#[path = "../codegen.rs"]
mod codegen;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../model.yaml");
    println!("cargo:rerun-if-changed={}", model.display());
    let parsed = quent_yaml::parse_from_file(&model)?;
    let (_, ui) = codegen::generate(&parsed.schema, &["Task", "QueryActivity"])?;
    codegen::write("fsm_declarations.rs", ui)?;
    Ok(())
}
