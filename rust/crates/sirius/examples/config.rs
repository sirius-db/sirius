//! Load a configuration without initializing the engine or accessing GPUs.

use sirius::{ConfigError, ContextConfigBuilder};

fn main() -> Result<(), ConfigError> {
    let builder = match std::env::args_os().nth(1) {
        Some(path) => ContextConfigBuilder::from_yaml(path)?,
        None => ContextConfigBuilder::new()?,
    };
    let _config = builder.build()?;
    println!("Configuration loaded successfully.");
    Ok(())
}
