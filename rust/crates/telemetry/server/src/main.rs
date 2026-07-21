use std::{net::ToSocketAddrs, path::PathBuf};

use clap::Parser;
use instrumentation_model::{Sirius, SiriusContext};
use quent_io::{ExporterOptions, FileSystemExporterOptions, FileSystemFormat};
use quent_query_engine_server::{
    analyzer_cache::index_query_engines, analyzer_service_router, collector_service,
    initialize_tracing,
};
use sirius_telemetry_analyzer::SiriusUiAnalyzer;
use tokio::net::TcpListener;

mod defaults {
    pub(crate) const QUENT_COLLECTOR_ADDRESS: &str = "[::]:7836";
    pub(crate) const QUENT_ANALYZER_ADDRESS: &str = "[::]:8080";
}

mod env {
    pub(crate) const QUENT_COLLECTOR_ADDRESS: &str = "QUENT_COLLECTOR_ADDRESS";
    pub(crate) const QUENT_COLLECTOR_OUTPUT_DIR: &str = "QUENT_COLLECTOR_OUTPUT_DIR";
    pub(crate) const QUENT_COLLECTOR_EXPORTER: &str = "QUENT_COLLECTOR_EXPORTER";
    pub(crate) const QUENT_ANALYZER_ADDRESS: &str = "QUENT_ANALYZER_ADDRESS";
    pub(crate) const QUENT_ANALYZER_CORS_ADDRESS: &str = "QUENT_ANALYZER_CORS_ADDRESS";
}

#[derive(Parser)]
struct Args {
    #[arg(long, default_value = "info")]
    log_level: String,

    #[arg(long, default_value = defaults::QUENT_COLLECTOR_ADDRESS, env = env::QUENT_COLLECTOR_ADDRESS)]
    collector_address: String,

    #[arg(long, default_value = "ndjson", env = env::QUENT_COLLECTOR_EXPORTER)]
    exporter: String,

    #[arg(long, default_value = "data", env = env::QUENT_COLLECTOR_OUTPUT_DIR)]
    output_dir: PathBuf,

    #[arg(long, default_value = defaults::QUENT_ANALYZER_ADDRESS, env = env::QUENT_ANALYZER_ADDRESS)]
    analyzer_address: String,

    #[arg(long, env = env::QUENT_ANALYZER_CORS_ADDRESS)]
    cors_address: Option<String>,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let Args {
        log_level,
        cors_address,
        collector_address,
        exporter,
        output_dir,
        analyzer_address,
    } = Args::parse();

    initialize_tracing(&log_level);

    let collector_addr = collector_address
        .to_socket_addrs()?
        .next()
        .ok_or_else(|| format!("unable to resolve socket address: {collector_address}"))?;
    let analyzer_addr = analyzer_address
        .to_socket_addrs()?
        .next()
        .ok_or_else(|| format!("unable to resolve socket address: {analyzer_address}"))?;

    let importer_output_dir = output_dir.clone();
    let lister_output_dir = output_dir.clone();

    let format = match exporter.as_str() {
        "ndjson" => FileSystemFormat::Ndjson,
        "msgpack" => FileSystemFormat::Msgpack,
        "postcard" => FileSystemFormat::Postcard,
        other => return Err(format!("unknown exporter: {other}").into()),
    };
    // Each context exports under `output_dir/<context-id>/<entity>/`, so the
    // collector writes per-entity streams beneath output_dir.
    let exporter_kind =
        ExporterOptions::FileSystem(FileSystemExporterOptions::new(format, output_dir));

    // The collector builds a fresh sink per incoming context, replaying each
    // remote source's events under that source's own context id.
    let collector = async {
        collector_service::<SiriusContext, _>(move |id| {
            SiriusContext::try_with_id(id, Some(exporter_kind.clone())).map_err(|e| e.to_string())
        })?
        .serve(collector_addr)
        .await
        .map_err(|error| -> Box<dyn std::error::Error> { Box::new(error) })
    };

    // Index the exported contexts by engine instance: each engine's telemetry is
    // the engine's own context plus its workers' contexts.
    let lister = move || index_query_engines(&lister_output_dir);

    // Reconstruct one context's umbrella event stream from its per-entity
    // subdirectories; the analyzer cache chains this across all the contexts that
    // make up an engine instance.
    let importer = move |context_id| {
        let dir = importer_output_dir.join(format!("{context_id}"));
        Ok(Sirius::import_events(&dir)?)
    };

    let analyzer = async {
        axum::serve(
            TcpListener::bind(analyzer_addr).await?,
            analyzer_service_router::<SiriusUiAnalyzer>(
                Box::new(importer),
                Box::new(lister),
                cors_address,
            )?
            .into_make_service(),
        )
        .await?;
        Ok::<(), Box<dyn std::error::Error>>(())
    };

    tracing::info!("listening on {collector_addr} and {analyzer_addr}");
    tokio::try_join!(collector, analyzer)?;
    Ok(())
}
