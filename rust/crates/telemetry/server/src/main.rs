use std::{
    net::ToSocketAddrs,
    path::{Path, PathBuf},
};

use clap::Parser;
use instrumentation_model::SiriusEvent;
use quent_collector::server::CollectorServiceOptions;
use quent_exporter::{
    ExporterOptions, ImporterOptions, MsgpackExporterOptions, MsgpackImporterOptions,
    NdjsonExporterOptions, NdjsonImporterOptions, PostcardExporterOptions, PostcardImporterOptions,
    create_importer,
};
use quent_query_engine_server::{
    analyzer_service_router, collector_service,
    error::{ServerError, ServerResult},
    initialize_tracing,
};
use sirius_telemetry_analyzer::SiriusUiAnalyzer;
use tokio::net::TcpListener;
use uuid::Uuid;

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

const TELEMETRY_EXTENSIONS: [&str; 3] = ["ndjson", "msgpack", "postcard"];

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

    let exporter = match exporter.as_str() {
        "ndjson" => ExporterOptions::Ndjson(NdjsonExporterOptions { output_dir }),
        "msgpack" => ExporterOptions::Msgpack(MsgpackExporterOptions { output_dir }),
        "postcard" => ExporterOptions::Postcard(PostcardExporterOptions { output_dir }),
        other => return Err(format!("unknown exporter: {other}").into()),
    };

    let collector = async {
        collector_service::<SiriusEvent>(CollectorServiceOptions { exporter })?
            .serve(collector_addr)
            .await
            .map_err(|error| -> Box<dyn std::error::Error> { Box::new(error) })
    };

    let lister = move || {
        let mut ids = std::collections::HashSet::new();
        for path in telemetry_file_paths(&lister_output_dir)? {
            let id = extract_engine_id(&path)?.or_else(|| telemetry_file_stem_uuid(&path));
            ids.extend(id);
        }
        Ok(ids.into_iter().collect())
    };

    let importer = move |engine_id| {
        let importer = importer_options_for_engine(&importer_output_dir, engine_id)?;
        Ok(Box::new(create_importer::<SiriusEvent>(&importer)?) as Box<dyn Iterator<Item = _>>)
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

fn telemetry_file_paths(output_dir: &Path) -> ServerResult<Vec<PathBuf>> {
    let mut paths = Vec::new();
    for entry in std::fs::read_dir(output_dir)? {
        let path = entry?.path();
        if path.is_file()
            && path
                .extension()
                .and_then(|extension| extension.to_str())
                .is_some_and(|extension| TELEMETRY_EXTENSIONS.contains(&extension))
        {
            paths.push(path);
        }
    }
    paths.sort();
    Ok(paths)
}

fn telemetry_file_stem_uuid(path: &Path) -> Option<Uuid> {
    path.file_stem()
        .and_then(|stem| stem.to_str())
        .and_then(|stem| Uuid::parse_str(stem).ok())
}

fn importer_options_for_path(path: PathBuf) -> Option<ImporterOptions> {
    match path.extension().and_then(|extension| extension.to_str()) {
        Some("postcard") => Some(ImporterOptions::Postcard(PostcardImporterOptions { path })),
        Some("msgpack") => Some(ImporterOptions::Msgpack(MsgpackImporterOptions { path })),
        Some("ndjson") => Some(ImporterOptions::Ndjson(NdjsonImporterOptions { path })),
        _ => None,
    }
}

fn extract_engine_id(path: &Path) -> ServerResult<Option<Uuid>> {
    let Some(importer) = importer_options_for_path(path.to_path_buf()) else {
        return Ok(None);
    };
    for event in create_importer::<SiriusEvent>(&importer)? {
        if let SiriusEvent::Engine(_) = event.data {
            return Ok(Some(event.id));
        }
    }
    Ok(None)
}

fn importer_options_for_engine(
    output_dir: &Path,
    engine_id: Uuid,
) -> ServerResult<ImporterOptions> {
    for path in telemetry_file_paths(output_dir)? {
        if extract_engine_id(&path)? == Some(engine_id) {
            return importer_options_for_path(path).ok_or_else(|| {
                ServerError::Cache("telemetry file has no supported extension".to_string())
            });
        }
    }

    for extension in ["postcard", "msgpack", "ndjson"] {
        let path = output_dir.join(format!("{engine_id}.{extension}"));
        if path.exists() {
            return importer_options_for_path(path).ok_or_else(|| {
                ServerError::Cache("telemetry file has no supported extension".to_string())
            });
        }
    }

    Err(ServerError::Io(std::io::Error::new(
        std::io::ErrorKind::NotFound,
        format!("no telemetry file found for engine {engine_id}"),
    )))
}
