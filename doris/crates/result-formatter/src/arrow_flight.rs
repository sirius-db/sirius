//! Arrow Flight result server.
//!
//! Serves query results from GPU execution as Arrow Flight streams.
//! The Doris FE connects to this server after calling `fetch_arrow_flight_schema`
//! on the gRPC service, using the fragment instance ID as the Flight ticket.

use std::pin::Pin;

use arrow_flight::flight_service_server::{FlightService, FlightServiceServer};
use arrow_flight::utils::batches_to_flight_data;
use arrow_flight::{
    Action, ActionType, Criteria, Empty, FlightData, FlightDescriptor, FlightInfo,
    HandshakeRequest, HandshakeResponse, PutResult, SchemaResult, Ticket,
};
use tonic::transport::Server;
use tonic::{Request, Response, Status, Streaming};
use tracing::{info, warn};

use crate::result_store::{FinstId, ResultStore};

type BoxStream<T> = Pin<Box<dyn futures::Stream<Item = Result<T, Status>> + Send + 'static>>;

/// Arrow Flight service implementation for Doris result delivery.
pub struct SiriusFlightService {
    store: ResultStore,
}

impl SiriusFlightService {
    pub fn new(store: ResultStore) -> Self {
        Self { store }
    }
}

#[tonic::async_trait]
impl FlightService for SiriusFlightService {
    type HandshakeStream = BoxStream<HandshakeResponse>;
    type ListFlightsStream = BoxStream<FlightInfo>;
    type DoGetStream = BoxStream<FlightData>;
    type DoPutStream = BoxStream<PutResult>;
    type DoActionStream = BoxStream<arrow_flight::Result>;
    type ListActionsStream = BoxStream<ActionType>;
    type DoExchangeStream = BoxStream<FlightData>;

    /// Stream result data for a fragment instance.
    ///
    /// The ticket contains a 16-byte fragment instance ID (hi + lo, little-endian).
    async fn do_get(
        &self,
        request: Request<Ticket>,
    ) -> Result<Response<Self::DoGetStream>, Status> {
        let ticket = request.into_inner();
        let finst_id = FinstId::from_bytes(&ticket.ticket).ok_or_else(|| {
            Status::invalid_argument("ticket must be 16 bytes (finst_id hi:lo LE)")
        })?;

        info!(%finst_id, "do_get: streaming results");

        let entry = self.store.get(&finst_id).ok_or_else(|| {
            warn!(%finst_id, "do_get: result not found");
            Status::not_found(format!("no result for finst_id {finst_id}"))
        })?;

        let flight_data =
            batches_to_flight_data(&entry.schema, entry.batches.clone())
                .map_err(|e| Status::internal(format!("failed to convert to flight data: {e}")))?;

        let stream = futures::stream::iter(flight_data.into_iter().map(Ok));
        Ok(Response::new(Box::pin(stream)))
    }

    async fn handshake(
        &self,
        _request: Request<Streaming<HandshakeRequest>>,
    ) -> Result<Response<Self::HandshakeStream>, Status> {
        let stream = futures::stream::empty();
        Ok(Response::new(Box::pin(stream)))
    }

    async fn list_flights(
        &self,
        _request: Request<Criteria>,
    ) -> Result<Response<Self::ListFlightsStream>, Status> {
        Err(Status::unimplemented("list_flights"))
    }

    async fn get_flight_info(
        &self,
        _request: Request<FlightDescriptor>,
    ) -> Result<Response<FlightInfo>, Status> {
        Err(Status::unimplemented("get_flight_info"))
    }

    async fn get_schema(
        &self,
        _request: Request<FlightDescriptor>,
    ) -> Result<Response<SchemaResult>, Status> {
        Err(Status::unimplemented("get_schema"))
    }

    async fn do_put(
        &self,
        _request: Request<Streaming<FlightData>>,
    ) -> Result<Response<Self::DoPutStream>, Status> {
        Err(Status::unimplemented("do_put"))
    }

    async fn do_action(
        &self,
        _request: Request<Action>,
    ) -> Result<Response<Self::DoActionStream>, Status> {
        Err(Status::unimplemented("do_action"))
    }

    async fn list_actions(
        &self,
        _request: Request<Empty>,
    ) -> Result<Response<Self::ListActionsStream>, Status> {
        Err(Status::unimplemented("list_actions"))
    }

    async fn do_exchange(
        &self,
        _request: Request<Streaming<FlightData>>,
    ) -> Result<Response<Self::DoExchangeStream>, Status> {
        Err(Status::unimplemented("do_exchange"))
    }

    async fn poll_flight_info(
        &self,
        _request: Request<FlightDescriptor>,
    ) -> Result<Response<arrow_flight::PollInfo>, Status> {
        Err(Status::unimplemented("poll_flight_info"))
    }
}

/// Start the Arrow Flight result server.
pub async fn start_flight_server(
    listen_addr: &str,
    store: ResultStore,
) -> Result<(), Box<dyn std::error::Error>> {
    let addr = listen_addr.parse()?;
    let service = SiriusFlightService::new(store);

    info!(addr = listen_addr, "starting Arrow Flight server");

    Server::builder()
        .add_service(FlightServiceServer::new(service))
        .serve(addr)
        .await?;

    Ok(())
}
