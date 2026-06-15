use std::io;

#[cfg(test)]
use crate::proto::brpc::policy::RpcRequestMeta;
use crate::proto::brpc::policy::{RpcMeta, RpcResponseMeta};
use anyhow::{Context, Result, anyhow};
use prost::Message;
#[cfg(test)]
use std::io::Read;
use tokio::io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt};

/// Magic prefix for Baidu PRPC frames. StarRocks FE sends backend RPCs with
/// this raw TCP frame format rather than gRPC over HTTP/2.
const PRPC_MAGIC: &[u8; 4] = b"PRPC";
const PRPC_HEAD_SIZE: usize = 12;
const MAX_PRPC_MESSAGE_SIZE: usize = 256 * 1024 * 1024;
// Cap the up-front body allocation so a frame that declares a large size but never sends the
// bytes cannot pin `MAX_PRPC_MESSAGE_SIZE` per connection; the buffer grows as data arrives.
const PRPC_READ_CHUNK: usize = 64 * 1024;
const PRPC_SUCCESS: i32 = 0;
// brpc/baidu-rpc status codes: ENOSERVICE/ENOMETHOD are distinct, and EINTERNAL covers the rest.
const PRPC_SERVICE_NOT_FOUND: i32 = 1001;
const PRPC_METHOD_NOT_FOUND: i32 = 1002;
const PRPC_ERROR: i32 = 2001;

/// Decoded PRPC request passed from the transport layer to a Tower service.
#[derive(Clone, Debug)]
pub(crate) struct Request {
    /// BRPC service name from request metadata.
    pub(crate) service_name: String,
    /// BRPC method name from request metadata.
    pub(crate) method_name: String,
    /// Protobuf-encoded method request body.
    pub(crate) body: Vec<u8>,
    /// Optional BRPC attachment; StarRocks sends thrift plan fragments here.
    pub(crate) attachment: Vec<u8>,
}

impl Request {
    /// Builds a decoded request for service dispatch and tests.
    pub(crate) fn new(
        service_name: impl Into<String>,
        method_name: impl Into<String>,
        body: Vec<u8>,
        attachment: Vec<u8>,
    ) -> Self {
        Self {
            service_name: service_name.into(),
            method_name: method_name.into(),
            body,
            attachment,
        }
    }
}

/// Service response before it is wrapped back into a PRPC frame.
#[derive(Clone, Debug)]
pub(crate) struct Response {
    /// Protobuf-encoded method response body.
    pub(crate) body: Vec<u8>,
    /// Optional response attachment. The current CN plan path does not use it.
    pub(crate) attachment: Vec<u8>,
}

impl Response {
    /// Builds a protobuf-body-only response.
    pub(crate) fn new(body: Vec<u8>) -> Self {
        Self {
            body,
            attachment: Vec::new(),
        }
    }
}

/// PRPC-level error, separate from StarRocks StatusPB method failures.
#[derive(Clone, Debug)]
pub(crate) struct Error {
    /// BRPC/PRPC error code returned in response metadata.
    code: i32,
    /// Human-readable error text returned in response metadata.
    text: String,
}

impl Error {
    /// Returns a PRPC error for an unknown service name.
    pub(crate) fn service_not_found(text: impl Into<String>) -> Self {
        Self {
            code: PRPC_SERVICE_NOT_FOUND,
            text: text.into(),
        }
    }

    /// Returns a PRPC error for an unknown method name.
    pub(crate) fn method_not_found(text: impl Into<String>) -> Self {
        Self {
            code: PRPC_METHOD_NOT_FOUND,
            text: text.into(),
        }
    }

    /// Returns the default generated-service response for an RPC we do not implement.
    pub(crate) fn method_not_implemented(method_name: impl std::fmt::Display) -> Self {
        Self::method_not_found(format!(
            "method '{method_name}' is not implemented by the Rust CN"
        ))
    }

    /// Returns a PRPC server error for a protobuf body that does not decode.
    pub(crate) fn invalid_request(
        method_name: impl std::fmt::Display,
        err: impl std::fmt::Display,
    ) -> Self {
        Self::server(format!("invalid {method_name} request: {err}"))
    }

    /// Returns a generic PRPC server error.
    fn server(text: impl Into<String>) -> Self {
        Self {
            code: PRPC_ERROR,
            text: text.into(),
        }
    }
}

impl std::fmt::Display for Error {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.text)
    }
}

impl std::error::Error for Error {}

/// Raw PRPC frame after the 12-byte TCP header has been decoded.
#[derive(Debug)]
pub(crate) struct Frame {
    /// BRPC metadata protobuf decoded from the frame prefix.
    meta: RpcMeta,
    /// Protobuf-encoded RPC request or response body.
    body: Vec<u8>,
    /// Optional raw attachment bytes following the protobuf body.
    attachment: Vec<u8>,
}

impl Frame {
    /// The PRPC correlation id, used to match a response frame back to its request.
    pub(crate) fn correlation_id(&self) -> Option<i64> {
        self.meta.correlation_id
    }

    /// Consumes the frame, moving its metadata, protobuf body, and attachment into a service
    /// request without cloning the (potentially large) body or attachment.
    pub(crate) fn into_request(self) -> std::result::Result<Request, Error> {
        let Some(request) = self.meta.request else {
            return Err(Error::server("missing PRPC request metadata"));
        };

        Ok(Request::new(
            request.service_name,
            request.method_name,
            self.body,
            self.attachment,
        ))
    }

    /// Builds a response frame for `correlation_id`, wrapping a service result in PRPC response
    /// metadata. `encode` is the single source of truth for `attachment_size`, so it is left unset.
    pub(crate) fn response_frame(
        correlation_id: Option<i64>,
        response: std::result::Result<Response, Error>,
    ) -> Self {
        let (error_code, error_text, body, attachment) = match response {
            Ok(response) => (PRPC_SUCCESS, None, response.body, response.attachment),
            Err(error) => (error.code, Some(error.text), Vec::new(), Vec::new()),
        };
        Self {
            meta: RpcMeta {
                request: None,
                response: Some(RpcResponseMeta {
                    error_code: Some(error_code),
                    error_text,
                }),
                compress_type: Some(0),
                correlation_id,
                attachment_size: None,
                chunk_info: None,
                authentication_data: None,
                stream_settings: None,
                user_fields: Default::default(),
            },
            body,
            attachment,
        }
    }

    /// Builds a request frame without going through the TCP reader.
    #[cfg(test)]
    pub(crate) fn for_request(
        service_name: impl Into<String>,
        method_name: impl Into<String>,
        body: Vec<u8>,
        attachment: Vec<u8>,
        correlation_id: Option<i64>,
    ) -> Self {
        Self {
            meta: RpcMeta {
                request: Some(RpcRequestMeta {
                    service_name: service_name.into(),
                    method_name: method_name.into(),
                    log_id: None,
                    trace_id: None,
                    span_id: None,
                    parent_span_id: None,
                    request_id: None,
                    timeout_ms: None,
                }),
                response: None,
                compress_type: Some(0),
                correlation_id,
                attachment_size: None,
                chunk_info: None,
                authentication_data: None,
                stream_settings: None,
                user_fields: Default::default(),
            },
            body,
            attachment,
        }
    }

    /// Reads one PRPC frame from a stream, returning `None` on normal connection close.
    #[cfg(test)]
    pub(crate) fn read(stream: &mut impl Read) -> Result<Option<Self>> {
        let mut header = [0u8; PRPC_HEAD_SIZE];
        match stream.read_exact(&mut header) {
            Ok(()) => {}
            Err(err)
                if matches!(
                    err.kind(),
                    io::ErrorKind::UnexpectedEof
                        | io::ErrorKind::ConnectionReset
                        | io::ErrorKind::ConnectionAborted
                ) =>
            {
                return Ok(None);
            }
            Err(err) => return Err(err).context("failed to read PRPC header"),
        }

        let sizes = FrameSizes::parse_header(&header)?;
        let payload = read_payload_sync(stream, sizes.message_size)?;
        Self::decode_payload(sizes.meta_size, payload).map(Some)
    }

    /// Reads one PRPC frame from an async stream, returning `None` on normal connection close.
    pub(crate) async fn read_async(stream: &mut (impl AsyncRead + Unpin)) -> Result<Option<Self>> {
        let mut header = [0u8; PRPC_HEAD_SIZE];
        match stream.read_exact(&mut header).await {
            Ok(_) => {}
            Err(err)
                if matches!(
                    err.kind(),
                    io::ErrorKind::UnexpectedEof
                        | io::ErrorKind::ConnectionReset
                        | io::ErrorKind::ConnectionAborted
                ) =>
            {
                return Ok(None);
            }
            Err(err) => return Err(err).context("failed to read PRPC header"),
        }

        let sizes = FrameSizes::parse_header(&header)?;
        let payload = read_payload(stream, sizes.message_size).await?;
        Self::decode_payload(sizes.meta_size, payload).map(Some)
    }

    /// Writes this encoded PRPC frame to an async stream.
    pub(crate) async fn write_async(&self, stream: &mut (impl AsyncWrite + Unpin)) -> Result<()> {
        let bytes = self.encode();
        stream
            .write_all(&bytes)
            .await
            .context("failed to write PRPC response")?;
        stream
            .flush()
            .await
            .context("failed to flush PRPC response")?;
        Ok(())
    }

    /// Encodes this PRPC frame as `PRPC` magic, message size, metadata size, and body.
    pub(crate) fn encode(&self) -> Vec<u8> {
        let mut meta = self.meta.clone();
        meta.attachment_size = Some(self.attachment.len().min(i32::MAX as usize) as i32);
        let meta_bytes = meta.encode_to_vec();
        let message_size = meta_bytes.len() + self.body.len() + self.attachment.len();

        let mut bytes = Vec::with_capacity(PRPC_HEAD_SIZE + message_size);
        bytes.extend_from_slice(PRPC_MAGIC);
        bytes.extend_from_slice(&(message_size as i32).to_be_bytes());
        bytes.extend_from_slice(&(meta_bytes.len() as i32).to_be_bytes());
        bytes.extend_from_slice(&meta_bytes);
        bytes.extend_from_slice(&self.body);
        bytes.extend_from_slice(&self.attachment);
        bytes
    }

    /// Decodes a complete PRPC payload after the fixed header has been parsed.
    fn decode_payload(meta_size: usize, payload: Vec<u8>) -> Result<Self> {
        let message_size = payload.len();
        let meta =
            RpcMeta::decode(&payload[..meta_size]).context("failed to decode PRPC metadata")?;

        if meta.compress_type.unwrap_or(0) != 0 {
            return Err(anyhow!(
                "compressed PRPC payloads are not supported by the Rust CN"
            ));
        }
        if meta.chunk_info.is_some() {
            return Err(anyhow!(
                "chunked PRPC payloads are not supported by the Rust CN"
            ));
        }
        if meta.stream_settings.is_some() {
            return Err(anyhow!(
                "streaming PRPC payloads are not supported by the Rust CN"
            ));
        }

        let attachment_size = meta.attachment_size.unwrap_or(0);
        if attachment_size < 0 {
            return Err(anyhow!("negative PRPC attachment size"));
        }
        let attachment_size = attachment_size as usize;
        let body_size = message_size - meta_size;
        if attachment_size > body_size {
            return Err(anyhow!("PRPC attachment size exceeds payload size"));
        }

        let body_start = meta_size;
        let body_end = message_size - attachment_size;
        let body = payload[body_start..body_end].to_vec();
        let attachment = payload[body_end..].to_vec();

        Ok(Self {
            meta,
            body,
            attachment,
        })
    }
}

/// Reads exactly `len` bytes from an async stream, growing the buffer in bounded chunks so a
/// large declared size cannot force a full up-front allocation before the bytes arrive.
async fn read_payload(stream: &mut (impl AsyncRead + Unpin), len: usize) -> Result<Vec<u8>> {
    let mut payload = Vec::with_capacity(len.min(PRPC_READ_CHUNK));
    while payload.len() < len {
        let chunk = (len - payload.len()).min(PRPC_READ_CHUNK);
        let start = payload.len();
        payload.resize(start + chunk, 0);
        stream
            .read_exact(&mut payload[start..])
            .await
            .context("failed to read PRPC body")?;
    }
    Ok(payload)
}

/// Synchronous counterpart to [`read_payload`] used by the test frame reader.
#[cfg(test)]
fn read_payload_sync(stream: &mut impl Read, len: usize) -> Result<Vec<u8>> {
    let mut payload = Vec::with_capacity(len.min(PRPC_READ_CHUNK));
    while payload.len() < len {
        let chunk = (len - payload.len()).min(PRPC_READ_CHUNK);
        let start = payload.len();
        payload.resize(start + chunk, 0);
        stream
            .read_exact(&mut payload[start..])
            .context("failed to read PRPC body")?;
    }
    Ok(payload)
}

/// Decoded payload sizes from the fixed PRPC header.
#[derive(Clone, Copy, Debug)]
struct FrameSizes {
    /// Total payload bytes following the fixed 12-byte header.
    message_size: usize,
    /// Bytes occupied by the metadata protobuf at the start of the payload.
    meta_size: usize,
}

impl FrameSizes {
    /// Validates a PRPC header and extracts body sizes.
    fn parse_header(header: &[u8; PRPC_HEAD_SIZE]) -> Result<Self> {
        if &header[..4] != PRPC_MAGIC {
            return Err(anyhow!("invalid PRPC magic code"));
        }

        let message_size = i32::from_be_bytes(header[4..8].try_into().unwrap());
        let meta_size = i32::from_be_bytes(header[8..12].try_into().unwrap());
        if message_size < 0 || meta_size < 0 {
            return Err(anyhow!("negative PRPC message size"));
        }
        let message_size = message_size as usize;
        let meta_size = meta_size as usize;
        if message_size > MAX_PRPC_MESSAGE_SIZE {
            return Err(anyhow!("PRPC message exceeds maximum size"));
        }
        if meta_size > message_size {
            return Err(anyhow!("PRPC meta size exceeds message size"));
        }

        Ok(Self {
            message_size,
            meta_size,
        })
    }
}

#[cfg(test)]
mod tests {
    use std::{
        io::Write,
        net::{TcpListener, TcpStream},
        thread,
    };

    use super::*;

    #[test]
    fn frame_round_trip_preserves_attachment_and_correlation() {
        // StarRocks uses the PRPC correlation id to match responses, while
        // FE-to-CN plan fragments travel in the attachment bytes.
        let request = Frame::for_request(
            "PInternalService",
            "exec_plan_fragment",
            b"request body".to_vec(),
            b"thrift attachment".to_vec(),
            Some(42),
        );

        let bytes = request.encode();
        let server = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = server.local_addr().unwrap();
        let join = thread::spawn(move || {
            let (mut stream, _) = server.accept().unwrap();
            Frame::read(&mut stream).unwrap().unwrap()
        });
        let mut client = TcpStream::connect(addr).unwrap();
        client.write_all(&bytes).unwrap();
        let decoded = join.join().unwrap();
        assert_eq!(decoded.correlation_id(), Some(42));

        let decoded_request = decoded.into_request().unwrap();
        assert_eq!(decoded_request.attachment, b"thrift attachment");
        assert_eq!(decoded_request.body, b"request body");
    }
}
